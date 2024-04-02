import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from functools import partial


class Gaussian(eqx.Module):
    mean: jnp.ndarray
    scaling: jnp.ndarray
    rotation: jnp.ndarray
    colour: jnp.ndarray
    opacity: jnp.ndarray

    velocity: jnp.ndarray
    acceleration: jnp.ndarray

    def __init__(self, width=256., height=256., *, key=jax.random.PRNGKey(0)):
        keys = jax.random.split(key, 7)

        ## Uniformly initialise parameters of a 2D gaussian
        self.mean = jax.random.uniform(keys[0], (2,), minval=0, maxval=min(width, height))
        self.scaling = jax.random.uniform(keys[1], (2,), minval=0, maxval=min(width, height)/10)
        self.rotation = jax.random.uniform(keys[2], (1,), minval=0, maxval=2*jnp.pi)
        self.colour = jax.random.uniform(keys[3], (3,), minval=0, maxval=1)
        self.opacity = jax.random.uniform(keys[4], (1,), minval=0, maxval=1)

        ## Initialise velocity and acceleration
        self.velocity = jax.random.uniform(keys[5], (2,), minval=-1, maxval=1)
        self.acceleration = jax.random.uniform(keys[6], (2,), minval=-1, maxval=1)


    def make_rotation_matrix(self, angle):
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        return jnp.array([[cos, -sin], [sin, cos]]).squeeze()

    def get_covariance(self) -> jnp.ndarray:
        """Calculate the covariance matrix of the gaussian."""
        scaling_matrix = jnp.diag(self.scaling)
        rotation_matrix = self.make_rotation_matrix(self.rotation)
        covariance = rotation_matrix @ scaling_matrix @ scaling_matrix.T @ rotation_matrix.T 
        return covariance

    def get_density(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculate the density of the gaussian at a given point."""
        x_ = (x - self.mean)[:, None]
        res = jnp.exp(-0.5 * x_.T @ jnp.linalg.inv(self.get_covariance()) @ x_).squeeze()
        return res

    def render(self, x: jnp.ndarray) -> jnp.ndarray:
        """Render the gaussian at a given point."""
        density = self.get_density(x)
        return density * self.colour * self.opacity




class Gaussians(eqx.Module):
    gaussians: list[Gaussian]

    def __init__(self, num_gaussians=10, width=256., height=256., *, key=jax.random.PRNGKey(0)):
        keys = jax.random.split(key, num_gaussians)
        self.gaussians = [Gaussian(width, height, key=keys[i]) for i in range(num_gaussians)]

    def get_densities(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculate the densities of all gaussians at a given point."""
        return jnp.array([g.get_density(x) for g in self.gaussians])
        # return jax.tree_map(lambda g: g.get_density(x), self.gaussians)
    
    def render_pixel(self, x: jnp.ndarray) -> jnp.ndarray:
        """Render all gaussians at a given point."""
        densities = self.get_densities(x)
        return jnp.array([d * g.colour * g.opacity for d, g in zip(densities, self.gaussians)]).sum(axis=0)
        # return jnp.array([d * g.colour for d, g in zip(densities, self.gaussians)]).sum(axis=0)
        # return jax.tree_map(lambda d, g: d * g.colour * g.opacity, (densities, self.gaussians))

    def render_image(self, width: int, height: int) -> jnp.ndarray:
        """Render all gaussians into an image."""

        meshgrid = jnp.meshgrid(jnp.arange(0, width), jnp.arange(0, height))
        pixels = jnp.stack(meshgrid, axis=0).T

        return jax.vmap(jax.vmap(self.render_pixel))(pixels).squeeze()






def mae_loss(gaussians: Gaussians, ref_image: jnp.ndarray):
    """Calculate the MSE loss between the rendered image and the reference image."""
    image = gaussians.render_image(ref_image.shape[0], ref_image.shape[1])
    return jnp.mean(jnp.abs(image - ref_image))


def mse_loss(gaussians: Gaussians, ref_image: jnp.ndarray):
    """Calculate the MSE loss between the rendered image and the reference image."""
    image = gaussians.render_image(ref_image.shape[0], ref_image.shape[1])
    ## Add penalty for values greater than 1
    return jnp.mean((image - ref_image) ** 2)


@partial(jax.jit, static_argnums=(3,4))
def train_step(gaussians, ref_image, opt_state, optimiser, clip_bound=1e-2):
    """Perform a single training step."""

    loss, grad = jax.value_and_grad(mse_loss)(gaussians, ref_image)

    ## clip the gradients
    if clip_bound is not None:
        grad = jax.tree_map(lambda x: jnp.clip(x, -clip_bound, clip_bound), grad)

    updates, new_opt_state = optimiser.update(grad, opt_state)
    new_scene = optax.apply_updates(gaussians, updates)

    return new_scene, new_opt_state, loss