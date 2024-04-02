
#%%
from gideo import *
import time
import matplotlib.pyplot as plt


import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
jax.config.update("jax_enable_x64", True)

import optax
from tqdm import tqdm


# key = jax.random.PRNGKey(42)
key = jax.random.PRNGKey(time.time_ns())
# key = None

ref_image = plt.imread('earth.jpeg')[...,:3]/255.

width, height = ref_image.shape[:2]
model = Gaussians(250, width, height, key=key)

image = model.render_image(width, height)

fig, (ax) = plt.subplots(1, 2)
sbimshow(image, title="Random init", ax=ax[0])
sbimshow(ref_image, title="Reference", ax=ax[1]);

#%%

nb_iter = 250
scheduler = optax.exponential_decay(1e-1, nb_iter, 0.8)
optimiser = optax.adam(scheduler)
opt_state = optimiser.init(model)

losses = []
start_time = time.time()
for i in tqdm(range(1, nb_iter+1), disable=True):
    model, opt_state, loss = train_step(model, ref_image, opt_state, optimiser, clip_bound=None)
    losses.append(loss)
    if i % 10 == 0 or i <= 3:
        print(f'Iteration: {i}        Loss: {loss:.3f}')
wall_time = time.time() - start_time

## Number of params in model
print("\nNumber of params:", jnp.size(model))
print("Number of pixels:", jnp.size(ref_image))
print("Wall time in h:m:s:", time.strftime("%H:%M:%S", time.gmtime(wall_time)))

image = model.render_image(width, height)

fig, (ax) = plt.subplots(1, 2)
sbimshow(image, title="Final render", ax=ax[0])
sbimshow(ref_image, title="Reference", ax=ax[1])


sbplot(losses, title="MAE loss history", y_scale='log', x_label="Iteration");

#%%
## Save the image
plt.imsave('earth_reconstructed.png', image)