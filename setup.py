from setuptools import setup

setup(
   name='gideo',
   version='0.1.0',
   author='ddrous',
   author_email='desmond.ngueguin@gmail.com',
   packages=['gideo'],
   url='http://pypi.python.org/pypi/gideo/',
   license='LICENSE.md',
   description='Video codecs in JAX',
   long_description=open('README.md', encoding="utf-8").read(),
   install_requires=[
         "jax >= 0.3.4",
         "optax >= 0.1.1",
         "seaborn",
   ],
)