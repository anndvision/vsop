from setuptools import setup, find_packages

setup(
    name="vsop",
    version="0.0.0",
    description="",
    url="",
    author="Anonymized",
    author_email="",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "ray",
        "six",
        "click",
        "pandas",
        "wandb",
        "seaborn",
        "shortuuid",
        "matplotlib",
        "tensorboard",
        "gymnasium[classic_control,atari,mujoco,accept-rom-license]",
        "jax[cuda]"
        "gymnax",
        "brax",
        "optax",
        "distrax==0.1.2",
        "dm-haiku",
        "flax",
        "mujoco==2.3.3",
        "bayesian-optimization",
    ],
    entry_points={
        "console_scripts": [

        ],
    },
)
