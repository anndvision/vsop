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
        "wandb==0.15.3",
        "seaborn",
        "shortuuid==1.0.11",
        "matplotlib",
        "tensorboard==2.13.0",
        "gymnasium[classic_control,atari,mujoco,accept-rom-license]",
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
            "ppo-mujoco=vsop.ppo_mujoco:main",
            "vsop-mujoco=vsop.vsop_mujoco:main",
            "a3c-mujoco=vsop.a3c_mujoco:main",
            "rmpg-mujoco=vsop.rmpg_mujoco:main",
        ],
    },
)
