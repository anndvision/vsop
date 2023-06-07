from setuptools import setup, find_packages

setup(
    name="vsop",
    version="0.0.0",
    description="",
    url="https://github.com/anndvision/vsop",
    author="Andrew Jesson and Chris Lu",
    author_email="andrew.jesson@cs.ox.ac.uk",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "ray==2.4.0",
        "six==1.16.0",
        "click==8.1.3",
        "pandas==2.0.1",
        "wandb==0.15.3",
        "seaborn==0.12.2",
        "shortuuid==1.0.11",
        "matplotlib==3.7.1",
        "tensorboard==2.13.0",
        "gymnasium[classic_control,atari,mujoco,accept-rom-license]==0.28.1",
        "mujoco==2.3.3",
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
