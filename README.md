# vsop

[Implementation of ReLU to the Rescue: Improve Your On-Policy Actor-Critic with Positive Advantages](https://arxiv.org/abs/2306.01460)

```.txt
@article{jesson2023relu,
  title={ReLU to the Rescue: Improve Your On-Policy Actor-Critic with Positive Advantages},
  author={Jesson, Andrew and Lu, Chris and Gupta, Gunshi and Filos, Angelos and Foerster, Jakob N. and Gal, Yarin},
  journal={arXiv preprint arXiv:2306.01460},
  year={2023}
}
```

## installation

### torch

Install this if using gymnasium

```.sh
conda env create -f environment-torch.yml

conda activate vsop-torch

pip install -e .
```

### jax

Install this if using gymnax

```.sh
conda env create -f environment-jax.yml

conda activate vsop-jax

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/

python3 -m pip install --upgrade "jax[cuda]==0.4.11" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python3 -m pip install gymnax==0.0.6 brax==0.9.0 optax==0.1.5 distrax==0.1.2 dm-haiku==0.0.9 flax==0.6.10 mujoco==2.3.3 ray[tune]==2.4.0 bayesian-optimization==1.4.3 seaborn==0.12.2
```

## run

### gymnasium

#### mujoco

VSOP

```.sh
vsop-mujoco --gpu-per-rep 0.5 --track True --env-id HalfCheetah-v4
```

PPO

```.sh
ppo-mujoco --gpu-per-rep 0.5 --track True --env-id HalfCheetah-v4
```

A3C

```.sh
a3c-mujoco --gpu-per-rep 0.5 --track True --env-id HalfCheetah-v4
```

RMPG

```.sh
rmpg-mujoco --gpu-per-rep 0.5 --track True --env-id HalfCheetah-v4
```

Results can be viewed by logging into Weights and Biases

#### procgen

VSOP

```.sh
vsop-procgen --gpu-per-rep 1.0 --track True env-id starpilot
```

PPO

```.sh
ppo-procgen --gpu-per-rep 1.0 --track True env-id starpilot
```

RMPG

```.sh
rmpg-procgen --gpu-per-rep 1.0 --track True env-id starpilot
```

### gymnax

#### brax-mujoco

VSOP

```.sh
python3 vsop/vsop_mujoco_jax.py --job-dir output/ --env-id Brax-hopper
```

PPO

```.sh
python3 vsop/ppo_mujoco_jax.py --job-dir output/ --env-id Brax-hopper
```

A3C

```.sh
python3 vsop/a3c_mujoco_jax.py --job-dir output/ --env-id Brax-hopper
```

#### minatar

VSOP

```.sh
python3 vsop/vsop_minatar.py --job-dir output/ --env-id Breakout-MinAtar
```

PPO

```.sh
python3 vsop/ppo_minatar.py --job-dir output/ --env-id Breakout-MinAtar
```

A3C

```.sh
python3 vsop/a3c_minatar.py --job-dir output/ --env-id Breakout-MinAtar
```

#### classic control

VSOP

```.sh
python3 vsop/vsop_classic.py --job-dir output/ --env-id Pendulum-v1
```

PPO

```.sh
python3 vsop/ppo_classic.py --job-dir output/ --env-id Pendulum-v1
```

A3C

```.sh
python3 vsop/a3c_classic.py --job-dir output/ --env-id Pendulum-v1
```

results can be viewed by running `plotting.ipynb`

## tune example

Here is an example to run hyperparameter tuning for VSOP on classic control using.

```.sh
python3 vsop/tune_vsop_classic.py
```
