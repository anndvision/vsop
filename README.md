# vsop

Implementation of ReLU to the Rescue: Improve Your On-Policy Actor-Critic with Positive Advantages

## installation

```.sh
conda env create -f environment.yml

conda activate vsop

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/

pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -e .
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
