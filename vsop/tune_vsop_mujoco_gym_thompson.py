from pathlib import Path

import jax
import ray
from ray import tune
from ray.air import session, RunConfig
from ray.tune.search import bohb
from ray.tune import schedulers
from vsop_mujoco_jax_ import make_train

base_config = {
    "ANNEAL_LR": True,
    "NUM_ENVS": 256,
    "NUM_STEPS": 8,
    "GAMMA": 0.99,
    "TOTAL_TIMESTEPS": int(1e6),
    "ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "ACTIVATION": "relu",
    "HSIZE": 256,
    "NORMALIZE": True,
    "SPECTRAL_NORM": True,
    "SN_COEF": 1.0,
    "THOMPSON": False,
    "NUM_WEIGHT_DECAY": 2048.0,
    "BACKEND": "positional",
    "CLIP_ACTION": True,
    "DEBUG": False,
}

search_space = {
    "LR": tune.qloguniform(1e-4, 1e-3, 0.00005),
    "GAE_LAMBDA": tune.quniform(0.5, 1.0, 0.01),
    "NUM_MINIBATCHES": tune.randint(0, 8),
    "UPDATE_EPOCHS": tune.randint(1, 11),
    "MAX_GRAD_NORM": tune.quniform(0.1, 10.0, 0.02),
    "DROPOUT_RATE": tune.quniform(0.0, 0.1, 0.005),
}

ray.init(
    num_gpus=4,
    num_cpus=8,
    dashboard_host="127.0.0.1",
    ignore_reinit_error=True,
)

algorithm = bohb.TuneBOHB(
    search_space,
    metric="mean_loss",
    mode="min",
)
scheduler = schedulers.HyperBandForBOHB(
    max_t=1,
)


def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)


def func(config):
    config["NUM_MINIBATCHES"] = 2 ** int(config["NUM_MINIBATCHES"])
    base_config.update(config)
    total_return = 0.0
    envs = [
        "Brax-reacher",
        "Brax-hopper",
        "Brax-humanoid",
    ]
    for env in envs:
        base_config["ENV_NAME"] = env
        num_seeds = 2
        rng = jax.random.PRNGKey(0)
        train_jv = jax.jit(jax.vmap(make_train(config=base_config)))
        rngs = jax.random.split(rng, num_seeds)
        out = train_jv(rngs)
        returns = out["metrics"]["return_info"][..., 1].mean(-1).reshape(num_seeds, -1)
        total_return += returns[:, -5:].mean()
    session.report(
        {
            "mean_loss": -total_return,
            "total_return": total_return,
        }
    )


num_envs = base_config["NUM_ENVS"]
job_dir = Path().resolve() / Path(f"output/tune_vsop/gym-thompson/{num_envs}/")
job_dir.mkdir(parents=True, exist_ok=True)

tuner = tune.Tuner(
    tune.with_resources(tune.with_parameters(func), resources={"cpu": 2, "gpu": 1}),
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algorithm,
        scheduler=scheduler,
        num_samples=100,
    ),
    run_config=RunConfig(
        name="bohb",
        local_dir=str(job_dir),
    ),
)
results = tuner.fit()
