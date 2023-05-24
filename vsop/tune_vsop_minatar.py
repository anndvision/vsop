from pathlib import Path

import jax

import ray
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch


from vsop_minatar import make_train


config = {
    "ANNEAL_LR": True,
    "TOTAL_TIMESTEPS": int(1e7),
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "ACTIVATION": "relu",
    "DROPOUT_RATE": 0.0,
    "NORMALIZE": False,
    "BACKEND": "positional",
    "SYMLOG_OBS": False,
    "CLIP_ACTION": True,
    "DEBUG": False,
}


search_space = {
    "LR": tune.uniform(1e-4, 1e-3),
    "NUM_STEPS": tune.uniform(2, 8),
    "NUM_ENVS": tune.uniform(5, 8),
    "UPDATE_EPOCHS": tune.uniform(1, 10),
    "NUM_MINIBATCHES": tune.uniform(0, 8),
    "GAMMA": tune.uniform(0.9, 1.0),
    "GAE_LAMBDA": tune.uniform(0.9, 1.0),
    "MAX_GRAD_NORM": tune.uniform(0.0, 12.0),
    "HSIZE": tune.uniform(6, 10),
}

ray.init(
    num_gpus=1,
    num_cpus=1,
    dashboard_host="127.0.0.1",
    ignore_reinit_error=True,
)

algorithm = BayesOptSearch(
    space=search_space,
    metric="mean_loss",
    mode="min",
    random_search_steps=30,
)


def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)


def func(config):
    config["LR"] = round_to_multiple(config["LR"], 0.00005)
    config["NUM_STEPS"] = 2 ** int(config["NUM_STEPS"])
    config["NUM_ENVS"] = 2 ** int(config["NUM_ENVS"])
    config["UPDATE_EPOCHS"] = int(config["UPDATE_EPOCHS"])
    config["NUM_MINIBATCHES"] = 2 ** int(config["NUM_MINIBATCHES"])
    config["GAMMA"] = round_to_multiple(config["GAMMA"], 0.002)
    config["GAE_LAMBDA"] = round_to_multiple(config["GAE_LAMBDA"], 0.002)
    config["MAX_GRAD_NORM"] = round_to_multiple(config["MAX_GRAD_NORM"], 0.1)
    config["HSIZE"] = 2 ** int(config["HSIZE"])
    total_return = 0.0
    envs = [
        "Asterix-MinAtar",
        "Breakout-MinAtar",
        "Freeway-MinAtar",
        "SpaceInvaders-MinAtar",
    ]
    for env in envs:
        config["ENV_NAME"] = env
        num_seeds = 2
        rng = jax.random.PRNGKey(0)
        train_jv = jax.jit(jax.vmap(make_train(config=config)))
        rngs = jax.random.split(rng, num_seeds)
        out = train_jv(rngs)
        returns = out["metrics"]["return_info"][..., 1].mean(-1).reshape(num_seeds, -1)
        total_return += returns.mean(0)[-1]
    tune.report(mean_loss=-total_return)


job_dir = Path("/users/andson/workbench/repos/mi/output/tune_vsop/minatar")
job_dir.mkdir(parents=True, exist_ok=True)

analysis = tune.run(
    run_or_experiment=func,
    metric="mean_loss",
    mode="min",
    name="bayesopt",
    resources_per_trial={"gpu": 1, "cpu": 1},
    num_samples=100,
    search_alg=algorithm,
    local_dir=str(job_dir),
    config=config,
)
