from pathlib import Path

import jax

import ray
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch


from dpo_classic import make_train


config = {
    "ANNEAL_LR": True,
    "GAMMA": 0.99,
    "TOTAL_TIMESTEPS": int(5e5),
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "CLIP_VLOSS": False,
    "ACTIVATION": "tanh",
    "HSIZE": 64,
    "NORMALIZE": False,
    "BACKEND": "positional",
    "SYMLOG_OBS": False,
    "CLIP_ACTION": True,
    "DEBUG": False,
}


search_space = {
    "LR": tune.uniform(1e-4, 1e-3),
    "NUM_STEPS": tune.uniform(2, 8),
    "NUM_ENVS": tune.uniform(2, 8),
    "UPDATE_EPOCHS": tune.uniform(1, 10),
    "NUM_MINIBATCHES": tune.uniform(0, 5),
    "GAE_LAMBDA": tune.uniform(0.0, 1.0),
    "MAX_GRAD_NORM": tune.uniform(0.0, 5.0),
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
    config["GAE_LAMBDA"] = round_to_multiple(config["GAE_LAMBDA"], 0.002)
    config["MAX_GRAD_NORM"] = round_to_multiple(config["MAX_GRAD_NORM"], 0.1)
    total_return = 0.0
    envs = [
        "CartPole-v1",
        "Acrobot-v1",
        "Pendulum-v1",
        "MountainCarContinuous-v0",
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


job_dir = Path().resolve() / Path("output/tune_dpo/classic")
job_dir.mkdir(parents=True, exist_ok=True)

analysis = tune.run(
    run_or_experiment=func,
    metric="mean_loss",
    mode="min",
    name="bayesopt",
    resources_per_trial={"gpu": 1, "cpu": 1},
    num_samples=200,
    search_alg=algorithm,
    local_dir=str(job_dir),
    config=config,
)
