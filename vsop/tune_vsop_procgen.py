from pathlib import Path
from types import SimpleNamespace

import numpy as np
import ray
import torch
from ray import tune
from ray.air import RunConfig, session
from ray.tune import schedulers
from ray.tune.search import bohb
from vsop_procgen import run_experiment

base_config = SimpleNamespace(
    torch_deterministic=True,
    temp_dir="/scratch-ssd/ray/",
    cuda=True,
    track=False,
    total_timesteps=int(4e6),
    learning_rate=None,
    num_envs=64,
    num_steps=256,
    anneal_lr=False,
    num_weight_decay=16384,
    gamma=0.999,
    clip_coef=0.2,
    vf_coef=0.5,
    max_grad_norm=0.5,
    thompson=True,
    num_advantage_samples=1,
    spectral_norm=True,
    batch_size=int(64 * 256),
)


search_space = {
    "learning_rate": tune.qloguniform(1e-4, 1e-3, 0.00005),
    "gae_lambda": tune.quniform(0.5, 1.0, 0.001),
    "num_minibatches": tune.randint(2, 5),
    "update_epochs": tune.randint(1, 6),
    "ent_coef": tune.randint(2, 8),
    "dropout_rate": tune.quniform(0.0, 0.1, 0.005),
}

ray.init(
    num_gpus=torch.cuda.device_count(),
    num_cpus=torch.cuda.device_count() * 24,
    dashboard_host="127.0.0.1",
    ignore_reinit_error=True,
)

algorithm = bohb.TuneBOHB(
    search_space,
    metric="total_return",
    mode="max",
)
scheduler = schedulers.HyperBandForBOHB(
    max_t=1,
)


def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)


def func(config):
    config["num_minibatches"] = 2 ** int(config["num_minibatches"])
    config["ent_coef"] = 10 ** int(-config["ent_coef"])
    base_config.learning_rate = config["learning_rate"]
    base_config.gae_lambda = config["gae_lambda"]
    base_config.num_minibatches = config["num_minibatches"]
    base_config.update_epochs = config["update_epochs"]
    base_config.ent_coef = config["ent_coef"]
    base_config.dropout_rate = config["dropout_rate"]
    base_config.minibatch_size = int(
        base_config.batch_size // base_config.num_minibatches
    )
    total_return = 0.0
    envs = [
        "bossfight",
        "dodgeball",
    ]
    seeds = [1331]
    results = []
    for env in envs:
        for repetition in seeds:
            base_config.env_id = env
            results.append(
                run_experiment(
                    exp_name=f"vsop-{env}", args=base_config, seed=repetition
                )
            )
    total_return = np.mean(results)
    session.report({"total_return": total_return})


job_dir = Path().resolve() / Path(f"output/tune_vsop/procgen_2/")
job_dir.mkdir(parents=True, exist_ok=True)

tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(func), resources={"CPU": 6.0, "GPU": 0.25}
    ),
    tune_config=tune.TuneConfig(
        metric="total_return",
        mode="max",
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
