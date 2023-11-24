from pathlib import Path
from types import SimpleNamespace

import numpy as np
import ray
from ray import tune
from ray.air import RunConfig, session
from ray.tune import schedulers
from ray.tune.search import bohb
from rmpg_mujoco import run_experiment

base_config = SimpleNamespace(
    torch_deterministic=True,
    cuda=True,
    track=False,
    wandb_entity="vsop",
    capture_video=False,
    temp_dir="/scratch-ssd/ray/",
    total_timesteps=int(1e6),
    num_envs=4,
    num_steps=512,
    anneal_lr=True,
    num_weight_decay=4 * 512,
    gamma=0.99,
    clip_coef=0.2,
    clip_vloss=False,
    ent_coef=0.0,
    vf_coef=1.0,
    norm_observations=True,
    norm_rewards=True,
    thompson=True,
    num_advantage_samples=1,
    mc_samples=100,
    dim_hidden=256,
    activation="relu",
    spectral_norm=True,
    orthogonal=True,
    batch_size=int(4 * 512),
)


search_space = {
    "learning_rate": tune.qloguniform(1e-4, 1e-3, 0.00005),
    "gae_lambda": tune.quniform(0.5, 1.0, 0.001),
    "num_minibatches": tune.randint(0, 8),
    "update_epochs": tune.randint(1, 11),
    "max_grad_norm": tune.quniform(0.1, 10.0, 0.02),
    "dropout_rate": tune.quniform(0.0, 0.1, 0.005),
}

ray.init(
    num_gpus=4,
    num_cpus=4 * 8,
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
    base_config.learning_rate = config["learning_rate"]
    base_config.gae_lambda = config["gae_lambda"]
    base_config.num_minibatches = config["num_minibatches"]
    base_config.update_epochs = config["update_epochs"]
    base_config.max_grad_norm = config["max_grad_norm"]
    base_config.dropout_rate = config["dropout_rate"]
    base_config.minibatch_size = int(
        base_config.batch_size // base_config.num_minibatches
    )
    total_return = 0.0
    envs = [
        "Reacher-v4",
        "Hopper-v4",
        "Humanoid-v4",
    ]
    seeds = [1331]
    results = []
    for env in envs:
        for repetition in seeds:
            base_config.env_id = env
            results.append(
                run_experiment(
                    exp_name=f"rmpg-{env}", args=base_config, seed=repetition
                )
            )
    total_return = np.mean(results)
    session.report(
        {
            "mean_loss": -total_return,
            "total_return": total_return,
        }
    )


job_dir = Path().resolve() / Path(f"output/tune_rmpg/mujoco/{base_config.num_envs}/")
job_dir.mkdir(parents=True, exist_ok=True)

tuner = tune.Tuner(
    tune.with_resources(tune.with_parameters(func), resources={"CPU": 2, "GPU": 1}),
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
