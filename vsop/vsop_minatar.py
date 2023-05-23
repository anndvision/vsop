import argparse
import json
from distutils.util import strtobool
from pathlib import Path
from typing import NamedTuple, Sequence

import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from gymnax.environments import spaces
from wrappers import (
    FlattenObservation,
    LogWrapper,
    VecEnv,
    NormalizeVecReward,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir",
        type=str,
        required=True,
        help="directory to write results",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="number of random repetitions",
    )
    parser.add_argument(
        "--debug",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    # Algorithm specific arguments
    parser.add_argument(
        "--env-id",
        type=str,
        default="Breakout-MinAtar",
        help="the id of the environment",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=int(1e7),
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=6.5e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=64,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="the discount factor gamma",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.9,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=16,
        help="the number of mini-batches",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=10,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="coefficient of the entropy",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="coefficient of the value function",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=2.4,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--normalize",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="normalize observations and rewards",
    )
    # Agent specific arguments
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="depth of neural network",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="hidden layer activation function",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.0,
        help="the dropout rate",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = {
        "LR": args.learning_rate,
        "ANNEAL_LR": args.anneal_lr,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "UPDATE_EPOCHS": args.update_epochs,
        "NUM_MINIBATCHES": args.num_minibatches,
        "GAMMA": args.gamma,
        "GAE_LAMBDA": args.gae_lambda,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": args.vf_coef,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "ENV_NAME": args.env_id,
        "HSIZE": args.width,
        "ACTIVATION": args.activation,
        "NORMALIZE": args.normalize,
        "DROPOUT_RATE": args.dropout_rate,
        "BACKEND": "positional",
        "SYMLOG_OBS": False,
        "CLIP_ACTION": True,
        "DEBUG": args.debug,
    }
    job_dir = Path(args.job_dir) / f"{args.env_id}" / "vsop"
    job_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(0)
    train_jv = jax.jit(jax.vmap(make_train(config=config)))
    rngs = jax.random.split(rng, args.num_seeds)
    out = train_jv(rngs)
    returns = out["metrics"]["return_info"][..., 1].mean(-1).reshape(args.num_seeds, -1)
    np.save(job_dir / "returns.npy", returns)

    with open(job_dir / "config.json", "w") as outfile:
        json.dump(config, outfile)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: dict

    @nn.compact
    def __call__(self, x):
        hsize = self.config["HSIZE"]
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        if self.config["DROPOUT_RATE"] > 0.0:
            actor_mean = nn.Dropout(
                rate=self.config["DROPOUT_RATE"], deterministic=False
            )(actor_mean)
        actor_mean = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        if self.config["DROPOUT_RATE"] > 0.0:
            actor_mean = nn.Dropout(
                rate=self.config["DROPOUT_RATE"], deterministic=False
            )(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        if self.config["CONTINUOUS"]:
            actor_logtstd = self.param(
                "log_std", nn.initializers.zeros, (self.action_dim,)
            )
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        if self.config["DROPOUT_RATE"] > 0.0:
            critic = nn.Dropout(rate=self.config["DROPOUT_RATE"], deterministic=False)(
                critic
            )
        critic = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        if self.config["DROPOUT_RATE"] > 0.0:
            critic = nn.Dropout(rate=self.config["DROPOUT_RATE"], deterministic=False)(
                critic
            )
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservation(env)
    env = LogWrapper(env)
    env = VecEnv(env)
    if config.get("NORMALIZE"):
        env = NormalizeVecReward(env, gamma=config["GAMMA"])

    config["CONTINUOUS"] = type(env.action_space(env_params)) == spaces.Box

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        if config["CONTINUOUS"]:
            network = ActorCritic(env.action_space(env_params).shape[0], config=config)
        else:
            network = ActorCritic(env.action_space(env_params).n, config=config)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        rng, _rng = jax.random.split(rng)
        if config["DROPOUT_RATE"] > 0.0:
            init_rngs = {
                "params": _rng,
                "dropout": _rng,
            }
        else:
            init_rngs = _rng
        network_params = network.init(init_rngs, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(
                    train_state.params,
                    last_obs,
                    rngs={
                        "dropout": jax.random.PRNGKey(np.random.randint(0, int(1e6)))
                    },
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )

                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(
                train_state.params,
                last_obs,
                rngs={"dropout": jax.random.PRNGKey(np.random.randint(0, int(1e6)))},
            )

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=8,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(
                            params,
                            traj_batch.obs,
                            rngs={
                                "dropout": jax.random.PRNGKey(
                                    np.random.randint(0, int(1e6))
                                )
                            },
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_loss = jnp.square(value - targets).mean()

                        # CALCULATE ACTOR LOSS
                        loss_actor = -(log_prob * nn.relu(gae)).mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )

                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            if config["DEBUG"]:
                metric = {}

                def callback(info):
                    stuff = info["return_info"][info["returned_episode"]]
                    print(stuff)

                jax.debug.callback(callback, traj_batch.info)
            else:
                metric = traj_batch.info

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    main()
