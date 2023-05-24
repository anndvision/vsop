import argparse
import json
from distutils.util import strtobool
from pathlib import Path
from typing import NamedTuple, Optional

import distrax
import flax.linen as nn
import gymnax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from gymnax.environments import spaces
from wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    FlattenObservation,
    LogWrapper,
    TransformObservation,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
)
from spectral_norm import SNParamsTree


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
        default=2,
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
        default=5e-3,
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
        default=128,
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
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=8,
        help="the number of mini-batches",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
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
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--spectral-norm",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles whether or not to use spectral normalization.",
    )
    parser.add_argument(
        "--sn-coef",
        type=float,
        default=1.0,
        help="the lipschitz constant",
    )
    parser.add_argument(
        "--symlog",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles whether or not to use symlog observation normalization.",
    )
    parser.add_argument(
        "--normalize",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="normalize observations and rewards",
    )
    # Agent specific arguments
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="depth of neural network",
    )
    parser.add_argument(
        "--num-weight-decay",
        type=int,
        default=np.inf,
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
        "SPECTRAL_NORM": args.spectral_norm,
        "SN_COEF": args.sn_coef,
        "DROPOUT_RATE": args.dropout_rate,
        "NUM_WEIGHT_DECAY": args.num_weight_decay,
        "BACKEND": "positional",
        "SYMLOG_OBS": args.symlog,
        "CLIP_ACTION": True,
        "DEBUG": args.debug,
    }
    job_dir = Path(args.job_dir) / f"{args.env_id}" / "vsop_sn"
    job_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(0)
    train_jv = jax.jit(jax.vmap(make_train(config=config)))
    rngs = jax.random.split(rng, args.num_seeds)
    out = train_jv(rngs)
    returns = out["metrics"]["return_info"][..., 1].mean(-1).reshape(args.num_seeds, -1)
    np.save(job_dir / "returns.npy", returns)

    with open(job_dir / "config.json", "w") as outfile:
        json.dump(config, outfile)


class ActorCritic(hk.Module):
    """Network head that produces a categorical distribution and value."""

    def __init__(
        self,
        config: dict,
        act_size: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.config = config
        self.act_size = act_size
        hsize = config["HSIZE"]
        if config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        self._action_body = hk.nets.MLP(
            [hsize, hsize],
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
            activate_final=True,
            activation=activation,
        )

        self._logit_layer = hk.Linear(
            act_size,
            w_init=hk.initializers.Orthogonal(0.01),
            b_init=hk.initializers.Constant(0),
        )

        self._value_body = hk.nets.MLP(
            [hsize, hsize],
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
            activate_final=True,
            activation=activation,
        )

        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1),
            b_init=hk.initializers.Constant(0),
        )

    def __call__(self, inputs: jnp.ndarray, dropout_rate, rng):
        rng_act, rng_critic = jax.random.split(rng)
        actor_mean = self._action_body(inputs, dropout_rate=dropout_rate, rng=rng_act)
        actor_mean = self._logit_layer(actor_mean)
        if self.config["CONTINUOUS"]:
            _log_std = hk.get_parameter("log_std", (self.act_size,), init=jnp.zeros)
            pi = (actor_mean, jnp.exp(_log_std))
        else:
            pi = actor_mean

        value = self._value_body(inputs, dropout_rate=dropout_rate, rng=rng_critic)
        value = jnp.squeeze(self._value_layer(value), axis=-1)
        return pi, value


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
    if "Brax-" in config["ENV_NAME"]:
        name = config["ENV_NAME"].split("Brax-")[1]
        env, env_params = (
            BraxGymnaxWrapper(name, backend=config.get("BACKEND", "generalized")),
            None,
        )
        env = LogWrapper(env)
        if config.get("SYMLOG_OBS"):
            env = TransformObservation(env, transform_obs=symlog)
        if config.get("CLIP_ACTION", True):
            env = ClipAction(env)
        env = VecEnv(env)
        if config.get("NORMALIZE"):
            env = NormalizeVecObservation(env)
            env = NormalizeVecReward(env, gamma=config["GAMMA"])
    else:
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
        def run_ac(input, rng):
            if config["CONTINUOUS"]:
                act_size = env.action_space(env_params).shape[0]
            else:
                act_size = env.action_space(env_params).n
            ac = ActorCritic(config=config, act_size=act_size)
            return ac(input, config["DROPOUT_RATE"], rng)

        net_init, net_apply = hk.transform(run_ac)

        init_x = jnp.zeros(env.observation_space(env_params).shape)
        rng, _rng_net, _rng_sn, _rng_d = jax.random.split(rng, 4)
        network_params = net_init(_rng_net, init_x, _rng_d)

        # We only apply spectralnorm to the weight matrices, not the biases. Also not on the last layer.
        sn_fn = hk.transform_with_state(
            lambda x: hk.SNParamsTree(eps=1e-12, ignore_regex="^(?!.*\/mlp[^b]*w).+$")(
                x, update_stats=True
            )
        )
        _, sn_state = sn_fn.init(_rng_sn, network_params)

        learning_rate = linear_schedule if config["ANNEAL_LR"] else config["LR"]
        if config["NUM_WEIGHT_DECAY"] < np.inf:
            weight_decay = (1 - config["DROPOUT_RATE"]) / (
                2 * config["NUM_WEIGHT_DECAY"]
            )
            tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        else:
            tx = optax.adam(learning_rate=learning_rate, eps=1e-5)

        if config["MAX_GRAD_NORM"] < np.inf:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), tx)
        train_state = TrainState.create(
            apply_fn=net_apply,
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
                train_state, sn_state, env_state, last_obs, rng = runner_state

                # RENAME THE RETURNED SN_STATE TO "_" TO NOT UPDATE ON INFERENCE
                if config["SPECTRAL_NORM"]:
                    sn_params, sn_state = sn_fn.apply(
                        None, sn_state, None, train_state.params
                    )
                else:
                    sn_params = train_state.params

                # SELECT ACTION
                rng, _rng, _rng_d = jax.random.split(rng, 3)
                pi, value = net_apply(sn_params, None, last_obs, _rng_d)
                if config["CONTINUOUS"]:
                    pi = distrax.MultivariateNormalDiag(pi[0], pi[1])
                else:
                    pi = distrax.Categorical(logits=pi)
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
                runner_state = (train_state, sn_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, sn_state, env_state, last_obs, rng = runner_state
            if config["SPECTRAL_NORM"]:
                sn_params, sn_state = sn_fn.apply(
                    None, sn_state, None, train_state.params
                )
            else:
                sn_params = train_state.params

            # RECALCULATE VALUES
            rng, _rng = jax.random.split(rng)
            all_obs = jnp.concatenate([traj_batch.obs, last_obs[None]], axis=0)

            # THIS VMAPPING IS TO MAKE THE RNG THE SAME FOR DROPOUT WITHIN THE ENVS
            # NOTE THAT WE COULD USE THE HAIKU RNG INPUT, WHICH IS WHY THERE IS A RANDOM NONE
            # NOTE ALSO THAT THIS IS USING THE SAME RNG WITHIN THE ENVS, BUT NOT ACROSS THEM
            _, vals = jax.vmap(net_apply, in_axes=(None, None, 0, None))(
                sn_params, None, all_obs, _rng
            )
            traj_batch = traj_batch._replace(value=vals[:-1])
            last_val = vals[-1]

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
                    train_state, sn_state, rng = train_state
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, sn_state, traj_batch, gae, targets, rng):
                        # RERUN NETWORK
                        if config["SPECTRAL_NORM"]:
                            sn_params, sn_state = sn_fn.apply(
                                None, sn_state, None, params
                            )
                        else:
                            sn_params = params
                        pi, value = jax.vmap(net_apply, in_axes=(None, None, 0, None))(
                            sn_params, None, traj_batch.obs, rng
                        )
                        if config["CONTINUOUS"]:
                            pi = distrax.MultivariateNormalDiag(pi[0], pi[1])
                        else:
                            pi = distrax.Categorical(logits=pi)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        # value_pred_clipped = traj_batch.value + (
                        #     value - traj_batch.value
                        # ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        # value_losses = jnp.square(value - targets)
                        # value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        # value_loss = (
                        #     0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        # )
                        value_loss = jnp.square(value - targets).mean()

                        # CALCULATE ACTOR LOSS
                        loss_actor = -(log_prob * nn.relu(gae)).mean()

                        # ENTROPY
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )

                        return total_loss, sn_state

                    rng, _rng = jax.random.split(rng)
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        sn_state,
                        traj_batch,
                        advantages,
                        targets,
                        _rng,
                    )
                    total_loss, sn_state = total_loss
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = (train_state, sn_state, rng)
                    return train_state, total_loss

                (
                    train_state,
                    sn_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
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
                rng, _rng = jax.random.split(rng)
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, (train_state, sn_state, _rng), minibatches
                )
                train_state, sn_state, rng = train_state
                update_state = (
                    train_state,
                    sn_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (train_state, sn_state, traj_batch, advantages, targets, rng)
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

            runner_state = (train_state, sn_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, sn_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    main()
