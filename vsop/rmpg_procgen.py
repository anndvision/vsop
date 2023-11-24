# MIT License

# Copyright (c) 2019 CleanRL developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import os
import random
import shutil
import time
from distutils.util import strtobool
from typing import cast

import gym
import numpy as np
import ray
import shortuuid
import torch
import torch.nn as nn
import torch.optim as optim
from procgen import ProcgenEnv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def generate_id() -> str:
    # ~3t run ids (36**3)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return cast(str, run_gen.random(3))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="unique experiment id to be shared over different seeds",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=4,
        help="number of random repetitions",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--gpu-per-rep",
        type=float,
        default=1.0,
        help="number of gpus per repetition",
    )
    parser.add_argument(
        "--cpu-per-rep",
        type=int,
        default=1,
        help="number of cpus per repetition",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="cleanRL",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="/scratch-ssd/ray/",
        help="path to write temp ray files",
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--env-id",
        type=str,
        default="starpilot",
        help="the id of the environment",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=int(25e6),
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.5e-4,
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
        default=256,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--num-weight-decay",
        type=float,
        default=np.inf,
        help="Number of examples used to calculate weight decay",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="the discount factor gamma",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.797,
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
        default=5,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=1e-5,
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
        "--thompson",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="do thompson sampling",
    )
    parser.add_argument(
        "--num-advantage-samples",
        type=int,
        default=1,
        help="number of samples from value posterior to calculate advantages",
    )

    # Agent specific arguments
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.015,
        help="dropout rate",
    )
    parser.add_argument(
        "--spectral-norm",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="apply spectral normalization",
    )

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ConsistentDropout(torch.nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(ConsistentDropout, self).__init__()
        self.q = 1 - p
        self.inplace = inplace

    def forward(self, x, seed=None):
        if self.q == 1.0:
            return x
        if self.training:
            mask = torch.distributions.Bernoulli(probs=self.q).sample(
                torch.Size([1]) + x.shape[1:]
            ).to(x.device) / (self.q)
            return x * mask
        return x


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        )
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        dropout_rate=0.0,
        spectral_norm=False,
    ):
        super().__init__()
        self.n = envs.single_action_space.n
        h, w, c = envs.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        ll = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        if spectral_norm:
            ll = nn.utils.spectral_norm(ll)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            ConsistentDropout(p=dropout_rate),
            ll,
            nn.ReLU(),
            ConsistentDropout(p=dropout_rate),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(256, envs.single_action_space.n), std=1)

    def get_value(self, x):
        # TODO the simple q critic
        x = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.actor(x)
        policy = Categorical(logits=logits)
        return policy.probs, self.critic(x)

    def get_action(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
        )

    def get_action_and_value(self, x, action):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        value = torch.gather(self.critic(hidden), 1, action.long().unsqueeze(1))
        return (
            probs.probs,
            probs.log_prob(action),
            probs.entropy(),
            value,
        )


def make_envs(num_envs, env_id, num_levels, gamma, distribution_mode="easy"):
    envs = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_id,
        num_levels=num_levels,
        start_level=0,
        distribution_mode=distribution_mode,
    )
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    return envs


def main():
    args = parse_args()
    ro = ray.init(
        num_gpus=torch.cuda.device_count(),
        dashboard_host="127.0.0.1",
        ignore_reinit_error=True,
        _temp_dir=args.temp_dir,
    )

    @ray.remote(
        num_gpus=args.gpu_per_rep,
        num_cpus=args.cpu_per_rep,
    )
    def run(**kwargs):
        func = run_experiment(**kwargs)
        return func

    exp_name = f"rmpg-{args.experiment_id}"
    results = []
    for repetition in range(args.repetitions):
        results.append(run.remote(exp_name=exp_name, args=args, seed=repetition))
    results = ray.get(results)
    ray.shutdown()
    shutil.rmtree(ro["session_dir"], ignore_errors=True)


def run_experiment(exp_name, args, seed):
    run_name = f"{args.env_id}__{exp_name}__{seed}__{int(time.time())}"
    config = vars(args)
    config["exp_name"] = exp_name
    if args.track:
        import wandb

        wandb.init(
            project=args.env_id,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = make_envs(
        num_envs=args.num_envs, env_id=args.env_id, num_levels=200, gamma=args.gamma
    )
    envs_test = make_envs(
        num_envs=args.num_envs, env_id=args.env_id, num_levels=0, gamma=args.gamma
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(
        envs,
        dropout_rate=args.dropout_rate,
        spectral_norm=args.spectral_norm,
    ).to(device)
    if args.num_weight_decay < np.inf:
        optimizer = optim.AdamW(
            agent.parameters(),
            lr=args.learning_rate,
            weight_decay=(1 - args.dropout_rate) / (2 * args.num_weight_decay),
        )
    else:
        optimizer = optim.Adam(
            agent.parameters(),
            lr=args.learning_rate,
            eps=1e-5,
        )

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to("cpu")
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_obs_test = torch.Tensor(envs_test.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    test_returns = []

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        if not args.thompson:
            agent.eval()
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs.to("cpu")
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                agent.train()
                action, logprob, _ = agent.get_action(next_obs)
                agent.eval()
                action_test, _, _ = agent.get_action(next_obs_test)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            next_obs_test, _, _, info_test = envs_test.step(action_test.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done, next_obs_test = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(done).to(device),
                torch.Tensor(next_obs_test).to(device),
            )

            for item in info:
                if "episode" in item.keys():
                    writer.add_scalar(
                        "train/episodic_return", item["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "train/episodic_length", item["episode"]["l"], global_step
                    )
                    break

            for item in info_test:
                if "episode" in item.keys():
                    test_returns.append(item["episode"]["r"])
                    writer.add_scalar(
                        "test/episodic_return", item["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "test/episodic_length", item["episode"]["l"], global_step
                    )
                    break

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_advantages, b_returns = [], []
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.cat([b_obs, next_obs.to("cpu")], dim=0)
            ),
            batch_size=256,
            shuffle=False,
        )
        state = torch.get_rng_state()
        for sample_idx in range(args.num_advantage_samples):
            with torch.no_grad():
                values = []
                probs = []
                for batch in dl:
                    torch.manual_seed(sample_idx)
                    p, v = agent.get_value(batch[0].to(device))
                    values.append(v)
                    probs.append(p)
                values = torch.cat(values, dim=0)
                probs = torch.cat(probs, dim=0)
                state_values = (probs * values).sum(-1, keepdims=True)
                advantages = values - state_values
                state_values = state_values.reshape(args.num_steps + 1, -1)
                next_value = state_values[-1:]
                state_values = state_values[:-1]
                returns = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = state_values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal
                    returns[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                b_advantages.append(advantages.unsqueeze(0))
                b_returns.append(returns.unsqueeze(0))
        torch.set_rng_state(state)

        # flatten the batch
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = torch.cat(b_advantages, dim=0).mean(0)
        b_returns = torch.cat(b_returns, dim=0).mean(0).reshape(-1)
        b_values = state_values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        agent.train()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                probs, _, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds].to(device), b_actions[mb_inds]
                )

                mb_advantages = b_advantages[mb_inds]

                # Policy loss
                pg_loss = (
                    -(torch.nn.functional.relu(mb_advantages) * probs).sum(-1).mean()
                )

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm < torch.inf:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()

    return np.mean(test_returns[-100:])


if __name__ == "__main__":
    main()
