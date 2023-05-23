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

import gymnasium
import numpy as np
import ray
import shortuuid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def generate_id() -> str:
    # ~3t run ids (36**3)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return cast(str, run_gen.random(3))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
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
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)",
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
        default="HalfCheetah-v4",
        help="the id of the environment",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=3000000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2048,
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
        default=32,
        help="the number of mini-batches",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=10,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
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
        "--target-kl",
        type=float,
        default=None,
        help="the target KL divergence threshold",
    )
    parser.add_argument(
        "--norm-observations",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="normalize observations",
    )
    parser.add_argument(
        "--norm-rewards",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="normalize rewards",
    )

    # Agent specific arguments
    parser.add_argument(
        "--dim-hidden",
        type=int,
        default=64,
        help="width of neural network",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="tanh",
        help="non-linearity of neural network",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.0,
        help="dropout rate",
    )
    parser.add_argument(
        "--spectral-norm",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="apply spectral normalization",
    )
    parser.add_argument(
        "--orthogonal",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="orthogonal weight initialization",
    )

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def make_env(
    env_id,
    idx,
    capture_video,
    run_name,
):
    def thunk():
        if capture_video:
            env = gymnasium.make(env_id, render_mode="rgb_array")
        else:
            env = gymnasium.make(env_id)
        env = gymnasium.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gymnasium.wrappers.ClipAction(env)
        return env

    return thunk


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


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        dim_hidden=64,
        activation="tanh",
        dropout_rate=0.0,
        spectral_norm=False,
        orthogonal=True,
    ):
        super().__init__()
        cl1 = nn.Linear(
            in_features=np.array(envs.single_observation_space.shape).prod(),
            out_features=dim_hidden,
        )
        cl2 = nn.Linear(in_features=dim_hidden, out_features=dim_hidden)
        al1 = nn.Linear(
            in_features=np.array(envs.single_observation_space.shape).prod(),
            out_features=dim_hidden,
        )
        al2 = nn.Linear(in_features=dim_hidden, out_features=dim_hidden)
        if orthogonal:
            cl1 = layer_init(cl1)
            cl2 = layer_init(cl2)
            al1 = layer_init(al1)
            al2 = layer_init(al2)
        if spectral_norm:
            cl1 = nn.utils.spectral_norm(cl1)
            cl2 = nn.utils.spectral_norm(cl2)
            al1 = nn.utils.spectral_norm(al1)
            al2 = nn.utils.spectral_norm(al2)
        if activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "relu":
            act_fn = nn.ReLU
        else:
            raise NotImplementedError("Unsupported activation function")
        self.critic = nn.Sequential(
            cl1,
            act_fn(),
            ConsistentDropout(p=dropout_rate),
            cl2,
            act_fn(),
            ConsistentDropout(p=dropout_rate),
            layer_init(nn.Linear(dim_hidden, 1), std=1.0)
            if orthogonal
            else nn.Linear(dim_hidden, 1),
        )
        self.actor_mean = nn.Sequential(
            al1,
            act_fn(),
            ConsistentDropout(p=dropout_rate),
            al2,
            act_fn(),
            ConsistentDropout(p=dropout_rate),
            layer_init(
                nn.Linear(dim_hidden, np.prod(envs.single_action_space.shape)),
                std=0.01,
            )
            if orthogonal
            else nn.Linear(dim_hidden, np.prod(envs.single_action_space.shape)),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
        )

    def get_action_and_value(self, x, action=None):
        action, logprob, entropy = self.get_action(x, action)
        return (
            action,
            logprob,
            entropy,
            self.get_value(x),
        )


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

    exp_name = f"ppo-{generate_id()}"
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

    # env setup
    envs = gymnasium.vector.SyncVectorEnv(
        [
            make_env(
                env_id=args.env_id,
                idx=i,
                capture_video=args.capture_video,
                run_name=run_name,
            )
            for i in range(args.num_envs)
        ]
    )
    if args.norm_observations:
        envs = gymnasium.wrappers.NormalizeObservation(envs)
        envs = gymnasium.wrappers.TransformObservation(
            envs, lambda obs: np.clip(obs, -10, 10)
        )
    if args.norm_rewards:
        envs = gymnasium.wrappers.NormalizeReward(envs, gamma=args.gamma)
        envs = gymnasium.wrappers.TransformReward(
            envs, lambda reward: np.clip(reward, -10, 10)
        )
    assert isinstance(
        envs.single_action_space, gymnasium.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(
        envs,
        dim_hidden=args.dim_hidden,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        spectral_norm=args.spectral_norm,
        orthogonal=args.orthogonal,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)

            # Only print when at least 1 env is done
            if "final_info" in infos:
                for info in infos["final_info"]:
                    # Skip the envs that are not done
                    if info is not None:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm < torch.inf:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

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
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        if args.track and args.capture_video:
            for filename in os.listdir(f"videos/{run_name}"):
                if filename not in video_filenames and filename.endswith(".mp4"):
                    wandb.log({f"videos": wandb.Video(f"videos/{run_name}/{filename}")})
                    video_filenames.add(filename)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
