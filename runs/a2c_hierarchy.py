import torch
import numpy as np
import argparse
from stable_baselines3.a2c import A2C
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from sc2_rl.envs.minier import MinierEnv
from sc2_rl.policies.dense import Policy


def run(args):
    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, 2**20)

    vecenv=make_vec_env(
        env_id=MinierEnv,
        n_envs=args.n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv)
    
    miner_policy = torch.load("path")
    military_policy = torch.load("path")

    model = A2C(
        policy=Policy,
        env=vecenv,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.val_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=args.log_dir,
        policy_kwargs=dict(
            miner_policy=OptimalMiner(),
            miner_observation_indices,
            miner_action_indices,
            military_policy=OptimalMilitary(),
            military_observation_indices,
            military_action_indices,
        ),
        verbose=1,
        seed=seed,
        device=args.device,
    )

    model.learn(
        args.total_timesteps,
        tb_log_name="SC2-minigame",
        log_interval=args.log_interval
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SC2-minigame")
    parser.add_argument("--n-envs", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")
    parser.add_argument("--hiddensize", type=int, default=128,
                        help="Hidden size of the policy")
    parser.add_argument("--n-steps", type=int, default=5,
                        help="Rollout Length")
    parser.add_argument("--gae-lambda", type=float, default=0.9,
                        help="GAE coefficient")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.995,
                        help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.05,
                        help="Entropy coefficient")
    parser.add_argument("--val-coef", type=float, default=0.25,
                        help="Value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum allowed graident norm")
    parser.add_argument("--total-timesteps", type=int, default=int(1e7),
                        help=("Training length interms of cumulative"
                              " environment timesteps"))

    parser.add_argument("--log-interval", type=int, default=2500,
                        help=("Logging interval in terms of training"
                              " iterations"))
    parser.add_argument("--log-dir", type=str, default=None,
                        help=("Logging dir"))

    args = parser.parse_args()

    run(args)
