import torch
from stable_baselines3.a2c import A2C
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env


def make_env(n_envs: int,
             seed: int):
    env = make_vec_env(env_id=Miner,
                       n_envs=n_envs,
                       seed=seed,
                       vec_env_cls=SubprocVecEnv)
    return env

if __name__ == "__main__":
    model = A2C(policy=Policy,
        env=make_env(16))