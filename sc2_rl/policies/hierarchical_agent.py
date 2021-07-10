import numpy as np
import torch
import gym
from typing import Dict, NamedTuple, List
from functools import partial, chain 
from stable_baselines3.a2c import A2C
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy


class CoreSkillTuple(NamedTuple):
    policy: torch.nn.Module
    observation_indices: torch.Tensor
    action_map: torch.Tensor
    action_indices: List[int]
    optimizer: torch.optim.Optimizer
    


class HierarchicalPolicy(torch.nn.Module):

    def __init__(self,
                 observation_space: gym.spaces,
                 action_space: gym.spaces,
                 lr_schedule: float,
                 use_sde: bool,
                 core_policy_tuple: List[CoreSkillTuple],
                 hidden_size: int = 128,
                 ortho_init: bool = True):
        super().__init__()
        self.observation_space = observation_space
        # self.action_space = action_space
        self.lr_schedule = lr_schedule
        self.core_policy_tuple = core_policy_tuple

        self.n_cores = len(self.core_policy_tuple)
        
        self.action_features = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),)
        self.action_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, self.n_cores),
        )
        self.value_features = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        )
        self.value_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
        )

        if ortho_init:
            module_gains = {
                self.value_features: np.sqrt(2),
                self.action_features: np.sqrt(2),
                self.action_layer: 0.01,
                self.value_layer: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(ActorCriticCnnPolicy.init_weights, gain=gain))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))

    def forward(self, observation):
        value, act_logits = self._forward(observation)
        act_dist = torch.distributions.Categorical(logits=act_logits)
        action = act_dist.sample()
        log_prob = act_dist.log_prob(action)

        core_actions = chain([core_policy.action_map[core_policy.policy(observation[core_policy.observation_indices])]
                        for core_policy in self.core_policy_tuple])
        actions = [*core_actions, action]
        
        return actions, value, log_prob

    def _forward(self, observation):
        val_feature = self.value_features(observation)
        act_feature = self.action_features(observation)

        value = self.value_layer(val_feature)
        act_logits = self.action_layer(act_feature)
        return value, act_logits
        
    def evaluate_actions(self, observation, actions):
        value, act_logits = self._forward(observation)
        act_dist = torch.distributions.Categorical(logits=act_logits)
        log_prob = act_dist.log_prob(actions)
        entropy = act_dist.entropy()
        return value, log_prob, entropy
    
