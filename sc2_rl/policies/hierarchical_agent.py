import numpy as np
import torch
import gym
from functools import partial
from stable_baselines3.a2c import A2C
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy


class HierarchicalPolicy(torch.nn.Module):

    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 use_sde,
                 miner_policy,
                 miner_observation_indices,
                 miner_action_indices,
                 military_policy,
                 military_observation_indices,
                 military_action_indices,
                 hidden_size=128,
                 ortho_init=True):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule
        
        self.action_features = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),)
        self.action_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, action_space.n),
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

        miner_actions = miner_policy(observations[miner_observation_indices])
        military_actions = military_policy(observations[military_observation_indices])
        actions = miner_actions_indices[miner_actions] * action + (1 - action) * military_actions_indices[military_actions]

        return action, value, log_prob

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
    
