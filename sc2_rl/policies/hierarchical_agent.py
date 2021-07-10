import numpy as np
import torch
import gym
from typing import Dict, NamedTuple, List
from functools import partial
from itertools import chain
from stable_baselines3.a2c import A2C
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy


class PolicyTuple(NamedTuple):
    policy: torch.nn.Module
    observation_indices: List[int]
    action_indices: List[List[int]] # Main Game | [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7]]
                                    # Example 1 | [[_, 1, _, 3], [_, _, _, _, 4, 5, _, _]] : Corresponding actions in the main game
                                    # Example 2 | [[_, 1, 2, 3], None]
                                    # [0, 5, 3, 5, 7, 8]
                                    #  ____  ++++++++++
    optimizer: torch.optim.Optimizer


class HierarchicalPolicy(torch.nn.Module):
    """ 
    Structure:
        - The Tree structure consists of Multiple managers (each being a node in the tree) in the
        tree while each skill is a leaf in the tree.
        - A HierarchicalPolicy object represents a single node within the tree and the root node,
        which itself is a HierarchicalPolicy object, is the main policy of the agent.
        - We use HierarchicalPolicy recursively to construct the main policy. For example, the
        root node may contain one or more HierarchicalPolicy where each HierarchicalPolicy may
        contain its own sub policies, and hence the tree structure.
        - In order to make all the skills(and the managers) trainable we need to store their
        actions in the buffer. So that we can take their log probabilities.
        - We can use multi-discrete action space that contains the actions of all the nodes
        (either skills or HierarchicalPolicy) within a single axis.
        - SB3 constructs the buffer using the env's action_space, therefore, we need to modify
        the action_space of the env to be a multi-discrete action space before feeding it to SB3.
        - We also need to have an action concatenating and partitioning functions for storing
        the actions in the buffer and using the sampled actions in the training phase,
        respectively.
    """

    def __init__(self,
                 observation_space: gym.spaces,
                 action_space: gym.spaces,
                 lr_schedule: float,
                 use_sde: bool,
                 policy_tuples: List[PolicyTuple],
                 hidden_size: int = 128,
                 ortho_init: bool = True,
                 use_value_function: bool = True):
        super().__init__()
        self.observation_space = observation_space
        # self.action_space = action_space
        self.lr_schedule = lr_schedule
        self.policy_tuples = policy_tuples

        self.n_cores = len(self.policy_tuples)
        
        self.action_features = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),)
        self.action_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, self.n_cores),
        )
        if use_value_function:
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
                self.action_features: np.sqrt(2),
                self.action_layer: 0.01,
            }
            if use_value_function:
                module_gains[self.value_layer] = 1
                module_gains[self.value_features] = np.sqrt(2)

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
        
    def evaluate_actions(self, observation, actions):
        value, act_logits = self._forward(observation)
        act_dist = torch.distributions.Categorical(logits=act_logits)
        log_prob = act_dist.log_prob(actions)
        entropy = act_dist.entropy()
        return value, log_prob, entropy
    
    def _forward(self, observation):
        val_feature = self.value_features(observation)
        act_feature = self.action_features(observation)

        value = self.value_layer(val_feature)
        act_logits = self.action_layer(act_feature)
        return value, act_logits

    def get_self_action(self, observation):
        act_feature = self.action_features(observation)
        act_logit = self.action_layer(act_feature)
        # raise NotImplementedError

    def evaluate_actions(self, observation, actions):
        selected_leaf_actions, paths = self.partition(actions)
        log_prob, entropy = self.nodewise_calculate(observation, path[0])

        # actions -> predecessor_leaf_actions, paths, from line144


        pred_log_prob, pred_entropy = [sub_pi.policy.evaluate_actions(observation, [selected_leaf_actions, sub_path])
                                        for sub_path, sub_pi in zip(paths[1:], self.policy_tuples) 
                                        if isinstance(sub_pi, HierarchicalPolicy)]
        return log_prob + pred_log_prob, entropy + pred_entropy

    def get_leaf_actions(self, observation):
        child_leaf_actions = [sub_pi.policy(observation[:, sub_pi.observation_indices])
                              for sub_pi in self.policy_tuples 
                              if not isinstance(sub_pi, HierarchicalPolicy)]
        child_leaf_actions = [self.map_actions(action, sub_pi.action_indices) 
                              for action, sub_pi in zip(child_leaf_actions, self.policy_tuples)]
        predecessor_leaf_actions, paths = list(zip(*[sub_pi.get_leaf_actions()
                                                    for sub_pi in self.policy_tuples 
                                                    if isinstance(sub_pi, HierarchicalPolicy)]))
        self_action = self.get_self_action(observation)
        leaf_actions = list(zip(*chain(child_leaf_actions, predecessor_leaf_actions)))
        action_tensors = [torch.stack(leaf_action, dim=1) for leaf_action in leaf_actions]
        selected_leaf_action = [act_tensor.gather(self_action, dim=1) for act_tensor in action_tensors]

        paths = [self.action] + list(chain(paths))
        return selected_leaf_action, paths

    @staticmethod
    def map_actions(action, action_indices):
        if not isinstance(action in (list, tuple)):
            action = [action]
        multidiscrete_action = []
        act_index = 0
        for space_index in action_indices:
            if space_index is None:
                multidiscrete_action.append(torch.zeros_like(action[0]))
            else:
                multidiscrete_action.append(action[act_index])
                act_index += 1
        return multidiscrete_action

