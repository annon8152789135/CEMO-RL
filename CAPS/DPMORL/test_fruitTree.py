import re
import sys
import gym
import numpy as np

import matplotlib
matplotlib.rcParams.update({'font.size': 22})

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from CAPS.DPMORL.utils import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import matplotlib.pyplot as plt
import mo_gymnasium
import gymnasium
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from CAPS.DPMORL.MORL_stablebaselines3.envs.wrappers.utility_env_wrapper import MultiEnv_UtilityFunction, ObsInfoWrapper
from CAPS.DPMORL.MORL_stablebaselines3.envs.wrappers.scalar_reward_wrapper import ScalarRewardEnv
from CAPS.DPMORL.MORL_stablebaselines3.utility_function.utility_function_parameterized import Utility_Function_Parameterized
from CAPS.DPMORL.MORL_stablebaselines3.utility_function.utility_function_programmed import Utility_Function_Programmed
from CAPS.DPMORL.MORL_stablebaselines3.utility_function.utility_function_programmed import Utility_Function_Linear
from CAPS.DPMORL.MORL_stablebaselines3.utility_function.utility_function_programmed import Utility_Function_Diverse_Goal
from gymnasium.spaces import Discrete
import math
from os import path
from typing import Callable, List, Dict, Tuple
import torch
from gym import spaces
from gym.utils import seeding
import time
import os
import argparse
from datetime import date
import glob

def make_eval_env(gym_id_name, utility_function, reward_shape, reward_dim_indices, seed=0, augment_state=False):
    depth_holder = {}
    def _env_fn():
        env = mo_gymnasium.make(gym_id_name)
        env.name = gym_id_name
        base = env.unwrapped
        depth_holder["tree_depth"] = (
                getattr(base, "tree_depth", None)
                or getattr(base, "depth", None)
                or getattr(base, "max_depth", None)
                or getattr(base, "horizon", None)
            )
        env = ObsInfoWrapper(env, reward_dim=reward_shape, reward_dim_indices=reward_dim_indices)
        return env
    # single env; can scale up if needed
    vec = DummyVecEnv([_env_fn], reward_dim=reward_shape)
    vec = MultiEnv_UtilityFunction(vec, utility_function, reward_dim=reward_shape, augment_state=augment_state)
    vec.update_utility_function(utility_function)
    
    #print("Tree Depth: ", depth_holder["tree_depth"])

    return vec, depth_holder["tree_depth"]

def get_reward_dim(env):
    try:
        reward_dim = env.reward_dim
    except:
        env = getattr(env, "unwrapped", env)
        reward_dim = env.reward_space.shape
    return reward_dim

def get_utility_function(reward_shape, idx=0,linear_utility=True, lamda=0.1, keep_scale= True, max_num_policies=6):
    
    print("reward_shape in get_utility_function: ", reward_shape)

    if linear_utility:
        utility_class_programmed = Utility_Function_Linear
    else:
        utility_class_programmed = Utility_Function_Programmed
    
    norm=True
    
    # Load pretrained utility functions
    assert os.path.isdir(f'CAPS/DPMORL/utility-model-selected/dim-{reward_shape}'), 'There is no pretrained utility functions provided. '
    num_pretrained_utility = len(glob.glob(f'CAPS/DPMORL/utility-model-selected/dim-{reward_shape}/*'))
    pretrained_utility_paths = [f'CAPS/DPMORL/utility-model-selected/dim-{reward_shape}/utility-{i}.pt'
                                for i in range(num_pretrained_utility)]
    
    pretrained_utility_functions = []
    for path in pretrained_utility_paths:
        model = Utility_Function_Parameterized(reward_shape=reward_shape, norm=norm, lamda=lamda, max_weight=0.5, keep_scale=keep_scale, size_factor=2)
        model.load_state_dict(torch.load(path))
        model.eval()
        model = model.cuda()
        pretrained_utility_functions.append(model)

    utility_function = utility_class_programmed(reward_shape=reward_shape, norm=norm, lamda=lamda, function_choice=0, keep_scale=keep_scale)
    num_utility_programmed = len(utility_function.utility_functions)

    if linear_utility:
        num_utility_pretrained = 0
    else:
        num_utility_pretrained = len(pretrained_utility_functions)

    num_total_policies = min(num_utility_programmed + num_utility_pretrained, max_num_policies)
    #print(f'{num_total_policies = }')

    print(f'Loading utility function for policy {idx} (out of {num_total_policies}), {idx < num_utility_programmed}')
    if idx < num_utility_programmed:
        utility_function = utility_class_programmed(reward_shape=reward_shape, norm=norm, lamda=lamda, function_choice=idx, keep_scale=keep_scale)
    else:
        utility_function = pretrained_utility_functions[idx - num_utility_programmed] 

    print(f'Utility function weights policy {pol_idx}: {utility_function.weights_list[utility_function.function_choice]}')
    
    return utility_function

def compute_entropy_sb3(model, obs, action):
    # obs_to_tensor puts obs on the correct device already
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    device = model.policy.device

    # Build action tensor on the SAME device
    if isinstance(action, np.ndarray):
        action_tensor = torch.as_tensor(action, device=device)
    else:
        action_tensor = torch.tensor([action], device=device)

    # Ensure 1D shape (batch,) for Discrete; match batch size of obs
    batch = obs_tensor.shape[0]
    action_tensor = action_tensor.reshape(-1)  # (N,)
    if action_tensor.numel() == 1 and batch > 1:
        action_tensor = action_tensor.repeat(batch)
    elif action_tensor.numel() > batch:
        action_tensor = action_tensor[:batch]

    # Correct dtype for Discrete actions
    if isinstance(model.action_space, Discrete):
        action_tensor = action_tensor.long()

    with torch.no_grad():  
        _, _, entropy = model.policy.evaluate_actions(obs_tensor, action_tensor)

    return float(entropy.item())

pol_idx = 3

def test(model_path, num_episodes=10, mode='ppo', augment_state=False, deterministic=True, reward_shape=6, reward_dim_indices=None):

    #print(f'Starting Test') 

    match = re.search(r'policy-program-(\d+)', os.path.basename(model_path))

    if match is None:
        raise ValueError(f"Could not extract policy index from model path: {model_path}")

    pol_idx = int(match.group(1))

    print(f'Model path: {model_path}, Reward shape: {reward_shape}, Reward dim indices: {reward_dim_indices}')
    
    if reward_dim_indices is None:
        reward_dim_indices = list(range(int(reward_shape)))

    print(f'Model path: {model_path}, Reward shape: {reward_shape}, Reward dim indices: {reward_dim_indices}')
        
    utility_function = get_utility_function(reward_shape, idx=pol_idx)
    #utility_function = get_utility_function(2, idx=pol_idx)

    env, depth = make_eval_env("fruit-tree-v0", utility_function, reward_shape, reward_dim_indices, augment_state=augment_state)
     
    print(f"{model_path}.zip is being loaded for testing.")
    if not os.path.exists(f'{model_path}.zip'): 
        raise f'{model_path} does not exist'
    model = PPO.load(f'{model_path}.zip')
    act_dim = env.action_space.n
    # observation space (vec env) -> single sub-env, use shape of env.observation_space
    obs = env.reset()
    num_feats = obs.shape[1]  # post-wrapper obs dim

    highlights_data = []
    #print('Num episodes: ', num_episodes)

    episodes_done = 0
    total_reward = 0.0
    episode_data = {'states': [], 'actions': [], 'entropy': [], 'dones': [], 'rewards': []}

    for episode in range(num_episodes):
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            # store pre-step state (copy to avoid aliasing)
            episode_data['states'].append(obs[0].copy())
            episode_data['actions'].append(int(action[0])) 
    
            obs, reward, done, infos = env.step(action) 

            # scalarized reward from the wrapper
            r = float(reward[0])
            episode_data['rewards'].append(r)
            episode_data['dones'].append(int(done[0]))
            episode_data['entropy'].append(compute_entropy_sb3(model,obs,action))

            total_reward += r 

            if done[0]:
                #print("episode length (steps): ",len(episode_data))
                highlights_data.append(episode_data)
                #print("Reward: ", total_reward)  
                total_reward = 0.0
                episode_data = {'states': [], 'actions': [], 'entropy': [], 'dones': [], 'rewards': []}
                obs = env.reset()
                break

    return highlights_data, model, num_feats, act_dim , depth

def calculate_fidelity(model_path, all_clusters, data, num_episodes=5, topin=False, apg_act=None, augment_state=False, deterministic=True):
    
    #print(f'Starting Test')  
    
    reward_shape = 6
    reward_dim_indices = list(range(int(reward_shape)))
    #print(f'{reward_dim_indices = }, {reward_shape = }')
    utility_function = get_utility_function(reward_shape, idx=pol_idx)
    #utility_function = get_utility_function(2, idx=pol_idx)

    env = make_eval_env("fruit-tree-v0", utility_function, reward_shape, reward_dim_indices, augment_state=augment_state)
     

    if not os.path.exists(f'{model_path}.zip'): 
        raise f'{model_path} does not exist'
    model = PPO.load(f'{model_path}.zip')
    act_dim = env.action_space.n

    if not topin:
        all_actions = data.actions 

    def get_cluster_action(clusters, num_feats, num_actions):
        if clusters == []:
            return np.random.randint(0, num_actions)
        taken_actions = np.zeros(num_actions)
        for cluster in clusters:
            ids = cluster.getInstanceIds()
            actions = all_actions[ids]
            for a in actions:
                taken_actions[int(a)] += 1
        policy = taken_actions / np.sum(taken_actions)
        return np.random.choice(np.arange(num_actions), p=policy)

    def find_clusters(obs, clusters, num_feats):
        # obs is vec: (1, feat_dim); flatten
        o = np.reshape(obs[0], [-1])
        valid = []
        for cluster in clusters:
            in_cluster = True
            for i in range(num_feats):
                lohi = cluster.get_bounds(i)
                if o[i] < lohi[1] or o[i] > lohi[0]:
                    in_cluster = False
                    break
            if in_cluster:
                valid.append(cluster)
        return valid

    obs = env.reset()
    num_feats = obs.shape[1]

    matches = []
    episodes_done = 0
    while episodes_done < num_episodes:
        if topin:
            abstract_action = apg_act(all_clusters, obs, act_dim)
        else:
            cls = find_clusters(obs, all_clusters, num_feats)
            abstract_action = get_cluster_action(cls, num_feats, act_dim)

        action, _ = model.predict(obs, deterministic=deterministic)
        matches.append(int(int(action[0]) == int(abstract_action)))

        obs, _, done, _ = env.step(action)
        if done[0]:
            obs = env.reset()
            episodes_done += 1

    fidelity = sum(matches) / len(matches) if len(matches) > 0 else 0.0
    return fidelity

def run_abstract_episode(all_clusters, data, utility_function, reward_shape, reward_dim_indices,
                            num_episodes=3, augment_state=False,
                            model_path='CAPS/DPMORL/experiments/FruitTree_test/DPMORL.FruitTree.LossNormLamda_0.1/policy-program-0',):

    #print(f'Starting Test')  

    os.makedirs(model_path, exist_ok=True)
    
    
    reward_shape = 6
    reward_dim_indices = list(range(int(reward_shape)))
    #print(f'{reward_dim_indices = }, {reward_shape = }')
    utility_function = get_utility_function(reward_shape, idx=pol_idx)
    #utility_function = get_utility_function(2, idx=pol_idx)

    env,_ = make_eval_env("fruit-tree-v0", utility_function, reward_shape, reward_dim_indices, augment_state=augment_state)
     
    if not os.path.exists(f'{model_path}.zip'): 
        raise f'{model_path} does not exist'
    model = PPO.load(f'{model_path}.zip')
    act_dim = env.action_space.n

    all_actions = data.actions
    num_feats = env.observation_space.shape[0]  # post-wrapper obs dim

    def get_cluster_action(clusters, num_feats, num_actions):
        if clusters == []:
            return np.random.randint(0, num_actions)
        taken_actions = np.zeros(num_actions)
        for cluster in clusters:
            ids = cluster.getInstanceIds()
            actions = all_actions[ids]
            for a in actions:
                taken_actions[int(a)] += 1
        policy = taken_actions / np.sum(taken_actions)
        return np.random.choice(np.arange(num_actions), p=policy)

    def find_clusters(obs, clusters, num_feats):
        o = np.reshape(obs[0], [-1])
        valid = []
        for cluster in clusters:
            in_cluster = True
            for i in range(num_feats):
                lohi = cluster.get_bounds(i)
                if o[i] < lohi[1] or o[i] > lohi[0]:
                    in_cluster = False
                    break
            if in_cluster:
                valid.append(cluster)
        return valid

    for ep in range(num_episodes):
        obs = env.reset()
        total_return = 0.0
        steps = 0
        while True:
            cls = find_clusters(obs, all_clusters, num_feats)
            abstract_action = get_cluster_action(cls, num_feats, act_dim)
            obs, reward, done, _ = env.step(np.array([abstract_action]))
            total_return += float(reward[0])  # scalarized reward via utility
            steps += 1
            if done[0] or steps > 200:
                break
        print(f"Episode {ep+1} with Abstract Policy. Reward: {total_return}")
        


if __name__ == '__main__':
    path = sys.argv[1]
    test(path)
    



    



