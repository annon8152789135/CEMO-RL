import gym
import numpy as np

import matplotlib
matplotlib.rcParams.update({'font.size': 22})

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
# Use your project's DummyVecEnv if needed for eval paths:
try:
    from DPMORL.utils import DummyVecEnv
except Exception:
    from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.utils import set_random_seed
import matplotlib.pyplot as plt
import mo_gymnasium
import gymnasium
import torch
import time
import os
import argparse
import glob
from typing import Callable

from MORL_stablebaselines3.envs.wrappers.utility_env_wrapper import MultiEnv_UtilityFunction, ObsInfoWrapper
from MORL_stablebaselines3.utility_function.utility_function_parameterized import Utility_Function_Parameterized
from MORL_stablebaselines3.utility_function.utility_function_programmed import Utility_Function_Programmed
from MORL_stablebaselines3.utility_function.utility_function_programmed import Utility_Function_Linear
from MORL_stablebaselines3.utility_function.utility_function_programmed import Utility_Function_Diverse_Goal
from DIPG.diverse_goal_env import DiverseGoalEnv

def make_env(env_name, rank, utility_function, reward_dim, reward_dim_indices, seed=None):
    def _init():
        if isinstance(env_name, str):
            env = mo_gymnasium.make(env_name)
            env.name = env_name
        else:
            env = env_name()
        env = ObsInfoWrapper(env, reward_dim=reward_dim, reward_dim_indices=reward_dim_indices)
        return env
    set_random_seed(seed)
    return _init

def choose_gpu(args):
    try:
        import pynvml
        pynvml.nvmlInit()
        if args.gpu == "all":
            memory_gpu = []
            for gpu_id in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_gpu.append(meminfo.free / 1024 / 1024)
            gpu1 = int(np.argmax(memory_gpu))
        else:
            gpu1 = int(args.gpu)
        print(f"****************************Chosen GPU : {gpu1}****************************")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu1)
    except Exception as e:
        print("pynvml not available or GPU query failed; proceeding with current CUDA_VISIBLE_DEVICES.")

def config_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    p = argparse.ArgumentParser("Multiprocessing-safe MORL trainer (thread outer, process inner)")
    p.add_argument('--env', type=str, default="Highway")
    p.add_argument('--exp_name', type=str, default='dpmorl')
    p.add_argument('--utility_epochs', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--lamda', type=float, default=1e-2)
    p.add_argument('--norm', type=bool, default=True)
    p.add_argument('--num_test_episodes', type=int, default=100)
    p.add_argument('--keep_scale', type=bool, default=True)
    p.add_argument('--reward_two_dim', type=bool, default=False)
    p.add_argument('--reward_dim_indices', type=str, default='')
    p.add_argument('--linear_utility', type=bool, default=False)
    p.add_argument('--augment_state', type=bool, default=False)
    p.add_argument('--test_only', type=bool, default=False)
    p.add_argument('--num_envs', type=int, default=8, help="subproc envs per learner")
    p.add_argument('--max_num_policies', type=int, default=8)
    p.add_argument('--total_timesteps', type=float, default=2e5)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--max_workers', type=int, default=2, help="how many policies to train concurrently (threads)")
    p.add_argument('--rollout_target', type=int, default=4096, help="target rollout size n_envs*n_steps")
    p.add_argument('--batch_size_cap', type=int, default=2048)
    return p.parse_args()

def env_functions(env_name):
    if env_name != "DiverseGoal":
        gym_id = get_id_name(env_name)
        return lambda: mo_gymnasium.make(gym_id) 
    return DiverseGoalEnv

def get_id_name(env_name):
    mapping = {
        "MountainCar": "mo-mountaincarcontinuous-v0",
        "BreakableBottles": "breakable-bottles-v0",
        "DeepSeaTreasure": "deep-sea-treasure-v0",
        "FishWood": "fishwood-v0",
        "FourRoom": "four-room-v0",
        "FruitTree": "fruit-tree-v0",
        "Highway": "mo-highway-v0",
        "LunarLander": "mo-lunar-lander-v2",
        "SuperMarioBros": "mo-supermario-v0",
        "HalfCheetah": "mo-halfcheetah-v4",
        "Hopper": "mo-hopper-v4",
        "Reacher": "mo-reacher-v4",
        "ReacherBullet": "mo-reacher-v0",
        "ResourceGathering": "resource-gathering-v0",
        "WaterReservoir": "water-reservoir-v0",
        "Minecart": "minecart-v0",
        "highway": "mo-highway-v0",
        "DiverseGoal": DiverseGoalEnv,
    }
    if env_name not in mapping:
        raise NotImplementedError("Please write the right and implemented env name!")
    return mapping[env_name]

class ReturnLogger(torch.nn.Module):
    def __init__(self, save_dir, env_name, algo_name, policy_id, it, seed):
        super().__init__()
        self.episode_vec_returns = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.env_name = env_name
        self.algo_name = algo_name
        self.seed = seed
        self.iter = it
        self.policy_id = policy_id
    def __call__(self, locals_, globals_):
        infos = locals_.get("infos", None)
        if isinstance(infos, (tuple, list)):
            for info in infos:
                if isinstance(info, dict) and "episode" in info:
                    self.episode_vec_returns.append(info["episode"]["r"])
        else:
            if locals_.get("done") and isinstance(infos, dict) and "episode" in infos:
                self.episode_vec_returns.append(infos["episode"]["r"])
        return True
    def training_end(self):
        file_name = f"MORL_{self.env_name}_{self.algo_name}_policy{self.policy_id}_seed{self.seed}_{self.iter}.npz"
        file_path = os.path.join(self.save_dir, file_name)
        np.savez_compressed(file_path, episode_vec_returns=self.episode_vec_returns)

def evaluate_policy(model, env, num_test_episodes):
    episode_returns = []
    obs = env.reset()
    trajectories = []
    current_trajectory = []
    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, infos = env.step(action)
        for info in infos:
            if "episode" in info:
                episode_returns.append(info["episode"]["r"])
                if len(episode_returns) % 10 == 0:
                    print(f'progress: {len(episode_returns)}/{num_test_episodes}')
        if done[0]:
            current_trajectory.append(infos[0].get('terminal_observation', obs[0]).copy())
            trajectories.append(np.array(current_trajectory))
            current_trajectory = []
        current_trajectory.append(obs[0].copy())
        if len(episode_returns) >= num_test_episodes:
            break
    episode_returns = episode_returns[:num_test_episodes]
    return np.array(episode_returns), trajectories

def train_one_policy(policy_idx, args, normalization_data, reward_shape, reward_dim_indices,
                     gym_id_name, utility_dir, num_utility_programmed, num_pretrained_utility):
    # limit torch intra-op threads to prevent oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # Utility selection
    norm_flag = True
    utility_class_programmed = Utility_Function_Linear if args.linear_utility else Utility_Function_Programmed
    if args.env == 'DiverseGoal':
        utility_class_programmed = Utility_Function_Diverse_Goal
        norm_flag = False

    if policy_idx < num_utility_programmed:
        uf = utility_class_programmed(reward_shape=reward_shape, norm=norm_flag,
                                      lamda=args.lamda, function_choice=policy_idx,
                                      keep_scale=args.keep_scale)
    else:
        pt_idx = policy_idx - num_utility_programmed
        model = Utility_Function_Parameterized(reward_shape=reward_shape, norm=norm_flag,
                                               lamda=args.lamda, max_weight=0.5,
                                               keep_scale=args.keep_scale, size_factor=1)
        path_pt = f'utility-model-selected/dim-{reward_shape}/utility-{pt_idx}.pt'
        model.load_state_dict(torch.load(path_pt, map_location='cpu'))
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        uf = model

    # normalization bounds
    if args.env in normalization_data:
        uf.min_val = normalization_data[args.env]['min'][0][reward_dim_indices]
        uf.max_val = normalization_data[args.env]['max'][0][reward_dim_indices]
        print(f'[Worker {policy_idx}] Using normalization data')
    else:
        print(f'[Worker {policy_idx}] No normalization data')

    # vectorized env WITH subprocesses (allowed, because outer is threading)
    env = SubprocVecEnv([
        make_env(gym_id_name, i, uf,
                 reward_dim=reward_shape,
                 reward_dim_indices=reward_dim_indices,
                 seed=args.seed + i)
        for i in range(args.num_envs)
    ])
    env = MultiEnv_UtilityFunction(env, uf, reward_dim=reward_shape, augment_state=args.augment_state)
    env.update_utility_function(uf)

    # rollout scaling
    n_envs = args.num_envs
    n_steps = max(128, int(args.rollout_target) // max(1, n_envs))
    batch_size = min(args.batch_size_cap, n_steps * n_envs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = PPO("MlpPolicy", env, verbose=1, device=device,
                 n_steps=n_steps, batch_size=batch_size,
                 n_epochs=5, learning_rate=args.lr)

    policy_name = f'program-{policy_idx}' if policy_idx < num_utility_programmed else f'pretrain-{policy_idx - num_utility_programmed}'
    os.makedirs(utility_dir, exist_ok=True)
    print(f"[Thread {policy_idx}] Training {policy_name} for {int(args.total_timesteps)} steps "
          f"(n_envs={n_envs}, n_steps={n_steps}, batch_size={batch_size}, device={device})")

    retlog = ReturnLogger(utility_dir, args.env, "PPO", policy_name, 0, args.seed + policy_idx)
    policy.learn(total_timesteps=int(args.total_timesteps), callback=retlog, progress_bar=True)
    retlog.training_end()
    policy.save(f'{utility_dir}/policy-{policy_name}')
    env.close()
    print(f"[Thread {policy_idx}] Done {policy_name}")
    return policy_name

if __name__ == "__main__":
    import pickle
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with open('normalization_data/data.pickle', 'rb') as file:
        normalization_data = pickle.load(file)
        
    args = config_args()
    choose_gpu(args)

    base_env_name = env_functions(args.env)
    test_env = base_env_name()
    try:
        reward_dim = test_env.reward_dim
    except:
        test_env = getattr(test_env, "unwrapped", test_env)
        reward_dim = test_env.reward_space.shape
    gym_id_name = get_id_name(args.env)

    utility_dir_root = 'experiments/' + args.exp_name
    os.makedirs(utility_dir_root, exist_ok=True)
    reward_shape = test_env.reward_dim
    if args.reward_two_dim:
        reward_shape = 2
    if args.reward_dim_indices == '':
        reward_dim_indices = list(range(reward_shape))
    else:
        reward_dim_indices = eval(args.reward_dim_indices)
        reward_shape = len(reward_dim_indices)
    print(f'{reward_dim_indices = }, {reward_shape = }')

    utility_class_programmed = Utility_Function_Linear if args.linear_utility else Utility_Function_Programmed
    norm = True
    if args.env == 'DiverseGoal':
        args.total_timesteps = max(1, int(args.total_timesteps // 20))
        utility_class_programmed = Utility_Function_Diverse_Goal
        norm = False
        print('Using Diverse Goal Utility Function')

    tmp_u = utility_class_programmed(reward_shape=reward_shape, norm=norm, lamda=args.lamda, function_choice=0, keep_scale=args.keep_scale)
    num_utility_programmed = len(tmp_u.utility_functions)

    assert os.path.isdir(f'utility-model-selected/dim-{reward_shape}'), 'There is no pretrained utility functions provided. '
    num_pretrained_utility = len(glob.glob(f'utility-model-selected/dim-{reward_shape}/*'))
    num_total_policies = min(num_utility_programmed + (0 if (args.linear_utility or args.env == 'DiverseGoal') else num_pretrained_utility),
                             args.max_num_policies)
    print(f'{num_total_policies = }')

    task_name = f"DPMORL.{args.env}.{'no_norm.' if args.norm == False else ''}LossNormLamda_{args.lamda}"
    utility_dir = os.path.join(utility_dir_root, task_name)

    if not args.test_only:
        max_workers = max(1, int(args.max_workers))
        print(f"Starting threaded parallel training with max_workers={max_workers} (SubprocVecEnv inside each thread)")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(train_one_policy, policy_idx, args, normalization_data, reward_shape, reward_dim_indices,
                          gym_id_name, utility_dir, num_utility_programmed, num_pretrained_utility)
                for policy_idx in range(num_total_policies)
            ]
            for fut in as_completed(futures):
                name = fut.result()
                print(f"Finished training {name}")
    else:
        # simple sequential eval
        for policy_idx in range(num_total_policies):
            policy_name = f'program-{policy_idx}' if policy_idx < num_utility_programmed else f'pretrain-{policy_idx - num_utility_programmed}'
            model_path = os.path.join(utility_dir, f'policy-{policy_name}.zip')
            if not os.path.exists(model_path):
                print(f'{policy_name} does not exist at {model_path}')
                continue
            policy = PPO.load(model_path)
            env = DummyVecEnv(
                [make_env(gym_id_name, i, None, reward_dim=reward_shape, reward_dim_indices=reward_dim_indices, seed=args.seed) for i in range(10)], 
            )
            returns, trajs = evaluate_policy(policy, env, args.num_test_episodes)
            np.savez_compressed(os.path.join(utility_dir, f"test_returns_policy_{policy_name}.npz"),
                                test_returns=returns)
