import sys
import time
import torch
from torch.autograd import Variable
import numpy as np
import random
import ray
from CAPS import explain, explain_auto_pred
from topin_baseline import gen_apg
from config import argparser
from Cartpole.test_cartpole import calculate_fidelity as calculate_fidelity_cart
from Cartpole.test_cartpole import test as test_cart
from data import Data
from abstract import APG
from translation import CartpolePredicates
"""from Gridworld.test_gridworld import calculate_fidelity as calculate_fidelity_grid
from Gridworld.test_gridworld import test as test_grid
from translation import GridworldPredicates"""
from MountainCar.test_mountaincar import calculate_fidelity as calculate_fidelity_mountain
from MountainCar.test_mountaincar import test as test_mountain
from translation import MountainCarPredicates
from zahavy_baseline import explain_zahavy
from LunarLander.test_lunarlander import calculate_fidelity as calculate_fidelity_lunar
from LunarLander.test_lunarlander import test as test_lunar
from translation import LunarLanderPredicates
from Blackjack.test_blackjack import calculate_fidelity as calculate_fidelity_blackjack
from Blackjack.test_blackjack import test as test_blackjack
from translation import BlackjackPredicates
from DPMORL.test_fruitTree import test as test_fruitTree
from DPMORL.test_fruitTree import calculate_fidelity as calculate_fidelity_fruitTree
from translation import FruitTreePredicates
from DPMORL.test_deepSea import test as test_deepSea
from DPMORL.test_deepSea import calculate_fidelity as calculate_fidelity_deepSea
from translation import HighwayPredicates
from DPMORL.test_highway import test as test_highway
from DPMORL.test_highway import calculate_fidelity as calculate_fidelity_highway
from translation import DeepSeaPredicates
from auto_translation import AutoPred





if __name__ == '__main__':

    args = argparser()
    model_path = args.path
    assert model_path != ''
    fidelity_fn = None

    if args.env == 'cart':
        data, model, num_feats, num_actions = test_cart(model_path, args.num_episodes, mode=args.alg)

        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_cart
        def value_fn(obs):
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        
        dataset = Data(data, value_fn)
        translator = CartpolePredicates(num_feats=num_feats)
    
        """elif args.env == 'grid':
        data, model, num_feats, num_actions = test_grid(model_path, args.num_episodes, mode=args.alg)
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_grid
        def value_fn(obs):
            int_state = obs[0]
            obs = np.zeros(48)
            obs[int_state] = 1
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        
        dataset = Data(data, value_fn)
        translator = GridworldPredicates(num_feats=num_feats)"""
    
    elif args.env == 'mountain':
        data, model, num_feats, num_actions = test_mountain(model_path, args.num_episodes, mode=args.alg)
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_mountain
        def value_fn(obs):
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        dataset = Data(data, value_fn)
        translator = MountainCarPredicates(num_feats=num_feats)
    
    elif args.env == 'lunar':
        data, model, num_feats, num_actions = test_lunar(model_path, args.num_episodes, mode=args.alg)
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_lunar
        def value_fn(obs):
            obs = np.reshape(obs, [1, -1])
            obs = Variable(torch.from_numpy(obs))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        dataset = Data(data, value_fn)
        translator = LunarLanderPredicates(num_feats=num_feats)

    elif args.env == 'blackjack':
        data, model, num_feats, num_actions = test_blackjack(model_path, args.num_episodes, mode=args.alg)
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_blackjack
        def value_fn(obs):
            obs = np.reshape(obs, [1, -1])
            obs = np.squeeze(obs)
            p = obs[0]
            d = obs[1]
            a = obs[2]
            s = np.zeros(45)
            s[p] = 1
            s[32+d] = 1
            s[43+a] = 1
            s = np.reshape(s, [1, -1])
            obs = Variable(torch.from_numpy(s))
            _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
            value = model.value_function().detach().numpy()[0]
            return value
        dataset = Data(data, value_fn)
        translator = BlackjackPredicates(num_feats=num_feats)
    
    elif args.env == 'MO_fruitTree':
        feature_names = [
            "lvl", "pos", 
        ]
        print(f'running MO_fruitTree')
        data, model, num_feats, num_actions, depth = test_fruitTree(model_path, args.num_episodes, mode=args.alg)
        print(len(data))
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_fruitTree
        def value_fn(obs): 
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            obs = np.reshape(obs, [1, -1]).astype(np.float32)
 
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
                value = model.policy.predict_values(obs_t)
                return float(value.squeeze().cpu().numpy())
        dataset = Data(data, value_fn)
        translator = FruitTreePredicates(num_feats=num_feats, depth= depth)

    elif args.env == 'MO_deepSea':
        feature_names = [
            "lvl", "pos", 
        ]
        print(f'running MO_deepSea')
        data, model, num_feats, num_actions = test_deepSea(model_path, args.num_episodes, mode=args.alg)
        print(len(data))
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_deepSea
        def value_fn(obs): 
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            obs = np.reshape(obs, [1, -1]).astype(np.float32)
 
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
                value = model.policy.predict_values(obs_t)
                return float(value.squeeze().cpu().numpy())
        dataset = Data(data, value_fn)
        translator = DeepSeaPredicates(num_feats=num_feats)

    elif args.env == 'MO_highway':
        feature_names = [
            "egoExists", "xEgo", "yEgo", "vxEgo", "vyEgo",
            "1stClosestCarExists", "xC1", "yC1", "vxC1", "vyC1",
            "2ndClosestCarExists", "xC2", "yC2", "vxC2", "vyC2",
            "3rdClosestCarExists", "xC3", "yC3", "vxC3", "vyC3",
            "4thClosestCarExists", "xC4", "yC4", "vxC4", "vyC4"
        ]
        print(f'running MO_highway')
        data, model, num_feats, num_actions, _ = test_highway(model_path, args.num_episodes, mode=args.alg)
        print(f'num_actions: {num_actions}')
        print(len(data))
        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_highway
        def value_fn(obs): 
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            else:
                obs = obs.astype(np.float32)

            if obs.ndim == 1:
                obs = obs.reshape(1, -1)
            elif obs.ndim != 2:
                raise ValueError(f"Unexpected obs shape: {obs.shape}")

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
                value = model.policy.predict_values(obs_t)
                value = value.detach().cpu().numpy()

            return value.reshape(-1, 1)
        dataset = Data(data, value_fn)
        translator = HighwayPredicates(num_feats=num_feats)

    else:
        raise ValueError('Enter valid environment')

    if args.zahavy_baseline:
        abstract_baseline = APG(num_actions, value_fn, translator)
        explain_zahavy(args, dataset, translator, abstract_baseline, num_actions, fidelity_fn, model_path, mode=args.alg)
    elif args.topin_baseline:
        info = {'states': dataset.states, 'actions': dataset.actions, 'next_states': dataset.next_states, 'dones': dataset.dones, 'entropies': dataset.entropies}
        abstract_baseline = APG(num_actions, value_fn, translator, info=info)
        gen_apg(abstract_baseline, model_path, fidelity_fn, mode=args.alg)
    elif args.autoPred:
        timestart = time.time()
        translator = AutoPred(num_feats=num_feats, feature_names=feature_names)
        abstract_baseline = APG(num_actions, value_fn, translator)
        print('Running AutoPred')
        print(args.shap_selection)
        print("Explanations with SHAP feature selection")
        explain_auto_pred(
            args=args,
            dataset=dataset,
            attr_names=feature_names,
            model_path=model_path,
            translator=translator,
            num_actions=num_actions,
            fidelity_fn=fidelity_fn,
            apg_baseline=abstract_baseline,
            num_feats=num_feats,
            mode=args.alg,
            shap_feature_selection=True #args.shap_selection
        )
        timestop = time.time()
        print(f'Time taken for AutoPred: {timestop - timestart} seconds')
        timestart = time.time()
        print("Explanations with tree path feature selection")
        explain_auto_pred(
            args=args,
            dataset=dataset,
            attr_names=feature_names,
            model_path=model_path,
            translator=translator,
            num_actions=num_actions,
            fidelity_fn=fidelity_fn,
            apg_baseline=abstract_baseline,
            num_feats=num_feats,
            mode=args.alg,
            shap_feature_selection=False # args.shap_selection
        )
        timestop = time.time()
        print(f'Time taken for AutoPred: {timestop - timestart} seconds')
    else:
        abstract_baseline = APG(num_actions, value_fn, translator)
        explain(args, dataset, model_path, translator, num_feats, num_actions, fidelity_fn, abstract_baseline, mode=args.alg)
        







