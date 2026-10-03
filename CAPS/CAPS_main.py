import sys
import time
import torch
from torch.autograd import Variable
import numpy as np
import random
import ray
import gc
from CAPS.CAPS_dict import explain, explain_auto_pred
from CAPS.topin_baseline import gen_apg
from CAPS.config import argparser
from CAPS.Cartpole.test_cartpole import calculate_fidelity as calculate_fidelity_cart
from CAPS.Cartpole.test_cartpole import test as test_cart
from CAPS.data import Data
from CAPS.abstract import APG
from CAPS.translation import CartpolePredicates
from CAPS.MountainCar.test_mountaincar import calculate_fidelity as calculate_fidelity_mountain
from CAPS.MountainCar.test_mountaincar import test as test_mountain
from CAPS.translation import MountainCarPredicates
from CAPS.zahavy_baseline import explain_zahavy
from CAPS.LunarLander.test_lunarlander import calculate_fidelity as calculate_fidelity_lunar
from CAPS.LunarLander.test_lunarlander import test as test_lunar
from CAPS.translation import LunarLanderPredicates
from CAPS.Blackjack.test_blackjack import calculate_fidelity as calculate_fidelity_blackjack
from CAPS.Blackjack.test_blackjack import test as test_blackjack
from CAPS.translation import BlackjackPredicates
from CAPS.DPMORL.test_fruitTree import test as test_fruitTree
from CAPS.DPMORL.test_fruitTree import calculate_fidelity as calculate_fidelity_fruitTree
from CAPS.translation import FruitTreePredicates
from CAPS.DPMORL.test_deepSea import test as test_deepSea
from CAPS.DPMORL.test_deepSea import calculate_fidelity as calculate_fidelity_deepSea
from CAPS.translation import HighwayPredicates
from CAPS.DPMORL.test_highway import test as test_highway
from CAPS.DPMORL.test_highway import calculate_fidelity as calculate_fidelity_highway
from CAPS.translation import DeepSeaPredicates
from CAPS.auto_translation import AutoPred





def CAPS_main(caps_args):

    args = caps_args
    model_path = args.path
    assert model_path != ''
    fidelity_fn = None

    print(f'Running CAPS_main with model path: {model_path} and environment: {args.env}', flush=True)

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
        #print(len(data))
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
            "x", "y", 
        ]
        print(f'running MO_deepSea')
        data, model, num_feats, num_actions = test_deepSea(model_path, args.num_episodes, mode=args.alg)
        #print(len(data))
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
        #print(len(data))
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
    
    sys.stdout.flush()
    print(f"Dataset collected for model {model_path}. Starting explanation generation...")

    if args.zahavy_baseline:
        abstract_baseline = APG(num_actions, value_fn, translator)
        explain_zahavy(args, dataset, translator, abstract_baseline, num_actions, fidelity_fn, model_path, mode=args.alg)
    elif args.topin_baseline:
        info = {'states': dataset.states, 'actions': dataset.actions, 'next_states': dataset.next_states, 'dones': dataset.dones, 'entropies': dataset.entropies}
        abstract_baseline = APG(num_actions, value_fn, translator, info=info)
        gen_apg(abstract_baseline, model_path, fidelity_fn, mode=args.alg)
    explanations = None
    if args.autoPred:
        timestart = time.time()
        translator = AutoPred(num_feats=num_feats, feature_names=feature_names)
        abstract_baseline = APG(num_actions, value_fn, translator)
        print('Running AutoPred')
        #print(args.shap_selection)
        if args.shap_selection:
            print("Explanations with SHAP feature selection")
        else:
            print("Explanations with decision tree path feature selection")
        explanations=explain_auto_pred(
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
            shap_feature_selection=args.shap_selection,
            use_all_features=args.use_all_features
        )
        timestop = time.time()
        print(f'Time taken for AutoPred: {timestop - timestart} seconds') 

        del abstract_baseline
        del dataset
        del data
        del model
        del translator

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
    else:
        abstract_baseline = APG(num_actions, value_fn, translator)
        explanations=explain(args, dataset, model_path, translator, num_feats, num_actions, fidelity_fn, abstract_baseline, mode=args.alg)
    return explanations
        







