import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='grid', help='Environment (MO_deepsea, MO_fruitTree, MO_highway)')
    parser.add_argument('--path', default='', help='Path to RLlib pre-trained model')
    parser.add_argument('--num_episodes', type=int, default=3, help='Number of episodes to collect data from')
    parser.add_argument('--calc_fidelity', type=str2bool, nargs='?', const=True, default=False, help='Calculate fidelity of generated graphs')
    parser.add_argument('--alpha', type=float, default=0.015, help='Alpha parameter')
    parser.add_argument('--k', type=int, default=3, help='Number of graphs to produce')
    parser.add_argument('--max_height', type=int, default=10, help='Maximum height of CLTree')
    parser.add_argument('--lmbda', type=float, default=1, help='Lambda value from RL training')
    parser.add_argument('--hayes_baseline', type=str2bool, nargs='?', const=True, default=False, help='Whether to use Hayes and Shah 2017 baseline for explanations')
    parser.add_argument('--topin_baseline', type=str2bool, nargs='?', const=True, default=False, help='Whether to use Topin and Veloso 2019 baseline for apg gen')
    parser.add_argument('--zahavy_baseline', type=str2bool, nargs='?', const=True, default=False, help='Whether to cluster states according to Zahavy methodology')
    parser.add_argument('--alg', default='DQN', help='Training algorithm. DQN and PPO supported currently')
    parser.add_argument('--experiment_name', default='test', help='Name of the experiment')
    ############## Policy Return Plot args ##############
    parser.add_argument('--plot_returns', type=str2bool, nargs='?', const=True, default=True, help='Whether to plot the return distributions of the trained policies')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of episodes to average into one plotted point')
    parser.add_argument('--final_episodes', type=int, default=100, help='Number of final episodes from each policy to plot')
    parser.add_argument('--reward_dims', nargs='+', type=int, default=None, help='Which reward dimensions to plot like [0,2,5] (if None, plots all)')
    ############## AutoPred args ##############
    parser.add_argument('--autoPred', type=str2bool, nargs='?', const=True, default=False, help='Whether to use the automatic predicate generation algorithm instead of hand-crafted predicates')
    parser.add_argument('--shap_selection', type=str2bool, nargs='?', const=True, default=False, help='Whether to use SHAP for feature selection in AutoPred (if False, uses feature selection based on decision tree path instead)')
    parser.add_argument('--use_all_features', type=str2bool, nargs='?', const=True, default=False, help='Whether to use all features for predicate generation')
    ############## graph selection args ##############
    parser.add_argument('--compare_criterion', default='combined', help='Criterion for comparing graphs ("combined", "wl", "sm")')
    ############## re-run policies arg ##############
    parser.add_argument('--use_existing', type=str2bool, nargs='?', const=True, default=False, help='Whether to use existing graphs or re-run the policies to collect new data (True or False)')
    ############## top x policies arg ##############
    parser.add_argument('--top_x_policies', type=int, default=1, help='Number of top policy pairs to consider')

    args = parser.parse_args()
    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif v.lower() in ('false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')