# ============================================================
# Create log directories if necessary
# ============================================================

mkdir -p outputs/logs/deepSea
mkdir -p outputs/logs/fruitTree
mkdir -p outputs/logs/highway


# ============================================================
# DEEP SEA + FRUIT TREE
# M: Generate/recreate graphs
# ============================================================

nohup python run.py --env=MO_deepSea --num_episodes=10 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/deepSea/deepSea10(m)ep.txt"
nohup python run.py --env=MO_fruitTree --num_episodes=10 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/fruitTree/fruitTree10(m)ep.txt"

nohup python run.py --env=MO_deepSea --num_episodes=50 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/deepSea/deepSea50(m)ep.txt"
nohup python run.py --env=MO_fruitTree --num_episodes=50 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/fruitTree/fruitTree50(m)ep.txt"

nohup python run.py --env=MO_deepSea --num_episodes=100 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/deepSea/deepSea100(m)ep.txt"
nohup python run.py --env=MO_fruitTree --num_episodes=100 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/fruitTree/fruitTree100(m)ep.txt"

nohup python run.py --env=MO_deepSea --num_episodes=200 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/deepSea/deepSea200(m)ep.txt"
nohup python run.py --env=MO_fruitTree --num_episodes=200 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/fruitTree/fruitTree200(m)ep.txt"

nohup python run.py --env=MO_deepSea --num_episodes=500 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/deepSea/deepSea500(m)ep.txt"
nohup python run.py --env=MO_fruitTree --num_episodes=500 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/fruitTree/fruitTree500(m)ep.txt"

echo 'done with graph regeneration experiments'


# ============================================================
# DEEP SEA + FRUIT TREE
# S: Use stored graphs
# ============================================================

nohup python run.py --env=MO_deepSea --num_episodes=10 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/deepSea/deepSea10(s)ep.txt"
nohup python run.py --env=MO_fruitTree --num_episodes=10 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/fruitTree/fruitTree10(s)ep.txt"

nohup python run.py --env=MO_deepSea --num_episodes=50 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/deepSea/deepSea50(s)ep.txt"
nohup python run.py --env=MO_fruitTree --num_episodes=50 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/fruitTree/fruitTree50(s)ep.txt"

nohup python run.py --env=MO_deepSea --num_episodes=100 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/deepSea/deepSea100(s)ep.txt"
nohup python run.py --env=MO_fruitTree --num_episodes=100 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/fruitTree/fruitTree100(s)ep.txt"

nohup python run.py --env=MO_deepSea --num_episodes=200 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/deepSea/deepSea200(s)ep.txt"
nohup python run.py --env=MO_fruitTree --num_episodes=200 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/fruitTree/fruitTree200(s)ep.txt"

nohup python run.py --env=MO_deepSea --num_episodes=500 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/deepSea/deepSea500(s)ep.txt"
nohup python run.py --env=MO_fruitTree --num_episodes=500 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/fruitTree/fruitTree500(s)ep.txt"

echo 'done with stored graph experiments'


# ============================================================
# HIGHWAY
# M: Generate/recreate graphs
# ============================================================

nohup python run.py --env=MO_highway --num_episodes=10 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/highway/highway10(m)ep.txt"

nohup python run.py --env=MO_highway --num_episodes=50 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/highway/highway50(m)ep.txt"

nohup python run.py --env=MO_highway --num_episodes=100 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/highway/highway100(m)ep.txt"

nohup python run.py --env=MO_highway --num_episodes=200 --alg=PPO --autoPred=True --shap_selection=True > "outputs/logs/highway/highway200(m)ep.txt"

echo 'done with Highway graph regeneration experiments'


# ============================================================
# HIGHWAY
# S: Use stored graphs
# ============================================================

nohup python run.py --env=MO_highway --num_episodes=10 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/highway/highway10(s)ep.txt"

nohup python run.py --env=MO_highway --num_episodes=50 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/highway/highway50(s)ep.txt"

nohup python run.py --env=MO_highway --num_episodes=100 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/highway/highway100(s)ep.txt"

nohup python run.py --env=MO_highway --num_episodes=200 --alg=PPO --autoPred=True --shap_selection=True --use_existing=True > "outputs/logs/highway/highway200(s)ep.txt"

echo 'done with Highway stored graph experiments'

echo 'done with all experiments'