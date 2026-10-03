mkdir -p outputs

nohup python -u main_policy.py --lamda=0.1 --env Highway --reward_dim_indices=[0,1,2] --exp_name highway_test --total_timesteps=1e6 > "outputs/Highway_0.1.txt"
echo "Running Highway lambda 0.1" 
nohup python -u main_policy.py --lamda=0.2 --env Highway --reward_dim_indices=[0,1,2] --exp_name highway_test --total_timesteps=1e6 > "outputs/Highway_0.2.txt"
echo "Running Highway lambda 0.2"