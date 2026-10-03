mkdir -p outputs
while read -r env; do
    echo "Running $env"
    nohup python -u main_policy.py --lamda=0.1 --env $env \
        --reward_two_dim --exp_name dpmorl --test_only True > outputs/${env}_test.txt 2>&1 &
    sleep 4s
done < env.txt
