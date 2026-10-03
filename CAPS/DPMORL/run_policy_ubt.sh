#!/bin/bash
mkdir -p outputs
while read -r env; do
    echo "Running $env"
    nohup python -u main_policy.py --lamda=0.1 --env "$env" \
        --reward_two_dim --exp_name dpmorl > "outputs/${env}.txt" 2>&1 &
    sleep 4s
done < env.txt
