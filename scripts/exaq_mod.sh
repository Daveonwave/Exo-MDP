# python main.py \
#     --env taxi \
#     --env_id taxi-traffic-v0 \
#     --algo exaq-mod \
#     --exp_name "taxiTraffic" \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --world "world.yaml" \
#     --n_episodes 100000 \
#     --gamma 1 \
#     --alpha 0.01 \
#     --tol 0.01 \
#     --max_no_improvement 10000 \
#     --eval_episodes 50 \
#     --eval_every 1000 \
#     --train_seeds 9 10 \
#     --eval_seed 1234 

python main.py \
    --env elevator \
    --env_id elevator-v0 \
    --algo exaq-mod \
    --exp_name "seedFixedTinyElev" \
    --dest_folder "/data2/salaorni/pcmdp" \
    --world "tinyWorld.yaml" \
    --n_episodes 10000 \
    --gamma 1 \
    --epsilon 1.0 \
    --epsilon_decay 0.9995 \
    --epsilon_min 0.05 \
    --alpha 0.001 \
    --tol 0.01 \
    --max_no_improvement 10000 \
    --eval_episodes 50 \
    --eval_every 10 \
    --train_seeds 1 2 3 4 5 6 7 8 9 10 \
    --eval_seed 1234 


# python main.py \
#     --env trading \
#     --env_id trading-v0 \
#     --algo exaq-mod \
#     --exp_name "fixedS0" \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --world "world.yaml" \
#     --n_episodes 20000 \
#     --gamma 1 \
#     --epsilon 1.0 \
#     --epsilon_decay 0.9998 \
#     --epsilon_min 0.05 \
#     --decay_type "exponential" \
#     --alpha 0.9 \
#     --tol 0.01 \
#     --max_no_improvement 20000 \
#     --eval_episodes 50 \
#     --eval_every 1 \
#     --train_seeds 1 2 3 4 5 6 7 8 9 10\
#     --eval_seed 1234 
