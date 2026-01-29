python main.py \
    --env taxi \
    --env_id taxi-traffic-v0 \
    --algo ql \
    --exp_name "provaTaxiTraffic" \
    --dest_folder "/data2/salaorni/pcmdp" \
    --world "world.yaml" \
    --n_episodes 15000 \
    --gamma 1 \
    --epsilon 1.0 \
    --epsilon_decay 0.9995 \
    --epsilon_min 0 \
    --decay_type "exponential" \
    --alpha 0.05 \
    --tol 0.01 \
    --max_no_improvement 10000 \
    --eval_episodes 50 \
    --eval_every 1 \
    --train_seeds 6 7 8 9 10 \
    --eval_seed 1234

# python main.py \
#     --env elevator \
#     --env_id elevator-v0 \
#     --algo ql \
#     --exp_name "localTinyElev" \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --world "tinyWorld.yaml" \
#     --n_episodes 7000 \
#     --gamma 1 \
#     --epsilon 1.0 \
#     --epsilon_decay 0.9995 \
#     --epsilon_min 0.05 \
#     --decay_type "exponential" \
#     --alpha 0.01 \
#     --tol 0.01 \
#     --max_no_improvement 7000 \
#     --eval_episodes 50 \
#     --eval_every 1 \
#     --train_seeds 1 \
#     --eval_seed 1234


# python main.py \
#     --env elevator \
#     --env_id elevator-v0 \
#     --algo ql \
#     --exp_name "localElev" \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --world "world.yaml" \
#     --n_episodes 100000 \
#     --gamma 1 \
#     --epsilon 1.0 \
#     --epsilon_decay 0.99997 \
#     --epsilon_min 0.05 \
#     --decay_type "exponential" \
#     --alpha 0.3 \
#     --tol 0.01 \
#     --max_no_improvement 100000 \
#     --eval_episodes 50 \
#     --eval_every 50 \
#     --train_seeds 1 \
#     --eval_seed 1234


# python main.py \
#     --env trading \
#     --env_id trading-v0 \
#     --algo ql \
#     --exp_name "provaFixedS0" \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --world "world.yaml" \
#     --n_episodes 30000 \
#     --gamma 1 \
#     --epsilon 1.0 \
#     --epsilon_decay 0.9998 \
#     --epsilon_min 0.05 \
#     --decay_type "mixed" \
#     --alpha 0.9 \
#     --tol 0.01 \
#     --max_no_improvement 30000 \
#     --eval_episodes 50 \
#     --eval_every 20 \
#     --train_seeds 1 \
#     --eval_seed 1234
