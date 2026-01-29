python main.py \
    --env taxi \
    --env_id taxi-traffic-v0 \
    --algo exaq \
    --exp_name "provaTaxiTraffic" \
    --dest_folder "/data2/salaorni/pcmdp" \
    --world "world.yaml" \
    --n_episodes 10000 \
    --gamma 1 \
    --alpha 0.01 \
    --tol 0.01 \
    --max_no_improvement 10000 \
    --eval_episodes 50 \
    --eval_every 1 \
    --train_seeds 1 \
    --eval_seed 1234 

# python main.py \
#     --env elevator \
#     --env_id elevator-v0 \
#     --algo exaq \
#     --exp_name "localTinyElev" \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --world "tinyWorld.yaml" \
#     --n_episodes 7000 \
#     --gamma 1 \
#     --epsilon 1.0 \
#     --epsilon_decay 0.9995 \
#     --epsilon_min 0.05 \
#     --alpha 0.001 \
#     --tol 0.01 \
#     --max_no_improvement 7000 \
#     --eval_episodes 50 \
#     --eval_every 1 \
#     --train_seeds 1 \
#     --eval_seed 1234 


# python main.py \
#     --env elevator \
#     --env_id elevator-v0 \
#     --algo exaq \
#     --exp_name "lowerArrivalRateProva" \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --world "world.yaml" \
#     --n_episodes 30000 \
#     --gamma 1 \
#     --epsilon 1.0 \
#     --epsilon_decay 0.999 \
#     --epsilon_min 0.05 \
#     --alpha 0.01 \
#     --tol 0.01 \
#     --max_no_improvement 30000 \
#     --eval_episodes 50 \
#     --eval_every 5 \
#     --train_seeds 1 \
#     --eval_seed 1234 


# python main.py \
#     --env trading \
#     --env_id trading-v0 \
#     --algo exaq \
#     --exp_name "prova" \
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
#     --eval_every 20 \
#     --train_seeds 1 \
#     --eval_seed 1234 
