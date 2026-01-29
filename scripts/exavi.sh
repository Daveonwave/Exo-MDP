python main.py \
    --env taxi \
    --env_id taxi-traffic-v0 \
    --algo exavi \
    --exp_name "provaTaxiTraffic" \
    --dest_folder "/data2/salaorni/pcmdp" \
    --world "world.yaml" \
    --n_episodes 5000 \
    --gamma 1 \
    --tol 0.01 \
    --max_no_improvement 5000 \
    --eval_episodes 50 \
    --eval_every 20 \
    --train_seeds 2 \
    --eval_seed 1234


# python main.py \
#     --env elevator \
#     --env_id elevator-v0 \
#     --algo exavi \
#     --exp_name "localTinyElev" \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --world "tinyWorld.yaml" \
#     --n_episodes 5000 \
#     --gamma 1 \
#     --tol 0.01 \
#     --max_no_improvement 5000 \
#     --eval_episodes 50 \
#     --train_seeds 1 \
#     --eval_every 1 \
#     --eval_seed 1234
