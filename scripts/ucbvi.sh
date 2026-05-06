python main.py \
    --env taxi \
    --env_id taxi-traffic-v0 \
    --algo ucbvi \
    --dest_folder "/data2/salaorni/pcmdp" \
    --exp_name "taxiTraffic" \
    --world "world.yaml" \
    --n_episodes 10000 \
    --gamma 1 \
    --tol 0.01 \
    --c_bonus 0.5 \
    --delta 0.000001 \
    --max_no_improvement 10000 \
    --eval_episodes 50 \
    --eval_every 10 \
    --train_seeds 1 \
    --eval_seed 1234
    # --R_given

# python main.py \
#     --env elevator \
#     --env_id elevator-v0 \
#     --algo ucbvi \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --exp_name "localTinyElev" \
#     --world "tinyWorld.yaml" \
#     --n_episodes 5000 \
#     --gamma 1 \
#     --tol 0.01 \
#     --c_bonus 0.5 \
#     --delta 0.000001 \
#     --max_no_improvement 5000 \
#     --eval_episodes 50 \
#     --eval_every 50 \
#     --train_seeds 1 \
#     --eval_seed 1234 \
#     #--R_given 



