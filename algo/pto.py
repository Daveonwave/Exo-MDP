import os
from tqdm.rich import trange
import numpy as np
import pickle
import json
from algo.utils import *
from algo.evaluation import validation_step


def train(env, args, eval_params, horizon, seed=None, model_file=None, settings=None):
    if args.get('dest_folder') is not None and os.path.exists(args['dest_folder']):
        dest_path = args['dest_folder']
    else:
        dest_path = './'
    writer, out_path = init_writer(f"{args['env']}/{args['exp_name']}/{args['algo']}", args, seed, dest_folder=dest_path)
    
    # Hyperparameters
    H = horizon
    n_episodes = args['n_episodes']
    rng = np.random.default_rng(seed=seed)
    eval_seed = args['eval_seed']
    
    best_eval_reward = -np.inf
    best_eval_episode = 0
    eval_counter = 0
    learning_curve = {}
    keys, multipliers = build_state_index_map(env)
    
    state_factorizability = False if env.spec.id == 'exomdp/elevator-v0' or env.spec.id == 'exomdp/taxi-traffic-v0' else True
    
    S = get_state_size(env)  # total number of flattened states
    A = env.action_space.n
    R = build_reward_matrix(env, keys, multipliers)
    print(f"State size: {S}, Action size: {A}, Horizon: {H}")
    print(f"State keys: {keys} Multipliers: {multipliers}")
    
    ctrl_keys, ctrl_multipliers = build_state_index_map(env, env.unwrapped.get_controllables())
    unctrl_keys, unctrl_multipliers = build_state_index_map(env, env.unwrapped.get_uncontrollables())
    
    S_ctrl = get_composite_state_size(env, keys=ctrl_keys)
    S_unctrl = get_composite_state_size(env, keys=unctrl_keys)
    
    # Special case: controllable dynamics depend on uncontrollable state
    if not state_factorizability:     
        P_ctrl = build_ctrl_transition_matrix(env, keys, multipliers, ctrl_keys, ctrl_multipliers)
    else:
        P_ctrl = build_ctrl_transition_matrix(env, ctrl_keys, ctrl_multipliers, ctrl_keys, ctrl_multipliers)

    # Visitation counters for uncontrollable dynamics
    N_ssp = np.zeros((H, S_unctrl, S_unctrl), dtype=np.int32)
    P_unctrl = np.zeros((H, S_unctrl, S_unctrl), dtype=np.float32)
    P_unctrl[:] = 1.0 / max(1, S_unctrl)
    
    # Value functions
    V = np.zeros((H+1, S), dtype=np.float32)
    Q = np.zeros((H, S, A), dtype=np.float32)
    
    # Precompute index mappings for all states
    s_ctrl_idx_map = np.zeros(S, dtype=int)
    s_unctrl_idx_map = np.zeros(S, dtype=int)
    
    # We also need the inverse: Map (ctrl, unctrl) -> Global S
    # We use -1 to catch any unreachable states if the grid isn't perfect
    grid_to_global_s = np.full((S_ctrl, S_unctrl), -1, dtype=int)
    
    for s in range(S):
        state = key_to_obs(s, env, keys, multipliers)
        sc = obs_to_key(state, ctrl_keys, ctrl_multipliers)
        su = obs_to_key(state, unctrl_keys, unctrl_multipliers)
        s_ctrl_idx_map[s] = sc
        s_unctrl_idx_map[s] = su
        grid_to_global_s[sc, su] = s

    def _update_unctrl_dynamics(h, s_unctrl, s_unctrl_next):
        """
        Update the empirical estimates of the MDP and the exploration bonus.

        Args:
            h (int): Time step.
            s (int): Current state.
            s_next (int): Next state.
        """
        s_idx = obs_to_key(s_unctrl, unctrl_keys, unctrl_multipliers)
        s_next_idx = obs_to_key(s_unctrl_next, unctrl_keys, unctrl_multipliers)
        
        N_ssp[h, s_idx, s_next_idx] += 1
        #P_unctrl[h, s_idx, s_next_idx] = N_ssp[h, s_idx, s_next_idx] / np.sum(N_ssp[h, s_idx, :])  # MLE
        row_sum = np.sum(N_ssp[h, s_idx, :])
        if row_sum > 0:
            P_unctrl[h, s_idx, :] = N_ssp[h, s_idx, :] / row_sum
            
    # Training procedure of PTO
    for episode in trange(n_episodes, desc="Training PTO"):
        # Validation step every 1000 episodes during training
        if episode % args['eval_every'] == 0:
            best_eval_reward, best_eval_episode, eval_counter = validation_step(
                env_name=args['env'],
                env_id=args['env_id'],
                eval_params=eval_params,
                episode=episode, 
                eval_episodes=args['eval_episodes'],
                Q=Q, 
                keys=keys, 
                multipliers=multipliers,
                tol=args['tol'], 
                eval_counter=eval_counter,
                exp_name=out_path,
                learning_curve=learning_curve,
                best_eval_reward=best_eval_reward,
                best_eval_episode=best_eval_episode,
                writer=writer,
                train_seed=seed,
                dest_path=dest_path,                
                eval_seed=eval_seed
                ) 
            
            if eval_counter == args['max_no_improvement']:
                break
            
        obs, _ = env.reset()
        obs_key = obs_to_key(obs, keys, multipliers)
        
        cumulated_reward = 0.
        h = 0
        done = False
                
        while not done:
            # randomized argmax on ties
            q_vals = Q[h, obs_key, :]
            action = int(rng.choice(np.flatnonzero(q_vals == q_vals.max())))  

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs_key = obs_to_key(next_obs, keys, multipliers)

            # Update empirical estimates of uncontrollable dynamics
            unctrl_obs = env.unwrapped.get_unctrl_obs(obs)
            unctrl_obs_next = env.unwrapped.get_unctrl_obs(next_obs)
            _update_unctrl_dynamics(h, unctrl_obs, unctrl_obs_next)
            
            cumulated_reward += reward
            done = terminated or truncated
            
            obs, obs_key = next_obs, next_obs_key
            h += 1

        writer.add_scalar('Training/MeanEpisodeReward', cumulated_reward, episode)
        writer.add_scalar('Training/MedianEpisodeReward', np.median(cumulated_reward), episode)
        
        # --- Split Bellman Backup (No P_hat) ---
        # Instead of P_hat (S,A,S), we use P_ctrl and P_unctrl separately.
        # Logic: E[V'] = Sum_sc_next( P_ctrl * Sum_su_next( P_unctrl * V(sc_next, su_next) ) )
        for h in reversed(range(H)):
            V_next = V[h + 1]
            
            # Transform V into a grid (S_ctrl, S_unctrl). This allows us to multiply with P_unctrl efficiently.
            #V_grid = V_next[grid_to_global_s] 
            V_next_padded = np.append(V_next, 0.0)
            V_grid = V_next_padded[grid_to_global_s]
            
            # Expectation over Uncontrollable Dynamics
            V_avg_unctrl = V_grid @ P_unctrl[h].T 
            
            # Expectation over Controllable Dynamics
            if state_factorizability:
                # P_ctrl is (S_ctrl, A, S_ctrl)
                Q_factorized = np.einsum("kax,xu->kau", P_ctrl, V_avg_unctrl)
                
                # Flatten back to global Q[s, a]
                Q[h] = R + Q_factorized[s_ctrl_idx_map, :, s_unctrl_idx_map]
                
            else:
                # P_ctrl is (S, A, S_ctrl) because dynamics depend on full state
                # Expand V_avg_unctrl to match global S
                V_relevant = V_avg_unctrl[:, s_unctrl_idx_map].T 
                
                # Compute expectation
                expectation = np.einsum("sax,sx->sa", P_ctrl, V_relevant)
                Q[h] = R + expectation
            
            # Update V for the next iteration (previous time step)
            V[h] = np.max(Q[h], axis=1)
            
        # # Assemble the P_hat from the controllable and uncontrollable parts
        # P_hat = np.zeros((S, A, S), dtype=np.float32)
        # for s in range(S):
        #     sc = s_ctrl_idx_map[s] if state_factorizability else s
        #     su = s_unctrl_idx_map[s]
        #     for a in range(A):
        #         for sp in range(S):
        #             scp = s_ctrl_idx_map[sp]
        #             sup = s_unctrl_idx_map[sp]
        #             P_hat[s, a, sp] = P_ctrl[sc, a, scp] * P_unctrl[su, sup]
        
        # # Now backward iteration
        # for h in reversed(range(H)):
        #     Q[h] = R + np.einsum("sax,x->sa", P_hat, V[h + 1])
        #     V[h] = np.max(Q[h], axis=1)
        
        writer.add_scalar('Q/Max', np.max(Q), episode)
        writer.add_scalar('Q/Min', np.min(Q), episode)
        writer.add_histogram('Q/Values', Q, episode)

    os.makedirs(f"{dest_path}/logs/results/{out_path}/{seed}/", exist_ok=True)
    with open(f"{dest_path}/logs/results/{out_path}/{seed}/learning_curve.json", "w", encoding="utf8") as output_file:
        json.dump(learning_curve, output_file)

    with open(f"{dest_path}/logs/results/{out_path}/{seed}/world_models.pkl", "wb") as model_output:
        pickle.dump({'V': V, 'N_ssp': N_ssp}, model_output)
    
    print("Training complete!")
    writer.close()

