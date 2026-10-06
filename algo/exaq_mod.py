import os
from tqdm.rich import tqdm, trange
import numpy as np
import json
import numpy as np
from exomdp import FunctionalElevatorEnv, FunctionalTaxiEnv, FunctionalTradingEnv, FunctionalTaxiTrafficEnv
from algo.utils import *
from algo.evaluation import validation_step


def train(env, args, eval_params, seed=None, model_file=None, settings=None):  
    if args.get('dest_folder') is not None and os.path.exists(args['dest_folder']):
        dest_path = args['dest_folder']
    else:
        dest_path = './'
    writer, out_path = init_writer(f"{args['env']}/{args['exp_name']}/{args['algo']}", args, seed, dest_folder=dest_path)

    # Hyperparameters
    alpha = args['alpha']
    gamma = args['gamma']
    epsilon = args['epsilon']
    epsilon_decay = args['epsilon_decay']
    epsilon_min = args['epsilon_min']
    n_episodes = args['n_episodes']
    rng = np.random.default_rng(seed=seed)
    eval_seed = args['eval_seed']
    
    best_eval_reward = -np.inf
    best_eval_episode = 0
    eval_counter = 0
    learning_curve = {}
    keys, multipliers = build_state_index_map(env)
    print(f"Keys: {keys}, Multipliers: {multipliers}")
    
    S = get_state_size(env)
    A = env.action_space.n
    Q = np.zeros((S, A))
    
    ctrl_vars = env.unwrapped.get_controllables()
    unctrl_vars = env.unwrapped.get_uncontrollables()
    
    # Functional environment for vectorized operations
    if args['env_id'] == 'elevator-v0':
        func_env = FunctionalElevatorEnv(settings=settings)
    elif args['env_id'] == 'taxi-v0':
        func_env = FunctionalTaxiEnv(num_states=S, num_actions=A)
    elif args['env_id'] == 'taxi-traffic-v0':
        func_env = FunctionalTaxiTrafficEnv(num_states=S, num_actions=A)
    elif args['env_id'] == 'trading-v0':
        func_env = FunctionalTradingEnv(settings=settings)
    else:
        raise ValueError("Unsupported environment for ExAQ.")
        
    for episode in trange(n_episodes, desc="Training ExAQ"):
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
            
        # Sample the controllable obs, action, and uncontrollable obs from the environment
        obs, _ = env.reset()  
        states = []
        actions = []
        uncontrollable_obs = []
        done = False
        
        while not done:
            obs_key = obs_to_key(obs, keys, multipliers)
            #action = rng.integers(A) 
            # if rng.uniform() < epsilon:
            #     action = rng.integers(A)    # Explore action space
            # else:
            #     action = np.argmax(Q[obs_key, :])
            action = int(np.argmax(Q[obs_key, :]))  # Greedy action selection
            
            states.append(obs)
            actions.append(action)
            uncontrollable_obs.append(env.unwrapped.get_unctrl_obs(obs))
            
            new_obs, reward, terminated, truncated, _ = env.step(action)
            obs = new_obs
            done = terminated or truncated
        
        # Add last uncontrollable observation after the episode ends       
        uncontrollable_obs.append(env.unwrapped.get_unctrl_obs(obs))
        batch_size = 1
        cumulated_reward = 0.0
        td_errors = []

        for state, action, unctrl_obs, next_unctrl_obs in zip(
            states,
            actions,
            uncontrollable_obs[:-1],
            uncontrollable_obs[1:],
        ):
            controllable_state = {key: np.asarray([state[key]]) for key in ctrl_vars}
            
            # For each controllable state, we add the current uncontrollable observation
            vec_state = compose_vec_state(controllable_state, unctrl_obs, batch_size)      
            vec_action = np.asanyarray([action])
            params = {"batch_size": batch_size, **(settings or {})}
                
            # Get the next state from the environment
            next_vec_state = func_env.transition(state=vec_state, action=vec_action, rng=rng, params=params)
            next_vec_state = compose_vec_state(next_vec_state, next_unctrl_obs, batch_size)
                        
            # Calculate rewards
            reward = func_env.reward(state=vec_state, action=vec_action, next_state=next_vec_state, rng=None, params=params)[0]
            cumulated_reward += reward
            
            # Q-update
            state_index = flatten_state(vec_state, keys, multipliers)[0]
            next_state_index = flatten_state(next_vec_state, keys, multipliers)[0]
            best_next_action = int(np.argmax(Q[next_state_index, :]))
            td_target = reward + gamma * Q[next_state_index, best_next_action] 
            td_error = td_target - Q[state_index, vec_action]
            Q[state_index, vec_action] += alpha * td_error
            td_errors.append(td_error)
    
        writer.add_scalar('Training/MeanEpisodeReward', cumulated_reward, episode)
        writer.add_scalar('Training/TD_Error', np.mean(td_errors), episode)
        
        writer.add_scalar('Q/Max', np.max(Q), episode)
        writer.add_scalar('Q/Min', np.min(Q), episode)
        writer.add_histogram('Q/Values', Q.flatten(), episode)

        # Decay epsilon
        # if args['decay_type'] == 'linear':
        #     epsilon -= (1.0 - epsilon_min) / (n_episodes)
        #     epsilon = max(epsilon_min, epsilon)
        # elif args['decay_type'] == 'exponential':
        #     epsilon = max(epsilon_min, epsilon * epsilon_decay)
        # elif args['decay_type'] == 'mixed': # linear first half, exponential second half
        #     if episode < n_episodes // 2:
        #         epsilon -= (1.0 - epsilon_min) / (n_episodes)
        #         epsilon = max(epsilon_min, epsilon)
        #     else:
        #         epsilon = max(epsilon_min, epsilon * epsilon_decay)
        # else:
        #     raise ValueError(f"Unsupported decay type: {args['decay_type']}")
        # writer.add_scalar('Exploration/Epsilon', epsilon, episode)
        
    os.makedirs(f"{dest_path}/logs/results/{out_path}/{seed}/", exist_ok=True)
    with open(f"{dest_path}/logs/results/{out_path}/{seed}/learning_curve.json", "w", encoding="utf8") as output_file:
        json.dump(learning_curve, output_file)
    
    print("Training complete!")
    writer.close()
