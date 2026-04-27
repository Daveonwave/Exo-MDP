from exomdp.elevator.simulator.passenger import generate_arrival_distribution
from exomdp.elevator.elevator_env import ElevatorEnv
from rich.pretty import pprint
from IPython.display import clear_output
from time import sleep
import numpy as np


def random_policy(env, args, eval_params, settings=None, writer=None):
    n_episodes = args['eval_episodes']
    exp_name = args['exp_name']

    returns = []

    for episode in range(n_episodes):
        _, _ = env.reset()
        done = False
        cumulated_reward = 0
        
        while not done:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            
            cumulated_reward += reward
            done = terminated or truncated
            # pprint(f"Episode: {episode}, Step: {info['current_time']}, Action: {action}, Reward: {reward}, Cumulative Reward: {cumulated_reward}")
        
        returns.append(cumulated_reward)
    print(f"Average return over {n_episodes} episodes: {np.mean(returns)}")


def longest_queue_first(env, args, eval_params, settings=None, writer=None):
    n_episodes = args['eval_episodes']
    exp_name = args['exp_name']
    returns = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        
        done = False
        cumulated_reward = 0
        
        while not done:
            if obs['n_passengers'] > 0:
                if obs['current_position'] == settings['min_floor']:
                    action = 1  # stay still
                else:
                    action = 0  # move down
            else:
                # Check if there are any passengers waiting at the current floor
                len_max_queue, idx_max_queue = max((queue, idx + 1) for idx, queue in enumerate(obs['floor_queues']))
                # TODO: how to deal with ties?
                
                if len_max_queue == 0:
                    action = 1  # stay still
                elif obs['current_position'] == (idx_max_queue) * settings['floor_height']:
                    action = 1  # stay still
                elif obs['current_position'] < (idx_max_queue) * settings['floor_height']:
                    action = 2  # move up
                elif obs['current_position'] > (idx_max_queue) * settings['floor_height']:
                    action = 0  # move down
                else:
                    action = 1  # stay still
                    
            obs, reward, terminated, truncated, _ = env.step(action)
            
            cumulated_reward += reward
            done = terminated or truncated
            # pprint(f"Episode: {episode}, Step: {info['current_time']}, Action: {action}, "
            #        f"Reward: {reward}, Cumulative Reward: {cumulated_reward}, "
            #        f"Current Pos: {obs['current_position']}, Passengers: {obs['n_passengers']}, "
            #        f"Queues: {obs['floor_queues']}")
        returns.append(cumulated_reward)
        #print(f"Episode {episode} finished with cumulative reward: {cumulated_reward}")
    print(f"Average return over {n_episodes} episodes: {np.mean(returns)}")
