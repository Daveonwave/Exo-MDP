import os
import json
import time
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from tqdm.rich import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.utils import *
from algo.evaluation import validation_step


class NormalizeWrapper(gym.Wrapper):
    """
    Transforms the Dict observation {'amount': 50, 'price': 200} 
    into a flat, normalized float vector [0.5, 0.2].
    """
    def __init__(self, env):
        super().__init__(env)        
        # Extract bounds from the env to normalize correctly
        self.max_amount = env.observation_space['amount'].n - 1
        self.max_price = env.observation_space['price'].n - 1
        self.S = 2 
        
        # For time-aware envs
        if 'time' in env.observation_space.spaces:
            self.max_time = env.observation_space['time'].n - 1
            self.max_time = env.unwrapped.N
            self.S = 3
        
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.S,), dtype=np.float32)

    def _normalize(self, obs):
        # Scale values to [0, 1] range
        norm_amount = obs['amount'] / self.max_amount
        norm_price = obs['price'] / self.max_price
        
        if self.S == 3:
            norm_time = obs['time'] / self.max_time
            return np.array([norm_amount, norm_price, norm_time], dtype=np.float32)
        
        return np.array([norm_amount, norm_price], dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info


def make_env(env_id, settings, rng):
    def thunk():
        env = gym.make(env_id, settings=settings, seed=rng.integers(0, 1e6))
        env = NormalizeWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, grid_bounds=None):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.grid_bounds = grid_bounds
        
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(), # Standard ReLU is generally preferred over LeakyReLU for DQN
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.actor(x)
    
    def get_policy_as_Q(self):
        """
        Reconstructs the full (S_discrete, A) table for validation, perfectly mirroring PPO implementation.
        """
        if self.grid_bounds is None:
            raise ValueError("grid_bounds not set in QNetwork init")
            
        max_amt, max_prc, max_time = self.grid_bounds
        
        # Generate all possible integer states
        amounts = np.arange(max_amt + 1)
        prices = np.arange(max_prc + 1)
                
        # Create grid: Meshgrid indexing 'ij' ensures amount varies fast, price varies slowly
        grid_prc, grid_amt = np.meshgrid(prices, amounts, indexing='ij')
        
        # Normalize (Mirroring the wrapper logic)
        norm_amt = grid_amt.flatten() / max_amt
        norm_prc = grid_prc.flatten() / max_prc
        
        flat_time = np.zeros_like(norm_amt, dtype=np.float32)
        
        if self.state_size == 3:
            all_states = np.stack((norm_amt, norm_prc, flat_time), axis=1)
        else:
            all_states = np.stack((norm_amt, norm_prc), axis=1)
            
        all_states_tensor = torch.FloatTensor(all_states).to(next(self.parameters()).device)
        
        # Predict Q-Values for the whole grid at once
        with torch.no_grad():
            q_values = self.forward(all_states_tensor)
            return q_values.cpu().numpy()


class Memory:
    def __init__(self, max_size, state_size, rng):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.rng = rng
        
        # Preallocate memory for massive speedups
        self.states = np.zeros((max_size, state_size), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int64)
        self.next_states = np.zeros((max_size, state_size), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

    def update(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = self.rng.integers(0, self.size, size=batch_size)
        
        # Slicing preallocated arrays is practically instantaneous
        batch_states = torch.as_tensor(self.states[idx], device=device)
        batch_actions = torch.as_tensor(self.actions[idx], device=device)
        batch_next_states = torch.as_tensor(self.next_states[idx], device=device)
        batch_rewards = torch.as_tensor(self.rewards[idx], device=device)
        batch_dones = torch.as_tensor(self.dones[idx], device=device)
        
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones

def train_step(batch_size, current, target, optim, memory, gamma, device, max_grad_norm):
    states, actions, next_states, rewards, is_done = memory.sample(batch_size, device)

    # 1. Obtain current Q(s, a)
    q_values = current(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # 2. Double DQN Logic: Select best action using current network, evaluate using target network
    with torch.no_grad():
        next_q_values_current = current(next_states)
        best_next_actions = torch.argmax(next_q_values_current, dim=1).unsqueeze(1)

        next_q_values_target = target(next_states)
        next_q_value = next_q_values_target.gather(1, best_next_actions).squeeze(1)

        expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    # 3. Optimize Temporal Difference Loss
    #loss = F.mse_loss(q_value, expected_q_value)
    loss = F.smooth_l1_loss(q_value, expected_q_value)

    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(current.parameters(), max_grad_norm)
    optim.step()
    
    return loss.item()


def train(env, args, eval_params, seed=None, model_file=None, settings=None):
    if args.get('dest_folder') is not None and os.path.exists(args['dest_folder']):
        dest_path = args['dest_folder']
    else:
        dest_path = './'
    writer, out_path = init_writer(f"{args['env']}/{args['exp_name']}/{args['algo']}", args, seed, dest_folder=dest_path)

    # Precompute state index mappings
    keys, multipliers = build_state_index_map(env)
    S = get_state_size(env)
    A = env.action_space.n
    print(f"State size: {S}, Action size: {A}")
    print(f"Keys: {keys}, Multipliers: {multipliers}")
    grid_bounds = (env.observation_space['amount'].n - 1, 
                   env.observation_space['price'].n - 1,
                   settings['horizon'])
    env.close()  

    # Hyperparameters mapped to your CLI configurations
    num_episodes = args['num_iterations']
    batch_size = args.get('minibatch_size', 64)
    gamma = args.get('gamma', 0.99)
    lr = args.get('alpha', 1e-3)
    max_grad_norm = args.get('max_grad_norm', 0.5)
    
    # DDQN Specific Hyperparameters
    eps = args.get('epsilon', 1.0)
    eps_decay = args.get('epsilon_decay', 0.998)
    eps_min = args.get('epsilon_min', 0.01)
    target_update_freq = 3000 
    max_memory_size = 10000

    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    device = torch.device("cuda" if torch.cuda.is_available() and args.get('cuda', False) else "cpu")
        
    # Environment Setup
    env_id = f"exomdp/{args['env_id']}" if "exomdp" not in args.get('env_id', '') else args['env_id']
    train_env = make_env(env_id=env_id, settings=settings, rng=rng)()
    
    # Check shape dynamically based on time dependency
    state_size = train_env.observation_space.shape[0]

    # Initialize DDQN Agent Networks
    Q_1 = QNetwork(state_size=state_size, action_size=A, grid_bounds=grid_bounds).to(device)
    Q_2 = QNetwork(state_size=state_size, action_size=A, grid_bounds=grid_bounds).to(device)
    
    # Synchronize and freeze target network
    Q_2.load_state_dict(Q_1.state_dict())
    for param in Q_2.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(Q_1.parameters(), lr=lr)
    memory = Memory(max_memory_size, state_size, rng)
    
    # Storage and Progress Setup
    global_step = 0
    start_time = time.time()
    
    best_eval_episode = 0
    best_eval_reward = -np.inf
    eval_counter = 0
    learning_curve = {}
    
    # Training Loop
    for episode in trange(0, num_episodes, desc="DDQN Episodes"):
        
        # Validation Step
        if episode % args['eval_every'] == 0 and episode > 0:
            best_eval_reward, best_eval_episode, eval_counter = validation_step(
                env_name=args['env'],
                env_id=args['env_id'],
                eval_params=eval_params,
                episode=episode, 
                eval_episodes=args['eval_episodes'],
                Q=None, 
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
                eval_seed=args['eval_seed'],
                agent=Q_1, 
                wrapper_class=NormalizeWrapper
            ) 
            
            if eval_counter == args['max_no_improvement']:
                break
        
        # Anneal Learning Rate
        if args.get('anneal_lr', True):
            frac = 1.0 - episode / num_episodes
            optimizer.param_groups[0]["lr"] = frac * lr
        
        # Episode execution
        state, info = train_env.reset()
        done = False
        
        while not done:
            global_step += 1
            
            # Epsilon-Greedy Action Selection
            if rng.uniform() <= eps:
                action = rng.integers(A)
            else:
                state_t = torch.Tensor(state).to(device)
                with torch.no_grad():
                    q_values = Q_1(state_t)
                action = torch.argmax(q_values).item()
            
            # Environment step
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            
            # Store transition
            memory.update(state, action, reward, next_state, terminated)
            state = next_state
            
            # Log exact episodic metrics
            if "episode" in info:
                ep_r = info["episode"]["r"][0] if isinstance(info["episode"]["r"], list) else info["episode"]["r"]
                ep_l = info["episode"]["l"][0] if isinstance(info["episode"]["l"], list) else info["episode"]["l"]
                writer.add_scalar("charts/episodic_return", ep_r, global_step)
                writer.add_scalar("charts/episodic_length", ep_l, global_step)
            
            # DDQN Optimization step
            if memory.size >= batch_size:
                loss = train_step(batch_size, Q_1, Q_2, optimizer, memory, gamma, device, max_grad_norm)
                writer.add_scalar("losses/q_loss", loss, global_step)
            
            # Target Network update hook
            if global_step % target_update_freq == 0:
                Q_2.load_state_dict(Q_1.state_dict())
        
        # Decay Epsilon Configuration
        eps = max(eps * eps_decay, eps_min)
        
        # Telemetry
        writer.add_scalar("charts/epsilon", eps, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
    # Standardize Saved Output
    os.makedirs(f"{dest_path}/logs/results/{out_path}/{seed}/", exist_ok=True)
    with open(f"{dest_path}/logs/results/{out_path}/{seed}/learning_curve.json", "w", encoding="utf8") as output_file:
        json.dump(learning_curve, output_file)
    
    torch.save(Q_1.state_dict(), f"{dest_path}/logs/results/{out_path}/{seed}/ddqn_model.pth")
    
    print("Training complete!")
    train_env.close()
    writer.close()