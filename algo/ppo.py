import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from tqdm.rich import trange
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

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
        
        #norm_time = self.env.unwrapped.k / self.max_time
        # norm_time = obs['time'] / self.max_time
        
        return np.array([norm_amount, norm_price], dtype=np.float32)
        #return np.array([norm_amount, norm_price, norm_time], dtype=np.float32)

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


class PPO(nn.Module):
    def __init__(self, state_size, action_size, grid_bounds=None):
        super().__init__()
        self.state_size = state_size
        self.grid_bounds = grid_bounds
        
        # Simple Actor-Critic Network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_size), std=0.01),
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def get_policy_as_Q(self):
        """
        Reconstructs the full (S_discrete, A) table for validation.
        """
        if self.grid_bounds is None:
            raise ValueError("grid_bounds not set in PPO init")
            
        max_amt, max_prc, max_time = self.grid_bounds
        
        # Generate all possible integer states
        amounts = np.arange(max_amt + 1)
        prices = np.arange(max_prc + 1)
                
        # Create grid: (Total_States, 2)
        # Meshgrid indexing 'ij' ensures with amount varies fast, price varies slowly (0,0), (1,0)...
        grid_prc, grid_amt = np.meshgrid(prices, amounts, indexing='ij')
        
        # Normalize (Mirroring the wrapper logic)
        norm_amt = grid_amt.flatten() / max_amt
        norm_prc = grid_prc.flatten() / max_prc
        
        # Fix time at 0.0 (Start of day policy) or 0.5 (Mid-day) to visualize the Q-table "slice"
        flat_time = np.zeros_like(norm_amt, dtype=np.float32)
        
        # Stack into (N, 2) tensor
        # This represents "Every possible state" in normalized format
        
        all_states = np.stack((norm_amt, norm_prc, flat_time), axis=1)
        #all_states = np.stack((norm_amt, norm_prc), axis=1)
        all_states_tensor = torch.FloatTensor(all_states).to(next(self.parameters()).device)
        
        # 3. Predict Probabilities for the whole grid at once
        with torch.no_grad():
            action_logits = self.actor(all_states_tensor)
            return action_logits.cpu().numpy()


def train(env, args, eval_params, seed=None, model_file=None, settings=None):
    if args.get('dest_folder') is not None and os.path.exists(args['dest_folder']):
        dest_path = args['dest_folder']
    else:
        dest_path = './'
    writer, out_path = init_writer(f"{args['env']}/{args['exp_name']}/{args['algo']}", args, seed, dest_folder=dest_path)

    # Precompute state index mappings on a single environment instance
    keys, multipliers = build_state_index_map(env)
    S = get_state_size(env)
    A = env.action_space.n
    print(f"State size: {S}, Action size: {A}")
    print(f"Keys: {keys}, Multipliers: {multipliers}")
    grid_bounds = (env.observation_space['amount'].n - 1, 
                   env.observation_space['price'].n - 1,
                   settings['horizon'])
    env.close()  # Close the single instance used for setup

    # Hyperparameters
    num_envs = args.get('num_envs', 8) # Parallel environments
    num_steps = args.get('steps_per_iteration', 128) # Steps per environment per update
    batch_size = int(num_envs * num_steps)
    minibatch_size = args.get('num_minibatch', 64)
    num_iterations = args['num_iterations'] # Total PPO update iterations
    
    # Standard PPO Params
    gamma = args.get('gamma', 0.99)
    lr = args.get('alpha', 2.5e-4)
    gae_lambda = args.get('gae_lambda', 0.95)
    clip_coef = args.get('eps_clip', 0.2)
    ent_coef = args.get('ent_coef', 0.01)
    vf_coef = args.get('vf_coef', 0.5)
    max_grad_norm = args.get('max_grad_norm', 0.5)
    update_epochs = args.get('update_epochs', 4)
    target_kl = None # Optional: Stop update if KL divergence is too high
    
    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    device = torch.device("cuda" if torch.cuda.is_available() and args.get('cuda', False) else "cpu")
        
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id=f"exomdp/{args['env_id']}", settings=settings, rng=rng) for _ in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Initialize PPO Agent
    #agent = PPO(state_size=3, action_size=A, grid_bounds=grid_bounds).to(device)
    agent = PPO(state_size=2, action_size=A, grid_bounds=grid_bounds).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    
    # ALGO Logic: Storage setup
    #obs = torch.zeros((num_steps, num_envs, 3)).to(device)
    obs = torch.zeros((num_steps, num_envs, 2)).to(device)
    actions = torch.zeros((num_steps, num_envs)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    
    best_eval_episode = 0
    best_eval_reward = -np.inf
    eval_counter = 0
    learning_curve = {}
    
    # Training Loop
    for iteration in trange(0, num_iterations, desc="PPO Updates"):
        # Validation step
        if iteration % args['eval_every'] == 0 and iteration > 0:
            #current_policy_table = agent.get_policy_as_Q()
            
            best_eval_reward, best_eval_episode, eval_counter = validation_step(
                env_name=args['env'],
                env_id=args['env_id'],
                eval_params=eval_params,
                episode=iteration, 
                eval_episodes=args['eval_episodes'],
                Q=None, # Passing probabilities as Q
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
                agent=agent,
                wrapper_class=NormalizeWrapper
                ) 
            
            if eval_counter == args['max_no_improvement']:
                break
        
        # Anneal learning rate if specified
        if args.get('anneal_lr', True):
            frac = 1.0 - iteration / num_iterations
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow
        
        # Collect experience
        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                
            actions[step] = action
            logprobs[step] = logprob

            # Execute step
            next_obs_numpy, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            #print("->>> Next Obs NumPy:", next_obs_numpy, "Reward:", reward, "Terminations:", terminations, "Truncations:", truncations)
            #print()
            next_done = np.logical_or(terminations, truncations)
                        
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_numpy).to(device)
            next_done = torch.Tensor(next_done).to(device)
            
            if "episode" in infos:
                for i in range(num_envs):
                    if "_episode" in infos and infos["_episode"][i]:
                        #print(f"global_step={global_step}, episodic_return={infos['episode']['r'][i]}")
                        writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], global_step)
                        writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)
        
        # GAE (Generalized Advantage Estimation)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            
        # Policy update
        #b_obs = obs.reshape((-1,3))
        b_obs = obs.reshape((-1,2))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        
        for _ in range(update_epochs):
            # np.random.shuffle(b_inds)
            rng.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.get('norm_adv', True): 
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.get('clip_vloss', False):
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        #print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
    # Save Final Results
    os.makedirs(f"{dest_path}/logs/results/{out_path}/{seed}/", exist_ok=True)
    with open(f"{dest_path}/logs/results/{out_path}/{seed}/learning_curve.json", "w", encoding="utf8") as output_file:
        json.dump(learning_curve, output_file)
    
    # Save the PyTorch Model
    torch.save(agent.state_dict(), f"{dest_path}/logs/results/{out_path}/{seed}/ppo_model.pth")
    
    print("Training complete!")
    envs.close()
    writer.close()