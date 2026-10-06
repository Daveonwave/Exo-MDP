from exomdp.utils import parameter_generator
from algo import ql, exaq, ucbvi, pto, ppo, exaq_mod
from algo.baselines import *
from rich.pretty import pprint
import gymnasium as gym
import multiprocessing as mp
import argparse
from algo.utils import init_writer


def run_single_seed(args, seed):
    """Run one independent training run with its own environment and seed."""
    print(f"\n=== Starting seed {seed} ===")

    # Reinitialize env settings per process (avoid cross-process sharing)
    env_settings = parameter_generator(world_file=f"exomdp/{args['env']}/{args['world']}")
    eval_env_settings = parameter_generator(world_file=f"exomdp/{args['env']}/{args['world']}")

    # Create environment
    if args['env'] == 'elevator':
        env = gym.make(f"exomdp/{args['env_id']}",
                       render_mode=None if not args['render'] else 'human',
                       settings=env_settings,
                       seed=seed)
    elif args['env'] == 'taxi':
        env = gym.make(f"exomdp/{args['env_id']}",
                           render_mode=None if not args['render'] else 'ansi',
                           settings=env_settings,
                           seed=seed)
    elif args['env'] == 'trading':
        env = gym.make(f"exomdp/{args['env_id']}",
                       render_mode=None,
                       settings=env_settings,
                       seed=seed)
    else:
        raise ValueError(f"Unsupported environment: {args['env']}")

    env.reset(seed=seed)

    # Run training for this seed
    algo = args['algo']
    if algo == 'ql':
        ql.train(env=env, args=args, settings=env_settings, eval_params=eval_env_settings, seed=seed)
    elif algo == 'exaq':
        exaq.train(env=env, args=args, settings=env_settings, eval_params=eval_env_settings, seed=seed)
    elif algo == 'exaq-mod':
        exaq_mod.train(env=env, args=args, settings=env_settings, eval_params=eval_env_settings, seed=seed)
    elif algo == 'ucbvi':
        ucbvi.train(env=env, args=args, settings=env_settings,
                    eval_params=eval_env_settings, horizon=env_settings['horizon'], seed=seed)
    elif algo == 'pto':
        pto.train(env=env, args=args, settings=env_settings, horizon=env_settings['horizon'], eval_params=eval_env_settings, seed=seed)
    elif algo == 'ppo':
        ppo.train(env=env, args=args, settings=env_settings, eval_params=eval_env_settings, seed=seed)
    elif algo == 'random':
        random_policy(env=env, args=args, settings=env_settings, eval_params=eval_env_settings)
    elif algo == 'longestFirst':
        longest_queue_first(env=env, args=args, settings=env_settings, eval_params=eval_env_settings)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    #env.close()
    print(f"=== Finished seed {seed} ===")



if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Run different RL algorithms on PCMDP environments.")
    
    # General parameters
    parser.add_argument('--env', type=str, choices=['elevator', 'taxi', 'trading'], required=True, help="Select the environment.")
    parser.add_argument('--env_id', type=str, default=None, help="Gym environment ID (if different from env).")
    parser.add_argument('--algo', type=str, choices=['ql', 'exaq', 'ucbvi', 'longestFirst', 'random', 'pto', 'ppo', 'exaq-mod'], required=True,
                        help="Algorithm to run: 'qlearning', 'qlearning_augmented', or 'ucbvi'.")
    parser.add_argument('--exp_name', type=str, help="Experiment name for logging.")
    parser.add_argument('--dest_folder', type=str, default=None, help="Path to server logs.")
    parser.add_argument('--world', type=str, default='world.yaml', help="World configuration file.")
    parser.add_argument('--n_episodes', type=int, default=1000, help="Number of training episodes.")
    parser.add_argument('--gamma', type=float, default=1.0, help="Discount factor (gamma).")
    parser.add_argument('--render', action='store_true', help="Render the environment (if supported).")
    
    # Evaluation parameters
    parser.add_argument('--tol', type=float, default=0.1, help="Tolerance for evaluation.")
    parser.add_argument('--max_no_improvement', type=int, default=100, help="Max episodes with no improvement before stopping.")
    parser.add_argument('--eval_every', type=int, default=10, help="Evaluate every N episodes.")
    parser.add_argument('--eval_episodes', type=int, default=30, help="Number of episodes for validation.")

    # Algorithm hyperparameters
    parser.add_argument('--epsilon', type=float, default=1.0, help="Initial epsilon for epsilon-greedy for Q-Learning.")
    parser.add_argument('--epsilon_decay', type=float, default=0.99, help="Epsilon decay rate for Q-Learning.")
    parser.add_argument('--epsilon_min', type=float, default=0.1, help="Minimum epsilon of Q-Learning.")
    parser.add_argument('--decay_type', type=str, choices=['exponential', 'linear', 'mixed'], default='exponential', help="Type of epsilon decay.")
    parser.add_argument('--alpha', type=float, default=0.1, help="Learning rate (alpha) for Q-Learning.")
    parser.add_argument('--c_bonus', type=float, default=1.0, help="Exploration bonus coefficient for UCBVI.")
    parser.add_argument('--delta', type=float, default=0.1, help="Confidence level for UCBVI.")
    parser.add_argument('--R_given', action='store_true', help="Whether the reward function is known in UCBVI.")
    parser.add_argument('--eps_clip', type=float, default=0.2, help="Clipping parameter for PPO.")
    parser.add_argument('--ent_coef', type=float, default=0.01, help="Entropy coefficient for PPO.")
    parser.add_argument('--vf_coef', type=float, default=0.5, help="Value function coefficient for PPO.")
    parser.add_argument('--gae_lambda', type=float, default=0.95, help="GAE lambda parameter for PPO.")
    parser.add_argument('--steps_per_iteration', type=int, default=128, help="Number of steps per environment per PPO update.")
    parser.add_argument('--num_envs', type=int, default=8, help="Number of parallel environments for PPO.")
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help="Max gradient norm for PPO.")
    parser.add_argument('--update_epochs', type=int, default=4, help="Number of update epochs per PPO iteration.")
    parser.add_argument('--minibatch_size', type=int, default=64, help="Minibatch size for PPO updates.")
    parser.add_argument('--num_iterations', type=int, default=1000, help="Number of PPO update iterations.")
    parser.add_argument('--anneal_lr', action='store_true', help="Whether to anneal the learning rate over time for PPO.")
    parser.add_argument('--norm_adv', action='store_true', help="Whether to normalize advantages for PPO.")
    
    # Seeds
    parser.add_argument('--train_seeds', nargs='+', type=int, default=[42], help="Random seed for reproducibility.")
    parser.add_argument('--eval_seed', type=int, default=None, help="Random seed for evaluation.")

    args = vars(parser.parse_args())
        
    # Use multiprocessing
    seeds = args['train_seeds']
    
    if len(seeds) > 1:
        n_processes = min(len(seeds), mp.cpu_count())
        
        print(f"Running {len(seeds)} seeds with up to {n_processes} parallel processes.")
        with mp.Pool(processes=n_processes) as pool:
            pool.starmap(run_single_seed, [(args, seed) for seed in seeds])
    else:
        run_single_seed(args, seeds[0])
        
    