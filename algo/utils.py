import numpy as np
import json
import gymnasium as gym
import itertools
import random
import numpy as np
import os
import time
from tensorboardX import SummaryWriter


def init_writer(exp_name, args, seed, dest_folder):
    """Initialize TensorBoard writer and log configuration."""
    time_identifier = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(
        dest_folder,
        "logs/tensorboard",
        exp_name,
        time_identifier,
        str(seed)
    )
    os.makedirs(out_path, exist_ok=True)
    writer = SummaryWriter(out_path)

    # Log hyperparameters as text for full readability
    writer.add_text("config/parameters", json.dumps(args, indent=4))
    with open(os.path.join(dest_folder, "logs/tensorboard", exp_name, time_identifier, "config.json"), "w") as f:
        json.dump(args, f, indent=4)

    return writer, os.path.join(exp_name, time_identifier)


def get_state_size(env):
    """
    Calculate the size of the state space based on the environment's observation space.

    Args:
        env (gym.Env): The environment to calculate the state size for.
    """
    if env.observation_space.__class__ == gym.spaces.Discrete:
        return env.observation_space.n
    elif env.observation_space.__class__ == gym.spaces.MultiDiscrete:
        return np.prod(env.observation_space.nvec)
    elif env.observation_space.__class__ == gym.spaces.Dict:
        return get_composite_state_size(env)
    else:
        raise ValueError(f"Unsupported observation space type: {type(env.observation_space)}")


def get_composite_state_size(env, keys=None):
    """
    Calculate the size of the composite state space.

    Args:
        env (gym.Env): The environment to calculate the state size for.
    """
    size = 1
    obs_space = env.observation_space
    
    if keys is None:
        keys = obs_space.spaces.keys()

    for key in keys:
        space = obs_space.spaces[key]
        if isinstance(space, gym.spaces.Discrete):
            size *= space.n
        elif isinstance(space, gym.spaces.MultiDiscrete):
            size *= np.prod(space.nvec)
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")
        
    return size


def build_state_index_map(env, keys=None):
    """
    Build a multiplier vector to convert state dict into unique flat indices.
    
    Returns:
        keys: list of observation keys used (order matters)
        multipliers: array of multipliers to convert each state component to index
    """
    if not isinstance(env.observation_space, gym.spaces.Dict):
        return None, None
    
    space = env.observation_space
    if keys is None:
        keys = env.unwrapped.get_space_vars()
    
    sizes = []
    
    for key in keys:
        subspace = space.spaces[key]
        if isinstance(subspace, gym.spaces.Discrete):
            sizes.append(subspace.n)
        elif isinstance(subspace, gym.spaces.MultiDiscrete):
            sizes.extend(subspace.nvec)
        else:
            raise ValueError(f"Unsupported space type: {type(subspace)} for key '{key}'")

    sizes = np.array(sizes)
    multipliers = np.cumprod([1] + list(sizes[:-1]))
    return keys, multipliers


def obs_to_key(obs, keys=None, multipliers=None):
    """
    Convert observation to flat Q-table index.
    
    Args:
        obs (_type_): _description_
        keys (_type_, optional): _description_. Defaults to None.
        multipliers (_type_, optional): _description_. Defaults to None.
    """
    if keys is None or multipliers is None:
        # Simple case: assume obs is already flat
        if isinstance(obs, dict):
            raise ValueError("Keys and multipliers must be provided for dict observations.")
        return obs  # Assume obs is already a flat index
    return flatten_state(obs, keys, multipliers, _no_batch=True)


def key_to_obs(index, env, keys=None, multipliers=None):
    """
    Convert flat Q-table index back to observation dict.
    
    Args:
        index (_type_): _description_
        env (_type_): _description_
        keys (_type_, optional): _description_. Defaults to None.       
        multipliers (_type_, optional): _description_. Defaults to None.
    """
    if keys is None or multipliers is None:
        raise ValueError("Keys and multipliers must be provided to convert index to observation.")
    
    obs = {}
    rem = index
    space = env.observation_space
    i = len(multipliers) - 1

    for key in reversed(keys):
        subspace = space.spaces[key]

        if isinstance(subspace, gym.spaces.Discrete):
            obs[key] = int(rem // multipliers[i])
            rem = rem % multipliers[i]
            i -= 1

        elif isinstance(subspace, gym.spaces.MultiDiscrete):
            vals = []
            for n in reversed(subspace.nvec):
                vals.append(int(rem // multipliers[i]))
                rem = rem % multipliers[i]
                i -= 1
            obs[key] = np.array(list(reversed(vals)), dtype=int)
        else:
            raise ValueError(f"Unsupported subspace type: {type(subspace)}")
    
    return obs
        

def flatten_state(state, keys, multipliers, _no_batch=False):
    """
    Convert a batch of states (dict of arrays) into flat Q-table indices.

    Args:
        state_dict: dict of arrays (shape: (batch_size, ...))
        keys: keys used in the state_dict
        multipliers: multipliers returned by build_state_index_map()

    Returns:
        np.ndarray: (batch_size,) Q-table indices
    """
    if not isinstance(state, dict):
        return state  # Assume already flat
    
    components = []
        
    if _no_batch:
        # For evaluation, we assume state_dict is not a batch
        for key in keys:
            vals = state[key]
            if isinstance(vals, np.ndarray):
                components.extend(vals)  # Flatten the array to a list
            else:
                components.append(vals)
        components = np.array(components, dtype=np.int64)
        
    else:
        for key in keys:
            vals = state[key] 
            if vals.ndim == 1:
                components.append(vals)
            else:  # MultiDiscrete case
                # Split MultiDiscrete into separate components
                for i in range(vals.shape[1]):
                    components.append(vals[:, i])

        # Stack all components shape: (num_components, batch_size)
        components = np.stack(components, axis=0)
    
    # Compute dot product with multipliers to get flattened index
    indices = np.tensordot(multipliers, components, axes=(0, 0))
    return indices  # shape: (batch_size,)

    
def get_controllable_space(env):
    """
    Generates all combinations of the controllable variables within the observation space.

    Args:
        env: The Gymnasium environment.

    Returns:
        dict of np.ndarray: Keys are observation components ('current_position', etc.), 
                            values are np.ndarrays of shape (num_combinations, ...) 
                            representing all possible states.
    """
    keys = env.unwrapped.get_controllables()
    spaces_to_include = {k: env.observation_space.spaces[k] for k in keys}

    values = []
    shapes = {}

    for key, space in spaces_to_include.items():
        if isinstance(space, gym.spaces.Discrete):
            vals = list(range(space.start, space.start + space.n))
            values.append(vals)
            shapes[key] = ()
            
        elif isinstance(space, gym.spaces.MultiDiscrete):
            dim_ranges = [list(range(s, s + n)) for n, s in zip(space.nvec, space.start)]
            multi_vals = list(itertools.product(*dim_ranges))  # Each is a tuple per MultiDiscrete
            values.append(multi_vals)
            shapes[key] = (len(space.nvec),)  # MultiDiscrete has multiple subdims

        else:
            raise ValueError(f"Unsupported space type: {type(space)} for key '{key}'")

    # Get Cartesian product over the controllable variables
    combinations = list(itertools.product(*values))

    # Split the product into per-key lists
    result = {k: [] for k in keys}

    for combo in combinations:
        for i, key in enumerate(keys):
            result[key].append(combo[i])

    # Convert lists to NumPy arrays, reshape if MultiDiscrete
    for key in keys:
        result[key] = np.array(result[key])
        # If MultiDiscrete, ensure correct final shape
        if len(result[key].shape) == 2 and shapes[key] != ():
            pass  # shape is (num_combinations, multi_discrete_dim)
        elif shapes[key] == ():
            result[key] = result[key].reshape(-1)  # shape (num_combinations,)

    return result

def compose_vec_state(ctrl_obs, unctrl_obs, batch_size):
    """
    Compose full observation from controllable and uncontrollable parts.

    Args:
        ctrl_obs: dict of controllable observation components
        unctrl_obs: dict of uncontrollable observation components
        batch_size: int, number of samples in the batch
    """
    # For other space types, we need to handle them differently
    vec_state = {}

    # Add controllable observations
    for key in ctrl_obs:
        vec_state[key] = ctrl_obs[key]
    
    # Add uncontrollable observations
    for key in unctrl_obs:
        shape = (batch_size, unctrl_obs[key].shape[0]) if isinstance(unctrl_obs[key], np.ndarray) else (batch_size)
        vec_state[key] = np.full(shape=shape, fill_value=unctrl_obs[key], dtype=np.int64)
    
    return vec_state


def build_ctrl_transition_matrix(env, state_keys, state_multipliers, next_state_keys, next_state_multipliers):
    """
    Build the transition matrix for the controllable part of the state space.

    Returns:
        np.ndarray: Transition matrix of shape (num_ctrl_states, num_actions, num_ctrl_states)
    """  
    S = get_composite_state_size(env, keys=state_keys)
    Sp = get_composite_state_size(env, keys=next_state_keys)
    A = env.action_space.n
    
    P = np.zeros((S, A, Sp), dtype=np.float32)
    
    for s in range(S):
        state = key_to_obs(s, env, keys=state_keys, multipliers=state_multipliers)
        for a in range(A):
            # Transition using the environment's transition function (transition can be stochastic)
            next_states, probs = env.unwrapped._ctrl_transitions(**{**state, "action": a})
            for sp, prob in zip(next_states, probs):
                next_ctrl_state = {key: sp[i] for i, key in enumerate(next_state_keys)}
                next_idx = obs_to_key(next_ctrl_state, keys=next_state_keys, multipliers=next_state_multipliers)
                P[s, a, next_idx] += prob
    return P


def build_reward_matrix(env, keys, multipliers):
    """
    Build the reward matrix for the environment.

    Returns:
        np.ndarray: Reward matrix of shape (num_states, num_actions)
    """
    S = get_composite_state_size(env, keys=keys)
    A = env.action_space.n
    
    R = np.zeros((S, A), dtype=np.float32)
    
    for s in range(S):
        state = key_to_obs(s, env, keys, multipliers)
        for a in range(A):
            R[s, a] = env.unwrapped.reward_from_transition(state, a)
    
    return R
