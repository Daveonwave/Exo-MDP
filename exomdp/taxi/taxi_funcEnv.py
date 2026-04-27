import numpy as np
from gymnasium import spaces
from gymnasium.experimental.functional import FuncEnv

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
] 


class FunctionalTaxiEnv(FuncEnv):
    """
    Functional version of the Taxi environment for reinforcement learning.
    This environment is used as a generative model, not episodic.
    
    Args:
        FuncEnv (_type_): _description_ 
    """
    def __init__(self, num_states=500, num_actions=6):        
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

    def initial(self, rng, params=None):
        # Initialize the environment state (this is never called)
        N = params.get('batch_size', 1)

        taxi_rows = rng.integers(0, params['n_rows'], size=(N,))
        taxi_cols = rng.integers(0, params['n_cols'], size=(N,))
        # Passenger and destination at one of the four locations or not spawned
        pass_idxs = np.zeros(N, dtype=np.int32)
        dest_idxs = np.zeros(N, dtype=np.int32)
        
        do_spawn = rng.binomial(1, params['spawn_prob'], size=(N,)).astype(bool)
        pass_idxs[do_spawn] = rng.integers(0, 4, size=np.sum(do_spawn))
        dest_idxs[do_spawn] = rng.integers(0, 4, size=np.sum(do_spawn))
        pass_idxs[~do_spawn] = 5  # Passenger not present
        dest_idxs[~do_spawn] = 4  # Destination not present

        state = {'row': taxi_rows,
                 'col': taxi_cols,
                 'pass_idx': pass_idxs,
                 'dest_idx': dest_idxs}
        return state

    def transition(self, state, action, rng, params=None):
        N = params.get('batch_size', 1)
        desc = np.array(MAP, dtype='c')
        locs = np.array(params["locations"])
        
        # Get current state variables
        rows, cols, pass_idxs, dest_idxs = state.values()
        
        # Copy current state
        new_rows = np.copy(rows)
        new_cols = np.copy(cols)
        new_pass_idxs = np.copy(pass_idxs)
        new_dest_idxs = np.copy(dest_idxs)
        
        # Move South (0)
        south_mask = (action == 0)
        new_rows[south_mask] = np.minimum(rows[south_mask] + 1, params['n_rows'] - 1)
        
        # Move North (1)
        north_mask = (action == 1)
        new_rows[north_mask] = np.maximum(rows[north_mask] - 1, 0)
        
        # Move East (2) if not wall
        east_mask = (action == 2) & (desc[1 + rows, 2 * cols + 2] == b":")
        new_cols[east_mask] = np.minimum(cols[east_mask] + 1, params['n_cols'] - 1)

        # Move West (3) if not wall
        west_mask = (action == 3) & (desc[1 + rows, 2 * cols] == b":")
        new_cols[west_mask] = np.maximum(cols[west_mask] - 1, 0)
        
        # Pickup (4)
        pickup_mask = (action == 4) & (pass_idxs < 4)
        at_pickup_loc = (rows == locs[pass_idxs, 0]) & (cols == locs[pass_idxs, 1])
        successful_pickup = pickup_mask & at_pickup_loc
        new_pass_idxs[successful_pickup] = 4  # Passenger is now in taxi

        # Dropoff (5)
        dropoff_mask = (action == 5) & (pass_idxs == 4)
        at_dest_loc = (rows == locs[dest_idxs, 0]) & (cols == locs[dest_idxs, 1])
        successful_dropoff = dropoff_mask & at_dest_loc
        
        # Spawn new passenger with some probability
        do_spawn = rng.binomial(1, params['spawn_prob'], size=(N,)).astype(bool)
        not_present = (pass_idxs == 5)
        new_pass_idxs[not_present & do_spawn] = rng.integers(0, 4, size=np.sum(not_present & do_spawn))
        new_dest_idxs[not_present & do_spawn] = rng.integers(0, 4, size=np.sum(not_present & do_spawn))
        new_pass_idxs[not_present & ~do_spawn] = 5  # Passenger not present
        new_dest_idxs[not_present & ~do_spawn] = 4  # Destination not
        
        # Respawn passenger and destination -> capire meglio (TODO) 
        if successful_dropoff.any():
            new_pass_idxs[successful_dropoff & do_spawn] = rng.integers(0, 4, size=np.sum(successful_dropoff & do_spawn))
            new_dest_idxs[successful_dropoff & do_spawn] = rng.integers(0, 4, size=np.sum(successful_dropoff & do_spawn))
            new_pass_idxs[successful_dropoff & ~do_spawn] = 5  # Passenger not present
            new_dest_idxs[successful_dropoff & ~do_spawn] = 4  # Destination
        
        new_state = {'row': new_rows,
                     'col': new_cols,
                     'pass_idx': new_pass_idxs,
                     'dest_idx': new_dest_idxs}
        return new_state

    def observation(self, state, rng, params=None):
        # Return the current state as observation
        return state
    
    def reward(self, state, action, next_state, rng, params=None):
        N = params.get('batch_size', 1)
        rows, cols, pass_idxs, dest_idxs = state.values()
        locs = np.array(params["locations"])
                
        rewards = np.full(N, params['step_reward'], dtype=np.float32)
                
        # Successful pickup
        pickup_mask = (action == 4) & (pass_idxs < 4) & (dest_idxs < 4)
        at_pickup_loc = (rows == locs[pass_idxs, 0]) & (cols == locs[pass_idxs, 1])
        successful_pickup = pickup_mask & at_pickup_loc

        # Successful dropoff
        dropoff_mask = (action == 5) & (pass_idxs == 4) & (dest_idxs < 4)
        at_dest_loc = (rows == locs[dest_idxs, 0]) & (cols == locs[dest_idxs, 1])
        successful_dropoff = dropoff_mask & at_dest_loc
        
        # Taxi is at *any* valid location (R, G, Y, B)
        at_any_dest = (pass_idxs == 4) & (dest_idxs < 4) & np.any((rows[:, None] == locs[:4, 0]) & (cols[:, None] == locs[:4, 1]), axis=1)
        
        # Illegal actions
        illegal_pickup = (action == 4) & ~successful_pickup
        illegal_dropoff = (action == 5) & ~at_any_dest
        
        # if any(dest_idxs >= 4) and any(action == 5):
        #     print(action)
        #     print(f"illegal_pickup: {illegal_pickup}")
        #     print(f"illegal_dropoff: {illegal_dropoff}")
        #     print(f"pass_idxs: {pass_idxs}")
        #     print(f"dest_idxs: {dest_idxs}")
        #     exit(1)
    
        # Assign rewards
        rewards[illegal_pickup | illegal_dropoff] = params['illegal_reward']
        rewards[successful_dropoff] = params['dropoff_reward']
        
        return rewards

    def terminal(self, state, params=None):
        # The environment is never terminal in this generative model
        return np.zeros(state.shape[0], dtype=bool)