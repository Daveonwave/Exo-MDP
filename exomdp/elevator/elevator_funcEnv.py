import numpy as np
from gymnasium import spaces
from gymnasium.experimental.functional import FuncEnv


class FunctionalElevatorEnv(FuncEnv):
    """
    Functional version of the Elevator environment for reinforcement learning.
    This environment is used as a generative model, not episodic.

    Args:
        FuncEnv (_type_): _description_
    """
    def __init__(self, settings):
        max_floor = settings['max_floor']
        min_floor = settings['min_floor']
        max_capacity = settings['max_capacity']
        max_queue_length = settings['max_queue_length']
        max_arrivals = settings['max_arrivals']
        movement_speed = settings['movement_speed']
        floor_height = settings['floor_height']

        self.observation_space = spaces.Dict({
            'pos': spaces.Discrete(int(floor_height * max_floor / movement_speed) + 1),
            'n_pass': spaces.Discrete(max_capacity + 1),
            'queues': spaces.MultiDiscrete(np.full(max_floor - min_floor, max_queue_length + 1)),
            'arrivals': spaces.MultiDiscrete(np.full(max_floor - min_floor, max_arrivals + 1))
        })

        self.action_space = spaces.Discrete(3)  # 0: down, 1: stay, 2: up

    def initial(self, rng, params=None):
        # Batch of N envs
        N = params.get('batch_size', 1)
        max_floor = params['max_floor']
        state = {
            'pos': rng.integers(0, max_floor, size=(N,)),  # random floor
            'n_pass': rng.integers(0, params['max_capacity'] + 1, size=(N,)),
            'queues': rng.integers(0, params['max_queue_length'] + 1, size=(N, max_floor)),
            'arrivals': np.zeros((N, max_floor)),
        }
        return state

    def transition(self, state, action, rng, params=None):   
        # Copy current state
        next_state = {
            'pos': np.copy(state['pos']),
            'n_pass': np.copy(state['n_pass']),
            'queues': np.copy(state['queues']),
            'arrivals': np.copy(state['arrivals']),
        }
                            
        # Move elevator according to action: 0=down, 1=stay, 2=up
        vertical_pos = state['pos'] * params['movement_speed'] / params['floor_height']  # Convert to vertical position in terms of floors   
        
        # Get indices for actions
        move_down_indices = np.where(action == 0)[0]  
        open_doors_indices = np.where(action == 1)[0] 
        move_up_indices = np.where(action == 2)[0]
        
        # CASE 1: THE ELEVATOR OPENS DOORS
        open_at_upper_floors_indices = np.intersect1d(open_doors_indices, np.where((vertical_pos % 1 == 0) & (vertical_pos != 0))[0])  # Indices where elevator is at a floor
        open_at_ground_floor_indices = np.intersect1d(open_doors_indices, np.where(vertical_pos == 0)[0])  # Indices where elevator is at ground floor
        
        # Update passengers at ground floor
        next_state['n_pass'][open_at_ground_floor_indices] = 0
        
        # Update passengers and queues at upper floors
        if len(open_at_upper_floors_indices) > 0:
            floors = vertical_pos[open_at_upper_floors_indices].astype(int)  # Get the floor indices where doors are opened
            
            current_pass = state['n_pass'][open_at_upper_floors_indices]
            waiting_pass = state['queues'][open_at_upper_floors_indices, floors - 1]
            
            # Compute loaded passengers at upper floors
            loaded_passengers = np.clip(current_pass + waiting_pass, 0, params['max_capacity'])
            boarded_passengers = loaded_passengers - current_pass
            
            # Update queue at upper floors
            next_state['n_pass'][open_at_upper_floors_indices] = loaded_passengers
            next_state['queues'][open_at_upper_floors_indices, floors - 1] -= boarded_passengers
            
        # CASE 2: THE ELEVATOR MOVES UP OR DOWN
        max_pos = params['max_floor'] * params['floor_height'] / params['movement_speed']
        next_state['pos'][move_up_indices] = np.clip(state['pos'][move_up_indices] + 1, 0, max_pos)  # Move up
        next_state['pos'][move_down_indices] = np.clip(state['pos'][move_down_indices] - 1, 0, max_pos)  # Move up
            
        # Update queues and arrivals
        next_state['queues'] = np.clip(next_state['queues'] + state['arrivals'], 0, params['max_queue_length']) 
        return next_state

    def observation(self, state, rng, params=None):
        # Return the current state as observation
        return {
            'pos': state['pos'],
            'n_pass': state['n_pass'],
            'queues': state['queues'],
            'arrivals': state['arrivals']
        }
        
    def reward(self, state, action, next_state, rng, params=None):
        rewards = np.zeros_like(state['pos'], dtype=np.float32)
        
        # Indices where elevator is opening doors
        open_doors_indices = np.where(action == 1)[0]
        # Indices where elevator is at ground floor
        at_ground_floor_indices = np.where(state['pos'] == 0)[0]  
        
        # Reward for delivering passengers to the ground floor
        if open_doors_indices.shape[0] != 0 and at_ground_floor_indices.shape[0] != 0:
            open_at_ground_floor_indices = np.intersect1d(open_doors_indices, at_ground_floor_indices) # Indices where elevator is at ground floor
            #print(open_at_ground_floor_indices)
            rewards[open_at_ground_floor_indices] += (
                state['n_pass'][open_at_ground_floor_indices] - 
                next_state['n_pass'][open_at_ground_floor_indices]
                ) * params['delivery_reward']  
            # Penalty for waiting passengers
            rewards += params['waiting_penalty'] * (np.sum(state['queues'], axis=1)) 
        else:
            # Penalty for waiting passengers
            rewards += params['waiting_penalty'] * (state['n_pass'] + np.sum(state['queues'], axis=1))  # Penalty for waiting passengers in the elevator and queues        
        return rewards
        
    def terminal(self, state, rng, params=None):
        # Done when step_count exceeds duration
        return np.zeros(state.shape[0], dtype=bool)