from gymnasium import Env, spaces
import matplotlib.pyplot as plt
import numpy as np
import pygame
from .simulator.elevator import Elevator
from .simulator.passenger import generate_arrival_distribution
import itertools


class ElevatorEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def _transitions(self, pos, n_pass, queues, arrivals, action):
        """
        
        
        Args:
            pos (_type_): _description_
            n_pass (_type_): _description_
            queues (_type_): _description_
            arrivals (_type_): _description_
            action (_type_): _description_
        """
        new_n_pass = n_pass
        new_queues = queues.copy()
        new_arrivals = arrivals.copy()
        vertical_high = pos * self._elevator.movement_speed
        
        reward = 0
        reward += self._waiting_penalty * (n_pass + sum(queues))
    
        # 0: move down
        if action == 0:
            new_pos = pos - 1
        # 1: stay still
        elif action == 1:
            new_pos = pos
            
            if pos == 0:
                # Add n_pass to compensate for served passengers
                reward += self._delivery_reward * n_pass + n_pass
                new_n_pass = 0
            
            elif vertical_high % self._elevator.movement_speed == 0:
                floor = int(vertical_high / self._elevator.movement_speed)
                #print("Serving floor:", floor)
                new_n_pass = np.clip(new_n_pass + queues[floor - 1], 0, self._elevator.max_capacity)
                new_queues[floor - 1] -= min(queues[floor - 1], self._elevator.max_capacity - n_pass)
        # 2: move up
        elif action == 2:
            new_pos = pos + 1
        
        # Update queues with new arrivals
        for i in range(len(queues)):
            new_queues[i] += arrivals[i]
            new_queues[i] = np.clip(new_queues[i], 0, self._elevator.queues[i + 1].max_queue_length)
            new_arrivals[i] = 0  # Reset arrivals after updating queues

        # Clip positions to valid range
        new_pos = np.clip(new_pos, 0, self.observation_space['pos'].n - 1)
        
        # print(f'--- Transition ---')
        # print(f'Old pos: {pos}, n_pass: {n_pass}, queues: {queues}, arrivals: {arrivals}, action: {action}')
        # print(f'New pos: {new_pos}, n_pass: {new_n_pass}, queues: {new_queues}, arrivals: {new_arrivals}, reward: {reward}')
        
        return new_pos, new_n_pass, new_queues, arrivals, reward
    
    def _ctrl_transitions(self, pos, n_pass, queues, arrivals, action):
        """
        Computes the next controllable state given the current state and action.
        
        Args:
            pos (int): Current elevator position.
            n_pass (int): Current number of passengers in the elevator.
            queues (np.ndarray): Current floor queues.
            action (int): Action taken by the agent.
        """    
        new_n_pass = n_pass
        new_queues = queues.copy()
        vertical_high = pos * self._elevator.movement_speed
            
        # 0: move down
        if action == 0:
            new_pos = pos - 1
        # 1: stay still
        elif action == 1:
            new_pos = pos
            
            if pos == 0:
                # Add n_pass to compensate for served passengers
                new_n_pass = 0
            
            elif vertical_high % self._elevator.movement_speed == 0:
                floor = int(vertical_high / self._elevator.movement_speed)
                #print("Serving floor:", floor)
                new_n_pass = np.clip(new_n_pass + queues[floor - 1], 0, self._elevator.max_capacity)
                new_queues[floor - 1] -= min(queues[floor - 1], self._elevator.max_capacity - n_pass)
        # 2: move up
        elif action == 2:
            new_pos = pos + 1
            
        # Clip positions to valid range
        new_pos = np.clip(new_pos, 0, self.observation_space['pos'].n - 1)
        
        # Update queues with new arrivals
        for i in range(len(queues)):
            new_queues[i] += arrivals[i]
            new_queues[i] = np.clip(new_queues[i], 0, self._elevator.queues[i + 1].max_queue_length)
    
        return [[new_pos, new_n_pass, new_queues]], [1.0]
    
    def reward_from_transition(self, state: dict, action: int):
        """
        Computes the reward for a given state and action.
        
        Args:
            state (dict): The state of the environment.
            action (int): The action taken by the agent.
        """
        return self._transitions(**state, action=action)[4]
    
    def __init__(self, settings:dict, seed=None, **kwargs):
        super().__init__()
        
        self._elevator = Elevator(max_floor=settings['max_floor'],
                                  min_floor=settings['min_floor'],
                                  max_capacity=settings['max_capacity'],
                                  movement_speed=settings['movement_speed'],
                                  floor_height=settings['floor_height'],
                                  max_arrivals=settings['max_arrivals'],
                                  max_queue_length=settings['max_queue_length'],
                                  )
        
        self.init_elevator_pos = settings['init_elevator_pos']
        self._arrival_distributions = settings['arrival_distributions']
        self._goal_floor = settings['goal_floor']
        self._delivery_reward = settings['delivery_reward']
        self._waiting_penalty = settings['waiting_penalty']
        
        self.current_time = 0
        self.horizon = settings['horizon']
        
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        
        self.observation_space = spaces.Dict({
            'pos': spaces.Discrete(int(self._elevator.floor_height * (self._elevator.max_floor) / self._elevator.movement_speed) + 1),
            'n_pass': spaces.Discrete(self._elevator.max_capacity + 1),
            'queues': spaces.MultiDiscrete(np.array([settings['max_queue_length'] + 1] * (self._elevator.max_floor - self._elevator.min_floor)), start=[0] * (self._elevator.max_floor - self._elevator.min_floor)),
            'arrivals': spaces.MultiDiscrete(np.array([settings['max_arrivals'] + 1] * (self._elevator.max_floor - self._elevator.min_floor)), start=[0] * (self._elevator.max_floor - self._elevator.min_floor))
        })
       
        # Action space -> 1: stay still, 2: move up, 0: move down
        self.action_space = spaces.Discrete(3)

        # Reward range (backward induction)
        # r_min = self._waiting_penalty * (self._elevator.max_capacity + sum([queue.max_queue_length for queue in self._elevator.queues]))
        # r_max = self._delivery_reward * self._elevator.max_capacity
        # self.reward_range = (r_min, r_max)
        
         # Pygame setup
        self.render_mode = kwargs['render_mode'] if 'render_mode' in kwargs else None
        if self.render_mode is not None and self.render_mode in self.metadata["render_modes"]:
            self._init_pygame()
        
    def _init_pygame(self):
        self.window_size = 512
        self.size = 5  
        self.window = None
        
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
        else:
            self.window = None
            self.clock = None
    
    def _get_obs(self):
        return {
            'pos': int(self._elevator.vertical_position // self._elevator.movement_speed),
            'n_pass': len(self._elevator.passengers),
            'queues': np.array([len(queue) for queue in self._elevator.queues if queue.floor != 0]).astype(np.int64),  
            'arrivals': np.array([len(queue.futures) for queue in self._elevator.queues if queue.floor != 0]).astype(np.int64),
        }
    
    def _get_info(self):
        return {
            'elevator_status': self._elevator.status(),
            'current_time': self.current_time,
            'goal_floor': self._goal_floor,
            'floor_height': self._elevator.floor_height,
            'movement_speed': self._elevator.movement_speed,
            'vertical_position': self._elevator.vertical_position,
        }
    
    def get_space_vars(self):
        return ['pos', 'n_pass', 'queues', 'arrivals']

    def get_controllables(self):
        return ['pos', 'n_pass', 'queues']

    def get_uncontrollables(self):
        return ['arrivals']
    
    def get_unctrl_obs(self, obs):
        return {'arrivals': obs['arrivals']}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_time = 0
        
        # Set the seed during the evaluation phase if provided
        if options is not None and 'seed' in options:
            rng = np.random.default_rng(options['seed'])
        else:
            rng = self._rng
        
        # Set the elevator position
        if self.init_elevator_pos is not None:
            self._elevator.reset(initial_position=self.init_elevator_pos)
        else:
            self._elevator.reset(initial_position=rng.integers(self.observation_space['pos'].n) * self._elevator.movement_speed)
        
        for queue in self._elevator.queues:
            if queue.floor != self._goal_floor:
                lambd = rng.uniform(self._arrival_distributions['lambda_min'], self._arrival_distributions['lambda_max'])
                queue.set_arrivals(arrivals=generate_arrival_distribution(lambd=lambd,
                                                                          total_time=self.horizon, 
                                                                          floor=queue.floor,
                                                                          goal_floor=self._goal_floor,
                                                                          rng=rng))
        
        self._elevator.update_queues(current_time=self.current_time)
        
        if self.render_mode == "human":
            self._render_frame()
        
        #info = self._get_info()
        info = {}
        return self._get_obs(), info
                
    def step(self, action):
        """
        Perform one step in the environment.

        Args:
            action (np.ndarray): The action to take.
            0: move down
            1: stay still
            2: move up
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
                observation (dict): The current state of the environment.
                reward (float): The reward received after taking the action.
                done (bool): Whether the episode has ended.
                truncated (bool): Whether the episode was truncated.
                info (dict): Additional information about the step.
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        # print('--- Step ---')
        # print(self._get_obs())
        # print(f'Action taken: {action}')
        
        served = []
        
        # 0: move down, 1: stay still, 2: move up
        if action == 2:
            self._elevator.move('up')
        elif action == 0:
            self._elevator.move('down')
        elif action == 1: 
            served = self._elevator.open_doors()
        else:
            raise ValueError(f"Invalid action {action}")
        
        # Calculate the reward: we penalize the waiting time of passengers
        reward = 0
        if len(served) > 0:
            reward += len(served) * self._delivery_reward
        reward += self._waiting_penalty * (len(self._elevator.passengers) + sum([len(queue) for queue in self._elevator.queues]))
        
        # Check arrivals at the current time
        self.current_time += 1
        self._elevator.update_queues(current_time=self.current_time)
        
        done = self.current_time >= self.horizon
        
        if self.render_mode == "human":
            self._render_frame()
        
        #info = self._get_info()
        info = {}
        # print(self._get_obs())
        # print(f'Reward received: {reward}\n')
        # print('----------------\n')
        
        return self._get_obs(), reward, done, False, info
        
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
            
    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        # The size of a single grid square in pixels
        pix_square_size = (self.window_size / self.size)  
        elevator_square_size = pix_square_size - 1

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                elevator_square_size * np.array([0,0]),
                (elevator_square_size, elevator_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.array([1,1]) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        
        # Vertical line
        pygame.draw.line(
                canvas,
                0,
                (pix_square_size, 0),
                (pix_square_size, self.window_size),
                width=3,
            )
        
        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
                
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

