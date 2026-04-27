import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils

class TradingEnv(Env):
    """
    A simple trading environment based on the paper Optimal Execution of Portfolio Transactions by Almgren and Chriss.
    The agent aims to execute trades over a fixed time horizon while minimizing market impact and transaction costs.
    In this version, we assume that the agent has not market impact.
    
    This environment matches our PCMDP framework by having a controllable part (the amount of securities hold by the agent)
    and an uncontrollable part (the market price evolution).

    ## Action Space
    The action shape is `(N,)` in the range `{0, X_max}` indicating the amount X_{t+1} of securities to hold at the next time step.
    The action space is of type `gym.spaces.Discrete(X_max + 1)`.
    
    ## Observation Space
    The observation shape is `(2,)` where the first element is the current amount of securities held by the agent X_t,
    and the second element is the current market price S_t of the security.
    The evolution of the market price S_t is uncontrollable by the agent and follows a geometric Brownian motion.
    The observation space is discretized to have finite states.
        
    """
    @staticmethod
    def _gen_profile(S0, mu, sigma, T, N, S_min, S_max, granularity, rng):
        """
        Simulates GBM with price discretization (tick size) and hard boundaries.
        """        
        # 1. Define time step
        tau = T / N
                
        # 2. Generate Brownian increments
        Z = rng.normal(0, 1, (N,))
        
        # 3. Calculate drift and diffusion
        drift_term = (mu - 0.5 * sigma**2) * tau
        diffusion_term = sigma * np.sqrt(tau) * Z
        
        # 4. Compute log returns
        log_return = drift_term + diffusion_term
        
        # 5. Construct raw price paths
        log_return_with_start = np.insert(log_return, 0, 0.0)
        cumulative_log_return = np.cumsum(log_return_with_start, axis=0)
        raw_path = S0 * np.exp(cumulative_log_return)
        
        # --- NEW LOGIC: DISCRETIZATION & BOUNDARIES ---
        
        # A. Apply Granularity (Tick Size)
        # Divide by tick size, round to nearest integer, multiply back
        discretized_path = np.round(raw_path / granularity) * granularity
        
        # B. Apply S_min and S_max (Clamping)
        # Values above S_max become S_max; values below S_min become S_min
        final_path = np.clip(discretized_path, S_min, S_max)        
        return final_path
    
    def __init__(self, settings: dict, seed: int = None, **kwargs):
        super().__init__()
        
        self.S0 = settings['initial_price']  # Initial price
        self.S_min = settings['min_price']  # Minimum price boundary
        self.S_max = settings['max_price']  # Maximum price boundary
        self.X_max = settings['max_amount']  # Maximum amount of securities
        self.granularity = settings['granularity']  # Price tick size
        
        self.mu = settings['drift']  # Drift of the price
        self.sigma = settings['volatility']  # Volatility of the price
        self.gamma = settings['permanent_impact']  # Permanent impact parameter
        self.epsilon = settings['transaction_cost']  # Transaction cost per trade unit    
        self.eta = settings['temporary_impact'] # Temporary impact parameter
        self.lamb = settings['risk_aversion']  # Risk aversion parameter
        
        self.N = settings['time_intervals']  # Number of discrete time intervals
        self.T = 1/252  # Total time horizon (1 trading day)
        self.tau = self.T / self.N  # Length of each time interval
        
        self.eta_tilde = self.eta - 0.5 * self.gamma * self.tau  # Adjusted temporary impact parameter
        #print(self.eta_tilde)
        
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(self.X_max + 1)
        self.observation_space = spaces.Dict({
            'amount': spaces.Discrete(self.X_max + 1),
            'price': spaces.Discrete(int((self.S_max - self.S_min) / self.granularity) + 1),
            'time': spaces.Discrete(self.N + 1)
        })
        
        self.profile = None
        self.k = 0
        self._state = None  
        
    def get_space_vars(self):
        return ['amount', 'price']
    
    def get_controllables(self):
        return ['amount']
    
    def get_uncontrollables(self):
        return ['price']
    
    def get_unctrl_obs(self, obs):
        return {'price': obs['price']}    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.k = 0
        
        # Set up RNG for profile generation (training and testing consistency)
        if options is not None and 'seed' in options:
            rng = np.random.default_rng(options['seed'])
        else:
            rng = self.rng
        
        # Generate initial state
        # Note: In optimal execution, we usually start with X0 = X_max (full inventory)
        X0 = self.X_max
        #S0_idx = rng.integers(200, 800)
        S0_idx = int((self.S0 - self.S_min) / self.granularity) - 1
        
        # Generate price profile for the episode
        self.profile = self._gen_profile(S0=self.S0, 
                                         mu=self.mu, 
                                         sigma=self.sigma,           
                                         T=self.T,
                                         N=self.N,
                                         S_min=self.S_min,
                                         S_max=self.S_max,
                                         granularity=self.granularity,
                                         rng=rng)
                
        self._state = {'amount': X0, 'price': S0_idx, 'time': self.k}
        return self._state, {}
    
    def step(self, action):        
        X = self._state['amount']
        S = self.S_min + self._state['price'] * self.granularity  # Convert discrete index to actual price
        X_next = action
        
        # Amount of traded securities. If n_k > 0 -> sold, n_k < 0 -> bought
        n_k = X - X_next 
        
        # Reward Calculation
        # Gain or cost from trading
        revenue = n_k * S
        
        # Transaction Cost (Temporary Impact + Fixed Fees) 
        # Cost = epsilon * |n| + (eta_tilde / tau) * n^2
        # Note: We use eta_tilde here, not raw eta
        #self.eta_tilde = 1e-9  #self.eta - 0.5 * self.gamma * self.tau
        execution_cost = (self.epsilon * abs(n_k)) + (self.eta_tilde / self.tau) * (n_k ** 2)
        
        # Risk = lambda * sigma^2 * tau * x_k^2
        # We pay a penalty for holding 'X_next' shares for the duration of interval tau
        holding_risk = self.lamb * (self.sigma ** 2) * self.tau * (X_next ** 2)
        # holding_risk = 0.0
        
        reward = revenue - (execution_cost + holding_risk)
        reward /= self.X_max * self.S_max  # Scale rewards
        
        # Update state
        self.k += 1
        terminated = self.k >= self.N or action == 0  # Episode ends if time horizon reached or all securities sold
        truncated = False
                
        S_next = self.profile[self.k]  # Next price from profile
        S_next_idx = int((S_next - self.S_min) / self.granularity)
                
        self._state = {'amount': X_next, 'price': S_next_idx, 'time': self.k}
        info = {'k': self.k, 'n_k': n_k}
        
        #print(f"Step {self.k}: X={X}, S={S:.2f}, X_next={X_next}, S_next={S_next:.2f}, n_k={n_k}, Reward={reward:.4f}")
        
        return self._state, reward, terminated, truncated, info
    
    def render(self):
        pass
    
    def close(self):
        pass