import numpy as np
from gymnasium import spaces
from gymnasium.experimental.functional import FuncEnv

class FunctionalTradingEnv(FuncEnv):
    """
    Functional version of the Trading environment for reinforcement learning.
    This environment is used as a generative model, not episodic.

    Args:
        FuncEnv (_type_): _description_
    """
    def __init__(self, settings):
        
        X_max = settings['max_amount']
        S_min = settings['min_price']
        S_max = settings['max_price']
        granularity = settings['granularity']
        
        self.action_space = spaces.Discrete(X_max + 1)  # Possible trade amounts from 0 to X_max
        self.observation_space = spaces.Dict({
            'amount': spaces.Discrete(X_max + 1), # Discretized inventory levels
            'price': spaces.Discrete(int((S_max - S_min) / granularity) + 1),  # Discretized price levels
        })

    def initial(self, rng, params = None):
        pass
        
    def transition(self, state, action, rng, params = None):
        # Copy current state
        Xs, Ss = state['amount'], state['price']
        
        Xs_next = np.zeros_like(Xs)
        Ss_next = np.zeros_like(Ss)
        
        Xs_next[:] = action  # Next inventory is determined by the action (trade amount)
        Ss_next[:] = Ss  # Price remains the same in this functional env (price changes are exogenous)
        
        # print(f"Transition step: Current amount: {Xs}, Action (next amount): {action}, Next amount: {Xs_next}, Current price: {Ss}, Next price: {Ss_next}")
        
        next_state = {'amount': Xs_next,
                      'price': Ss_next}
        
        return next_state
        
    def reward(self, state, action, next_state, rng, params=None):
        
        # Extract parameters
        T = 1/252
        N = params['time_intervals']
        tau = T / N  # Time step size
        
        S_min = params['min_price']
        S_max = params['max_price']
        granularity = params['granularity']
        sigma = params['volatility']
        epsilon = params['transaction_cost']
        eta_tilde = params['temporary_impact'] - 0.5 * params['permanent_impact'] * tau
        lamb = params['risk_aversion']
        
        # Extract state variables
        Xs, Ss = state['amount'], state['price']
        Xs_next = next_state['amount']
        
        # Amount of traded securities. If n_k > 0 -> sold, n_k < 0 -> bought
        ns = Xs - Xs_next  
        
        # Gain or cost from trading
        revenue = ns * (Ss * granularity + S_min)
        
        # Transaction Cost (Temporary Impact + Fixed Fees)
        execution_cost = (epsilon * np.abs(ns)) + (eta_tilde / tau) * (ns ** 2)
        
        # Risk-averse component: penalty for holding inventory
        holding_risk = lamb * (sigma ** 2) * tau * (Xs_next ** 2)
        
        rewards = revenue - (execution_cost + holding_risk)
        rewards /= self.observation_space['amount'].n * S_max  # Scale rewards
        
        return rewards
    
    def terminal(self, state, params=None):
        return np.where(state['amount'] == 0, True, False)
        