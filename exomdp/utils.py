import yaml


def read_yaml(yaml_file: str):
    with open(yaml_file, 'r') as fin:
        params = yaml.safe_load(fin)
    return params


def parameter_generator(world_file: str, **kwargs) -> dict:
    """
    Generates the parameters dict for `ElevatorEnv`.
    """
    world_settings = read_yaml(world_file)
    
    params = {}
    
    # Generic settings
    params['horizon'] = kwargs.get('horizon', world_settings['horizon'])
    params['timestep'] = kwargs.get('timestep', world_settings['timestep'])
    
    # Elevator settings
    if 'elevator' in world_settings:
        params['min_floor'] = kwargs.get('min_floor', world_settings['elevator']['min_floor'])
        params['max_floor'] = kwargs.get('max_floor', world_settings['elevator']['max_floor'])
        params['max_capacity'] = kwargs.get('max_capacity', world_settings['elevator']['max_capacity'])
        params['movement_speed'] = kwargs.get('movement_speed', world_settings['elevator']['movement_speed'])
        params['floor_height'] = kwargs.get('floor_height', world_settings['elevator']['floor_height'])
        params['max_arrivals'] = kwargs.get('max_arrivals', world_settings['floors']['max_arrivals'])
        params['max_queue_length'] = kwargs.get('max_queue_length', world_settings['floors']['max_queue_length'])
        params['goal_floor'] = kwargs.get('goal_floor', world_settings['goal_floor'])
        params['init_elevator_pos'] = kwargs.get('init_elevator_pos', world_settings['init_elevator_pos'])
        params['random_init_state'] = kwargs.get('random_init_state', world_settings['random_init_state'])
        params['delivery_reward'] = kwargs.get('delivery_reward', world_settings['delivery_reward'])
        params['waiting_penalty'] = kwargs.get('waiting_penalty', world_settings['waiting_penalty'])
        params['arrival_distributions'] = {
            'lambda_min': kwargs.get('lambda_min', world_settings['floors']['lambda_min']),
            'lambda_max': kwargs.get('lambda_max', world_settings['floors']['lambda_max']),
            # 'seed': kwargs.get('seed', world_settings['floors']['seed']),
        }
    
    # Taxi settings
    elif 'taxi' in world_settings:
        params['fickle_prob'] = kwargs.get('fickle_prob', world_settings['taxi']['fickle_prob'])
        params['spawn_prob'] = kwargs.get('spawn_prob', world_settings['taxi']['spawn_prob'])
        params['dropoff_reward'] = kwargs.get('dropoff_reward', world_settings['taxi']['dropoff_reward'])
        params['step_reward'] = kwargs.get('step_reward', world_settings['taxi']['step_reward'])
        params['illegal_reward'] = kwargs.get('illegal_reward', world_settings['taxi']['illegal_reward'])
        # params['passengers_destinations_seed'] = kwargs.get('passengers_destinations_seed', world_settings['taxi']['passengers_destinations_seed'])
        params['n_rows'] = kwargs.get('n_rows', world_settings['taxi']['n_rows'])
        params['n_cols'] = kwargs.get('n_cols', world_settings['taxi']['n_cols'])
        params['locations'] = kwargs.get('locations', world_settings['taxi']['locations'])
        params['traffic_locs'] = kwargs.get('traffic_locs', world_settings['taxi']['traffic_locs'])
        params['traffic_prob'] = kwargs.get('traffic_prob', world_settings['taxi']['traffic_prob'])
    
    # Trading settings
    elif 'trading' in world_settings:
        params['time_intervals'] = kwargs.get('time_intervals', world_settings['trading']['time_intervals'])
        params['max_amount'] = kwargs.get('max_amount', world_settings['trading']['max_amount'])
        params['initial_price'] = kwargs.get('initial_price', world_settings['trading']['initial_price'])
        params['min_price'] = kwargs.get('min_price', world_settings['trading']['min_price'])
        params['max_price'] = kwargs.get('max_price', world_settings['trading']['max_price'])
        params['granularity'] = kwargs.get('granularity', world_settings['trading']['granularity'])
        params['transaction_cost'] = kwargs.get('transaction_cost', world_settings['trading']['transaction_cost'])
        params['permanent_impact'] = kwargs.get('permanent_impact', world_settings['trading']['permanent_impact'])
        params['temporary_impact'] = kwargs.get('temporary_impact', world_settings['trading']['temporary_impact'])
        params['risk_aversion'] = kwargs.get('risk_aversion', world_settings['trading']['risk_aversion'])
        params['volatility'] = kwargs.get('volatility', world_settings['trading']['volatility'])
        params['drift'] = kwargs.get('drift', world_settings['trading']['drift'])
    
    else:
        raise ValueError("Unsupported environment settings in the world file.")
    
    return params