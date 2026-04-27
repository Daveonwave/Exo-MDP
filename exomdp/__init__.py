from .taxi.taxi_funcEnv import FunctionalTaxiEnv
from .taxi.taxi_traffic_funcEnv import FunctionalTaxiTrafficEnv
from .elevator.elevator_funcEnv import FunctionalElevatorEnv
from .trading.trading_funcEnv import FunctionalTradingEnv

from gymnasium.envs.registration import register

register(
    id='pcmdp/elevator-v0',
    entry_point='pcmdp.elevator.elevator_env:ElevatorEnv'
    )

register(
    id='pcmdp/taxi-v0',
    entry_point='pcmdp.taxi.taxi_env:TaxiEnv'
    )

register(
    id='pcmdp/taxi-traffic-v0',
    entry_point='pcmdp.taxi.taxi_traffic_env:TaxiEnv'
    )

register(
    id='pcmdp/trading-v0',
    entry_point='pcmdp.trading.trading_env:TradingEnv'
    )

register(
    id='pcmdp/trading-v1',
    entry_point='pcmdp.trading.trading_time_env:TradingEnv'
    )