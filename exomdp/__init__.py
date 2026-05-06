from .taxi.taxi_funcEnv import FunctionalTaxiEnv
from .taxi.taxi_traffic_funcEnv import FunctionalTaxiTrafficEnv
from .elevator.elevator_funcEnv import FunctionalElevatorEnv
from .trading.trading_funcEnv import FunctionalTradingEnv

from gymnasium.envs.registration import register

register(
    id='exomdp/elevator-v0',
    entry_point='exomdp.elevator.elevator_env:ElevatorEnv'
    )

register(
    id='exomdp/taxi-v0',
    entry_point='exomdp.taxi.taxi_env:TaxiEnv'
    )

register(
    id='exomdp/taxi-traffic-v0',
    entry_point='exomdp.taxi.taxi_traffic_env:TaxiEnv'
    )

register(
    id='exomdp/trading-v0',
    entry_point='exomdp.trading.trading_env:TradingEnv'
    )

register(
    id='exomdp/trading-v1',
    entry_point='exomdp.trading.trading_time_env:TradingEnv'
    )