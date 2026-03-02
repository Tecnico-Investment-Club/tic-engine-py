# Import from the new subfolders
from .example_strat.example_strat import ExampleStrat
from .example_strat_two.example_strat_two import ExampleStratTwo
from .base import IStrategy

def get_strategy(strategy_name: str, params: dict) -> IStrategy:
    safe_params = params or {}

    if strategy_name == "ExampleStrategy":
        return ExampleStrat(**safe_params)
    elif strategy_name == "ExampleStrategyTwo":
        return ExampleStratTwo(**safe_params)
    else:
        raise ValueError(f"Strategy {strategy_name} not found in factory.")