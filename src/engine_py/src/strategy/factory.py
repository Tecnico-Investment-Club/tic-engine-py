from .base import IStrategy
from .example_strat import MovingAverageCrossStrategy

def get_strategy(strategy_name: str, params: dict) -> IStrategy:
    """
    Returns an instance of the requested strategy.
    """
    if strategy_name == "MovingAverageCrossStrategy":
        return MovingAverageCrossStrategy(
            short_window=params.get("short_window", 9),
            long_window=params.get("long_window", 21)
        )
    # Add future strategies here
    else:
        raise ValueError(f"Strategy {strategy_name} not found in factory.")