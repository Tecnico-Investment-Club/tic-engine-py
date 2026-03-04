import importlib
from typing import Dict, Type

from .base import IStrategy
from .example_strat.example_strat import ExampleStrat
from .example_strat_two.example_strat_two import ExampleStratTwo


# Backwards-compatible registry mapping simple names to concrete classes.
_NAME_REGISTRY: Dict[str, Type[IStrategy]] = {
    "ExampleStrategy": ExampleStrat,
    "ExampleStrategyTwo": ExampleStratTwo,
}


def _load_class_from_path(class_path: str) -> Type[IStrategy]:
    """
    Dynamically load a strategy class from a fully qualified class path.

    Example: "src.engine_py.src.strategy.example_strat.ExampleStrat"
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not issubclass(cls, IStrategy):
        raise TypeError(
            f"{class_path} is not a subclass of IStrategy (got {cls!r})."
        )
    return cls


def get_strategy(
    strategy_name: str,
    params: dict,
    class_path: str | None = None,
) -> IStrategy:
    """
    Factory for strategy instances.

    Preferred usage is via `class_path`, which allows completely decoupled
    strategy implementations without modifying engine code. For legacy
    configs, `strategy_name` is resolved via an internal registry.
    """
    safe_params = params or {}

    # 1) Fully-qualified class path takes precedence if provided.
    if class_path:
        cls = _load_class_from_path(class_path)
        return cls(**safe_params)

    # 2) Fall back to name-based registry for backwards compatibility.
    if strategy_name in _NAME_REGISTRY:
        return _NAME_REGISTRY[strategy_name](**safe_params)

    raise ValueError(
        f"Strategy '{strategy_name}' not found. "
        f"Provide a valid 'strategy.name' or 'strategy.class_path' in config."
    )