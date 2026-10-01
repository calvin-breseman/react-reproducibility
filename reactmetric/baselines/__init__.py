"""Baseline forecasters and a registry for benchmark.add_baseline."""

from ._kinematics import CalibratedNoise
from .constant_velocity import ConstantVelocity

__all__ = ["ConstantVelocity", "CalibratedNoise", "available", "build"]

# Lazily import heavier baselines so the core install stays light.
_OPTIONAL = {}
try:
    from .ctrv import CTRV  # noqa: F401

    _OPTIONAL["CTRV"] = CTRV
    __all__.append("CTRV")
except Exception:  # pragma: no cover
    pass
try:
    from .imm import IMM  # noqa: F401

    _OPTIONAL["IMM"] = IMM
    __all__.append("IMM")
except Exception:  # pragma: no cover
    pass


_REGISTRY = {"ConstantVelocity": ConstantVelocity, **_OPTIONAL}

_GRU_NAMES = ("GRU", "GRUPosition", "GRUOrientation")


def available() -> list:
    """Names usable with benchmark.add_baseline / build()."""
    names = list(_REGISTRY.keys())
    try:
        import torch  # noqa: F401

        from .gru import GRU, GRUOrientation, GRUPosition  # noqa: F401

        names.extend(["GRUPosition", "GRUOrientation", "GRU"])
    except Exception:  # pragma: no cover
        pass
    return names


def build(name: str, **kwargs):
    """Construct a baseline forecaster by name."""
    if name in _GRU_NAMES:
        from .gru import GRU, GRUOrientation, GRUPosition

        if name == "GRUOrientation":
            return GRUOrientation(**kwargs)
        # GRU and GRUPosition both resolve to the position checkpoint.
        return GRUPosition(**kwargs) if name != "GRU" else GRU(**kwargs)
    if name not in _REGISTRY:
        raise KeyError(f"unknown baseline '{name}'; available: {available()}")
    return _REGISTRY[name](**kwargs)


# Re-export GRU classes when torch is present so `rm.baselines.GRUPosition` works.
try:
    from .gru import GRU, GRUOrientation, GRUPosition, default_checkpoint_path

    __all__ += [
        "GRU",
        "GRUPosition",
        "GRUOrientation",
        "default_checkpoint_path",
    ]
except Exception:  # pragma: no cover
    pass
