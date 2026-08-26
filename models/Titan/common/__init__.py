# src/modeling_module/models/Titan/common/__init__.py
from __future__ import annotations

from .configs import TitanConfig

__all__ = [
    "TitanConfig",
    "TitanDecoder",
    "TitanDecoderLayer",
    "MemoryAttention",
    "PositionWiseFFN",
    "HardLocalMemoryMatcher",
    "LMM",
    "GatedSoftMemory",
    "SurpriseGatedMemory",
    "TitansMACEncoder",
    "TitansMemoryState",
    "TitansNeuralMemory",
    "TPPGatedMemoryState",
    "TPPSpecificGatedMemory",
]

_LAZY = {
    "TitanDecoder": (".decoder", "TitanDecoder"),
    "TitanDecoderLayer": (".decoder", "TitanDecoderLayer"),
    "MemoryAttention": (".memory", "MemoryAttention"),
    "PositionWiseFFN": (".memory", "PositionWiseFFN"),
    "HardLocalMemoryMatcher": (".memory", "HardLocalMemoryMatcher"),
    "LMM": (".memory", "LMM"),
    "GatedSoftMemory": (".memory", "GatedSoftMemory"),
    "SurpriseGatedMemory": (".memory", "SurpriseGatedMemory"),
    "TitansMACEncoder": (".titans_mac", "TitansMACEncoder"),
    "TitansMemoryState": (".titans_mac", "TitansMemoryState"),
    "TitansNeuralMemory": (".titans_mac", "TitansNeuralMemory"),
    "TPPGatedMemoryState": (".tpp_gated_memory", "TPPGatedMemoryState"),
    "TPPSpecificGatedMemory": (".tpp_gated_memory", "TPPSpecificGatedMemory"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path, attr = _LAZY[name]
    from importlib import import_module

    mod = import_module(module_path, package=__name__)
    value = getattr(mod, attr)
    globals()[name] = value  # cache
    return value
