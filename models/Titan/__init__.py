# src/modeling_module/models/Titan/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING
from .common.configs import TitanConfig

__all__ = [
    "TitanConfig",
    "TitanBaseModel",
    "TitanLMMModel",
    "TitanSeq2SeqModel",
    "TitanBackbone",
    "MemoryEncoder",
    # common exports (optional convenience)
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

if TYPE_CHECKING:
    from .Titans import TitanBaseModel, TitanLMMModel, TitanSeq2SeqModel
    from .backbone import TitanBackbone, MemoryEncoder
    from .common.decoder import TitanDecoder, TitanDecoderLayer
    from .common.memory import (
        GatedSoftMemory,
        HardLocalMemoryMatcher,
        LMM,
        MemoryAttention,
        PositionWiseFFN,
        SurpriseGatedMemory,
    )
    from .common.titans_mac import (
        TitansMACEncoder,
        TitansMemoryState,
        TitansNeuralMemory,
    )
    from .common.tpp_gated_memory import (
        TPPGatedMemoryState,
        TPPSpecificGatedMemory,
    )

_LAZY = {
    "TitanBaseModel": (".Titans", "TitanBaseModel"),
    "TitanLMMModel": (".Titans", "TitanLMMModel"),
    "TitanSeq2SeqModel": (".Titans", "TitanSeq2SeqModel"),
    "TitanBackbone": (".backbone", "TitanBackbone"),
    "MemoryEncoder": (".backbone", "MemoryEncoder"),
    "TitanDecoder": (".common.decoder", "TitanDecoder"),
    "TitanDecoderLayer": (".common.decoder", "TitanDecoderLayer"),
    "MemoryAttention": (".common.memory", "MemoryAttention"),
    "PositionWiseFFN": (".common.memory", "PositionWiseFFN"),
    "HardLocalMemoryMatcher": (".common.memory", "HardLocalMemoryMatcher"),
    "LMM": (".common.memory", "LMM"),
    "GatedSoftMemory": (".common.memory", "GatedSoftMemory"),
    "SurpriseGatedMemory": (".common.memory", "SurpriseGatedMemory"),
    "TitansMACEncoder": (".common.titans_mac", "TitansMACEncoder"),
    "TitansMemoryState": (".common.titans_mac", "TitansMemoryState"),
    "TitansNeuralMemory": (".common.titans_mac", "TitansNeuralMemory"),
    "TPPGatedMemoryState": (
        ".common.tpp_gated_memory",
        "TPPGatedMemoryState",
    ),
    "TPPSpecificGatedMemory": (
        ".common.tpp_gated_memory",
        "TPPSpecificGatedMemory",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path, attr = _LAZY[name]
    from importlib import import_module

    mod = import_module(module_path, package=__name__)
    value = getattr(mod, attr)
    globals()[name] = value
    return value
