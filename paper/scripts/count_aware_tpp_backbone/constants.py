"""Experiment identifiers shared by the count-aware runner components."""

from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    LOGNORMAL_VARIANT,
    TAIL_HEAD_ONLY_VARIANT,
    TAIL_SHARED_VARIANT,
    TAIL_VARIANTS,
)
from paper.scripts.run_intermittent_log_backbone_control import (
    BACKBONES as LEGACY_BACKBONES,
)


SEEDS = (42, 52, 62)
BACKBONES = (*LEGACY_BACKBONES, "nhp", "sahp")
TITAN_HISTORICAL_MEMORY_BACKBONES = (
    "titantpp_no_memory",
    "titantpp_gated_soft_memory",
    "titantpp_surprise_memory",
)
TITAN_PERSISTENT_MEMORY_BACKBONES = (
    "titantpp_persistent_only",
    "titantpp_persistent_surprise_memory",
    "titantpp_dual_memory_shared",
    "titantpp_dual_memory_adapter_only",
)
TITAN_MEMORY_BACKBONES = (
    *TITAN_HISTORICAL_MEMORY_BACKBONES,
    *TITAN_PERSISTENT_MEMORY_BACKBONES,
)
SUPPORTED_BACKBONES = (*BACKBONES, *TITAN_MEMORY_BACKBONES)
VARIANT = LOG_MSE_VARIANT
FROZEN_TAIL_LAMBDA = 0.09111380335463036
QUANTITY_VARIANT_ALIASES = {
    "log_mse": VARIANT,
    VARIANT: VARIANT,
    "lognormal_k1": LOGNORMAL_VARIANT,
    LOGNORMAL_VARIANT: LOGNORMAL_VARIANT,
    "tail_shared": TAIL_SHARED_VARIANT,
    TAIL_SHARED_VARIANT: TAIL_SHARED_VARIANT,
    "tail_head_only": TAIL_HEAD_ONLY_VARIANT,
    TAIL_HEAD_ONLY_VARIANT: TAIL_HEAD_ONLY_VARIANT,
}
BACKBONE_LABELS = {
    "rmtpp": "Count-aware RMTPP",
    "thp": "Count-aware THP",
    "titantpp": "Count-aware TitanTPP",
    "nhp": "Adapted NHP",
    "sahp": "Adapted SAHP",
    "titantpp_no_memory": "TitanTPP No Memory",
    "titantpp_gated_soft_memory": "TitanTPP Gated Soft Memory",
    "titantpp_surprise_memory": "TitanTPP Surprise Memory",
    "titantpp_persistent_only": "TitanTPP Persistent Only",
    "titantpp_persistent_surprise_memory": (
        "TitanTPP Persistent Surprise Memory"
    ),
    "titantpp_dual_memory_shared": "TitanTPP Dual Memory Shared",
    "titantpp_dual_memory_adapter_only": (
        "TitanTPP Dual Memory Adapter-only"
    ),
}


__all__ = [
    "BACKBONES",
    "BACKBONE_LABELS",
    "FROZEN_TAIL_LAMBDA",
    "LOGNORMAL_VARIANT",
    "QUANTITY_VARIANT_ALIASES",
    "SEEDS",
    "SUPPORTED_BACKBONES",
    "TAIL_HEAD_ONLY_VARIANT",
    "TAIL_SHARED_VARIANT",
    "TAIL_VARIANTS",
    "TITAN_HISTORICAL_MEMORY_BACKBONES",
    "TITAN_MEMORY_BACKBONES",
    "TITAN_PERSISTENT_MEMORY_BACKBONES",
    "VARIANT",
]
