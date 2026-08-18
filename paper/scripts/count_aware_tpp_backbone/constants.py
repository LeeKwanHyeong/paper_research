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
}


__all__ = [
    "BACKBONES",
    "BACKBONE_LABELS",
    "FROZEN_TAIL_LAMBDA",
    "LOGNORMAL_VARIANT",
    "QUANTITY_VARIANT_ALIASES",
    "SEEDS",
    "TAIL_HEAD_ONLY_VARIANT",
    "TAIL_SHARED_VARIANT",
    "TAIL_VARIANTS",
    "VARIANT",
]
