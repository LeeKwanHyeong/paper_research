"""Experiment identifiers shared by the count-aware runner components."""

import math

from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    LOGNORMAL_VARIANT,
    TAIL_HEAD_ONLY_VARIANT,
    TAIL_SHARED_VARIANT,
    TAIL_VARIANTS,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    TIME_HEAD_MODE_LOGNORMAL_DURATION,
    TIME_HEAD_MODE_SCALED_EXACT,
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
MODEL_ROLE_EXPERIMENTAL = "experimental"
MODEL_ROLE_T0_COMMON_CONTROL = "t0_common_control"
MODEL_ROLE_T1_INCUMBENT = "t1_incumbent"
MODEL_ROLE_T1_BACKBONE_COMPARISON = "t1_backbone_comparison"
MODEL_ROLE_TIME_HEAD_DIAGNOSTIC = "time_head_diagnostic"
MODEL_ROLES = (
    MODEL_ROLE_EXPERIMENTAL,
    MODEL_ROLE_T0_COMMON_CONTROL,
    MODEL_ROLE_T1_INCUMBENT,
    MODEL_ROLE_T1_BACKBONE_COMPARISON,
    MODEL_ROLE_TIME_HEAD_DIAGNOSTIC,
)
T0_COMMON_BACKBONES = ("rmtpp", "thp", "nhp", "sahp", "titantpp")
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


def validate_model_role_contract(
    *,
    model_role: str,
    backbones: tuple[str, ...],
    quantity_variants: tuple[str, ...],
    time_head_mode: str,
    lambda_tail: float,
) -> None:
    """Reject official-role runs that drift from the frozen baseline contract."""
    if model_role == MODEL_ROLE_EXPERIMENTAL:
        return
    if model_role == MODEL_ROLE_T0_COMMON_CONTROL:
        invalid = sorted(set(backbones) - set(T0_COMMON_BACKBONES))
        if invalid:
            raise ValueError(f"T0 common control has unsupported backbones: {invalid}")
        if quantity_variants != (VARIANT,):
            raise ValueError("T0 common control requires the direct log-MSE variant")
        if time_head_mode != TIME_HEAD_MODE_LEGACY_CLAMPED:
            raise ValueError("T0 common control requires legacy_clamped_rmtpp")
        if not math.isclose(lambda_tail, 0.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("T0 common control requires lambda_tail=0")
        return
    if model_role == MODEL_ROLE_T1_INCUMBENT:
        if backbones != ("titantpp",):
            raise ValueError("T1 incumbent requires backbone=titantpp")
    elif model_role == MODEL_ROLE_T1_BACKBONE_COMPARISON:
        invalid = sorted(set(backbones) - {"titantpp", *TITAN_MEMORY_BACKBONES})
        if invalid:
            raise ValueError(f"T1 backbone comparison has non-Titan backbones: {invalid}")
        if "titantpp" not in backbones or len(backbones) < 2:
            raise ValueError(
                "T1 backbone comparison requires fresh titantpp plus at least one candidate"
            )
    elif model_role == MODEL_ROLE_TIME_HEAD_DIAGNOSTIC:
        if backbones != ("titantpp",):
            raise ValueError("Time-head diagnostic requires backbone=titantpp")
        if time_head_mode not in (
            TIME_HEAD_MODE_SCALED_EXACT,
            TIME_HEAD_MODE_LOGNORMAL_DURATION,
        ):
            raise ValueError("Time-head diagnostic is limited to H0 or H3")
    else:
        raise ValueError(f"Unsupported model role: {model_role}")

    if quantity_variants != (TAIL_SHARED_VARIANT,):
        raise ValueError("T1-based roles require the tail-shared quantity variant")
    if model_role != MODEL_ROLE_TIME_HEAD_DIAGNOSTIC:
        if time_head_mode != TIME_HEAD_MODE_LEGACY_CLAMPED:
            raise ValueError("T1 incumbent/backbone comparison requires legacy_clamped_rmtpp")
    if not math.isclose(
        lambda_tail,
        FROZEN_TAIL_LAMBDA,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"T1-based roles require lambda_tail={FROZEN_TAIL_LAMBDA}")


__all__ = [
    "BACKBONES",
    "BACKBONE_LABELS",
    "FROZEN_TAIL_LAMBDA",
    "LOGNORMAL_VARIANT",
    "MODEL_ROLES",
    "MODEL_ROLE_EXPERIMENTAL",
    "MODEL_ROLE_T0_COMMON_CONTROL",
    "MODEL_ROLE_T1_BACKBONE_COMPARISON",
    "MODEL_ROLE_T1_INCUMBENT",
    "MODEL_ROLE_TIME_HEAD_DIAGNOSTIC",
    "QUANTITY_VARIANT_ALIASES",
    "SEEDS",
    "SUPPORTED_BACKBONES",
    "TAIL_HEAD_ONLY_VARIANT",
    "TAIL_SHARED_VARIANT",
    "TAIL_VARIANTS",
    "T0_COMMON_BACKBONES",
    "TITAN_HISTORICAL_MEMORY_BACKBONES",
    "TITAN_MEMORY_BACKBONES",
    "TITAN_PERSISTENT_MEMORY_BACKBONES",
    "VARIANT",
    "validate_model_role_contract",
]
