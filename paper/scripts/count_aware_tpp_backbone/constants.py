"""Experiment identifiers shared by the count-aware runner components."""

import math

from models.Titan.common.key_value_memory import KEY_VALUE_BACKBONE, KEY_VALUE_ROLE
from models.Titan.common.elapsed_age import ELAPSED_AGE_BACKBONE, ELAPSED_AGE_ROLE
from models.TPPs.CountAwareTHPStaticMemory import (
    THP_STATIC_MEMORY_BACKBONE,
    THP_STATIC_MEMORY_ROLE,
)
from models.TPPs.CountAwareTitanInterLayerMemory import (
    INTERLAYER_MEMORY_BACKBONE,
    INTERLAYER_MEMORY_ROLE,
)
from models.TPPs.CountAwareTitanMemoryFiLM import (
    MEMORY_FILM_BACKBONE,
    MEMORY_FILM_ROLE,
)
from models.TPPs.CountAwareTitanCausalQKV import (
    CAUSAL_QKV_BACKBONE,
    CAUSAL_QKV_ROLE,
)
from models.TPPs.CountAwareTitanBoundedQK import (
    BOUNDED_QK_BACKBONE,
    BOUNDED_QK_ROLE,
)
from models.TPPs.CountAwareTitanLevelHistoryQKV import (
    LEVEL_HISTORY_QKV_BACKBONE,
    LEVEL_HISTORY_QKV_ROLE,
)
from models.TPPs.CountAwareTitanDualTimescale import (
    DUAL_TIMESCALE_BACKBONE, DUAL_TIMESCALE_ROLE,
)
from models.TPPs.CountAwareTitanValueNorm import (
    VALUE_NORM_BACKBONE,
    VALUE_NORM_ROLE,
)

from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    LOGNORMAL_VARIANT,
    QUANTILE_ADAPTIVE_VARIANT,
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
    "titantpp_titans_mac",
    "titantpp_tpp_gated_memory",
    "titantpp_weighted_static_memory",
    "titantpp_hard_memory_local_time",
    KEY_VALUE_BACKBONE,
    ELAPSED_AGE_BACKBONE,
    INTERLAYER_MEMORY_BACKBONE,
    MEMORY_FILM_BACKBONE,
    CAUSAL_QKV_BACKBONE,
    BOUNDED_QK_BACKBONE,
    LEVEL_HISTORY_QKV_BACKBONE,
    VALUE_NORM_BACKBONE,
    DUAL_TIMESCALE_BACKBONE,
)
SUPPORTED_BACKBONES = (*BACKBONES, *TITAN_MEMORY_BACKBONES, THP_STATIC_MEMORY_BACKBONE)
VARIANT = LOG_MSE_VARIANT
FROZEN_TAIL_LAMBDA = 0.09111380335463036
MODEL_ROLE_EXPERIMENTAL = "experimental"
MODEL_ROLE_T0_COMMON_CONTROL = "t0_common_control"
MODEL_ROLE_T1_INCUMBENT = "t1_incumbent"
MODEL_ROLE_T1_BACKBONE_COMPARISON = "t1_backbone_comparison"
MODEL_ROLE_TIME_HEAD_DIAGNOSTIC = "time_head_diagnostic"
MODEL_ROLE_TITAN_B012_SCREENING = "titan_b012_screening"
MODEL_ROLE_WEIGHTED_STATIC = "t0_weighted_static_retrieval"
MODEL_ROLE_HARD_LOCAL_TIME = "t0_hard_memory_local_time"
MODEL_ROLE_QUANTILE_CHECKPOINT_ALIGNMENT = "quantile_checkpoint_alignment"
MODEL_ROLE_RAW_RMSE_BASELINE_ALIGNMENT = "raw_rmse_baseline_alignment"
MODEL_ROLE_INTERLAYER_MEMORY = INTERLAYER_MEMORY_ROLE
MODEL_ROLE_MEMORY_FILM = MEMORY_FILM_ROLE
MODEL_ROLE_CAUSAL_QKV = CAUSAL_QKV_ROLE
MODEL_ROLE_BOUNDED_QK = BOUNDED_QK_ROLE
MODEL_ROLE_LEVEL_HISTORY_QKV = LEVEL_HISTORY_QKV_ROLE
MODEL_ROLE_VALUE_NORM = VALUE_NORM_ROLE
MODEL_ROLE_DUAL_TIMESCALE = DUAL_TIMESCALE_ROLE
CHECKPOINT_MONITOR_JOINT = "validation_joint_objective"
CHECKPOINT_MONITOR_RAW_QUANTITY_RMSE = "validation_raw_quantity_rmse"
CHECKPOINT_HISTORY_RAW_QUANTITY_RMSE = "val_qty_rmse"
QUANTILE_ADAPTIVE_QUANTILES = (0.5, 0.9, 0.95, 0.99)
QUANTILE_ADAPTIVE_RAW_WEIGHTS = (1.0, 1.0, 1.5, 2.0, 3.0)
QUANTILE_ADAPTIVE_STRENGTH = 1.0
MODEL_ROLES = (
    ELAPSED_AGE_ROLE,
    KEY_VALUE_ROLE,
    MODEL_ROLE_EXPERIMENTAL,
    MODEL_ROLE_T0_COMMON_CONTROL,
    MODEL_ROLE_T1_INCUMBENT,
    MODEL_ROLE_T1_BACKBONE_COMPARISON,
    MODEL_ROLE_TIME_HEAD_DIAGNOSTIC,
    MODEL_ROLE_TITAN_B012_SCREENING,
    MODEL_ROLE_WEIGHTED_STATIC,
    MODEL_ROLE_HARD_LOCAL_TIME,
    MODEL_ROLE_QUANTILE_CHECKPOINT_ALIGNMENT,
    MODEL_ROLE_RAW_RMSE_BASELINE_ALIGNMENT,
    MODEL_ROLE_INTERLAYER_MEMORY,
    MODEL_ROLE_MEMORY_FILM,
    MODEL_ROLE_CAUSAL_QKV,
    MODEL_ROLE_BOUNDED_QK,
    MODEL_ROLE_LEVEL_HISTORY_QKV,
    MODEL_ROLE_VALUE_NORM,
    MODEL_ROLE_DUAL_TIMESCALE,
    THP_STATIC_MEMORY_ROLE,
)
T0_COMMON_BACKBONES = ("rmtpp", "thp", "nhp", "sahp", "titantpp")
TITAN_B012_BACKBONES = (
    "titantpp",
    "titantpp_titans_mac",
    "titantpp_tpp_gated_memory",
)
QUANTITY_VARIANT_ALIASES = {
    "log_mse": VARIANT,
    VARIANT: VARIANT,
    "lognormal_k1": LOGNORMAL_VARIANT,
    LOGNORMAL_VARIANT: LOGNORMAL_VARIANT,
    "quantile_adaptive": QUANTILE_ADAPTIVE_VARIANT,
    QUANTILE_ADAPTIVE_VARIANT: QUANTILE_ADAPTIVE_VARIANT,
    "tail_shared": TAIL_SHARED_VARIANT,
    TAIL_SHARED_VARIANT: TAIL_SHARED_VARIANT,
    "tail_head_only": TAIL_HEAD_ONLY_VARIANT,
    TAIL_HEAD_ONLY_VARIANT: TAIL_HEAD_ONLY_VARIANT,
}
BACKBONE_LABELS = {
    ELAPSED_AGE_BACKBONE: "TitanTPP-HardLMM Elapsed-Age Encoder",
    KEY_VALUE_BACKBONE: "TitanTPP-HardLMM Separate-Key Sparse Retrieval",
    THP_STATIC_MEMORY_BACKBONE: "Count-aware THP + Static Hard Memory",
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
    "titantpp_titans_mac": "TitanTPP Faithful Titans-MAC",
    "titantpp_tpp_gated_memory": "TitanTPP TPP-specific Gated Memory",
    "titantpp_weighted_static_memory": "Hard-LMM Similarity-Weighted Static Retrieval",
    "titantpp_hard_memory_local_time": "Hard-LMM Quantity Memory / Local Time",
    INTERLAYER_MEMORY_BACKBONE: "Hard-LMM Inter-layer Shared-bank Read",
    MEMORY_FILM_BACKBONE: "Hard-LMM Shared-bank Feature Modulation",
    CAUSAL_QKV_BACKBONE: "Hard-LMM Causal QKV Convolution",
    BOUNDED_QK_BACKBONE: "Hard-LMM Bounded QK with Causal V",
    LEVEL_HISTORY_QKV_BACKBONE: "Hard-LMM Level-preserving History-confidence QKV",
    VALUE_NORM_BACKBONE: "Hard-LMM Prototype Value-Norm Consistency",
    DUAL_TIMESCALE_BACKBONE: "Hard-LMM Local/Global Transition Memory",
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
    if model_role == ELAPSED_AGE_ROLE or ELAPSED_AGE_BACKBONE in backbones:
        if model_role != ELAPSED_AGE_ROLE or backbones != (ELAPSED_AGE_BACKBONE,):
            raise ValueError("Elapsed-age memory requires its dedicated single-backbone role")
        if (quantity_variants != (VARIANT,) or time_head_mode != TIME_HEAD_MODE_LEGACY_CLAMPED
                or lambda_tail != 0.0):
            raise ValueError("Elapsed-age role requires direct log-MSE, legacy time head and no tail loss")
        return
    candidate_roles = {
        MODEL_ROLE_INTERLAYER_MEMORY: INTERLAYER_MEMORY_BACKBONE,
        MODEL_ROLE_MEMORY_FILM: MEMORY_FILM_BACKBONE,
        MODEL_ROLE_CAUSAL_QKV: CAUSAL_QKV_BACKBONE,
        MODEL_ROLE_BOUNDED_QK: BOUNDED_QK_BACKBONE,
        MODEL_ROLE_LEVEL_HISTORY_QKV: LEVEL_HISTORY_QKV_BACKBONE,
        MODEL_ROLE_VALUE_NORM: VALUE_NORM_BACKBONE,
        MODEL_ROLE_DUAL_TIMESCALE: DUAL_TIMESCALE_BACKBONE,
    }
    matched_candidate_roles = [
        role
        for role, candidate in candidate_roles.items()
        if model_role == role or candidate in backbones
    ]
    if matched_candidate_roles:
        if len(matched_candidate_roles) != 1:
            raise ValueError("Backbone candidate roles cannot be mixed")
        role = matched_candidate_roles[0]
        candidate = candidate_roles[role]
        if model_role != role or backbones != (candidate,):
            raise ValueError(
                f"{candidate} requires its dedicated single-backbone role"
            )
        if (
            quantity_variants != (VARIANT,)
            or time_head_mode != TIME_HEAD_MODE_LEGACY_CLAMPED
            or not math.isclose(lambda_tail, 0.0, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise ValueError(
                f"{candidate} requires direct log-MSE, legacy time head and no tail loss"
            )
        return
    if model_role == KEY_VALUE_ROLE or KEY_VALUE_BACKBONE in backbones:
        if model_role != KEY_VALUE_ROLE or backbones != (KEY_VALUE_BACKBONE,):
            raise ValueError("Separate-key memory requires its dedicated single-backbone role")
        if (quantity_variants != (VARIANT,) or time_head_mode != TIME_HEAD_MODE_LEGACY_CLAMPED
                or lambda_tail != 0.0):
            raise ValueError("Separate-key role requires direct log-MSE, legacy time head and no tail loss")
        return
    if model_role == THP_STATIC_MEMORY_ROLE or THP_STATIC_MEMORY_BACKBONE in backbones:
        if model_role != THP_STATIC_MEMORY_ROLE or backbones != (THP_STATIC_MEMORY_BACKBONE,):
            raise ValueError("THP static memory requires its dedicated single-backbone role")
        if (quantity_variants != (VARIANT,) or time_head_mode != TIME_HEAD_MODE_LEGACY_CLAMPED
                or lambda_tail != 0.):
            raise ValueError("THP static memory requires direct log-MSE, legacy head and no tail loss")
        return
    if model_role == MODEL_ROLE_HARD_LOCAL_TIME or "titantpp_hard_memory_local_time" in backbones:
        if model_role != MODEL_ROLE_HARD_LOCAL_TIME or backbones != ("titantpp_hard_memory_local_time",):
            raise ValueError("Local-time candidate requires its dedicated single-backbone role")
        if (quantity_variants != (VARIANT,) or time_head_mode != TIME_HEAD_MODE_LEGACY_CLAMPED
                or lambda_tail != 0.0):
            raise ValueError("Local-time role requires direct log-MSE, legacy time head and no tail loss")
        return
    if model_role == MODEL_ROLE_EXPERIMENTAL:
        return
    if model_role == MODEL_ROLE_RAW_RMSE_BASELINE_ALIGNMENT:
        if not backbones or any(backbone not in {"rmtpp", "thp"} for backbone in backbones):
            raise ValueError(
                "Raw-RMSE baseline alignment is limited to RMTPP and THP"
            )
        if quantity_variants != (VARIANT,):
            raise ValueError(
                "Raw-RMSE baseline alignment requires the direct log-MSE variant"
            )
        if time_head_mode != TIME_HEAD_MODE_LEGACY_CLAMPED:
            raise ValueError(
                "Raw-RMSE baseline alignment requires legacy_clamped_rmtpp"
            )
        if not math.isclose(lambda_tail, 0.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("Raw-RMSE baseline alignment requires lambda_tail=0")
        return
    if model_role == MODEL_ROLE_QUANTILE_CHECKPOINT_ALIGNMENT:
        if backbones != ("titantpp",):
            raise ValueError(
                "Quantile checkpoint alignment requires backbone=titantpp"
            )
        if quantity_variants != (VARIANT, QUANTILE_ADAPTIVE_VARIANT):
            raise ValueError(
                "Quantile checkpoint alignment requires ordered unweighted and "
                "quantile-adaptive variants"
            )
        if time_head_mode != TIME_HEAD_MODE_LEGACY_CLAMPED:
            raise ValueError(
                "Quantile checkpoint alignment requires legacy_clamped_rmtpp"
            )
        if not math.isclose(lambda_tail, 0.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("Quantile checkpoint alignment requires lambda_tail=0")
        return
    if model_role == MODEL_ROLE_WEIGHTED_STATIC:
        if backbones != ("titantpp_weighted_static_memory",):
            raise ValueError("Weighted static role requires only the registered W0 candidate")
        if quantity_variants != (VARIANT,) or time_head_mode != TIME_HEAD_MODE_LEGACY_CLAMPED:
            raise ValueError("Weighted static role requires direct log-MSE and legacy_clamped_rmtpp")
        if not math.isclose(lambda_tail, 0.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("Weighted static role requires lambda_tail=0")
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
    if model_role == MODEL_ROLE_TITAN_B012_SCREENING:
        if backbones != TITAN_B012_BACKBONES:
            raise ValueError(
                "Titan B0/B1/B2 screening requires ordered backbones="
                f"{TITAN_B012_BACKBONES}"
            )
        if quantity_variants != (VARIANT,):
            raise ValueError(
                "Titan B0/B1/B2 screening requires the direct log-MSE variant"
            )
        if time_head_mode != TIME_HEAD_MODE_LEGACY_CLAMPED:
            raise ValueError(
                "Titan B0/B1/B2 screening requires legacy_clamped_rmtpp"
            )
        if not math.isclose(lambda_tail, 0.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("Titan B0/B1/B2 screening requires lambda_tail=0")
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
    "CHECKPOINT_HISTORY_RAW_QUANTITY_RMSE",
    "CHECKPOINT_MONITOR_JOINT",
    "CHECKPOINT_MONITOR_RAW_QUANTITY_RMSE",
    "LOGNORMAL_VARIANT",
    "MODEL_ROLES",
    "MODEL_ROLE_EXPERIMENTAL",
    "MODEL_ROLE_T0_COMMON_CONTROL",
    "MODEL_ROLE_T1_BACKBONE_COMPARISON",
    "MODEL_ROLE_T1_INCUMBENT",
    "MODEL_ROLE_TITAN_B012_SCREENING",
    "MODEL_ROLE_WEIGHTED_STATIC",
    "MODEL_ROLE_HARD_LOCAL_TIME",
    "MODEL_ROLE_QUANTILE_CHECKPOINT_ALIGNMENT",
    "MODEL_ROLE_RAW_RMSE_BASELINE_ALIGNMENT",
    "MODEL_ROLE_BOUNDED_QK",
    "MODEL_ROLE_LEVEL_HISTORY_QKV",
    "MODEL_ROLE_VALUE_NORM",
    "MODEL_ROLE_DUAL_TIMESCALE",
    "MODEL_ROLE_TIME_HEAD_DIAGNOSTIC",
    "QUANTITY_VARIANT_ALIASES",
    "QUANTILE_ADAPTIVE_QUANTILES",
    "QUANTILE_ADAPTIVE_RAW_WEIGHTS",
    "QUANTILE_ADAPTIVE_STRENGTH",
    "QUANTILE_ADAPTIVE_VARIANT",
    "SEEDS",
    "SUPPORTED_BACKBONES",
    "TAIL_HEAD_ONLY_VARIANT",
    "TAIL_SHARED_VARIANT",
    "TAIL_VARIANTS",
    "T0_COMMON_BACKBONES",
    "TITAN_B012_BACKBONES",
    "TITAN_HISTORICAL_MEMORY_BACKBONES",
    "TITAN_MEMORY_BACKBONES",
    "TITAN_PERSISTENT_MEMORY_BACKBONES",
    "THP_STATIC_MEMORY_BACKBONE",
    "THP_STATIC_MEMORY_ROLE",
    "VARIANT",
    "validate_model_role_contract",
]
