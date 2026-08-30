"""Build the forward-looking TitanTPP-MAC candidate without editing frozen B1."""

from __future__ import annotations

from typing import Any

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.Titan.common.titans_mac_optimized import (
    apply_titantpp_mac_semantic_optimization,
    optimization_metadata,
)
from paper.scripts.count_aware_titantpp_mac_contract import (
    MODEL_ROLE_TITANTPP_MAC_PRIMARY,
    TITANTPP_MAC_PRIMARY_BACKBONES,
    TITANTPP_MAC_PAPER_NAME,
    validate_titantpp_mac_primary_contract,
)
from paper.scripts.count_aware_tpp_backbone.constants import VARIANT


def build_count_aware_titantpp_mac_primary(
    *,
    optimize_execution: bool = True,
    **factory_kwargs: Any,
) -> tuple[CountAwareTitanTPP, dict[str, Any]]:
    """Build B1 and attach only forward-looking paper/runtime metadata."""
    validate_titantpp_mac_primary_contract(
        backbones=TITANTPP_MAC_PRIMARY_BACKBONES,
        quantity_variants=(str(factory_kwargs.get("quantity_variant", VARIANT)),),
        time_head_mode=str(
            factory_kwargs.get("time_head_mode", "legacy_clamped_rmtpp")
        ),
        lambda_tail=float(factory_kwargs.get("lambda_tail", 0.0)),
    )
    model, frozen_metadata = build_count_aware_model(
        "titantpp_titans_mac",
        **factory_kwargs,
    )
    if not isinstance(model, CountAwareTitanTPP):
        raise TypeError("TitanTPP-MAC factory returned an unexpected model")
    if optimize_execution:
        apply_titantpp_mac_semantic_optimization(model)
    metadata = {
        **frozen_metadata,
        "paper_model_name": TITANTPP_MAC_PAPER_NAME,
        "model_positioning": "primary_candidate",
        "model_role": MODEL_ROLE_TITANTPP_MAC_PRIMARY,
        "semantic_optimization_enabled": bool(optimize_execution),
    }
    if optimize_execution:
        metadata.update(optimization_metadata())
    return model, metadata


__all__ = ["build_count_aware_titantpp_mac_primary"]
