"""Forward-looking TitanTPP-MAC role kept separate from frozen B012 constants."""

from __future__ import annotations

import math

from models.TPPs.CountAwareTPP import TIME_HEAD_MODE_LEGACY_CLAMPED
from paper.scripts.count_aware_tpp_backbone.constants import VARIANT


MODEL_ROLE_TITANTPP_MAC_PRIMARY = "titantpp_mac_primary"
TITANTPP_MAC_PRIMARY_BACKBONES = ("titantpp_titans_mac",)
TITANTPP_MAC_PAPER_NAME = "Count-aware TitanTPP-MAC"


def validate_titantpp_mac_primary_contract(
    *,
    backbones: tuple[str, ...],
    quantity_variants: tuple[str, ...],
    time_head_mode: str,
    lambda_tail: float,
) -> None:
    if backbones != TITANTPP_MAC_PRIMARY_BACKBONES:
        raise ValueError(
            "TitanTPP-MAC primary runs require "
            f"backbones={TITANTPP_MAC_PRIMARY_BACKBONES}"
        )
    if quantity_variants != (VARIANT,):
        raise ValueError("TitanTPP-MAC primary requires direct log-MSE")
    if time_head_mode != TIME_HEAD_MODE_LEGACY_CLAMPED:
        raise ValueError("TitanTPP-MAC primary requires legacy_clamped_rmtpp")
    if not math.isclose(lambda_tail, 0.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("TitanTPP-MAC primary requires lambda_tail=0")


__all__ = [
    "MODEL_ROLE_TITANTPP_MAC_PRIMARY",
    "TITANTPP_MAC_PAPER_NAME",
    "TITANTPP_MAC_PRIMARY_BACKBONES",
    "validate_titantpp_mac_primary_contract",
]
