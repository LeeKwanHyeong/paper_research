"""Compatibility wrapper for the canonical :mod:`models.TPPs` package."""

from models.TPPs.SelfAttentiveHawkesTPP import CountAwareSAHP, SAHPEncoderBlock

__all__ = ["CountAwareSAHP", "SAHPEncoderBlock"]
