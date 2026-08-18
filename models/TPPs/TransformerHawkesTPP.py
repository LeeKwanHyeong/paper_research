"""Canonical TPP namespace for the existing THP implementation."""

from models.RMTPPs.TransformerHawkesTPP import (
    THPEncoderLayer,
    THPTemporalEncoder,
    TransformerHawkesTPP,
)

__all__ = ["THPEncoderLayer", "THPTemporalEncoder", "TransformerHawkesTPP"]
