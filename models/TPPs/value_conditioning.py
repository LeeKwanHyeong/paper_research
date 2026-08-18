"""Canonical TPP import path for value-conditioning helpers.

The implementation remains in the legacy package during the compatibility
migration. New code should import these helpers from ``models.TPPs``.
"""

from models.RMTPPs.value_conditioning import *  # noqa: F403
