"""Legacy RMTPP namespace.

New code should import temporal point process models from :mod:`models.TPPs`.
Count-aware exports stay lazy here to avoid a compatibility import cycle.
"""

from models.RMTPPs.RMTPP import RMTPP
from models.RMTPPs.TitanTPP import TitanTPP
from models.RMTPPs.TransformerHawkesTPP import TransformerHawkesTPP
from models.RMTPPs.config import RMTPPConfig, THPConfig

__all__ = [
    "CountAwareNHP",
    "CountAwareRMTPP",
    "CountAwareSAHP",
    "CountAwareTHP",
    "CountAwareTitanTPP",
    "RMTPP",
    "SharedTimeCountModel",
    "TitanTPP",
    "TransformerHawkesTPP",
    "build_count_aware_model",
    "RMTPPConfig",
    "THPConfig",
]

_TPP_COMPAT_EXPORTS = {
    "CountAwareNHP",
    "CountAwareRMTPP",
    "CountAwareSAHP",
    "CountAwareTHP",
    "CountAwareTitanTPP",
    "SharedTimeCountModel",
    "build_count_aware_model",
}


def __getattr__(name: str):
    if name not in _TPP_COMPAT_EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    from models import TPPs

    value = getattr(TPPs, name)
    globals()[name] = value
    return value
