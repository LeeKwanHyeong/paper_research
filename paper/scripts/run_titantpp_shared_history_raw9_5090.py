"""Host-only 5090 adapter for the immutable raw9 shared-history joint worker.

The user moved the same nine approved uncapped fits to idle 5090 while the
5080 Cap campaign continues. Only the host allowlist and isolated contract /
stage schema names change. The source SHA and exact replacement counts make
any scientific implementation change a hard import failure.
"""
from pathlib import Path as _Path
import hashlib as _hashlib


FROZEN_WORKER_SHA256 = "100de1166d6e59dad906e0df21d3baadddc90d7834e948676cc124b822effb1c"
_FROZEN_WORKER_PATH = _Path(__file__).with_name("run_titantpp_shared_history_raw9.py")
_frozen_source = _FROZEN_WORKER_PATH.read_bytes()
if _hashlib.sha256(_frozen_source).hexdigest() != FROZEN_WORKER_SHA256:
    raise ValueError("Immutable raw9 joint worker changed; 5090 adapter refused")

_ADAPTATION_REPLACEMENTS = (
    ('SCHEMA = "titantpp_shared_history_raw9_stage_v1"',
     'SCHEMA = "titantpp_shared_history_raw9_5090_stage_v1"', 1),
    ('CONTRACT_SCHEMA = "titantpp_shared_history_raw9_v1"',
     'CONTRACT_SCHEMA = "titantpp_shared_history_raw9_5090_v1"', 1),
    ("5080", "5090", 5),
)
_adapted_source = _frozen_source.decode("utf-8")
for _original, _replacement, _expected_count in _ADAPTATION_REPLACEMENTS:
    if _adapted_source.count(_original) != _expected_count:
        raise ValueError("Immutable raw9 host-only adaptation count changed")
    _adapted_source = _adapted_source.replace(_original, _replacement)

# Define functions in this module's own globals so stage invocation and tests
# see a separate adapter rather than mutating the immutable parent's globals.
exec(compile(_adapted_source, str(_FROZEN_WORKER_PATH) + " [5090 adapter]", "exec"), globals())
