"""Unchanged original-data comparison arithmetic, bound to native5090."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

_PATH = Path(__file__).with_name("compare_titantpp_shared_history_raw9.py")
_SHA = "9642639de7f293ee0aea4e7210b071d1f86eb4e2e564ec04fc1bda544919a524"
if hashlib.sha256(_PATH.read_bytes()).hexdigest() != _SHA:
    raise ValueError("Preserved raw9 comparator changed")
_source = _PATH.read_text()
_changes = {
    "from paper.scripts import manage_titantpp_shared_history_raw9 as manager":
        ("from paper.scripts import manage_titantpp_shared_history_raw9_5090 as manager", 1),
    "from paper.scripts import prepare_titantpp_shared_history_raw9 as preparation":
        ("from paper.scripts import prepare_titantpp_shared_history_raw9_5090 as preparation", 1),
    '"5080"': ('"5090"', 2),
    '"retrieval/5080"': ('"retrieval/5090"', 1),
    "new candidate uses5080/cuDNN91900.": ("new candidate uses5090/cuDNN92000.", 1),
}
for _before, (_after, _count) in _changes.items():
    if _source.count(_before) != _count:
        raise ValueError("Unexpected host-only comparison adaptation: " + _before)
    _source = _source.replace(_before, _after)
_spec = importlib.util.spec_from_file_location(__name__ + "_private_math", _PATH)
_adapter = importlib.util.module_from_spec(_spec)
exec(compile(_source, str(_PATH), "exec"), _adapter.__dict__)
_adapter.__file__ = __file__
for _name, _value in tuple(_adapter.__dict__.items()):
    if not _name.startswith("_"):
        globals()[_name] = _value

if __name__ == "__main__":
    main()
