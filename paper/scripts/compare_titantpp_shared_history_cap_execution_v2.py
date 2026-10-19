"""Unchanged comparison math, bound only to the isolated execution v2 manager."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

_PATH = Path(__file__).with_name("compare_titantpp_shared_history_cap_execution.py")
_SHA = "a8c5cae739df9a539cad9e7638d8a19f62063393a089ddddd6327da35ed3e395"
if hashlib.sha256(_PATH.read_bytes()).hexdigest() != _SHA:
    raise ValueError("Preserved execution v1 comparison changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_private_v1", _PATH)
_adapter = importlib.util.module_from_spec(_spec)
_before = "from paper.scripts import manage_titantpp_shared_history_cap_execution as manager"
_source = _PATH.read_text()
if _source.count(_before) != 1:
    raise ValueError("Unexpected preserved comparison manager import")
_source = _source.replace(_before, "from paper.scripts import manage_titantpp_shared_history_cap_execution_v2 as manager", 1)
exec(compile(_source, str(_PATH), "exec"), _adapter.__dict__)
_adapter.__file__ = __file__
for _name, _value in tuple(_adapter.__dict__.items()):
    if not _name.startswith("_"):
        globals()[_name] = _value

if __name__ == "__main__":
    main()
