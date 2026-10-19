"""New isolated attempt after a reproduced pre-fit /proc system-owner lookup bug.

All v1 originals and scientific code stay immutable. Only the corrected runtime,
new root, matching finite completion and comparator entrypoints differ.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

_PATH = Path(__file__).with_name("manage_titantpp_shared_history_cap_execution.py")
_SHA = "c5804e4341a5a5a2ca3009c64e21281fa3a1889bcd3faaf0e42f2b854ed41135"
if hashlib.sha256(_PATH.read_bytes()).hexdigest() != _SHA:
    raise ValueError("Preserved execution v1 manager changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_private_v1", _PATH)
_adapter = importlib.util.module_from_spec(_spec)
_source = _PATH.read_text()
_replacements = {
    "from paper.scripts import titantpp_shared_history_cap_execution_runtime as runtime":
        "from paper.scripts import titantpp_shared_history_cap_execution_v2_runtime as runtime",
    ' / "execution_v1"': ' / "execution_v2"',
    'MODULE = "paper.scripts.titantpp_shared_history_cap_execution_runtime"':
        'MODULE = "paper.scripts.titantpp_shared_history_cap_execution_v2_runtime"',
    '{NAME}_execution_v1_5080': '{NAME}_execution_v2_5080',
    'shared_history18_5080_20261019_v1': 'shared_history18_5080_20261019_v2',
    'NAME + "/execution_v1"': 'NAME + "/execution_v2"',
    '"new_execution_source_count": 4': '"new_execution_source_count": 7',
    'from paper.scripts.compare_titantpp_shared_history_cap_execution import main as compare':
        'from paper.scripts.compare_titantpp_shared_history_cap_execution_v2 import main as compare',
    '"paper.scripts.manage_titantpp_shared_history_cap_execution", "--mode", "finish"':
        '"paper.scripts.manage_titantpp_shared_history_cap_execution_v2", "--mode", "finish"',
    '"from paper.scripts import titantpp_shared_history_cap_execution_runtime as r"':
        '"from paper.scripts import titantpp_shared_history_cap_execution_v2_runtime as r"',
}
for _before, _after in _replacements.items():
    _expected = 2 if _before == ' / "execution_v1"' else 1
    if _source.count(_before) != _expected:
        raise ValueError("Unexpected preserved execution-manager adaptation: " + _before)
    _source = _source.replace(_before, _after)
exec(compile(_source, str(_PATH), "exec"), _adapter.__dict__)
_adapter.NEW_SOURCE = (*_adapter.NEW_SOURCE,
    "paper/scripts/manage_titantpp_shared_history_cap_execution_v2.py",
    "paper/scripts/titantpp_shared_history_cap_execution_v2_runtime.py",
    "paper/scripts/compare_titantpp_shared_history_cap_execution_v2.py")
for _name, _value in tuple(_adapter.__dict__.items()):
    if not _name.startswith("_"):
        globals()[_name] = _value

if __name__ == "__main__":
    main()
