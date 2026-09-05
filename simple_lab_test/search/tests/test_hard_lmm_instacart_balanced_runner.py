"""Stage barriers fail closed before the next phase can load restricted data."""
import hashlib

import pytest

from paper.scripts.run_hard_lmm_instacart_balanced import require_gate


def test_failed_or_wrong_stage_cannot_admit_next_inputs():
    with pytest.raises(ValueError, match="did not pass"):
        require_gate({"stage": "input", "passed": False, "frozen_hashes": {}}, "input")
    with pytest.raises(ValueError, match="Wrong gate stage"):
        require_gate({"stage": "input", "passed": True, "frozen_hashes": {}}, "body")


def test_changed_frozen_file_invalidates_previously_passed_gate(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text('{"cut":1}')
    marker = {"stage": "input", "passed": True,
              "frozen_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}}
    require_gate(marker, "input")
    path.write_text('{"cut":2}')
    with pytest.raises(ValueError, match="changed"):
        require_gate(marker, "input")
