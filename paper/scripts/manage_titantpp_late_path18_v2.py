"""Fixed-log-input late-path18 directly after the existing loss comparison.

Preserve the previous preparation and every historical source/result byte.
Only the dependency, common-loss selector and isolated execution root change.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

_PREVIOUS = Path(__file__).with_name("manage_titantpp_late_path18.py")
_PREVIOUS_SHA = "30dfd093b53b65fc2886c2474d647fab460458851dcbd9d0caab5096dd3f99d4"
if hashlib.sha256(_PREVIOUS.read_bytes()).hexdigest() != _PREVIOUS_SHA:
    raise ValueError("Preserved late-path manager changed")
_spec = importlib.util.spec_from_file_location("paper.scripts._late_path18_v2_manager_private", _PREVIOUS)
_adapter = importlib.util.module_from_spec(_spec)
_source = _PREVIOUS.read_text()
_REPLACEMENTS = {
    "from paper.scripts import titantpp_late_path18_runtime as _runtime":
        "from paper.scripts import titantpp_late_path18_v2_runtime as _runtime",
    '"titantpp_mlp16_late_path18_20261019_v1"': '"titantpp_mlp16_late_path18_20261019_v2"',
    '"paper.scripts.titantpp_late_path18_runtime"': '"paper.scripts.titantpp_late_path18_v2_runtime"',
    'PARENT_NAME = "titantpp_raw_time_mlp27_20261019_v1"':
        'PARENT_NAME = "titantpp_quantity_loss_mlp18_20261019_v2"',
    'tmux=f"late_path18_{host}_20261019_v1"': 'tmux=f"late_path18_{host}_20261019_v2"',
    '"parent_raw_time27": binding': '"parent_quantity_loss27": binding',
    '2026-10-19 user approved two late-path candidates x Cap3datasets x3seeds after loss/time comparisons, same one common condition, max3 per5080/5090; CNN unchanged; no Test':
        '2026-10-19 user deferred raw-time27; two late-path candidates x Cap3datasets x3seeds directly after loss comparison; both inputs fixed log1p; one shared loss selected from3; max3 per5080/5090; CNN unchanged; no Test',
}
for _before, _after in _REPLACEMENTS.items():
    if _source.count(_before) != 1:
        raise ValueError("Unexpected preserved manager adaptation: " + _before)
    _source = _source.replace(_before, _after, 1)
exec(compile(_source, str(_PREVIOUS), "exec"), _adapter.__dict__)
_impl = _adapter._impl
_impl.REQUIRED_NEW_SOURCE = (
    "paper/scripts/manage_titantpp_late_path18_v2.py",
    "paper/scripts/titantpp_late_path18_v2_runtime.py",
    "paper/scripts/select_compare_titantpp_late_path18_v2.py",
)
_SOURCE_READY_PATH = _impl.ROOT / "reports/titantpp_mlp16_late_path18_20261019_v1/SOURCE_READY.json"
_SOURCE_READY_SHA = "ef8b043e7214a5ab855c0823f25ce62e8fdf388904ee40affefb2c68834f2bd0"
PARENT_NAME = "titantpp_quantity_loss_mlp18_20261019_v2"


def verify_parent_completion():
    """Verify only completed loss27 originals; no raw-time fit is needed/run."""
    from paper.scripts import manage_titantpp_raw_time_mlp27 as verifier
    from paper.scripts.compare_titantpp_quantity_loss_mlp18 import aggregate
    from paper.scripts.select_compare_titantpp_late_path18_v2 import select_common_condition
    verified = verifier.verify_parent_completion()
    contract, final, comparison, binding = verified
    references = verifier.baseline_validation_references(verified=verified)
    _impl.require(len(references) == 27, "Expected27 completed loss endpoints")
    for key, value in aggregate(comparison["records"]).items():
        _impl.require(comparison[key] == value, "Preserved loss aggregation differs: " + key)
    for row in references:
        row.update(time_input="log1p", time_input_transform="log1p_nonnegative_float32")
    selection = select_common_condition(references)
    _impl.require(selection["selected_condition"]["time_input"] == "log1p"
                  and len(selection["condition_scores"]) == 3, "Only loss may be selected")
    return contract, final, references, selection, binding


def source_snapshot():
    """Inherit the sealed960 bytes and add exactly three new operation files."""
    _impl.require(_impl.sha(_SOURCE_READY_PATH) == _SOURCE_READY_SHA, "Preserved source960 inventory changed")
    previous = _impl.read(_SOURCE_READY_PATH)
    original = previous["files"]
    _impl.require(len(original) == 960 and _impl.canonical(original) == previous["files_sha256"],
                  "Expected preserved source960")
    for name, digest in original.items():
        _impl.require(_impl.sha(_impl.ROOT / name) == digest, "Preserved source changed: " + name)
    required = set(_impl.REQUIRED_NEW_SOURCE)
    tracked = set(_impl.subprocess.check_output(["git", "ls-files", "--", *required],
                                               cwd=_impl.ROOT, text=True).splitlines())
    _impl.require(tracked == required and not _impl.subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--", *required], cwd=_impl.ROOT, text=True).strip(),
        "Commit new fixed-input operation files before prepare")
    files = {**original, **{name: _impl.sha(_impl.ROOT / name) for name in required}}
    revision = _impl.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_impl.ROOT, text=True).strip()
    return {"base_git_revision": revision, "files": files, "files_sha256": _impl.canonical(files),
        "model_science_source_revision": revision, "model_science_source_files_sha256": _impl.canonical(files),
        "parent_source_files_sha256": previous["files_sha256"], "parent_source_file_count": 960,
        "model_worker_unchanged": True, "raw_time_experiment_required": False}


_adapter.verify_parent_completion = verify_parent_completion
_adapter.source_snapshot = source_snapshot
_impl.source_snapshot = source_snapshot
_original_collection_program = _impl.collection_program


def collection_program(contract, host):
    """Bind the preserved collector to this campaign's loss-parent validator."""
    source = _original_collection_program(contract, host)
    needle = "from paper.scripts import titantpp_late_path18_runtime as r"
    _impl.require(source.count(needle) == 1, "Unexpected preserved collection program")
    return source.replace(needle, "from paper.scripts import titantpp_late_path18_v2_runtime as r", 1)


_impl.collection_program = collection_program
for _name, _value in tuple(_impl.__dict__.items()):
    if not _name.startswith("_"):
        globals()[_name] = _value

if __name__ == "__main__":
    raise SystemExit(_impl.main())
