"""Build an exclusive source/data bundle after the new runner is committed."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE / "execution_v2"
CAMPAIGN = BASE.name + "_execution_v2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    OUT.mkdir(exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tracked = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT, text=True).splitlines()
    prefixes = ("paper/", "models/", "data_loader/", "simple_lab_test/", "utils/")
    selected = [p for p in tracked if p.startswith(prefixes)]
    assert "paper/scripts/run_phen_instacart_3seed.py" in selected
    files = {}
    for relative in selected:
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink(), relative
        committed = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT)
        assert committed == path.read_bytes(), f"Uncommitted source: {relative}"
        files[relative] = digest(path)

    old = json.loads((ROOT / "paper/contracts/local_detail_instacart_replication_v1.json").read_text())
    data = next(d for d in old["datasets"] if d["dataset_id"] == "insta_market_basket")
    for key in ("additional_history_boundaries", "history_boundaries", "potentially_active_lags",
                "window_limitation", "expected_global_steps"):
        data.pop(key, None)
    original_identity = json.loads(json.dumps(data["inherited_data_identity"]))
    input_dir = ROOT / "search_artifacts/titantpp_instacart_norm_5080_20260930_v1/data/insta_market_basket"
    inputs = {"sample_data/insta_market_basket/train_validation.parquet": input_dir / "train_validation.parquet",
              "sample_data/insta_market_basket/split_manifest.json": input_dir / "split_manifest.json"}
    hashes = {p: digest(source) for p, source in inputs.items()}
    assert hashes["sample_data/insta_market_basket/train_validation.parquet"] == "2593041835f1bfc16bdb7ced6d5aab177c6c0006d6d970ba75aeccb97a82b505"
    assert hashes["sample_data/insta_market_basket/split_manifest.json"] == "81f1c59686dc49bc7b9183f29680d4567e63341fe3d2fac7c6d2c0f4dbaaf71c"
    for key, name in (("data", "train_validation.parquet"), ("split_manifest", "split_manifest.json")):
        relative = "sample_data/insta_market_basket/" + name
        data["inherited_data_identity"][key] = {"path": relative, "sha256": hashes[relative]}

    architecture = {
        "name": "PHEN", "component_arm": "full", "hidden_dim": 64, "encoder_blocks": 2,
        "attention_heads": 4, "persistent_memory": 16, "prototype_count": 64, "prototype_topk": 4,
        "history_basis": 15, "history_branches": 8, "history_parameter_count": 9600,
        "history_availability_thresholds": [1, 2, 4, 8, 16, 32, 64, 128],
        "relation": "shared current/previous observed state pair; gates use available history length",
        "factory": "models.TPPs.CountAwareComponentAblation.build_model(data, 'full')",
        "input_time": "log1p", "input_quantity": "log1p", "target_values_excluded": True,
        "objective": "recorded duration NLL + log quantity MSE; alpha1; lambda_tail0",
    }
    hosts = {}
    for host in ("5080", "5090"):
        observed = json.loads((BASE / f"{host}_runtime.stdout").read_text())
        hosts[host] = {
            "root": f"/home/leekwanhyeong/workspace/paper_research/search_artifacts/{CAMPAIGN}_{host}",
            "python": observed["python"], "boot_id": observed["boot_id"],
            "gpu_uuid": observed["gpu_csv"].split(",")[0],
            "paused_processes": observed["paused_processes"],
            "native_runtime": {k: observed[k] for k in ("python_version", "torch_version", "cuda_version", "cudnn_version")},
            "tmux": "/usr/bin/tmux" if host == "5080" else "/opt/miniconda3/envs/ai_env/bin/tmux",
        }
    contract = {
        "schema": "phen_instacart_3seed_v1", "campaign": CAMPAIGN,
        "model_role": "observed_time_joint_v1", "evaluation_scope": "validation_only",
        "source": {"base_git_revision": revision, "files": files, "files_sha256": canonical(files)},
        "input_files": hashes, "hosts": hosts, "datasets": [data],
        "architecture": architecture, "design_sha256": canonical(architecture),
        "seeds": [42, 52, 62], "host_by_seed": {"42": "5080", "52": "5090", "62": "5090"},
        "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40, "batch_size": 128,
                     "monitor": "validation_raw_quantity_rmse", "warm_start": False,
                     "tie": "strict_earliest_finite_minimum", "optimizer": "AdamW", "lr": .001,
                     "weight_decay": .01, "grad_clip": 1.},
        "provenance": {"original_instacart_identity": original_identity,
                       "original_contract_sha256": digest(ROOT / "paper/contracts/local_detail_instacart_replication_v1.json"),
                       "historical_test_exposure": True, "fresh_fit_count": 3, "reused_fit_count": 0},
        "policy": {"scientific_retries": 0, "overwrite": False, "resume": False,
                   "maximum_parallel_fits_per_gpu": 1, "test_evaluation": False,
                   "foreign_process_changes": False, "MAC_executed": False},
    }
    save(OUT / "EXECUTION_CONTRACT.json", contract)
    bundle = OUT / "source_inputs.tar.gz"
    with tarfile.open(bundle, "x:gz") as archive:
        for relative in sorted(files):
            archive.add(ROOT / relative, arcname="source/" + relative, recursive=False)
        for relative, path in inputs.items():
            archive.add(path, arcname="source/" + relative, recursive=False)
        archive.add(OUT / "EXECUTION_CONTRACT.json", arcname="EXECUTION_CONTRACT.json", recursive=False)
    # Verify bootstrap resolution in the exact extracted layout, outside the
    # original repository. Local in-repo tests cannot expose missing sentinels.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="phen-instacart-source-") as tmp:
        dest = Path(tmp)
        with tarfile.open(bundle) as archive:
            archive.extractall(dest, filter="data")
        check = "from pathlib import Path; from simple_lab_test.common.pathing import resolve_project_root; root=Path.cwd().resolve(); assert resolve_project_root(__file__ if '__file__' in globals() else root)==root; from paper.scripts.count_aware_tpp_backbone import core; import utils.magnitude_pipeline; assert Path(utils.magnitude_pipeline.__file__).resolve().is_relative_to(root); print('isolated bootstrap and scientific imports PASS')"
        result = subprocess.run([sys.executable, "-c", check], cwd=dest / "source", capture_output=True, text=True)
        save(OUT / "ISOLATED_IMPORT_CHECK.json", {"actual_exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        assert result.returncode == 0, result.stderr
    save(OUT / "BUNDLE.json", {"source_revision": revision, "source_files": len(files),
                              "source_files_sha256": canonical(files), "input_files": hashes,
                              "contract_sha256": canonical(contract), "contract_raw_sha256": digest(OUT / "EXECUTION_CONTRACT.json"),
                              "bundle_sha256": digest(bundle), "bundle_bytes": bundle.stat().st_size})
    print(json.dumps(json.loads((OUT / "BUNDLE.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
