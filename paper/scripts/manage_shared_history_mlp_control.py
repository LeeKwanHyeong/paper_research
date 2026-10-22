"""Isolated controls for the validation-only Shared15 / single-MLP study.

The prior A-D implementation remains immutable. Low-level process supervision
is adapted here with explicit new modules, cardinalities and phase identities.
No Test stage exists in this campaign.
"""

from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time
import traceback

NAME = "shared_history_mlp_control_20261022_v1"
SCHEMA = "shared_history_mlp_control_execution_v1"
ARMS = ("shared15", "single_mlp", "encoder_only")
DATASETS = ("raf_spare_parts", "yellow_trip_hourly", "intermittent_frozen_5000")
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / NAME
BUNDLE = ROOT / "search_artifacts" / NAME
PARENT = ROOT / "search_artifacts/shared_component_ablation_20261022_v1"
NEW_SOURCE = (
    "models/TPPs/CountAwareHistoryMLPControl.py",
    "paper/scripts/run_shared_history_mlp_control.py",
    "paper/scripts/profile_shared_history_mlp_control.py",
    "paper/scripts/manage_shared_history_mlp_control.py",
)


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def below(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Evidence path must stay below the isolated root",
    )
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Evidence symlink escaped the isolated root")
    return path


def validate(c):
    require(
        c.get("schema") == SCHEMA and c.get("campaign") == NAME,
        "Campaign identity changed",
    )
    require(
        c.get("scope") == "execution"
        and c.get("evaluation_scope") == "validation_only"
        and c.get("execution_scope") == "validation_only"
        and c.get("held_out_test_authorized") is False,
        "Only Validation and Train profiling are authorized",
    )
    require(set(c.get("hosts", {})) == {"5080", "5090"}, "Both approved hosts required")
    expected = {(d, a, s) for d in DATASETS for a in ARMS for s in (42, 52, 62)}
    jobs = c.get("jobs", [])
    require(
        len(jobs) == 27
        and len({j.get("id") for j in jobs}) == 27
        and {(j.get("dataset"), j.get("arm"), j.get("seed")) for j in jobs} == expected,
        "Exactly 27 matched conditions required",
    )
    for j in jobs:
        require(
            j["id"] == f"{j['dataset']}__{j['arm']}__seed{j['seed']}"
            and j["host"] == ("5080" if j["seed"] == 42 else "5090"),
            "Fixed job/host assignment changed",
        )
        require(
            bool(j.get("reuse")) == (j["arm"] != "single_mlp"),
            "Only B/D may reuse original fits",
        )
        if j.get("reuse"):
            reuse = j["reuse"]
            underlying = "no_prototype" if j["arm"] == "shared15" else "encoder_only"
            require(
                reuse.get("underlying_arm") == underlying
                and reuse.get("parent_job_id")
                == f"{j['dataset']}__{underlying}__seed{j['seed']}"
                and reuse.get("parent_contract_sha256")
                == c["parent"]["canonical_contract_sha256"]
                and reuse.get("parent_source_files_sha256")
                == c["parent"]["source_files_sha256"],
                "Original B/D identity may not be retagged",
            )
            require(
                all(
                    isinstance(reuse.get(k), dict)
                    and set(reuse[k]) == {"path", "sha256"}
                    for k in (
                        "checkpoint",
                        "history",
                        "train_receipt",
                        "validation_receipt",
                        "train_exit",
                        "validation_exit",
                        "validation_predictions",
                        "train_stdout",
                        "train_stderr",
                        "validation_stdout",
                        "validation_stderr",
                    )
                ),
                "Original reuse proof is incomplete",
            )
    require(
        c.get("fresh_fit_count") == 9 and c.get("reused_fit_count") == 18,
        "Fit/reuse counts changed",
    )
    require(
        c["common_condition"]
        == {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0},
        "Common objective changed",
    )
    require(
        c["training"]
        == {
            "maximum_epochs": 300,
            "minimum_epochs": 40,
            "patience": 40,
            "batch_size": 128,
            "monitor": "validation_raw_quantity_rmse",
        },
        "Training budget changed",
    )
    require(
        c["source"]["files"]
        and canonical(c["source"]["files"]) == c["source"]["files_sha256"],
        "Source inventory changed",
    )
    require(
        len(c["input_files"]) == 9 and len(c["legacy_files"]) >= 198,
        "All inputs and original B/D evidence required",
    )
    require(
        c["cost"]
        == {
            "batch_size": 128,
            "preparation_seed": 42,
            "warmup_batches": 5,
            "measured_batches": 32,
            "repetitions": 3,
            "modes": ["train", "eval"],
            "fresh_process_per_rep": True,
            "same_fit_host": True,
            "maximum_parallel_workers": 1,
            "automatic_retry": False,
        },
        "Train profiling conditions changed",
    )
    require(
        c.get("MAC_deferred") is True and c.get("maximum_parallel_workers") == 1,
        "Scope or GPU concurrency changed",
    )
    for relative in (*c["source"]["files"], *c["input_files"], *c["legacy_files"]):
        require(
            not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "Inventory escaped root",
        )


def verify_files(c, source):
    validate(c)
    for inventory in (
        c["source"]["files"],
        c["input_files"],
        c.get("legacy_files", {}),
    ):
        for path, digest in inventory.items():
            require(
                sha(below(source, path)) == digest, "Frozen file SHA changed: " + path
            )


def verify_parent_preservation():
    value = read(REPORT / "PRESERVATION.json")
    require(
        value.get("status") == "frozen" and bool(value.get("files")),
        "Parent preservation inventory missing",
    )
    for name, item in value["files"].items():
        p = Path(name)
        require(
            p.is_file()
            and not p.is_symlink()
            and p.stat().st_size == item["bytes"]
            and sha(p) == item["sha256"],
            "Original campaign evidence changed: " + name,
        )
    return value


def audit_reuse(parent):
    """Read only the parent's fit/Validation collections, never Test metrics."""
    import torch
    from models.TPPs import CountAwareComponentAblation as old_model
    from paper.scripts import advance_shared_component_ablation as old_control
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    from paper.scripts import run_titantpp_shared_history_raw9 as raw9

    roots = {
        host: old_control.validate_collection(
            parent,
            host,
            "fit_validation",
            PARENT / "retrieval/fit_validation" / host,
            extract=False,
        )
        for host in ("5080", "5090")
    }
    audit = []
    for job in parent["jobs"]:
        if job["arm"] not in ("no_prototype", "encoder_only"):
            continue
        base = roots[job["host"]]
        folder = base / "run" / job["id"]
        train, validation = (
            read(folder / "train_receipt.json"),
            read(folder / "validation_receipt.json"),
        )
        checkpoint = below(base, train["selected_checkpoint"])
        history = checkpoint.parent / "history.json"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        old_model.validate_checkpoint(payload, job["arm"])
        require(
            payload.get("seed") == job["seed"]
            and payload.get("backbone") == job["arm"],
            "Original B/D checkpoint identity changed",
        )
        require(
            payload.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
            and raw9.prepared.select_best_epoch(read(history)["history"])
            == payload["best_epoch"]
            == train["selected_epoch"]
            == validation["selected_epoch"],
            "Original first strict Validation selection changed",
        )
        require(
            sha(checkpoint)
            == train["checkpoint_sha256"]
            == validation["checkpoint_sha256"],
            "Original binary changed",
        )
        state = canonical_state_dict_sha256(payload["model_state_dict"])
        require(
            state
            == payload["model_state_sha256"]
            == train["state_sha256"]
            == validation["state_sha256"],
            "Original state changed",
        )
        require(
            payload.get("interface_meta", {}).get("source_files_sha256")
            == parent["source"]["files_sha256"],
            "Parent training source identity changed",
        )
        for stage in ("train", "validation"):
            exit_value = read(base / "control" / f"{job['id']}.{stage}.exit.json")
            require(
                exit_value.get("actual_os_returncode") == 0
                and exit_value.get("timeout") is False
                and exit_value.get("resource_incident") is None
                and exit_value.get("supervisor_error") is None,
                "Original scientific exit was not successful",
            )
            for stream in ("stdout", "stderr"):
                require(
                    sha(base / "control" / f"{job['id']}.{stage}.{stream}.log")
                    == exit_value[stream + "_sha256"],
                    "Original raw output changed",
                )
        require(
            validation.get("validation_replay_passed") is True
            and validation.get("individual_predictions_match") is True,
            "Original individual Validation replay missing",
        )
        prediction = base / Path(validation["prediction_path"]).relative_to(
            parent["hosts"][job["host"]]["root"]
        )
        reference = train["validation_reference"]
        from paper.scripts.run_shared_component_ablation import compare_prediction_files

        parent_reference = base / Path(reference["prediction_path"]).relative_to(
            parent["hosts"][job["host"]]["root"]
        )
        compare_prediction_files(prediction, parent_reference)
        files = {
            "checkpoint": checkpoint,
            "history": history,
            "train_receipt": folder / "train_receipt.json",
            "validation_receipt": folder / "validation_receipt.json",
            "train_exit": base / "control" / f"{job['id']}.train.exit.json",
            "validation_exit": base / "control" / f"{job['id']}.validation.exit.json",
            "validation_predictions": prediction,
        }
        files.update(
            {
                stage + "_" + stream: base
                / "control"
                / f"{job['id']}.{stage}.{stream}.log"
                for stage in ("train", "validation")
                for stream in ("stdout", "stderr")
            }
        )
        audit.append(
            {
                "parent_job_id": job["id"],
                "dataset": job["dataset"],
                "seed": job["seed"],
                "host": job["host"],
                "underlying_arm": job["arm"],
                "selected_epoch": train["selected_epoch"],
                "canonical_tensor_sha256": state,
                "files": {
                    key: {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}
                    for key, p in files.items()
                },
            }
        )
    require(len(audit) == 18, "All eighteen B/D checkpoints must be audited")
    return audit


def prepare():
    spec = read(REPORT / "SCIENCE_SPEC.json")
    auth = read(REPORT / "AUTHORIZATION.json")
    require(
        spec["status"] == "locked"
        and auth["status"] == "approved"
        and auth["science_spec_sha256"] == sha(REPORT / "SCIENCE_SPEC.json"),
        "Science or explicit authorization changed",
    )
    require(not BUNDLE.exists(), "Prepared campaign exists; preserve it")
    parent_path = PARENT / "execution_contract.json"
    parent = read(parent_path)
    require(
        sha(parent_path) == spec["parent_contract_raw_sha256"],
        "Parent contract changed",
    )
    for name, digest in parent["source"]["files"].items():
        require(
            sha(ROOT / name) == digest,
            "Frozen parent scientific source changed: " + name,
        )
    if not (REPORT / "PRESERVATION.json").exists():
        inventory = {}
        for old_root in (
            PARENT,
            ROOT / "reports/shared_component_ablation_20261022_v1",
        ):
            for p in sorted(old_root.rglob("*")):
                require(not p.is_symlink(), "Original evidence contains symlink")
                if p.is_file():
                    inventory[str(p)] = {"sha256": sha(p), "bytes": p.stat().st_size}
        write(
            REPORT / "PRESERVATION.json",
            {
                "status": "frozen",
                "files": inventory,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    verify_parent_preservation()
    rows = audit_reuse(parent)
    write(
        REPORT / "B_D_REUSE_AUDIT.json",
        {
            "status": "PASS",
            "scope": "validation_only",
            "new_inference": 0,
            "parent_contract_sha256": canonical(parent),
            "checkpoint_rows": rows,
        },
    )
    c = {
        key: deepcopy(parent[key])
        for key in ("datasets", "input_files", "training", "limits", "common_condition")
    }
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    c.update(
        schema=SCHEMA,
        campaign=NAME,
        scope="execution",
        created_at=datetime.now(timezone.utc).isoformat(),
        science_spec_sha256=sha(REPORT / "SCIENCE_SPEC.json"),
        science=spec,
        authorization_sha256=sha(REPORT / "AUTHORIZATION.json"),
        preservation_sha256=sha(REPORT / "PRESERVATION.json"),
        model_role="observed_time_joint_v1",
        data_variant="original_uncapped",
        quantity_capping_applied=False,
        execution_scope="validation_only",
        evaluation_scope="validation_only",
        held_out_test_authorized=False,
        maximum_parallel_workers=1,
        fresh_fit_count=9,
        reused_fit_count=18,
        MAC_deferred=True,
        source={
            "files": deepcopy(parent["source"]["files"]),
            "base_git_revision": revision,
        },
        parent={
            "path": str(parent_path),
            "contract_file_sha256": sha(parent_path),
            "canonical_contract_sha256": canonical(parent),
            "source_files_sha256": parent["source"]["files_sha256"],
            "legacy_contract": {
                "path": "legacy/parent_contract.json",
                "sha256": sha(parent_path),
            },
        },
        architecture=deepcopy(spec["architecture"]),
        cost={
            key: spec["cost"][key]
            for key in (
                "batch_size",
                "preparation_seed",
                "warmup_batches",
                "measured_batches",
                "repetitions",
                "modes",
                "fresh_process_per_rep",
                "same_fit_host",
            )
        },
        implementation_qualification={"status": "pending", "training_allowed": False},
        native_qualification_required=True,
        execution_authorization={
            "real_data_training": True,
            "real_data_inference": True,
            "remote_execution": True,
            "test_evaluation": False,
        },
    )
    c["cost"].update(maximum_parallel_workers=1, automatic_retry=False)
    c["source"]["files"].update({name: sha(ROOT / name) for name in NEW_SOURCE})
    c["source"]["files_sha256"] = canonical(c["source"]["files"])
    c["hosts"] = deepcopy(parent["hosts"])
    for host, h in c["hosts"].items():
        preflight = read(REPORT / "preflight" / (host + ".stdout"))
        remote = (
            "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/"
            + NAME
            + "_"
            + host
        )
        h.update(
            root=remote,
            source_root=remote + "/source",
            tmux=NAME + "_" + host,
            absolute_host_deadline_unix=time.time() + c["limits"]["queue_wall_seconds"],
            boot_id=preflight["boot_id"],
        )
        h.pop("completed_predecessor", None)
        h.pop("absolute_host_deadline_binding", None)
        h["environment"].update(
            SOURCE_REVISION=revision,
            PYTHONPATH=remote + "/source",
            MPLCONFIGDIR=remote + "/cache/matplotlib",
            XDG_CACHE_HOME=remote + "/cache",
            TORCHINDUCTOR_CACHE_DIR=remote + "/cache/torchinductor",
        )
    c["jobs"], c["legacy_files"] = [], {}
    BUNDLE.mkdir(parents=True)
    for data in c["datasets"]:
        for seed in (42, 52, 62):
            for arm in ARMS:
                job = {
                    "id": f"{data['dataset_id']}__{arm}__seed{seed}",
                    "dataset": data["dataset_id"],
                    "seed": seed,
                    "arm": arm,
                    "host": "5080" if seed == 42 else "5090",
                    "time_input": "log1p",
                    "loss_mode": "log_mse",
                    "time_alpha": 1.0,
                }
                if arm != "single_mlp":
                    underlying = "no_prototype" if arm == "shared15" else "encoder_only"
                    row = next(
                        v
                        for v in rows
                        if v["dataset"] == data["dataset_id"]
                        and v["seed"] == seed
                        and v["underlying_arm"] == underlying
                    )
                    reuse = {
                        k: row[k]
                        for k in (
                            "selected_epoch",
                            "canonical_tensor_sha256",
                            "parent_job_id",
                            "underlying_arm",
                        )
                    }
                    reuse.update(
                        parent_contract_sha256=canonical(parent),
                        parent_source_files_sha256=parent["source"]["files_sha256"],
                    )
                    for key, item in row["files"].items():
                        source = Path(item["path"])
                        relative = (
                            "legacy/" + job["id"] + "/" + key + "".join(source.suffixes)
                        )
                        dest = BUNDLE / "input_snapshot" / relative
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with source.open("rb") as src, dest.open("xb") as dst:
                            shutil.copyfileobj(src, dst)
                        require(sha(dest) == item["sha256"], "Legacy byte copy changed")
                        c["legacy_files"][relative] = item["sha256"]
                        reuse[key] = {"path": relative, "sha256": item["sha256"]}
                    job["reuse"] = reuse
                c["jobs"].append(job)
    for relative, source in [
        (name, PARENT / "input_snapshot" / name) for name in c["input_files"]
    ] + [("legacy/parent_contract.json", parent_path)]:
        digest = sha(source)
        dest = BUNDLE / "input_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, dest.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        require(sha(dest) == digest, "Input copy changed")
        if relative.startswith("legacy/"):
            c["legacy_files"][relative] = digest
        else:
            require(c["input_files"][relative] == digest, "Original input differs")
    validate(c)
    verify_parent_preservation()
    write(BUNDLE / "prepared_contract.json", c)
    return {
        "status": "prepared_not_trainable",
        "new_fit": 9,
        "reused": 18,
        "validation": 27,
        "cost": 162,
    }


def seal(tests_receipt):
    c = read(BUNDLE / "prepared_contract.json")
    qa = read(REPORT / "INDEPENDENT_QA.json")
    tests = read(tests_receipt)
    require(
        qa.get("status") == "PASS" and tests.get("actual_os_returncode") == 0,
        "Local independent/test gates not passed",
    )
    reviewed = qa["files_reviewed_sha256"]
    require(
        all(name in reviewed for name in NEW_SOURCE)
        and all(sha(ROOT / name) == digest for name, digest in reviewed.items()),
        "Reviewed source changed",
    )
    require(
        all(
            tests.get("files_checked_sha256", {}).get(name)
            == c["source"]["files"][name]
            for name in NEW_SOURCE
        )
        and tests.get("source_unchanged_during_tests") is True,
        "Tests are not bound to prepared source",
    )
    require(
        c["science_spec_sha256"] == sha(REPORT / "SCIENCE_SPEC.json"),
        "Science changed after preparation",
    )
    require(
        all(
            sha(ROOT / name) == digest for name, digest in c["source"]["files"].items()
        ),
        "Prepared scientific source changed",
    )
    verify_parent_preservation()
    c["implementation_qualification"] = {
        "status": "passed",
        "training_allowed": True,
        "independent_qa_sha256": sha(REPORT / "INDEPENDENT_QA.json"),
        "tests_receipt_sha256": sha(tests_receipt),
    }
    write(BUNDLE / "execution_contract.json", c)
    with (
        (BUNDLE / "deployment.tar.gz").open("xb") as stream,
        tarfile.open(fileobj=stream, mode="w:gz") as archive,
    ):
        archive.add(
            BUNDLE / "execution_contract.json", arcname="execution_contract.json"
        )
        for name in c["source"]["files"]:
            archive.add(ROOT / name, arcname="source/" + name)
        for name, digest in {**c["input_files"], **c["legacy_files"]}.items():
            p = BUNDLE / "input_snapshot" / name
            require(sha(p) == digest, "Prepared data/legacy copy changed")
            archive.add(p, arcname="source/" + name)
        marker = tarfile.TarInfo("source/sample_data")
        marker.type = tarfile.DIRTYPE
        archive.addfile(marker)
    result = {
        "status": "sealed",
        "contract_sha256": sha(BUNDLE / "execution_contract.json"),
        "canonical_contract_sha256": canonical(c),
        "archive_sha256": sha(BUNDLE / "deployment.tar.gz"),
        "source_files_sha256": c["source"]["files_sha256"],
        "git_revision": c["source"]["base_git_revision"],
        "science_spec_sha256": c["science_spec_sha256"],
        "independent_qa_sha256": sha(REPORT / "INDEPENDENT_QA.json"),
        "tests_receipt_sha256": sha(tests_receipt),
    }
    write(BUNDLE / "integrity.json", result)
    return result


def proc_identity(pid):
    try:
        path = Path("/proc") / str(pid)
        values = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "state": values[0],
            "start_ticks": int(values[19]),
            "pgid": int(values[2]),
            "argv": (path / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
            "cwd": os.readlink(path / "cwd"),
        }
    except FileNotFoundError:
        return None


def guard(c, host, active=()):
    h = c["hosts"][host]
    require(time.time() < h["absolute_host_deadline_unix"], "Campaign deadline reached")
    require(
        Path("/proc/sys/kernel/random/boot_id").read_text().strip() == h["boot_id"],
        "Server boot identity changed",
    )
    paused = h.get("paused_processes", [])
    for expected in paused:
        live = proc_identity(expected["pid"])
        require(
            live is not None
            and live["state"] == "T"
            and all(
                live[k] == expected[k] for k in ("start_ticks", "pgid", "argv", "cwd")
            ),
            "Historical paused process changed",
        )
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    pids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
    require(
        pids <= set(active) | {p["pid"] for p in paused},
        "Unknown GPU owner; no new work",
    )
    return {"compute_pids": sorted(pids), "paused_preserved": len(paused)}


def _require_identity_absent(identity):
    keys = ("pid", "start_ticks", "pgid", "argv", "cwd")
    require(
        isinstance(identity, dict)
        and all(key in identity for key in keys)
        and type(identity["pid"]) is int
        and type(identity["start_ticks"]) is int,
        "Actual process identity is missing",
    )
    live = proc_identity(identity["pid"])
    require(
        live is None or any(live[key] != identity[key] for key in keys),
        "Completed scientific process is still alive",
    )


def _verify_parent_exit(c, host, phase, root):
    """Validate the separate parent's observation; never infer its own OS exit."""
    terminal = read(root / (phase + ".supervisor_exit.json"))
    claim = read(root / (phase + ".claim.json"))
    parent = read(root / (phase + ".supervisor.intent.json"))
    h = c["hosts"][host]
    command = [
        h["python"],
        "-u",
        "-m",
        "paper.scripts.manage_shared_history_mlp_control",
        "--mode",
        "dispatch",
        "--contract",
        h["root"] + "/execution_contract.json",
        "--contract-sha",
        sha(root / "execution_contract.json"),
        "--host",
        host,
        "--phase",
        phase,
    ]
    require(
        terminal.get("host") == host
        and terminal.get("phase") == phase
        and terminal.get("contract_sha256") == canonical(c)
        and claim.get("contract_sha256") == canonical(c)
        and parent.get("contract_sha256") == canonical(c)
        and terminal.get("actual_os_returncode") == 0
        and terminal.get("command") == command
        and parent.get("command") == command
        and claim.get("identity", {}).get("argv") == command
        and claim["identity"].get("cwd") == h["source_root"]
        and parent.get("identity", {}).get("argv")
        == ["supervise" if item == "dispatch" else item for item in command]
        and parent["identity"].get("cwd") == h["source_root"],
        "Dispatcher actual exit or fixed parent identity differs",
    )
    for stream in ("stdout", "stderr"):
        require(
            sha(root / (phase + "." + stream + ".log"))
            == terminal.get(stream + "_sha256"),
            "Actual dispatcher output changed",
        )
    _require_identity_absent(claim["identity"])
    return terminal


def verify_native_qualification(c, host, root=None):
    root = Path(root or c["hosts"][host]["root"])
    value = read(root / "qualification.complete.json")
    require(
        value.get("status") == "complete"
        and value.get("host") == host
        and value.get("contract_sha256") == canonical(c),
        "Native qualification identity differs",
    )
    exits = value.get("actual_child_exits", [])
    require(
        len(exits) == 12
        and all(
            e.get("actual_os_returncode") == 0
            and e.get("timeout") is False
            and e.get("supervisor_error") is None
            and e.get("resource_incident") is None
            for e in exits
        ),
        "Native qualification requires actual successful exits for nine integration and three capacity checks",
    )
    jobs = {j["id"]: j for j in c["jobs"]}
    for entry in exits:
        require(
            entry.get("host") == host
            and entry.get("contract_sha256") == canonical(c)
            and jobs.get(entry.get("job_id"), {}).get("host") == host
            and entry.get("stage") in {"synthetic", "capacity"},
            "Native child scientific identity differs",
        )
        raw = root / "control" / (entry["job_id"] + "." + entry["stage"] + ".exit.json")
        require(
            read(raw) == entry,
            "Native qualification differs from the original actual child exit",
        )
        for stream in ("stdout", "stderr"):
            path = (
                root
                / "control"
                / (entry["job_id"] + "." + entry["stage"] + "." + stream + ".log")
            )
            require(
                sha(path) == entry.get(stream + "_sha256"),
                "Actual native child output changed",
            )
        _require_identity_absent(entry.get("identity"))
    require(
        {jobs[e["job_id"]]["arm"] for e in exits if e["stage"] == "capacity"}
        == set(ARMS),
        "All three full-batch capacity exits are required",
    )
    require(
        {
            (jobs[e["job_id"]]["dataset"], jobs[e["job_id"]]["arm"])
            for e in exits
            if e["stage"] == "synthetic"
        }
        == {(d, a) for d in DATASETS for a in ARMS},
        "All dataset/arm integration checks are required",
    )
    for arm, item in value["capacity_receipts"].items():
        path = below(root, item["path"])
        require(sha(path) == item["sha256"], "Native capacity evidence changed")
        receipt = read(path)
        require(
            receipt["status"] == "qualified"
            and receipt["arm"] == arm
            and receipt["host"] == host
            and receipt["contract_sha256"] == canonical(c)
            and receipt["source_files_sha256"] == c["source"]["files_sha256"],
            "Capacity receipt identity differs",
        )
        require(
            receipt.get("synthetic") is True
            and receipt.get("real_data_loaded") is False
            and receipt.get("cost_complete") is False
            and receipt.get("checkpoint_saved") is False,
            "Capacity qualification must be synthetic and preserve checkpoints",
        )
        rows = receipt.get("rows", [])
        require(
            len(rows) == 2 and {r.get("mode") for r in rows} == {"train", "eval"},
            "Both capacity modes are required",
        )
        for row in rows:
            require(
                row.get("batch_size") == 128
                and row.get("padded_length") == 256
                and row.get("optimizer_steps") == int(row["mode"] == "train"),
                "Full-batch capacity shape or optimizer step differs",
            )
            elapsed = row.get("elapsed_seconds")
            require(
                type(elapsed) in (float, int)
                and math.isfinite(elapsed)
                and elapsed > 0,
                "Capacity time must be finite and positive",
            )
            for kind in ("allocated", "reserved"):
                baseline = row.get("memory_baseline_bytes", {}).get(kind)
                peak = row.get("memory_peak_bytes", {}).get(kind)
                require(
                    type(baseline) is int
                    and type(peak) is int
                    and 0 <= baseline <= peak
                    and peak > 0,
                    "Capacity memory measurement is missing or invalid",
                )
    require(
        set(value["capacity_receipts"]) == set(ARMS),
        "Three native capacity receipts are required",
    )
    _verify_parent_exit(c, host, "qualification", root)
    return value


def verify_phase_receipt(c, host, phase, root=None):
    require(
        phase in ("qualification", "fit_validation", "cost"),
        "Unsupported phase; Test is not authorized",
    )
    root = Path(root or c["hosts"][host]["root"])
    complete = read(root / (phase + ".complete.json"))
    terminal = _verify_parent_exit(c, host, phase, root)
    require(
        complete.get("status") == "complete"
        and all(
            v.get("host") == host
            and v.get("phase") == phase
            and v.get("contract_sha256") == canonical(c)
            for v in (complete, terminal)
        )
        and terminal.get("actual_os_returncode") == 0,
        "Phase or dispatcher exit identity differs",
    )
    entries = complete.get("actual_child_exits", [])
    assigned = [j for j in c["jobs"] if j["host"] == host]
    if phase == "qualification":
        verify_native_qualification(c, host, root)
    else:
        expected = {
            (j["id"], "reuse" if j.get("reuse") else "train") for j in assigned
        } | {(j["id"], "validation") for j in assigned}
        if phase == "cost":
            expected = {(assigned[0]["id"], "prepare_batches")} | {
                (j["id"], f"cost_{mode}_{rep}")
                for j in assigned
                for mode in ("train", "eval")
                for rep in (1, 2, 3)
            }
        require(
            len(entries) == len(expected)
            and {(e.get("job_id"), e.get("stage")) for e in entries} == expected,
            "Phase is missing expected scientific or cost workers",
        )
    for entry in entries:
        require(
            entry.get("host") == host
            and entry.get("contract_sha256") == canonical(c)
            and entry.get("actual_os_returncode") == 0
            and entry.get("timeout") is False
            and entry.get("supervisor_error") is None
            and entry.get("resource_incident") is None,
            "Phase has an unsuccessful child exit",
        )
        stem = root / "control" / (entry["job_id"] + "." + entry["stage"])
        require(
            read(str(stem) + ".exit.json") == entry,
            "Phase differs from original child exit",
        )
        for stream in ("stdout", "stderr"):
            require(
                sha(str(stem) + "." + stream + ".log") == entry[stream + "_sha256"],
                "Actual child output changed",
            )
        _require_identity_absent(entry.get("identity"))
    return complete, terminal


def verify_global_gate(c, root):
    root = Path(root)
    gate = read(root / "global_validation_gate.json")
    jobs = {j["id"]: j for j in c["jobs"]}
    require(
        gate.get("status") == "passed"
        and gate.get("contract_sha256") == canonical(c)
        and set(gate.get("jobs", {})) == set(jobs),
        "All 27 individual Validation replays must pass before Train profiling",
    )
    for job_id, evidence in gate["jobs"].items():
        job = jobs[job_id]
        require(evidence["host"] == job["host"], "Global Validation host changed")
        selected = {}
        for stage in ("train", "validation"):
            for kind in ("receipt", "exit"):
                item = evidence[f"{stage}_{kind}"]
                path = below(root, item["path"])
                require(
                    sha(path) == item["sha256"], "Global original evidence SHA changed"
                )
                value = read(path)
                expected_stage = (
                    "reuse"
                    if stage == "train" and kind == "exit" and job.get("reuse")
                    else stage
                )
                require(
                    value.get("job_id") == job_id
                    and value.get("stage") == expected_stage
                    and value.get("host") == job["host"]
                    and value.get("contract_sha256") == canonical(c),
                    "Global scientific identity differs",
                )
                if kind == "exit":
                    require(
                        value.get("actual_os_returncode") == 0
                        and value.get("timeout") is False
                        and value.get("supervisor_error") is None
                        and value.get("resource_incident") is None,
                        "Actual successful child exits are required",
                    )
                else:
                    require(
                        value.get("status") == "complete"
                        and value.get("host") == job["host"],
                        "Receipt is not complete",
                    )
                    require(
                        value.get("source_files_sha256") == c["source"]["files_sha256"]
                        and value.get("held_out_test_evaluated") is False
                        and value.get("training_origin")
                        == ("reused_verified_fit" if job.get("reuse") else "fresh_fit"),
                        "Global selected source or training origin differs",
                    )
                    selected[stage] = value
                    if stage == "validation":
                        require(
                            value.get("validation_replay_passed") is True
                            and value.get("individual_predictions_match") is True,
                            "Individual Validation replay missing",
                        )
        selection_keys = (
            "selected_checkpoint",
            "selected_epoch",
            "checkpoint_sha256",
            "state_sha256",
            "training_origin",
        )
        require(
            all(
                selected["train"].get(key) is not None
                and selected["train"][key] == selected["validation"].get(key)
                for key in selection_keys
            )
            and isinstance(selected["train"]["selected_checkpoint"], str)
            and bool(selected["train"]["selected_checkpoint"])
            and type(selected["train"]["selected_epoch"]) is int
            and 1
            <= selected["train"]["selected_epoch"]
            <= c["training"]["maximum_epochs"],
            "Global Validation must use the exact selected training checkpoint",
        )
        if job.get("reuse"):
            require(
                selected["train"]["selected_epoch"] == job["reuse"]["selected_epoch"]
                and selected["train"]["checkpoint_sha256"]
                == job["reuse"]["checkpoint"]["sha256"]
                and selected["train"]["state_sha256"]
                == job["reuse"]["canonical_tensor_sha256"],
                "Global reused selection differs from the fixed original",
            )
    return gate


def terminate_owned(process, command, identity):
    if process.poll() is not None:
        return
    live = proc_identity(process.pid)
    require(
        identity is not None
        and live is not None
        and live["argv"] == command
        and all(live[k] == identity[k] for k in ("pid", "start_ticks", "pgid", "cwd")),
        "Refuse to terminate unverified process",
    )
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def stage_child(c, host, job, stage, contract_path):
    allowed = {
        "synthetic",
        "train",
        "reuse",
        "validation",
        "capacity",
        "prepare_batches",
    } | {f"cost_{mode}_{rep}" for mode in ("train", "eval") for rep in (1, 2, 3)}
    require(stage in allowed, "Unsupported child stage; Test is not authorized")
    h = c["hosts"][host]
    root = Path(h["root"])
    guard(c, host)
    stem = root / "control" / (job["id"] + "." + stage)

    def path(suffix):
        return Path(str(stem) + suffix)

    command = [
        h["python"],
        "-u",
        "-m",
        "paper.scripts."
        + (
            "profile_shared_history_mlp_control"
            if stage == "capacity"
            or stage == "prepare_batches"
            or stage.startswith("cost_")
            else "run_shared_history_mlp_control"
        ),
        "--contract",
        str(contract_path),
        "--contract-sha",
        sha(contract_path),
        "--host",
        host,
    ]
    if stage == "capacity":
        command += ["--job", job["id"], "--mode", "qualification"]
    elif stage == "prepare_batches":
        command += ["--mode", "prepare-batches"]
    elif stage.startswith("cost_"):
        _, mode, rep = stage.split("_")
        command += ["--job", job["id"], "--mode", mode, "--rep", rep]
    else:
        command += ["--job", job["id"], "--stage", stage]
    write(
        path(".intent.json"),
        {
            "job": job,
            "stage": stage,
            "contract_sha256": canonical(c),
            "command": command,
        },
    )
    started = time.monotonic()
    identity = None
    timed_out = False
    error = incident = None
    last_guard = -1e9
    limit = c["limits"][
        "fit_wall_seconds" if stage == "train" else "evaluation_wall_seconds"
    ]
    with path(".stdout.log").open("xb") as out, path(".stderr.log").open("xb") as err:
        process = subprocess.Popen(
            command,
            cwd=h["source_root"],
            env={**os.environ, **h["environment"]},
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                live = proc_identity(process.pid)
                if identity is None and live is not None and live["argv"] == command:
                    identity = live
                    write(path(".launch.json"), identity)
                if time.monotonic() - last_guard >= 30 and incident is None:
                    last_guard = time.monotonic()
                    try:
                        guard(c, host, active=(process.pid,))
                    except RuntimeError as fault:
                        incident = str(fault)
                        write(
                            path(".resource_incident.json"),
                            {
                                "error": incident,
                                "policy": "drain_current_child_preserve_foreign_and_paused_block_next_stage",
                            },
                        )
                if (
                    time.monotonic() - started >= limit
                    or time.time() >= h["absolute_host_deadline_unix"]
                ):
                    timed_out = True
                    terminate_owned(process, command, identity)
                    break
                time.sleep(0.2)
        except BaseException as fault:
            error = type(fault).__name__ + ": " + str(fault)
        finally:
            if process.poll() is None:
                terminate_owned(process, command, identity)
            code = process.wait()
    value = {
        "job_id": job["id"],
        "stage": stage,
        "host": host,
        "command": command,
        "identity": identity,
        "actual_os_returncode": code,
        "timeout": timed_out,
        "supervisor_error": error,
        "resource_incident": incident,
        "contract_sha256": canonical(c),
        "elapsed_seconds": time.monotonic() - started,
        "stdout_sha256": sha(path(".stdout.log")),
        "stderr_sha256": sha(path(".stderr.log")),
    }
    write(path(".exit.json"), value)
    require(
        code == 0 and not timed_out and error is None and incident is None,
        "Child or resource guard failed; preserve attempt; no retry",
    )
    guard(c, host)
    return value


def dispatch(c, host, phase, contract_path):
    require(
        phase in ("qualification", "fit_validation", "cost"),
        "Unsupported phase; Test is not authorized",
    )
    verify_files(c, c["hosts"][host]["source_root"])
    q = c["implementation_qualification"]
    require(
        q.get("status") == "passed" and q.get("training_allowed") is True,
        "Independent implementation gate is pending",
    )
    root = Path(c["hosts"][host]["root"])
    if phase == "fit_validation":
        verify_native_qualification(c, host, root)
    if phase == "cost":
        verify_global_gate(c, root)
    guard(c, host)
    write(
        root / (phase + ".claim.json"),
        {
            "identity": proc_identity(os.getpid()),
            "contract_sha256": canonical(c),
            "started_at_unix": time.time(),
        },
    )
    jobs = [j for j in c["jobs"] if j["host"] == host]
    exits = []
    capacities = {}
    if phase == "qualification":
        unique = {(j["dataset"], j["arm"]): j for j in jobs}
        for job in unique.values():
            exits.append(stage_child(c, host, job, "synthetic", contract_path))
        for arm in ARMS:
            job = next(j for j in jobs if j["arm"] == arm)
            exits.append(stage_child(c, host, job, "capacity", contract_path))
            relative = "cost/qualification/" + arm + ".json"
            capacities[arm] = {"path": relative, "sha256": sha(root / relative)}
    elif phase == "fit_validation":
        for job in jobs:
            exits.append(
                stage_child(
                    c,
                    host,
                    job,
                    "reuse" if job.get("reuse") else "train",
                    contract_path,
                )
            )
        for job in jobs:
            exits.append(stage_child(c, host, job, "validation", contract_path))
    elif phase == "cost":
        exits.append(stage_child(c, host, jobs[0], "prepare_batches", contract_path))
        for job in jobs:
            for mode in ("train", "eval"):
                for rep in (1, 2, 3):
                    exits.append(
                        stage_child(c, host, job, f"cost_{mode}_{rep}", contract_path)
                    )
    else:
        raise RuntimeError("Unknown phase")
    write(
        root / (phase + ".complete.json"),
        {
            "status": "complete",
            "phase": phase,
            "host": host,
            "contract_sha256": canonical(c),
            "actual_child_exits": exits,
            "capacity_receipts": capacities,
            "gpu": guard(c, host),
            "finished_at_unix": time.time(),
            "supervisor_own_os_exit_directly_observed": False,
        },
    )


def supervise(c, host, phase, contract_path):
    """A separate parent observes the dispatcher's actual OS return code."""
    require(
        phase in ("qualification", "fit_validation", "cost"),
        "Unsupported phase; Test is not authorized",
    )
    h = c["hosts"][host]
    root = Path(h["root"])
    verify_files(c, h["source_root"])
    guard(c, host)
    command = [
        h["python"],
        "-u",
        "-m",
        "paper.scripts.manage_shared_history_mlp_control",
        "--mode",
        "dispatch",
        "--contract",
        str(contract_path),
        "--contract-sha",
        sha(contract_path),
        "--host",
        host,
        "--phase",
        phase,
    ]
    write(
        root / (phase + ".supervisor.intent.json"),
        {
            "command": command,
            "identity": proc_identity(os.getpid()),
            "contract_sha256": canonical(c),
        },
    )
    started = time.time()
    with (
        (root / (phase + ".stdout.log")).open("xb") as out,
        (root / (phase + ".stderr.log")).open("xb") as err,
    ):
        result = subprocess.run(
            command,
            cwd=h["source_root"],
            env={**os.environ, **h["environment"]},
            stdout=out,
            stderr=err,
        )
    value = {
        "actual_os_returncode": result.returncode,
        "command": command,
        "host": host,
        "phase": phase,
        "contract_sha256": canonical(c),
        "started_at_unix": started,
        "observed_exit_at_unix": time.time(),
        "stdout_sha256": sha(root / (phase + ".stdout.log")),
        "stderr_sha256": sha(root / (phase + ".stderr.log")),
        "parent_own_os_exit_directly_observed": False,
    }
    write(root / (phase + ".supervisor_exit.json"), value)
    return value


def pack(c, host, phase):
    """One immutable transfer archive after the phase has actually exited."""
    h = c["hosts"][host]
    root = Path(h["root"])
    verify_files(c, h["source_root"])
    gpu = guard(c, host)
    complete, exit_receipt = verify_phase_receipt(c, host, phase, root)
    for identity in [
        read(root / (phase + ".claim.json"))["identity"],
        *[e.get("identity") for e in complete["actual_child_exits"]],
    ]:
        require(identity is not None, "A scientific child identity was not captured")
        live = proc_identity(identity["pid"])
        require(
            live is None
            or any(live[k] != identity[k] for k in ("start_ticks", "argv", "cwd")),
            "Phase or scientific worker is still alive",
        )
    prefix = "collection_" + phase
    write(
        root / (prefix + ".intent.json"),
        {"contract_sha256": canonical(c), "host": host, "phase": phase},
    )
    inventory = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        legacy_evidence = relative.parts[:2] == ("source", "legacy")
        if (
            relative.parts[0] == "cache"
            or (relative.parts[0] == "source" and not legacy_evidence)
            or relative.name.startswith("collection_")
        ):
            continue
        require(not path.is_symlink(), "Collection refuses symlinks")
        if path.is_file():
            require(
                path.stat().st_size > 0 or path.suffix == ".log",
                "Empty scientific evidence file",
            )
            inventory[str(relative)] = {
                "sha256": sha(path),
                "bytes": path.stat().st_size,
            }
    require(bool(inventory), "Empty collection inventory")
    manifest = {
        "host": host,
        "phase": phase,
        "contract_sha256": canonical(c),
        "files": inventory,
        "source_files": c["source"]["files"],
        "input_files": c["input_files"],
        "legacy_files": c["legacy_files"],
        "all_source_input_legacy_actual_sha_verified": True,
        "gpu": gpu,
    }
    manifest_path = root / (prefix + ".manifest.json")
    write(manifest_path, manifest)
    archive_path = root / (prefix + ".tar.gz")
    with (
        archive_path.open("xb") as stream,
        tarfile.open(fileobj=stream, mode="w:gz") as archive,
    ):
        archive.add(manifest_path, arcname=manifest_path.name)
        for relative, item in inventory.items():
            require(
                sha(root / relative) == item["sha256"], "File changed while collecting"
            )
            archive.add(root / relative, arcname=relative)
    value = {
        "status": "complete",
        "host": host,
        "phase": phase,
        "archive": str(archive_path),
        "archive_sha256": sha(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "manifest_sha256": sha(manifest_path),
        "files": len(inventory),
        "contract_sha256": canonical(c),
    }
    write(root / (prefix + ".receipt.json"), value)
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("prepare", "seal", "dispatch", "supervise", "pack"),
        required=True,
    )
    p.add_argument("--tests-receipt", type=Path)
    p.add_argument("--contract", type=Path)
    p.add_argument("--contract-sha")
    p.add_argument("--host", choices=("5080", "5090"))
    p.add_argument("--phase", choices=("qualification", "fit_validation", "cost"))
    a = p.parse_args(argv)
    if a.mode == "prepare":
        result = prepare()
    elif a.mode == "seal":
        result = seal(a.tests_receipt)
    else:
        require(
            a.contract_sha and sha(a.contract) == a.contract_sha,
            "Externally pinned contract SHA is required",
        )
        c = read(a.contract)
        if a.mode == "supervise":
            result = supervise(c, a.host, a.phase, a.contract)
            print(json.dumps(result))
            return 0 if result["actual_os_returncode"] == 0 else 1
        if a.mode == "pack":
            print(json.dumps(pack(c, a.host, a.phase)))
            return 0
        try:
            dispatch(c, a.host, a.phase, a.contract)
        except BaseException as error:
            path = Path(c["hosts"][a.host]["root"]) / (a.phase + ".failure.json")
            if not path.exists():
                write(
                    path,
                    {
                        "error": type(error).__name__ + ": " + str(error),
                        "traceback": traceback.format_exc(),
                        "contract_sha256": canonical(c),
                        "automatic_retry": False,
                    },
                )
            raise
        result = {"status": "complete"}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
