#!/usr/bin/env python3
"""Continue the source-89 BOUNDED-QK campaign after the diagnosed NVRTC failure.

This is deliberately a one-incident controller.  It reuses only the five
completed jobs from the immutable failed campaign, adds the verified CUDA 13
library directory to child processes, and delegates the queue and gates to the
original committed campaign implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET


RECOVERY_ID = "bounded_qk_nvrtc_recovery_v1"
DIAGNOSIS = (
    "Installed CUDA 13 NVRTC builtins were not found by JIT dynamic loader; "
    "process-local library path correction only. Original failed run remains immutable."
)
TRAINING_REVISION = "89cd700c28dd169cd59bfccbd694ef8967bac823"
CONTRACT_SHA256 = "09464ee99907b7a3f0c3e35637c5bd36e5bb105111f1898d07843a6ca3b72997"
DEPLOYMENT_SHA256 = "dc0a0e861efab8cd884a16c31be0ea408b91bbc9ee9b7a90d84c84927f35e4ea"
FAILED_STATUS_SHA256 = "48dbb5d8b85d9594134cfccf24ad8e965bbc8b1a48c6e5398ee2454b7f433b2c"
FAILED_JOB = "normalized_e1_FULL_yellow_trip_hourly"
LIBRARY_DIRECTORY = Path(
    "/opt/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib"
)
LIBRARY_SHA256 = {
    "libnvrtc.so.13": "a49e67e8e74590f1e98de55c39c6287efd3f59e3c3797464d7bbe0fe01349b11",
    "libnvrtc-builtins.so.13.0": "91dcd944d01da9c0f08fff5d779db136a47f6d62bdb63bae900d2f481e92c3a2",
}
DATASETS = ("yellow_trip_hourly", "intermittent_frozen_5000", "insta_market_basket")
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "POLARS_MAX_THREADS": "4",
    "PYTHONNOUSERSITE": "1",
}
REQUIRED_CUDA_TESTS = {
    "test_zero_initialization_preserves_b_full_outputs_common_and_kernel_gradients[64-cuda]",
    "test_operating_extremes_finite_and_optimizer_roundtrip_next_step[cuda]",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value and value not in {"TBD", "None"}, f"Missing {label}")
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, f"{label} must be an absolute path")
    return path


def _sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"Invalid {label}")
    return value


def _revision(value: Any, label: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
            f"Invalid {label}")
    return value


def expected_reused_files() -> set[str]:
    common = {
        "cuda_qualification/results.xml", "cuda_qualification/audit.json",
        "cost_profile/profile.json", "cost_profile/audit.json",
    }
    job_files = {
        "audit.json", "execution_receipt.json", "history_seed_metrics.csv",
        "history_summary.csv", "launch_contract.json", "quantity_seed_metrics.csv",
        "quantity_summary.csv", "run_summaries.csv", "runner.log",
    }
    run_files = {
        "best_val_qty_rmse_model.pt", "history.json", "last_epoch_state.pt",
        "summary.json", "train.log",
    }
    for dataset in DATASETS:
        prefix = f"jobs/e1_{dataset}"
        common.update(f"{prefix}/{name}" for name in job_files)
        run = f"{prefix}/runs/titantpp_hard_memory_bounded_qk/count_only_log_regression/seed_42"
        common.update(f"{run}/{name}" for name in run_files)
    return common


def validate_recovery_manifest(manifest: Mapping[str, Any]) -> dict[str, Path]:
    required_top = {
        "schema_version", "recovery_id", "status", "diagnosis", "training",
        "controller", "original", "new_output_root", "nvrtc", "reused_files",
        "failed_duration_cache", "held_out_test_evaluated", "additional_seeds_authorized",
    }
    require(set(manifest) == required_top, "Recovery manifest fields drift")
    require(manifest["schema_version"] == 1 and manifest["recovery_id"] == RECOVERY_ID,
            "Recovery identity drift")
    require(manifest["status"] == "frozen_before_recovery_outputs", "Recovery status drift")
    require(manifest["held_out_test_evaluated"] is False
            and manifest["additional_seeds_authorized"] is False, "Recovery scope drift")
    require(manifest["diagnosis"] == DIAGNOSIS,
            "This controller only accepts the diagnosed NVRTC incident")

    training = manifest["training"]
    require(isinstance(training, dict), "Missing training identity")
    require(training.get("source_revision") == TRAINING_REVISION, "Training revision drift")
    _absolute(training.get("source_root"), "training source root")
    _absolute(training.get("python_executable"), "training Python")
    require(isinstance(training.get("runtime"), dict) and isinstance(training.get("dependencies"), dict),
            "Runtime identity is incomplete")
    require(isinstance(training.get("gpu_uuid"), str) and training["gpu_uuid"], "GPU UUID missing")

    controller = manifest["controller"]
    require(isinstance(controller, dict), "Missing controller identity")
    _absolute(controller.get("source_root"), "controller source root")
    _absolute(controller.get("path"), "controller path")
    _revision(controller.get("source_revision"), "controller revision")
    _sha(controller.get("file_sha256"), "controller file digest")

    original = manifest["original"]
    require(isinstance(original, dict), "Missing original execution identity")
    require(original.get("deployment_manifest_sha256") == DEPLOYMENT_SHA256,
            "Original deployment digest drift")
    require(original.get("failed_status_sha256") == FAILED_STATUS_SHA256,
            "Original failure digest drift")
    require(original.get("failed_job") == FAILED_JOB, "Original failed job drift")
    for key in ("output_root", "deployment_manifest_path", "failed_status_path"):
        _absolute(original.get(key), f"original {key}")

    new_root = _absolute(manifest["new_output_root"], "new output root")
    require(new_root != _absolute(original["output_root"], "original output root"),
            "Recovery output must be separate from the failed campaign")
    require(not new_root.is_relative_to(_absolute(original["output_root"], "original output root")),
            "Recovery output cannot be inside the immutable failed campaign")

    nvrtc = manifest["nvrtc"]
    require(isinstance(nvrtc, dict) and nvrtc.get("library_directory") == str(LIBRARY_DIRECTORY),
            "NVRTC library directory drift")
    require(nvrtc.get("ld_library_path") == str(LIBRARY_DIRECTORY), "NVRTC loader path drift")
    files = nvrtc.get("files")
    require(isinstance(files, list) and len(files) == len(LIBRARY_SHA256), "NVRTC file set drift")
    indexed = {Path(row.get("path", "")).name: row for row in files if isinstance(row, dict)}
    require(set(indexed) == set(LIBRARY_SHA256), "NVRTC file identities drift")
    for name, expected in LIBRARY_SHA256.items():
        require(indexed[name] == {"path": str(LIBRARY_DIRECTORY / name), "sha256": expected},
                f"NVRTC binding drift: {name}")

    reused = manifest["reused_files"]
    require(isinstance(reused, dict) and set(reused) == expected_reused_files(),
            "Completed artifact population drift")
    for relative, digest in reused.items():
        logical = Path(relative)
        require(not logical.is_absolute() and ".." not in logical.parts, "Unsafe reused artifact path")
        _sha(digest, f"reused artifact digest {relative}")
    require(manifest["failed_duration_cache"] == {"reuse": False},
            "The unqualified failed duration cache must not be reused")
    return {
        "training_root": _absolute(training["source_root"], "training source root"),
        "controller_root": _absolute(controller["source_root"], "controller source root"),
        "old_root": _absolute(original["output_root"], "original output root"),
        "new_root": new_root,
        "deployment": _absolute(original["deployment_manifest_path"], "deployment manifest"),
        "failed_status": _absolute(original["failed_status_path"], "failed status"),
    }


def verify_controller_source(manifest: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    spec = manifest["controller"]
    live = Path(__file__).resolve()
    require(live == Path(spec["path"]).resolve(), "Controller path drift")
    require(sha256(live) == spec["file_sha256"], "Controller file digest drift")
    root = paths["controller_root"].resolve()
    require(live.is_relative_to(root), "Controller is outside its committed checkout")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(revision == spec["source_revision"], "Controller checkout revision drift")
    relative = live.relative_to(root).as_posix()
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=root)
    require(hashlib.sha256(committed).hexdigest() == spec["file_sha256"],
            "Controller file is not the committed revision")


def load_training_campaign(training_root: Path) -> Any:
    campaign_path = training_root / "paper/scripts/run_hard_lmm_bounded_qk_campaign.py"
    require(campaign_path.is_file(), "Original campaign runner is missing")
    for name, module in tuple(sys.modules.items()):
        if name == "paper" or name.startswith(("paper.", "models.", "simple_lab_test.")):
            location = getattr(module, "__file__", None)
            if location is not None:
                require(Path(location).resolve().is_relative_to(training_root.resolve()),
                        f"Local module was imported outside source89: {name}")
    sys.path.insert(0, str(training_root))
    spec = importlib.util.spec_from_file_location("_bounded_qk_source89_campaign", campaign_path)
    require(spec is not None and spec.loader is not None, "Cannot load original campaign runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == campaign_path.resolve(), "Original campaign import drift")
    dependency = importlib.import_module("paper.scripts.run_hard_lmm_backbone_candidate_campaign")
    require(Path(dependency.__file__).resolve().is_relative_to(training_root.resolve()),
            "Original campaign dependency import drift")
    return module


def verify_pinned_files(root: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    observed = {}
    for relative, expected in manifest["reused_files"].items():
        path = root / relative
        require(path.is_file(), f"Missing completed artifact: {relative}")
        actual = sha256(path)
        require(actual == expected, f"Completed artifact drift: {relative}")
        observed[relative] = actual
    return observed


def verify_original_failure(manifest: Mapping[str, Any], paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    original = manifest["original"]
    require(sha256(paths["deployment"]) == DEPLOYMENT_SHA256, "Deployment manifest changed")
    require(sha256(paths["failed_status"]) == FAILED_STATUS_SHA256, "Failed status changed")
    deployment = read_json(paths["deployment"])
    failure = read_json(paths["failed_status"])
    training = manifest["training"]
    require(deployment.get("source_root") == str(paths["training_root"])
            and deployment.get("source_revision") == TRAINING_REVISION, "Original source identity drift")
    require(deployment.get("output_root") == str(paths["old_root"]), "Original output root drift")
    for key in ("python_executable", "runtime", "dependencies", "gpu_uuid"):
        require(deployment.get(key) == training.get(key), f"Original {key} drift")
    require(deployment.get("execution_contract_sha256") == CONTRACT_SHA256,
            "Original campaign contract drift")
    require(deployment.get("environment", {}).get("OMP_NUM_THREADS") == "4"
            and all(deployment.get("environment", {}).get(k) == v for k, v in THREAD_ENVIRONMENT.items()),
            "Original four-thread environment drift")
    require(failure.get("status") == "failed_execution" and failure.get("current_job") == FAILED_JOB,
            "Unexpected original failure")
    require(failure.get("source_revision") == TRAINING_REVISION
            and failure.get("deployment_manifest_sha256") == DEPLOYMENT_SHA256,
            "Original failure lineage drift")
    require("run_backbone_normalized_duration.py" in str(failure.get("error"))
            and FAILED_JOB in failure.get("jobs", {}), "Original failed job evidence drift")
    require(failure["jobs"][FAILED_JOB].get("status") == "failed", "Original failed job state drift")
    completed = {"cuda_qualification", "cost_profile", *(f"e1_{dataset}" for dataset in DATASETS)}
    require(all(failure.get("jobs", {}).get(job, {}).get("status") == "passed" for job in completed),
            "Original completed-job state drift")
    for job in completed | {FAILED_JOB}:
        require(failure["jobs"][job].get("output") == str(paths["old_root"] / (
            job if job in {"cuda_qualification", "cost_profile"} else f"jobs/{job}"
        )), f"Original job output drift: {job}")
    return deployment, failure


def verify_libraries(manifest: Mapping[str, Any]) -> None:
    indexed = {Path(row["path"]).name: row for row in manifest["nvrtc"]["files"]}
    for name, expected in LIBRARY_SHA256.items():
        path = Path(indexed[name]["path"])
        require(path.is_file() and sha256(path) == expected, f"NVRTC library drift: {name}")


def child_environment(manifest: Mapping[str, Any], new_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    environment["MPLCONFIGDIR"] = str(new_root / "matplotlib")
    environment["TORCH_KERNEL_CACHE_PATH"] = str(new_root / "torch_kernel_cache")
    environment["LD_LIBRARY_PATH"] = manifest["nvrtc"]["ld_library_path"]
    return environment


def audit_reused_execution(campaign: Any, contract: Mapping[str, Any], manifest: Mapping[str, Any],
                           paths: Mapping[str, Path]) -> dict[str, Any]:
    old = paths["old_root"]
    # CUDA qualification is accepted only after reconstructing the XML result.
    junit = old / "cuda_qualification/results.xml"
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    require(len(cases) == 67 and all(case.find("failure") is None and case.find("error") is None
                                     and case.find("skipped") is None for case in cases),
            "Reused CUDA qualification is incomplete")
    names = {case.attrib.get("name") for case in cases}
    require(REQUIRED_CUDA_TESTS <= names, "Required actual CUDA qualification is missing")
    cuda_audit = read_json(old / "cuda_qualification/audit.json")
    require(cuda_audit == {"status": "passed", "test_count": 67,
                           "required_actual_cuda_tests": sorted(REQUIRED_CUDA_TESTS),
                           "junit_sha256": sha256(junit)}, "Reused CUDA audit drift")

    profiler = importlib.import_module("paper.scripts.profile_hard_lmm_bounded_qk")
    require(Path(profiler.__file__).resolve().is_relative_to(paths["training_root"].resolve()),
            "Cost auditor was not imported from source89")
    profile_path = old / "cost_profile/profile.json"
    profile = read_json(profile_path)
    recalculated = profiler.summarize_cost(profile.get("rows", []))
    require(profile.get("status") == "passed" and profile.get("cost_gate") == recalculated
            and len(profile.get("rows", [])) == 27, "Reused cost proof failed reconstruction")
    require(profile.get("source_file_sha256") == sha256(Path(profiler.__file__)),
            "Reused cost profiler source drift")
    require(Path(profiler.original.__file__).resolve().is_relative_to(paths["training_root"].resolve())
            and profile.get("reused_profile_file_sha256") == sha256(Path(profiler.original.__file__)),
            "Reused base profiler source drift")
    cost_audit = read_json(old / "cost_profile/audit.json")
    require(cost_audit == {"status": "passed", "cost_gate": recalculated,
                           "profile_sha256": sha256(profile_path)}, "Reused cost audit drift")

    auditor = importlib.import_module("paper.scripts.audit_hard_lmm_bounded_qk")
    require(Path(auditor.__file__).resolve().is_relative_to(paths["training_root"].resolve()),
            "Joint auditor was not imported from source89")
    joint = {}
    for dataset in DATASETS:
        output = old / "jobs" / f"e1_{dataset}"
        recomputed = auditor.audit_bounded_qk_job(
            output, candidate=contract["candidate"], dataset=dataset,
            source_revision=TRAINING_REVISION, expected_epochs=1,
            binding=contract["data_bindings"][dataset],
        )
        require(recomputed == read_json(output / "audit.json"), f"Reused e1 audit drift: {dataset}")
        joint[dataset] = recomputed
    return {"cuda_qualification": cuda_audit, "cost_profile": cost_audit, "joint_e1": joint}


class Continuation:
    def __init__(self, *, manifest_path: Path, manifest_sha256: str, manifest: dict[str, Any],
                 paths: dict[str, Path], campaign: Any, deployment: dict[str, Any],
                 contract: dict[str, Any], reused: dict[str, Any]) -> None:
        self.manifest_path = manifest_path
        self.manifest_sha256 = manifest_sha256
        self.manifest = manifest
        self.paths = paths
        self.campaign = campaign
        self.deployment = deployment
        self.contract = contract
        self.reused = reused
        self.environment = child_environment(manifest, paths["new_root"])
        self.status_path = paths["new_root"] / "campaign_status.json"
        self.status: dict[str, Any] = {
            "schema_version": 1, "recovery_id": RECOVERY_ID,
            "training_source_revision": TRAINING_REVISION,
            "controller_source_revision": manifest["controller"]["source_revision"],
            "recovery_manifest_sha256": manifest_sha256,
            "original_deployment_manifest_sha256": DEPLOYMENT_SHA256,
            "original_failed_status_sha256": FAILED_STATUS_SHA256,
            "status": "prepared", "started_at": utc_now(), "current_job": None,
            "held_out_test_evaluated": False, "additional_seeds_launched": False,
            "jobs": {}, "decisions": {},
            "reused_from": {"output_root": str(paths["old_root"]), "jobs": {}},
            "environment_overrides": {
                **THREAD_ENVIRONMENT, "MPLCONFIGDIR": self.environment["MPLCONFIGDIR"],
                "TORCH_KERNEL_CACHE_PATH": self.environment["TORCH_KERNEL_CACHE_PATH"],
                "LD_LIBRARY_PATH": self.environment["LD_LIBRARY_PATH"],
            },
        }

    def persist(self) -> None:
        self.status["updated_at"] = utc_now()
        save_json(self.status_path, self.status)

    def verify_boundary(self) -> dict[str, Any]:
        require(sha256(self.manifest_path) == self.manifest_sha256,
                "Recovery manifest changed after outputs")
        validate_recovery_manifest(read_json(self.manifest_path))
        verify_controller_source(self.manifest, self.paths)
        verify_libraries(self.manifest)
        verify_original_failure(self.manifest, self.paths)
        verify_pinned_files(self.paths["old_root"], self.manifest)
        self.campaign.verify_source(self.deployment)
        preflight = self.campaign.gpu_preflight("RTX 5090")
        runtime_command = [
            self.manifest["training"]["python_executable"], "-s", "-c",
            "import json,torch,sys,importlib.metadata; "
            "from paper.scripts.run_backbone_normalized_duration import runtime_identity; "
            "print(json.dumps({'runtime':runtime_identity(torch.device('cuda')),'executable':sys.executable,"
            "'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','polars','pyarrow','pytest']}}))",
        ]
        observed = json.loads(subprocess.check_output(
            runtime_command, cwd=self.paths["training_root"], env=self.environment, text=True,
        ))
        training = self.manifest["training"]
        require(observed == {"runtime": training["runtime"], "executable": training["python_executable"],
                             "dependencies": training["dependencies"]}, "Runtime identity drift")
        gpu_uuid = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True,
        ).strip()
        require(gpu_uuid == training["gpu_uuid"], "Physical GPU identity drift")
        return preflight

    def run_command(self, job_id: str, command: list[str], output: Path) -> None:
        preflight = self.verify_boundary()
        output.mkdir(parents=True, exist_ok=False)
        receipt = {
            "status": "running", "started_at": utc_now(), "output": str(output),
            "command": command, "gpu_preflight": preflight,
            "training_source_revision": TRAINING_REVISION,
            "controller_source_revision": self.manifest["controller"]["source_revision"],
            "recovery_manifest_sha256": self.manifest_sha256,
            "subprocess_ld_library_path": self.environment["LD_LIBRARY_PATH"],
        }
        self.status.update(status=f"running_{job_id}", current_job=job_id)
        self.status["jobs"][job_id] = receipt
        save_json(output / "execution_receipt.json", receipt)
        self.persist()
        with (output / "runner.log").open("x") as log:
            subprocess.run(command, cwd=self.paths["training_root"], env=self.environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        verify_pinned_files(self.paths["old_root"], self.manifest)

    def complete(self, job_id: str, audit: Mapping[str, Any]) -> None:
        require(audit.get("status") == "passed", f"Job audit failed: {job_id}")
        self.status["jobs"][job_id].update(dict(audit), status="passed", completed_at=utc_now())
        output = Path(self.status["jobs"][job_id]["output"])
        save_json(output / "audit.json", dict(audit))
        self.status["current_job"] = None
        self.persist()

    def qualify_nvrtc(self) -> None:
        job_id = "nvrtc_normalized_likelihood_qualification"
        output = self.paths["new_root"] / "jobs" / job_id
        program = (
            "import json,torch; "
            "x=torch.tensor([-9.,-1.,0.,2.],device='cuda',dtype=torch.float64,requires_grad=True); "
            "loss=-torch.special.log_ndtr(x).mean(); loss.backward(); "
            "assert bool(torch.isfinite(loss)) and x.grad is not None and bool(torch.isfinite(x.grad).all()); "
            "print(json.dumps({'status':'passed','dtype':str(x.dtype),'device':str(x.device),"
            "'loss':float(loss),'gradient_norm':float(x.grad.norm())}))"
        )
        command = [self.manifest["training"]["python_executable"], "-s", "-c", program]
        self.run_command(job_id, command, output)
        lines = (output / "runner.log").read_text(encoding="utf-8").splitlines()
        require(lines, "NVRTC qualification emitted no evidence")
        observed = json.loads(lines[-1])
        require(observed.get("status") == "passed" and observed.get("dtype") == "torch.float64"
                and str(observed.get("device", "")).startswith("cuda")
                and all(float(observed[k]) > 0 for k in ("loss", "gradient_norm")),
                "Normalized likelihood CUDA qualification failed")
        self.complete(job_id, {"status": "passed", "observed": observed,
                               "runner_log_sha256": sha256(output / "runner.log")})

    def joint(self, dataset: str, phase_name: str) -> dict[str, Any]:
        if phase_name == "e1":
            audit = self.reused["joint_e1"][dataset]
            self.status["reused_from"]["jobs"][f"e1_{dataset}"] = {
                "status": "reused_after_read_only_reaudit",
                "output": str(self.paths["old_root"] / "jobs" / f"e1_{dataset}"),
                "audit": audit,
            }
            self.persist()
            return audit
        require(phase_name == "seed42_screening", "Unexpected joint phase")
        auditor = importlib.import_module("paper.scripts.audit_hard_lmm_bounded_qk")
        phase = self.contract["phases"][phase_name]
        job_id = f"{phase_name}_{dataset}"
        output = self.paths["new_root"] / "jobs" / job_id
        command = self.campaign.job_command(
            python=self.manifest["training"]["python_executable"], candidate=self.contract["candidate"],
            dataset=dataset, output=output, source_revision=TRAINING_REVISION,
            phase=phase, host_role="5090",
        )
        self.run_command(job_id, command, output)
        audit = auditor.audit_bounded_qk_job(
            output, candidate=self.contract["candidate"], dataset=dataset,
            source_revision=TRAINING_REVISION, expected_epochs=phase["epochs"],
            binding=self.contract["data_bindings"][dataset],
        )
        self.complete(job_id, audit)
        return audit

    def duration(self, dataset: str, role: str, full_fit: bool,
                 joint_audit: Mapping[str, Any] | None) -> dict[str, Any]:
        source = (self.contract["datasets"][dataset]["FULL"]["source"] if role == "FULL"
                  else self.campaign.candidate_source(joint_audit))
        job_manifest = self.campaign.duration_manifest(
            dataset=dataset, role=role, source=source, deployment=self.deployment,
        )
        manifest_dir = self.paths["new_root"] / "duration_manifests"
        manifest_dir.mkdir(exist_ok=True)
        manifest_path = manifest_dir / f"{role}_{dataset}.json"
        if manifest_path.exists():
            require(read_json(manifest_path) == job_manifest, "Duration manifest drift")
        else:
            save_json(manifest_path, job_manifest)
        stage = "full" if full_fit else "e1"
        job_id = f"normalized_{stage}_{role}_{dataset}"
        output = self.paths["new_root"] / "jobs" / job_id
        command = [
            self.manifest["training"]["python_executable"], "-s", str(self.campaign.DURATION_RUNNER),
            "--job-manifest", str(manifest_path), "--expected-job-manifest-sha256", sha256(manifest_path),
            "--output-dir", str(output), "--device", "cuda", "--source-revision", TRAINING_REVISION,
        ]
        if full_fit:
            cache = self.paths["new_root"] / "jobs" / f"normalized_e1_{role}_{dataset}" / "cache"
            require(cache.is_dir(), "Full fit requires its qualified continuation e1 cache")
            command += ["--feature-cache-dir", str(cache)]
        else:
            command += ["--allow-partial-contract", "--max-epochs", "1"]
        self.run_command(job_id, command, output)
        audit = self.campaign.audit_duration(
            output, dataset=dataset, role=role, full_fit=full_fit, source=source,
        )
        self.complete(job_id, audit)
        return audit

    def record_gate(self, dataset: str, decision: dict[str, Any]) -> None:
        self.status["decisions"][dataset] = decision
        self.persist()


def run_recovery(manifest_path: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    require(sha256(manifest_path) == expected_manifest_sha256, "Recovery manifest digest drift")
    manifest = read_json(manifest_path)
    paths = validate_recovery_manifest(manifest)
    require(not paths["new_root"].exists(), "Recovery output exists; automatic restart is forbidden")
    verify_controller_source(manifest, paths)
    verify_libraries(manifest)
    deployment, _ = verify_original_failure(manifest, paths)
    verify_pinned_files(paths["old_root"], manifest)
    campaign = load_training_campaign(paths["training_root"])
    require(sha256(campaign.CONTRACT_PATH) == CONTRACT_SHA256, "Frozen screening contract drift")
    contract = campaign.read_json(campaign.CONTRACT_PATH)
    campaign.validate_contract(contract)
    campaign.verify_source(deployment)
    input_evidence = campaign.verify_inputs(contract)
    reused = audit_reused_execution(campaign, contract, manifest, paths)

    paths["new_root"].mkdir(parents=True, exist_ok=False)
    (paths["new_root"] / "matplotlib").mkdir()
    (paths["new_root"] / "torch_kernel_cache").mkdir()
    continuation = Continuation(
        manifest_path=manifest_path, manifest_sha256=expected_manifest_sha256,
        manifest=manifest, paths=paths, campaign=campaign, deployment=deployment,
        contract=contract, reused=reused,
    )
    continuation.status["input_verification"] = input_evidence
    continuation.status["reused_from"].update({
        "cuda_qualification": reused["cuda_qualification"],
        "cost_profile": reused["cost_profile"],
        "verified_file_count": len(manifest["reused_files"]),
    })
    continuation.persist()
    try:
        continuation.qualify_nvrtc()
        decision = campaign.execute_phases(
            contract, joint=continuation.joint, duration=continuation.duration,
            record_gate=continuation.record_gate,
        )
        continuation.status.update(decision, current_job=None, completed_at=utc_now())
        final = {
            **decision, "training_source_revision": TRAINING_REVISION,
            "controller_source_revision": manifest["controller"]["source_revision"],
            "contract_sha256": CONTRACT_SHA256,
            "recovery_manifest_sha256": expected_manifest_sha256,
            "original_failed_status_sha256": FAILED_STATUS_SHA256,
            "held_out_test_evaluated": False, "additional_seeds_launched": False,
            "recorded_at": utc_now(),
        }
        save_json(paths["new_root"] / "final_decision.json", final)
        verify_pinned_files(paths["old_root"], manifest)
        continuation.persist()
        return final
    except BaseException as error:
        current = continuation.status.get("current_job")
        if current in continuation.status["jobs"]:
            continuation.status["jobs"][current].update(
                status="failed", error=repr(error), completed_at=utc_now(),
            )
        continuation.status.update(status="failed_execution", error=repr(error), completed_at=utc_now())
        continuation.persist()
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--expected-recovery-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    result = run_recovery(args.recovery_manifest.resolve(), args.expected_recovery_manifest_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
