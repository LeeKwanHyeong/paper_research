#!/usr/bin/env python3
"""Freeze local source/contract or assemble a permit; never launch remote work."""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts import task_routed_episode_parallel_common as common


def build_contract():
    """Only configuration and prior synthetic receipts are read."""
    for receipt_path in (common.CPU_RECEIPT, common.IMPLEMENTATION_RECEIPT):
        receipt = json.loads((ROOT / receipt_path).read_text())
        for name, digest in receipt["source_sha256"].items():
            common.require(common.sha_file(ROOT / name) == digest,
                           "Previously verified implementation source changed: " + name)
    datasets = deepcopy(json.loads((ROOT / common.INHERITED).read_text())["datasets"])
    for data in datasets:
        data["epochs"] = 120
    reference = json.loads((ROOT / common.RUNTIME_REFERENCE).read_text())["hosts"]
    remote = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/task_routed_episode_seed42_parallel_20260916_v1"
    hosts = {}
    for alias, assignment in common.ASSIGNMENTS.items():
        hosts[alias] = {key: deepcopy(reference[alias][key]) for key in
                       ("python", "gpu_uuid", "runtime_expected", "tmux_binary")}
        hosts[alias].update(root=remote, source_root=remote + "/source", output_dir=remote + "/run",
            assigned_datasets=list(assignment), environment=deepcopy(common.ENVIRONMENT),
            tmux=f"task_routed_episode_seed42_parallel_20260916_{alias}")
    files = common.source_manifest()
    return {
        "schema": common.SCHEMA, "status": "frozen_ready_for_separate_gpu_execution_approval",
        "approval_required": True,
        "authorization_note": "Current instruction authorizes local runner/contract preparation. No GPU approval, qualification or start permit has been issued for this new contract.",
        "purpose_ko": "같은 관측 메모리와 두 검색을 사용하면서, 시간·수량에 공유 평균을 전달하는 방식과 각각의 검색 결과를 배정하는 Backbone 차이를 검증한다.",
        "arms": list(common.ARMS), "seed": 42, "epochs": 120,
        "total_optimizer_steps": 6816240, "endpoint_replays": 18,
        "datasets": datasets, "hosts": hosts,
        "limits": deepcopy(common.LIMITS), "policy": deepcopy(common.POLICY),
        "acceptance": deepcopy(common.ACCEPTANCE), "cost_gates": deepcopy(common.COST_GATES),
        "design_sha256": common.sha_file(ROOT / common.DESIGN),
        "inherited_data_contract_sha256": common.sha_file(ROOT / common.INHERITED),
        "cpu_receipt_sha256": common.sha_file(ROOT / common.CPU_RECEIPT),
        "implementation_receipt_sha256": common.sha_file(ROOT / common.IMPLEMENTATION_RECEIPT),
        "source": {"files": files, "files_sha256": common.sha_json(files),
            "base_git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
            "worktree": "preexisting_dirty_edits_preserved; listed_files_are_authoritative"},
        "comparison": {
            "order": "5080:Intermittent then Taxi; 5090:Instacart; each B then shared-mixture control then task-specific candidate",
            "selector": "earliest strict finite validation raw quantity RMSE minimum",
            "metrics": "RMSE/MAE/body/tail/legacy clamped time loss from one selected checkpoint; final120 and last30 mean/sampleSD separately",
            "mechanism_gate": "candidate passes all seven guards against BOTH fresh B and shared-mixture control on every dataset",
            "scope": "exploratory seed42 reused validation; no held-out or benchmark superiority claim",
            "initialization": "fresh shared B state and RNG; same new trainable state; only routing_code buffer differs; time/quantity queries are distinct within each model; zero output projections preserve initial B outputs/objective/preclip common gradients, not necessarily the first globally clipped optimizer step",
            "exposure": "exact same per-epoch train/validation batch content and order hashes across three arms",
            "server_comparison": "all three arms of each dataset stay on one server; native environments differ between datasets",
            "mechanism_claim": "comparison tests task assignment versus shared mixture; it also changes gradient coupling, not the shared memory contents, and is not evidence of independent task training or equal function classes",
            "common_time_likelihood": "legacy clamped time loss is a provisional engineering guard; normalized common time likelihood and benchmark/three-seed confirmation are later separate work",
            "body_tail_definition": "body: q<=train boundary[2]; tail: q>train boundary[3]; intermediate and all frozen quantity/history bins also reported, never selected on validation targets",
            "practical_significance": "strict RMSE improvement is an exploratory gate, not a significance test; tiny improvements must not be promoted directly to paper claims",
        },
        "cost": {
            "cpu_receipt": common.CPU_RECEIPT, "cuda_measured": False,
            "expected_training_hours": None,
            "qualification": "each host: batch128 lengths64/256 with matching max_seq_len, 7 updates per arm/length (2 warmup+5 measured), plus 3 restore updates per new arm; 48 synthetic updates within900 seconds",
            "budget_origin": "common48-hour wall/monotonic ceiling begins at earliest NEW native qualification; absolute deadline exists only after qualification start",
            "prior_execution_budget_reused": False,
            "scope": "CPU cost is local evidence only; CUDA step+allocated-memory+runtime checks must pass on both hosts",
            "failure_action": "stop only owned run and report; no automatic retry/resume/runtime mutation",
        },
        "launch": {
            "qualification_entrypoint": "paper/scripts/qualify_task_routed_episode_parallel.py",
            "training_entrypoint": "paper/scripts/run_task_routed_episode_parallel.py",
            "approval_file": "separate task_routed_episode_parallel_approval_v1 bound to canonical contract SHA",
            "native_qualification": "both host receipts must pass, with raw cost measurements verified, before either training partition starts",
            "root_must_be_new": True, "gpu_must_be_idle": True,
            "current_gpu_availability": "not_checked_in_local_preparation; recheck immediately before qualification and training",
            "runtime_reference": common.RUNTIME_REFERENCE,
            "source_sync": "frozen bundle to new isolated roots only; include empty sample_data sentinel; no delete or public Runtime mutation",
            "scheduler": "remains paused; no scheduler modification is authorized by this contract",
            "supervision": "owned process group, parent pipe authority, exclusive run directory, deadlines, storage and GPU exclusivity checks; preserves other jobs",
        },
    }


def approval_template(contract):
    return {"schema": "task_routed_episode_parallel_approval_v1", "approved": False,
        "contract_sha256": common.sha_json(contract), "hosts": ["5080", "5090"],
        "scope": "dual_native_synthetic_cuda_qualification_and_fresh_nine_arm_validation",
        "user_instruction": ""}


def write_bundle(path, contract):
    common.require(not Path(path).exists(), "Fresh bundle path required")
    common.verify_source(contract)
    with Path(path).open("xb") as target, tarfile.open(fileobj=target, mode="w:gz") as archive:
        sentinel = tarfile.TarInfo("source/sample_data")
        sentinel.type, sentinel.mode = tarfile.DIRTYPE, 0o755
        archive.addfile(sentinel)
        for name, digest in sorted(contract["source"]["files"].items()):
            common.require(common.sha_file(ROOT / name) == digest, "Source changed during packaging")
            encoded = (ROOT / name).read_bytes()
            entry = tarfile.TarInfo("source/" + name)
            entry.size, entry.mode = len(encoded), 0o644
            archive.addfile(entry, io.BytesIO(encoded))
        encoded = (json.dumps(contract, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        entry = tarfile.TarInfo("frozen_execution/execution_contract.json")
        entry.size, entry.mode = len(encoded), 0o644
        archive.addfile(entry, io.BytesIO(encoded))
    common.verify_source(contract)


def freeze(contract_path, output):
    common.require(not contract_path.exists() and not output.exists(), "Fresh contract/output paths required")
    contract = build_contract()
    output.mkdir(parents=True, exist_ok=False)
    common.write_json(contract_path, contract, exclusive=True)
    for host in common.ASSIGNMENTS:
        common.read_contract(contract_path, host)
    common.write_json(output / "approval_template_NOT_APPROVED.json", approval_template(contract), exclusive=True)
    write_bundle(output / "source_bundle.tar.gz", contract)
    commands = {}
    for alias, host in contract["hosts"].items():
        remote = Path(host["root"])
        commands[alias] = {
            "cwd": host["source_root"], "environment": host["environment"],
            "tmux": host["tmux"], "tmux_binary": host["tmux_binary"],
            "qualification_argv": [host["python"], contract["launch"]["qualification_entrypoint"],
                "--contract", str(remote / "frozen_execution/execution_contract.json"),
                "--approval", str(remote / "approval.json"), "--host", alias],
            "training_argv": [host["python"], contract["launch"]["training_entrypoint"], "execute",
                "--contract", str(remote / "frozen_execution/execution_contract.json"),
                "--permit", str(remote / "start_permit.json"), "--host", alias],
        }
    common.write_json(output / "prepared_commands_NOT_EXECUTED.json", commands, exclusive=True)
    receipt = {"status": "prepared_not_launched", "contract": str(contract_path),
        "contract_sha256": common.sha_json(contract), "contract_file_sha256": common.sha_file(contract_path),
        "source_files_sha256": contract["source"]["files_sha256"],
        "bundle_sha256": common.sha_file(output / "source_bundle.tar.gz"),
        "source_file_count": len(contract["source"]["files"]),
        "approved": False, "native_cuda_executed": False, "training_launched": False,
        "source_uploaded": False, "real_data_loaded": False, "held_out_accessed": False,
        "scheduler_changed": False, "full_training_hours_estimate": None}
    common.write_json(output / "preparation_receipt.json", receipt, exclusive=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--contract", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    permit_parser = subparsers.add_parser("permit")
    permit_parser.add_argument("--contract", type=Path, required=True)
    permit_parser.add_argument("--approval", type=Path, required=True)
    permit_parser.add_argument("--qualification-5080", type=Path, required=True)
    permit_parser.add_argument("--qualification-5090", type=Path, required=True)
    permit_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(args.contract, args.output), ensure_ascii=False))
        return
    common.require(not args.output.exists(), "Fresh permit output required")
    contract = common.read_contract(args.contract, "5090")
    approval = json.loads(args.approval.read_text())
    common.verify_approval(contract, approval)
    bindings = {}
    for alias in common.ASSIGNMENTS:
        path = getattr(args, "qualification_" + alias)
        receipt = json.loads(path.read_text())
        bindings[alias] = {"path": str(Path(contract["hosts"][alias]["root"]) / "qualification/receipt.json"),
            "sha256": common.sha_file(path), "receipt_sha256": common.sha_json(receipt), "receipt": receipt}
    common.write_json(args.output, common.create_permit(contract, approval, bindings), exclusive=True)
    print(json.dumps({"status": "permit_created_no_training_launched", "output": str(args.output)}))


if __name__ == "__main__":
    main()
