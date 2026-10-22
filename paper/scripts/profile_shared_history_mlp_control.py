"""Train-only cost measurement for the locked history single-MLP control.

The module deliberately has no training scheduler: every invocation measures one
already selected checkpoint in one fresh process.  It never writes a scientific
checkpoint and it uses the unchanged Shared target-output path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from paper.scripts.profile_shared_component_ablation import (
    CudaHooks,
    measure_batches,
    validate_batch_cache,
    _target_outputs,
    _optimizer,
)

SCHEMA = "shared_history_mlp_control_cost_v1"
ARMS = ("shared15", "single_mlp", "encoder_only")
BATCH_SIZE, WARMUP, MEASURED, PREPARATION_SEED = 128, 5, 32, 42
PROTOCOL = {
    "batch_size": BATCH_SIZE,
    "preparation_seed": PREPARATION_SEED,
    "warmup_batches": WARMUP,
    "measured_batches": MEASURED,
    "repetitions": 3,
    "modes": ["train", "eval"],
    "fresh_process_per_rep": True,
    "same_fit_host": True,
    "maximum_parallel_workers": 1,
    "automatic_retry": False,
}


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "JSON object required")
    return value


def read_sealed_contract(path: Path, expected_sha256: str) -> dict:
    raw = Path(path).read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == expected_sha256, "Contract raw SHA differs"
    )
    value = json.loads(raw)
    require(isinstance(value, dict), "Contract JSON object required")
    return value


def write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def tensor_sha(values: dict[str, torch.Tensor]) -> str:
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    return canonical_state_dict_sha256(values)


def validate_cost_contract(contract: dict) -> None:
    manager = _manager()
    manager.validate(contract)
    cost = contract.get("cost", {})
    for key, expected in PROTOCOL.items():
        actual = cost.get(
            key,
            contract.get("maximum_parallel_workers")
            if key == "maximum_parallel_workers"
            else None,
        )
        require(actual == expected, "Cost protocol differs: " + key)
    require(cost.get("automatic_retry") is False, "Cost retries are forbidden")


def _manager():
    from paper.scripts import manage_shared_history_mlp_control

    return manage_shared_history_mlp_control


def _runner():
    from paper.scripts import run_shared_history_mlp_control

    return run_shared_history_mlp_control


def select_job(contract: dict, host: str, job_id: str) -> tuple[dict, dict, Path, Path]:
    validate_cost_contract(contract)
    job, data, root, folder = _runner().select_job(contract, host, job_id)
    require(
        job["host"] == host and job["id"] == job_id and job["arm"] in ARMS,
        "Job admission differs",
    )
    return job, data, Path(root), Path(folder)


def _root(contract: dict, host: str) -> Path:
    return Path(contract["hosts"][host]["root"])


def _runtime(contract: dict, host: str) -> dict:
    manager = _manager()
    from paper.scripts.quantity_comparison_runtime import (
        configure_runtime,
        runtime_identity,
    )

    spec = contract["hosts"][host]
    require(
        Path(sys.executable).resolve() == Path(spec["python"]).resolve(),
        "Native Python differs",
    )
    for key, value in spec["environment"].items():
        require(os.environ.get(key) == value, "Native environment differs: " + key)
    manager.guard(contract, host, active=(os.getpid(),))
    configure_runtime("cuda:0", threads=spec["runtime_expected"]["threads"])
    identity = runtime_identity("cuda:0")
    require(
        all(identity.get(k) == v for k, v in spec["runtime_expected"].items()),
        "Native runtime differs",
    )
    require(identity["gpu"]["uuid"] == spec["gpu_uuid"], "GPU identity differs")
    return identity


def _guard(contract: dict, host: str) -> dict:
    manager = _manager()
    return manager.guard(contract, host, active=(os.getpid(),))


def _model(data: dict, arm: str):
    from models.TPPs.CountAwareHistoryMLPControl import build_model

    return build_model(data, arm)


def _load_selected(
    contract: dict, host: str, job_id: str
) -> tuple[torch.nn.Module, dict, dict, dict]:
    runner = _runner()
    runner._check_source(contract, host)
    job, data, root, folder = runner.select_job(contract, host, job_id)
    receipt = read(folder / "train_receipt.json")
    # Reuse the new runner's complete selection/proof/history validation rather
    # than weakening admission for the cost path. The loaded model is a fresh
    # in-memory copy; train/eval profiling never writes its parent checkpoint.
    model, selected = runner._load_model(contract, data, job, receipt, root, "cpu")
    model.requires_grad_(True)
    checkpoint = Path(selected["checkpoint_path"])
    before = sha_file(checkpoint)
    require(
        before == selected["checkpoint_sha256"], "Selected checkpoint binary changed"
    )
    digest = tensor_sha(model.state_dict())
    require(
        digest == tensor_sha(selected["checkpoint_payload"]["model_state_dict"]),
        "Selected tensors differ",
    )
    return model, selected, {"binary_sha256": before, "tensor_sha256": digest}, data


def qualification(
    contract: dict, host: str, job_id: str, *, contract_file_sha256: str | None = None
) -> dict:
    job, _job_data, root, _folder = select_job(contract, host, job_id)
    output = root / "cost" / "qualification" / f"{job['arm']}.json"
    require(not output.exists(), "Qualification is immutable")
    write_exclusive(
        output.with_suffix(".intent.json"),
        {"host": host, "arm": job["arm"], "pid": os.getpid()},
    )
    manager = _manager()
    manager.verify_files(contract, Path(contract["hosts"][host]["source_root"]))
    identity = _runtime(contract, host)
    # Qualification must exercise the required physical B128×256 shape even
    # when the supplied arm job is RAF (max length 84).
    choices = [d for d in contract["datasets"] if d["loader"]["max_seq_len"] == 256]
    require(choices, "A length-256 qualification dataset is required")
    data = choices[0]
    model, _ = _model(data, job["arm"])
    model.to("cuda:0")
    optimizer = _optimizer(model, data)
    batch = {
        "dts": torch.ones(BATCH_SIZE, 256),
        "quantities": torch.ones(BATCH_SIZE, 256),
        "mask": torch.ones(BATCH_SIZE, 256, dtype=torch.bool),
    }
    rows = []
    for mode in ("eval", "train"):
        _guard(contract, host)
        model.train(mode == "train")
        model.requires_grad_(mode == "train")
        optimizer.zero_grad(set_to_none=True)
        hooks = CudaHooks()
        hooks.sync()
        baseline = hooks.memory()
        hooks.reset_peak()
        start = hooks.clock()
        b = hooks.transfer(batch)
        with torch.set_grad_enabled(mode == "train"):
            result = _target_outputs(model, b)
            loss = result["joint_loss"].mean()
            if mode == "train":
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), data["optimizer"]["grad_clip"]
                )
                optimizer.step()
        hooks.sync()
        elapsed = hooks.clock() - start
        require(bool(torch.isfinite(loss)), "Synthetic capacity loss is nonfinite")
        if mode == "train":
            require(
                bool(torch.isfinite(norm)), "Synthetic capacity gradient is nonfinite"
            )
        require(
            all(
                bool(torch.isfinite(parameter).all())
                for parameter in model.parameters()
            ),
            "Synthetic capacity parameters are nonfinite",
        )
        rows.append(
            {
                "mode": mode,
                "elapsed_seconds": elapsed,
                "batch_size": BATCH_SIZE,
                "padded_length": 256,
                "memory_baseline_bytes": baseline,
                "memory_peak_bytes": hooks.peak(),
                "optimizer_steps": int(mode == "train"),
            }
        )
    _guard(contract, host)
    result = {
        "schema": SCHEMA,
        "status": "qualified",
        "host": host,
        "arm": job["arm"],
        "pid": os.getpid(),
        "runtime": identity,
        "contract_sha256": canonical_sha(contract),
        "contract_file_sha256": contract_file_sha256,
        "source_files_sha256": contract["source"]["files_sha256"],
        "rows": rows,
        "synthetic": True,
        "real_data_loaded": False,
        "checkpoint_loaded": False,
        "checkpoint_saved": False,
        "scientific_fit": False,
        "cost_complete": False,
    }
    write_exclusive(output, result)
    return result


def profile(
    contract: dict,
    host: str,
    job_id: str,
    mode: str,
    rep: int,
    *,
    contract_file_sha256: str | None = None,
) -> dict:
    require(
        mode in {"train", "eval"} and rep in {1, 2, 3}, "Invalid cost mode/repetition"
    )
    job, _job_data, root, _folder = select_job(contract, host, job_id)
    manager = _manager()
    manager.verify_global_gate(contract, root)
    output = root / "cost" / "results" / job_id / f"{mode}_rep{rep}.json"
    require(not output.exists(), "Completed cost repetition is immutable")
    write_exclusive(
        output.with_suffix(".intent.json"),
        {"host": host, "job_id": job_id, "mode": mode, "rep": rep, "pid": os.getpid()},
    )
    model, selected, checkpoint, data = _load_selected(contract, host, job_id)
    batch_path = root / "cost" / "batches" / f"{job['dataset']}.pt"
    manifest_path = batch_path.with_suffix(".json")
    manifest = read(manifest_path)
    require(sha_file(batch_path) == manifest["file_sha256"], "Batch file changed")
    batches = torch.load(batch_path, map_location="cpu", weights_only=True)
    validate_batch_cache(
        manifest,
        batches,
        data,
        contract=contract,
        contract_file_sha256=contract_file_sha256,
    )
    identity = _runtime(contract, host)
    model.to("cuda:0")
    optimizer = _optimizer(model, data) if mode == "train" else None
    metrics = measure_batches(
        model,
        batches,
        mode,
        optimizer,
        CudaHooks(),
        _target_outputs,
        grad_clip=data["optimizer"]["grad_clip"],
        guard=lambda: _guard(contract, host),
    )
    if mode == "eval":
        require(
            tensor_sha(model.state_dict()) == checkpoint["tensor_sha256"],
            "Eval mutated selected parameters",
        )
    require(
        sha_file(Path(selected["checkpoint_path"])) == checkpoint["binary_sha256"],
        "Selected checkpoint changed during cost measurement",
    )
    selected_identity = {
        "checkpoint_path": str(selected["checkpoint_path"]),
        "checkpoint_sha256": checkpoint["binary_sha256"],
        "state_sha256": checkpoint["tensor_sha256"],
        "selected_epoch": selected["selected_epoch"],
        "training_origin": selected["training_origin"],
        "provenance": selected["provenance"],
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "mode": mode,
        "host": host,
        "job_id": job_id,
        "dataset": job["dataset"],
        "arm": job["arm"],
        "training_seed": job["seed"],
        "timing_rep": rep,
        "runtime": identity,
        "contract_sha256": canonical_sha(contract),
        "contract_file_sha256": contract_file_sha256,
        "source_files_sha256": contract["source"]["files_sha256"],
        "selected": selected_identity,
        "checkpoint_binary_unchanged": True,
        "checkpoint_saved": False,
        "scientific_fit": False,
        "input_split": "train",
        "synthetic": False,
        "accuracy_evaluated": False,
        "preparation_seed": PREPARATION_SEED,
        "batch_manifest_sha256": sha_file(manifest_path),
        "target_ids_sha256": manifest["target_ids_sha256"],
        "batch_tensor_sha256": [r["tensor_sha256"] for r in manifest["batches"]],
        "parameters": sum(p.numel() for p in model.parameters()),
        **metrics,
    }
    write_exclusive(output, result)
    return result


def prepare_batches(
    contract: dict, host: str, *, contract_file_sha256: str | None = None
) -> dict:
    """Create one deterministic, Train-only cache per dataset for every arm/seed."""
    validate_cost_contract(contract)
    manager = _manager()
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from data_loader.event_seq_data_module import collate_week_lookback

    manager.verify_files(contract, Path(contract["hosts"][host]["source_root"]))
    root, results = _root(contract, host), {}
    manager.verify_global_gate(contract, root)
    for data in contract["datasets"]:
        identity = data["inherited_data_identity"]
        source = Path(contract["hosts"][host]["source_root"]).resolve()
        relative = Path(identity["data"]["path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Input escaped source root",
        )
        input_path = (source / relative).resolve()
        require(input_path.is_relative_to(source), "Input escaped source root")
        require(
            contract["input_files"].get(str(relative)) == identity["data"]["sha256"]
            and sha_file(input_path) == identity["data"]["sha256"],
            "Train input SHA changed",
        )
        path = root / "cost" / "batches" / f"{data['dataset_id']}.pt"
        manifest_path = path.with_suffix(".json")
        if manifest_path.exists():
            manifest = read(manifest_path)
            require(
                sha_file(path) == manifest["file_sha256"], "Existing batch file changed"
            )
            validate_batch_cache(
                manifest,
                torch.load(path, map_location="cpu", weights_only=True),
                data,
                contract=contract,
                contract_file_sha256=contract_file_sha256,
            )
            results[data["dataset_id"]] = {
                "cached": True,
                "manifest_sha256": sha_file(manifest_path),
                "target_ids_sha256": manifest["target_ids_sha256"],
            }
            continue
        require(not path.exists(), "Partial batch preparation exists")
        write_exclusive(
            path.with_suffix(".intent.json"),
            {
                "host": host,
                "dataset": data["dataset_id"],
                "contract_sha256": canonical_sha(contract),
            },
        )
        import polars as pl

        frame = (
            pl.scan_parquet(input_path)
            .filter(pl.col("chronological_split") == "train")
            .collect()
            .sort(["oper_part_no", "seq"])
        )
        require(
            frame["chronological_split"].unique().to_list() == ["train"],
            "Train-only predicate failed",
        )
        loader = make_loader(
            prepare_count_frame(frame),
            target_split="train",
            shuffle=True,
            generator=torch.Generator().manual_seed(PREPARATION_SEED),
            **{
                key: data["loader"][key]
                for key in ("batch_size", "lookback_weeks", "max_seq_len")
            },
        )
        total = BATCH_SIZE * (WARMUP + MEASURED)
        require(len(loader.dataset) >= total, "Insufficient distinct Train targets")
        indices = torch.randperm(
            len(loader.dataset),
            generator=torch.Generator().manual_seed(PREPARATION_SEED),
        ).tolist()[:total]
        batches, rows, target_ids = [], [], []
        for number in range(WARMUP + MEASURED):
            chunk = indices[number * BATCH_SIZE : (number + 1) * BATCH_SIZE]
            _, dts, mask, _, quantities = collate_week_lookback(
                [loader.dataset[i] for i in chunk]
            )
            batch = {"dts": dts, "mask": mask, "quantities": quantities}
            targets = [
                [
                    str(loader.dataset.parts[p]),
                    int(loader.dataset.seq_lists[p][pos + 1]),
                ]
                for p, pos in (loader.dataset.index[i] for i in chunk)
            ]
            batches.append(batch)
            target_ids.extend(targets)
            rows.append(
                {
                    "batch": number,
                    "phase": "warmup" if number < WARMUP else "measured",
                    "target_split": "train",
                    "targets": targets,
                    "tensor_sha256": tensor_sha(batch),
                }
            )
        manifest = {
            "schema": SCHEMA,
            "dataset": data["dataset_id"],
            "split": "train",
            "synthetic": False,
            "contract_sha256": canonical_sha(contract),
            "contract_file_sha256": contract_file_sha256,
            "preparation_seed": PREPARATION_SEED,
            "input_sha256": identity["data"]["sha256"],
            "target_ids_sha256": canonical_sha(target_ids),
            "batches": rows,
        }
        validate_batch_cache(
            manifest,
            batches,
            data,
            contract=contract,
            contract_file_sha256=contract_file_sha256,
        )
        with path.open("xb") as stream:
            torch.save(batches, stream)
        manifest["file_sha256"] = sha_file(path)
        write_exclusive(manifest_path, manifest)
        results[data["dataset_id"]] = {
            "cached": False,
            "manifest_sha256": sha_file(manifest_path),
            "target_ids_sha256": manifest["target_ids_sha256"],
        }
    return {
        "schema": SCHEMA,
        "status": "prepared",
        "host": host,
        "datasets": results,
        "real_data_loaded": True,
        "data_split": "train",
        "gpu_execution": False,
        "cost_complete": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--job")
    parser.add_argument(
        "--mode",
        choices=("qualification", "prepare-batches", "train", "eval"),
        required=True,
    )
    parser.add_argument("--rep", type=int, choices=(1, 2, 3))
    args = parser.parse_args(argv)
    contract = read_sealed_contract(args.contract, args.contract_sha)
    if args.mode == "prepare-batches":
        require(
            args.job is None and args.rep is None, "Preparation has no job/repetition"
        )
        value = prepare_batches(
            contract, args.host, contract_file_sha256=args.contract_sha
        )
    elif args.mode == "qualification":
        require(args.job and args.rep is None, "Qualification requires job only")
        value = qualification(
            contract, args.host, args.job, contract_file_sha256=args.contract_sha
        )
    else:
        require(args.job and args.rep, "Cost requires job and repetition")
        value = profile(
            contract,
            args.host,
            args.job,
            args.mode,
            args.rep,
            contract_file_sha256=args.contract_sha,
        )
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
