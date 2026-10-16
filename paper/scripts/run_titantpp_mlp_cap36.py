"""Fresh Cap-data fits and selected-checkpoint evaluation of four existing MLPs.

This module does not change model, loader, loss or head implementations.  Each
CLI invocation owns one fresh stage.  The supervisor owns GPU admission and the
host-wide Validation gate; this worker independently checks that gate for Test.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("titantpp_history_mlp", "titantpp_history_mlp_width16",
        "titantpp_all_available_history_mlp", "titantpp_current_only_param_matched")
SEEDS = (42, 52, 62)
DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
VARIANT = "count_only_log_regression"
SCHEMA = "titantpp_mlp_cap36_stage_v1"
MONITOR = "validation_raw_quantity_rmse"
REPLAY_RTOL = REPLAY_ATOL = 1e-5
POPULATION_KEYS = ("target_count", "target_identity_sha256", "target_quantity_sha256")
# These are the existing Factory defaults for an INACTIVE auxiliary head.  Cap
# quantiles may coincide; they are reporting strata, not legal T1 parameters.
# count_only_log_regression never reads these constants in its active loss.
INACTIVE_TAIL = {"tail_threshold": 46., "tail_normalization_scale": 46.,
                 "tail_clip_cap": 187., "tail_huber_delta": 1.}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def resolve_under(root, path):
    root = Path(root).resolve()
    path = Path(path)
    resolved = (path if path.is_absolute() else root / path).resolve()
    require(resolved.is_relative_to(root), "Artifact path escapes the host root")
    return resolved


def file_inventory(root, paths):
    return {str(resolve_under(root, p).relative_to(Path(root).resolve())): sha_file(p)
            for p in paths}


def verify_inventory(root, inventory):
    require(isinstance(inventory, dict) and bool(inventory), "Missing artifact inventory")
    for path, expected in inventory.items():
        require(sha_file(resolve_under(root, path)) == expected, "Artifact SHA mismatch: " + path)


def select_job(contract, host, job_id):
    require(host in ("5080", "5090") and host in contract["hosts"], "Unapproved host")
    jobs = [j for j in contract["jobs"] if j["id"] == job_id]
    require(len(jobs) == 1, "Job ID must select exactly one condition")
    job = jobs[0]
    require(job["host"] == host and job["arm"] in ARMS and job["seed"] in SEEDS
            and job["dataset"] in DATASETS, "Job scope mismatch")
    require(Path(job_id).name == job_id and job_id not in ("", ".", ".."), "Unsafe job ID")
    datasets = [d for d in contract["datasets"] if d["dataset_id"] == job["dataset"]]
    require(len(datasets) == 1, "Job dataset is not uniquely bound")
    data = datasets[0]
    cap = float(data["cap_value"])
    bounds = data["quantity_boundaries_all_train_rows"]
    require(math.isfinite(cap) and cap > 0 and len(bounds) == 4
            and all(math.isfinite(float(x)) for x in bounds)
            and all(a <= b for a, b in zip(bounds, bounds[1:]))
            and float(bounds[-1]) == cap, "Invalid Cap quantity boundaries")
    training = contract["training"]
    for key, expected in (("maximum_epochs", 300), ("minimum_epochs", 40),
                          ("patience", 40), ("batch_size", 128), ("monitor", MONITOR)):
        require(training.get(key) == expected, "Training condition changed: " + key)
    require(training.get("warm_start", False) is False, "Warm start is not permitted")
    require(data["model"]["quantity_variant"] == VARIANT
            and data["model"]["hidden_dim"] == 64
            and data["model"]["lambda_log_qty"] == 1.
            and data["model"]["lambda_tail"] == 0.
            and data["model"]["time_head_mode"] == "heteroscedastic_lognormal_duration",
            "Common model/head/loss changed")
    root = Path(contract["hosts"][host]["root"]).resolve()
    return job, data, root, root / "run" / job_id


def install_hooks():
    from paper.scripts import run_titantpp_history_width as width
    width.install_hooks()


def role_for(arm):
    require(arm in ARMS, "Only the four existing MLP arms are admitted")
    if arm == ARMS[0]:
        return "observed_time_core_ablation_v1"
    if arm == ARMS[1]:
        return "observed_time_history_mlp_width16_v1"
    return "observed_time_history_controls_v1"


def build_model(data, arm):
    install_hooks()
    require(arm in ARMS, "Foreign model")
    from models.TPPs.CountAwareFactory import build_count_aware_model
    config = {k: v for k, v in data["model"].items()
              if k not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    config.update(INACTIVE_TAIL)
    return build_count_aware_model(arm, **config,
        train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"],
        max_seq_len=data["loader"]["max_seq_len"])


def training_args(contract, data, output, arm):
    install_hooks()
    from paper.scripts import run_multilag_detail_execution as base
    args = base.training_args({**contract, "model_role": role_for(arm)}, data, output)
    args.epochs, args.min_epochs, args.early_stopping_patience = 300, 40, 40
    args.execution_role = "fresh_existing_mlp_cap36_validation"
    for key, value in INACTIVE_TAIL.items():
        setattr(args, key, value)
    return args


def population_identity(dataset, split):
    """The existing target identity format, extended only to the Test split."""
    import numpy as np
    require(split in ("train", "validation", "test"), "Foreign target split")
    ix = np.asarray(dataset.index, dtype=np.int64).reshape(-1, 2)
    require(len(ix) > 0, "Empty target population")
    part, position = ix[:, 0], ix[:, 1] + 1
    seq = np.asarray([dataset.seq_lists[p][t] for p, t in zip(part, position)], dtype=np.int64)
    q = np.asarray([dataset.val_lists[p][t] for p, t in zip(part, position)], dtype=np.float64)
    require(all(dataset.split_lists[p][t] == split for p, t in zip(part, position)), "Target split changed")
    require(np.isfinite(q).all() and (q >= 0).all(), "Invalid target values")
    h = hashlib.sha256(b"hard_lmm_target_identity_v1\0" + split.encode() + b"\0")
    h.update(json.dumps([str(p) for p in dataset.parts], ensure_ascii=False, separators=(",", ":")).encode())
    for values in (part, position, seq):
        h.update(values.astype("<i8").tobytes())
    qh = hashlib.sha256(b"hard_lmm_target_quantity_v1\0" + q.astype("<f8").tobytes()).hexdigest()
    return q, {"target_count": len(q), "target_identity_sha256": h.hexdigest(), "target_quantity_sha256": qh}


def verify_population(actual, expected):
    require(actual == {k: expected[k] for k in POPULATION_KEYS}, "Exact target population changed")


class MetricAccumulator:
    """Finite raw-quantity metrics; a strict upper tail may be empty after Cap."""
    def __init__(self, cap_value):
        self.cap = float(cap_value)
        require(math.isfinite(self.cap) and self.cap > 0, "Invalid cap")
        self.values = {key: [0, 0., 0., 0., 0.] for key in ("overall", "tail", "cap_hit")}

    def add(self, truth, prediction, temporal):
        import numpy as np
        q, p, t = [np.asarray(x, dtype=np.float64).reshape(-1) for x in (truth, prediction, temporal)]
        require(q.shape == p.shape == t.shape and all(np.isfinite(x).all() for x in (q, p, t)),
                "Nonfinite or mismatched metric rows")
        require((q >= 0).all() and (q <= self.cap).all(), "Truth is outside the capped task")
        for key, mask in (("overall", np.ones(q.shape, dtype=bool)), ("tail", q > self.cap),
                          ("cap_hit", q == self.cap)):
            e = p[mask] - q[mask]
            a = self.values[key]
            a[0] += int(mask.sum())
            a[1] += float(np.abs(e).sum())
            a[2] += float(np.square(e).sum())
            a[3] += float(t[mask].sum())
            a[4] += float(e.sum())

    def finish(self):
        def one(a):
            n, ae, se, temporal, signed = a
            return {"count": n, "qty_mae": ae / n if n else None,
                    "qty_rmse": math.sqrt(se / n) if n else None,
                    "time_nll": temporal / n if n else None, "qty_bias": signed / n if n else None,
                    "qty_sse": se, "qty_absolute_error_sum": ae,
                    "time_nll_sum": temporal, "status": "available" if n else "N/A_empty"}
        require(self.values["overall"][0] > 0, "No evaluated targets")
        result = {key: one(value) for key, value in self.values.items()}
        result.update(tail_rule="q > cap_value", cap_hit_rule="q == cap_value", cap_value=self.cap,
                      time_metric="recorded_positive_integer_time_nll")
        return result


def compare_metrics(actual, expected):
    for group in ("overall", "tail", "cap_hit"):
        require(actual[group]["count"] == expected[group]["count"], "Metric support changed")
        for key in ("qty_rmse", "qty_mae", "time_nll"):
            a, b = actual[group][key], expected[group][key]
            if a is None or b is None:
                require(a is None and b is None, "Empty-tail metric changed")
            else:
                require(math.isclose(a, b, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL),
                        "Validation replay changed: " + group + "." + key)


def compare_prediction_files(actual, expected):
    from itertools import zip_longest
    count = 0
    with gzip.open(actual, "rt") as new, gzip.open(expected, "rt") as old:
        for a, b in zip_longest(new, old):
            require(a is not None and b is not None, "Validation row count changed")
            x, y = json.loads(a), json.loads(b)
            for key in ("target_index", "entity_id", "target_seq", "raw_quantity", "history_length", "split"):
                require(x[key] == y[key], "Validation target row changed: " + key)
            for key in ("predicted_raw_quantity", "time_nll"):
                require(math.isclose(x[key], y[key], rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL),
                        "Validation individual prediction changed")
            count += 1
    require(count > 0, "Empty Validation prediction replay")
    return count


def _load_selected(data, job, train_receipt, root, device):
    install_hooks()
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint
    checkpoint = resolve_under(root, train_receipt["selected_checkpoint"])
    require(sha_file(checkpoint) == train_receipt["checkpoint_sha256"], "Selected checkpoint bytes changed")
    cp = torch_load_checkpoint(checkpoint, map_location="cpu")
    require(cp["backbone"] == job["arm"] and cp["seed"] == job["seed"]
            and cp["variant"] == VARIANT and cp["checkpoint_monitor"] == MONITOR
            and cp["best_epoch"] == train_receipt["selected_epoch"]
            and cp["evaluation_scope"] == "validation_only" and cp["held_out_test_evaluated"] is False,
            "Selected checkpoint route/selection changed")
    validate_checkpoint_route(cp, job["arm"])
    digest = canonical_state_dict_sha256(cp["model_state_dict"])
    require(digest == cp["model_state_sha256"] == train_receipt["state_sha256"], "Selected tensor SHA changed")
    model, _ = build_model(data, job["arm"])
    model.load_state_dict(cp["model_state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False), checkpoint, digest


def evaluate_selected(data, job, train_receipt, root, frame, split, output, device="cuda:0"):
    """Reuse the established frozen-loader/target_outputs path and persist every row."""
    import numpy as np
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    model, checkpoint, digest = _load_selected(data, job, train_receipt, root, device)
    loader = make_loader(frame, target_split=split, shuffle=False, generator=None,
        **{k: data["loader"][k] for k in ("batch_size", "lookback_weeks", "max_seq_len")})
    q, identity = population_identity(loader.dataset, split)
    binding = data["parent_data_identity"] if split == "test" else data["inherited_data_identity"]
    verify_population(identity, binding["populations"][split])
    metrics = MetricAccumulator(data["cap_value"])
    processed = 0
    output = Path(output)
    with output.open("xb") as binary, gzip.GzipFile(fileobj=binary, mode="wb", mtime=0) as compressed:
        with torch.no_grad():
            for _, dts, mask, part_ids, quantities in loader:
                value = target_outputs(model, dts.to(device), mask.to(device), quantities.to(device), lambda_log_qty=1.)
                arrays = {k: value[k].detach().cpu().numpy() for k in ("true_qty", "pred_qty", "time_loss", "history_length")}
                n = len(arrays["true_qty"])
                require(np.array_equal(arrays["true_qty"].astype(np.float64), q[processed:processed+n]), "Inference truth changed")
                metrics.add(arrays["true_qty"], arrays["pred_qty"], arrays["time_loss"])
                for i in range(n):
                    p, end = loader.dataset.index[processed+i]
                    require(int(part_ids[i]) == int(p), "Loader order changed")
                    row = {"target_index": processed+i, "entity_id": str(loader.dataset.parts[p]),
                        "target_seq": int(loader.dataset.seq_lists[p][end+1]), "split": split,
                        "raw_quantity": float(arrays["true_qty"][i]), "predicted_raw_quantity": float(arrays["pred_qty"][i]),
                        "time_nll": float(arrays["time_loss"][i]), "history_length": int(arrays["history_length"][i])}
                    compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode())
                processed += n
    require(processed == identity["target_count"], "Incomplete evaluation")
    require(canonical_state_dict_sha256(model.state_dict()) == digest, "Evaluation changed model state")
    require(sha_file(checkpoint) == train_receipt["checkpoint_sha256"], "Checkpoint bytes changed during evaluation")
    result = {"metrics": metrics.finish(), "target_identity": identity,
              "prediction_path": str(output.resolve()), "prediction_rows": processed,
              "parameters_unchanged": True, "state_sha256": digest}
    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def verify_train_receipt(contract, host, job, root, receipt):
    require(receipt.get("status") == "complete" and receipt.get("stage") == "train"
            and receipt.get("job_id") == job["id"] and receipt.get("host") == host
            and receipt.get("contract_sha256") == sha_json(contract), "Training receipt binding changed")
    verify_inventory(root, receipt["files"])


def verify_validation_gate(contract, host, root):
    gate = read(Path(root) / "validation_gate.json")
    require(gate.get("status") == "passed" and gate.get("host") == host
            and gate.get("contract_sha256") == sha_json(contract), "Host Validation gate is not valid")
    assigned = {j["id"]: j for j in contract["jobs"] if j["host"] == host}
    require(set(gate.get("jobs", {})) == set(assigned), "Host Validation gate is incomplete")
    for job_id, job in assigned.items():
        path = Path(root) / "run" / job_id / "validation_receipt.json"
        require(sha_file(path) == gate["jobs"][job_id]["validation_receipt_sha256"], "Validation receipt SHA changed")
        r = read(path)
        require(r.get("status") == "complete" and r.get("stage") == "validation"
                and r.get("job_id") == job_id and r.get("contract_sha256") == sha_json(contract)
                and r.get("validation_replay_passed") is True, "Validation replay gate did not pass")
        verify_inventory(root, r["files"])
        tr = read(Path(root) / "run" / job_id / "train_receipt.json")
        verify_train_receipt(contract, host, job, root, tr)
        require(r["state_sha256"] == tr["state_sha256"] and r["selected_checkpoint"] == tr["selected_checkpoint"],
                "Validation gate checkpoint changed")
    return gate


def _base_receipt(contract, host, job, stage):
    return {"schema": SCHEMA, "status": "complete", "stage": stage, "job_id": job["id"],
        "host": host, "dataset": job["dataset"], "arm": job["arm"], "seed": job["seed"],
        "contract_sha256": sha_json(contract), "held_out_test_evaluated": stage == "test",
        "inactive_tail_constructor": {**INACTIVE_TAIL, "lambda_tail": 0.,
            "active_loss_uses_these_constants": False, "reporting_tail_uses_cap_value": True}}


def run_stage(contract, host, job_id, stage):
    require(stage in ("train", "validation", "test"), "Invalid scientific stage")
    job, data, root, folder = select_job(contract, host, job_id)
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / (stage + "_receipt.json")
    require(not receipt_path.exists(), "Completed stage must not be executed again")
    if stage == "test":
        verify_validation_gate(contract, host, root)
    write_exclusive(folder / (stage + ".intent.json"), {"contract_sha256": sha_json(contract),
        "job_id": job_id, "stage": stage, "started_unix": time.time()})
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    configure_runtime("cuda:0", 4)
    install_hooks()
    from paper.scripts import run_multilag_detail_execution as base
    if stage == "train":
        from paper.scripts.count_aware_tpp_backbone import training
        require(not (folder / "runs").exists(), "Fresh fit required; resume is forbidden")
        frame, input_receipt = base.prepare_admitted_data(data)
        write_exclusive(folder / "input_receipt.json", input_receipt)
        args = training_args(contract, data, folder, job["arm"])
        args.seeds = str(job["seed"])
        interface = base.time_interface(data, frame, contract)
        interface["backbone_design"] = {"schema": SCHEMA, "arm": job["arm"],
            "cap_value": data["cap_value"], "training": contract["training"]}
        quantity = {"boundaries": data["quantity_boundaries_all_train_rows"],
            "strata": [{"stratum": f"quantity_{i}", "stratum_order": i, "stratum_label": f"Cap Train bin {i}"}
                       for i in range(5)]}
        summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
            interface_meta=interface, backbone=job["arm"], quantity_variant=VARIANT, seed=job["seed"])
        checkpoint = resolve_under(root, summary["checkpoint_path"])
        receipt = {**_base_receipt(contract, host, job, stage), "selected_checkpoint": str(checkpoint),
            "selected_epoch": summary["best_epoch"], "checkpoint_sha256": sha_file(checkpoint),
            "state_sha256": summary["checkpoint_state_sha256"], "summary": summary}
        prediction = folder / "training_validation_predictions.jsonl.gz"
        reference = evaluate_selected(data, job, receipt, root, frame, "validation", prediction)
        for key, value in (("qty_rmse", summary["best_val_qty_rmse"]), ("qty_mae", summary["best_val_qty_mae"]),
                           ("time_nll", summary["best_val_time_nll"])):
            require(math.isclose(reference["metrics"]["overall"][key], value, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL),
                    "Selected Validation differs from training summary")
        receipt.update(validation_reference=reference, target_identity=reference["target_identity"])
        files = [p for p in (folder / "runs").rglob("*") if p.is_file()]
        receipt["files"] = file_inventory(root, [*files, prediction, folder / "input_receipt.json"])
    else:
        tr_path = folder / "train_receipt.json"
        tr = read(tr_path)
        verify_train_receipt(contract, host, job, root, tr)
        if stage == "validation":
            frame, _ = base.prepare_admitted_data(data)
        else:
            import polars as pl
            from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
            binding = data["parent_data_identity"]
            for key in ("data", "split_manifest"):
                require(sha_file(binding[key]["path"]) == binding[key]["sha256"], "Test input SHA changed")
            raw = pl.read_parquet(binding["data"]["path"])
            require(set(raw["chronological_split"].unique().to_list()) == {"train", "validation", "test"},
                    "Full Cap input has wrong split labels")
            frame = prepare_count_frame(raw.sort(["oper_part_no", "seq"]))
        prediction = folder / (stage + "_predictions.jsonl.gz")
        evaluated = evaluate_selected(data, job, tr, root, frame, stage, prediction)
        if stage == "validation":
            compare_metrics(evaluated["metrics"], tr["validation_reference"]["metrics"])
            require(evaluated["target_identity"] == tr["validation_reference"]["target_identity"], "Replay targets changed")
            compare_prediction_files(prediction, tr["validation_reference"]["prediction_path"])
        receipt = {**_base_receipt(contract, host, job, stage), **evaluated,
            "selected_checkpoint": tr["selected_checkpoint"], "selected_epoch": tr["selected_epoch"],
            "checkpoint_sha256": tr["checkpoint_sha256"], "validation_replay_passed": True,
            "files": file_inventory(root, [prediction, tr_path, resolve_under(root, tr["selected_checkpoint"])])}
    write_exclusive(receipt_path, receipt)
    return receipt


def fabricated_data():
    return {"model": {"hidden_dim": 64, "quantity_variant": VARIANT,
        "time_head_mode": "heteroscedastic_lognormal_duration", "time_scale": 1.,
        "time_initial_location": .2, "time_initial_scale": .8,
        "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": "hour", "top_code": None}},
        "statistics": {"train_log_mean": 1.2, "train_log_std": .8}, "loader": {"max_seq_len": 256}}


def synthetic_check(contract=None, host=None, *, device=None):
    """Only fabricated inputs, all four models, no research rows/checkpoints."""
    import torch
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    device = device or ("cuda:0" if contract is not None else "cpu")
    configure_runtime(device, 4)
    data = fabricated_data()
    peaks, counts, initial_states = {}, {}, {}
    for arm in ARMS:
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(42)
        model, _ = build_model(data, arm)
        initial_states[arm] = canonical_state_dict_sha256(model.state_dict())
        model = model.to(device).train()
        dt = torch.tensor([[1., 2., 1., 3., 1.], [1., 1., 2., 1., 3.]], device=device)
        q = torch.tensor([[1., 3., 2., 4., 2.], [2., 4., 1., 3., 1.]], device=device)
        mask = torch.ones_like(dt, dtype=torch.bool)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = target_outputs(model, dt, mask, q, lambda_log_qty=1.)["joint_loss"].mean()
            require(bool(torch.isfinite(loss)), "Nonfinite synthetic loss")
            loss.backward()
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Nonfinite gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        model.eval()
        before = canonical_state_dict_sha256(model.state_dict())
        with torch.no_grad():
            value = target_outputs(model, dt, mask, q, lambda_log_qty=1.)
            changed = q.clone()
            changed[:, -1] += 100.
            altered = target_outputs(model, dt, mask, changed, lambda_log_qty=1.)
            changed_dt = dt.clone()
            changed_dt[:, -1] += 10.
            altered_dt = target_outputs(model, changed_dt, mask, q, lambda_log_qty=1.)
            single = target_outputs(model, dt[:1], mask[:1], q[:1], lambda_log_qty=1.)
            for key in ("pred_qty", "time_loss"):
                torch.testing.assert_close(value[key], altered[key], rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(value[key][:1], single[key], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(value["pred_qty"], altered_dt["pred_qty"], rtol=1e-5, atol=1e-5)
            state = io.BytesIO()
            torch.save(model.state_dict(), state)
            state.seek(0)
            restored, _ = build_model(data, arm)
            restored.load_state_dict(torch.load(state, map_location="cpu", weights_only=True), strict=True)
            restored = restored.to(device).eval()
            replay = target_outputs(restored, dt, mask, q, lambda_log_qty=1.)
            for key in ("pred_qty", "time_loss"):
                torch.testing.assert_close(value[key], replay[key], rtol=1e-5, atol=1e-5)
        require(before == canonical_state_dict_sha256(model.state_dict()), "Synthetic evaluation mutated state")
        counts[arm] = sum(p.numel() for p in model.parameters())
        del restored
        memory_shape = None
        if str(device).startswith("cuda"):
            require(contract is not None and host in contract["hosts"], "Native memory admission needs its host contract")
            assigned = {j["dataset"] for j in contract["jobs"] if j["host"] == host}
            lengths = [int(d["loader"]["max_seq_len"]) for d in contract["datasets"] if d["dataset_id"] in assigned]
            require(bool(lengths) and contract["training"]["batch_size"] == 128, "Missing full-shape admission contract")
            memory_shape = [128, max(lengths)]
            full_dt = torch.ones(memory_shape, device=device)
            full_q = torch.randint(1, 11, memory_shape, device=device).float()
            full_mask = torch.ones(memory_shape, dtype=torch.bool, device=device)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            full_loss = target_outputs(model, full_dt, full_mask, full_q, lambda_log_qty=1.)["joint_loss"].mean()
            require(bool(torch.isfinite(full_loss)), "Nonfinite full-shape synthetic loss")
            full_loss.backward()
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                    "Nonfinite full-shape synthetic gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            torch.cuda.synchronize()
            peaks[arm] = {"max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                          "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "full_training_shape": memory_shape, "includes_optimizer_step": True}
            del full_dt, full_q, full_mask, full_loss
        else:
            peaks[arm] = {"max_memory_reserved_bytes": 0, "full_training_shape": None,
                          "includes_optimizer_step": False}
        del model, optimizer
        gc.collect()
    return {"status": "passed", "device": str(device), "host": host, "arms": list(ARMS),
        "cuda_memory_peaks": peaks, "parameter_count": counts, "initial_state_sha256": initial_states,
        "checks": {"finite_forward_backward": True, "checkpoint_roundtrip": True,
                   "target_quantity_blocked": True, "target_time_blocked_for_quantity": True,
                   "batch_independence": True, "state_unchanged_during_evaluation": True},
        "real_data_loaded": False, "held_out_test_evaluated": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--host", required=True, choices=("5080", "5090"))
    parser.add_argument("--job")
    parser.add_argument("--stage", required=True, choices=("train", "validation", "test", "synthetic"))
    args = parser.parse_args(argv)
    contract = read(args.contract)
    if args.stage == "synthetic":
        result = synthetic_check(contract, args.host)
    else:
        require(args.job is not None, "Scientific stage requires --job")
        result = run_stage(contract, args.host, args.job, args.stage)
    print(json.dumps({k: result[k] for k in ("status", "job_id", "stage", "host", "real_data_loaded") if k in result}, sort_keys=True))


if __name__ == "__main__":
    main()
