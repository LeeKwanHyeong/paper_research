#!/usr/bin/env python3
"""Train-only full 130-parameter time-head refit above an immutable quantity-selected CNN+GRU54 checkpoint.

Import is inert. The CLI owns no remote operations. It admits Train/Validation
only, freezes the quantity function, and saves an E0-capable calibration artifact
separately from the original checkpoint. Native code is loaded only from the
SHA-bound scientific snapshot in the execution contract.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

SCHEMA = "titantpp_time_head_full_refit_v1"
SOURCE_CLOSURE = "4e94229fae002dc678025ecf056862f832c64459969f403b3cb5ac8cea8ba7b0"
ARM = "titantpp_cnn_gru54"
PARAMETERS = ("v_t.weight", "b_t", "w_raw", "time_scale_weight.weight")
PARAMETER_SHAPES = {"v_t.weight": (1, 64), "b_t": (1,), "w_raw": (1,), "time_scale_weight.weight": (1, 64)}
PARAMETER_COUNT = 130
FITTING = {"parameters": list(PARAMETERS), "maximum_epochs": 40, "patience": 10,
    "lr": 0.001, "batch_size": 128, "grad_clip": 1.0, "optimizer": "Adam",
    "weight_decay": 0.0, "selector": "validation_time_nll",
    "include_identity_epoch0": True, "tie": "strict_earliest_finite_minimum"}
DATASETS = ("yellow_trip_hourly", "raf_spare_parts")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    os.replace(temporary, path)


def confined(root, name):
    root = Path(root).resolve()
    require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe source path")
    path = (root / name).resolve()
    require(path.is_relative_to(root), "Source symlink escaped frozen root")
    return path


def validate_contract(c, *, verify_files=True):
    require(c.get("schema") == SCHEMA and c.get("approved") is True, "Unapproved refit contract")
    require(c.get("evaluation_scope") == "Train_fit_Validation_development_selection" and c.get("held_out_test_evaluated") is False,
        "Refit scope must be Train fitting and Validation selection")
    require(c.get("fit_split", "train") == "train" and c.get("selection_split", "validation") == "validation", "Foreign fit/selector split")
    require(c.get("host") == "5080", "Only 5080 is admitted")
    require(c.get("device") == "cuda:0", "Native 5080 CUDA device required")
    limits = c["limits"]
    require(limits["per_condition_seconds"] == 3600 and limits["total_wall_seconds"] == 23400
        and limits["workers_per_host"] == 1 and limits["automatic_retry"] is False
        and limits["max_cpu_rss_bytes"] == 16 * 1024**3 and limits["peak_device_fraction_max"] == .8, "Approved resource limits changed")
    require(c.get("fitting") == FITTING, "Only the approved full 130-parameter time-head fitting stage is admitted")
    require(c["source"]["files_sha256"] == SOURCE_CLOSURE == canonical(c["source"]["files"]), "Frozen source closure changed")
    require(Path(c["data_root"]).resolve() == Path(c["source"]["root"]).resolve()
        == Path(c["runtime"]["source_root"]).resolve(), "Source/data root changed")
    datasets = {d["dataset_id"]: d for d in c["datasets"]}
    require(len(datasets) == len(c["datasets"]) and set(datasets) <= set(DATASETS), "Foreign/duplicate dataset")
    require(1 <= len(c["jobs"]) <= 6 and len({j["id"] for j in c["jobs"]}) == len(c["jobs"]), "Foreign/duplicate jobs")
    require(len({(j["dataset"], j["seed"]) for j in c["jobs"]}) == len(c["jobs"]), "Duplicate dataset/seed")
    for j in c["jobs"]:
        require(j.get("arm") == ARM and j["dataset"] in datasets and type(j["seed"]) is int and j["seed"] in (42, 52, 62), "Foreign job")
        data = datasets[j["dataset"]]
        require(data["loader"]["batch_size"] == 128 and data["loader"]["drop_last"] is False
            and data["loader"]["validation_shuffle"] is False, "Canonical batch/population policy changed")
        require(data["model"]["hidden_dim"] == 64 and data["model"]["time_head_mode"] == "heteroscedastic_lognormal_duration",
            "Frozen head/model changed")
        require(data["model"]["time_observation_contract"]["mode"] == "positive_integer_round_clamp_v1"
            and data["model"]["time_observation_contract"]["top_code"] is None, "Recorded-time law changed")
        require(j["checkpoint"]["epoch"] >= 1 and len(j["checkpoint"]["state_sha256"]) == 64, "Checkpoint binding missing")
        require(j["baseline_metrics"]["count"] == data["inherited_data_identity"]["populations"]["validation"]["target_count"],
            "Full Validation population changed")
        require(Path(j["output_dir"]).is_absolute() and Path(j["output_dir"]).resolve().is_relative_to(Path(c["root"]).resolve() / "run"),
            "Output must be inside the explicit owned run root")
        require(not Path(j["checkpoint"]["path"]).resolve().is_relative_to(Path(j["output_dir"]).resolve()), "Output overlaps input checkpoint")
        if verify_files:
            require(sha(j["checkpoint"]["path"]) == j["checkpoint"]["sha256"], "Input checkpoint SHA changed")
            for key in ("source_terminal", "source_endpoint"):
                require(sha(j[key]["path"]) == j[key]["sha256"], "Original metadata SHA changed: " + key)
            terminal, endpoint = read(j["source_terminal"]["path"]), read(j["source_endpoint"]["path"])
            require(terminal.get("scientific_success") is True and terminal.get("status") == "complete"
                and terminal.get("held_out_test_evaluated") is False, "Incomplete original source fit")
            original_id = j["canonical_original_id"]
            for document in (terminal, endpoint):
                original_job = document.get("job", {})
                require(original_job.get("id") == original_id and original_job.get("dataset") == j["dataset"]
                    and original_job.get("seed") == j["seed"] and original_job.get("arm") == ARM, "Original metadata job identity changed")
            selected_key = f"runs/{ARM}/count_only_log_regression/seed_{j['seed']}/best_val_qty_rmse_model.pt"
            endpoint_key = f"runs/{ARM}/count_only_log_regression/seed_{j['seed']}/endpoint_replays.json"
            require(terminal["files"][selected_key] == j["checkpoint"]["sha256"]
                and terminal["files"][endpoint_key] == j["source_endpoint"]["sha256"], "Original terminal selected/endpoint SHA mismatch")
            require(endpoint.get("best_epoch") == j["checkpoint"]["epoch"]
                and endpoint["selected"]["state_sha256"] == j["checkpoint"]["state_sha256"], "Original selected endpoint state/epoch changed")
            for name in ("count", "qty_rmse", "qty_mae", "time_nll", "tail"):
                require(endpoint["selected"][name] == j["baseline_metrics"][name], "Original baseline metric binding changed")
            require(endpoint.get("evaluation_scope") == "validation_only" and endpoint.get("held_out_test_evaluated") is False,
                "Original endpoint is not Validation-only")
    if verify_files:
        for name, expected in c["source"]["files"].items():
            require(sha(confined(c["source"]["root"], name)) == expected, "Frozen source SHA changed: " + name)
        if "runtime_sha256" in c:
            require(sha(__file__) == c["runtime_sha256"], "Runtime SHA changed")
    return datasets


def verify_operational_authorization(c, contract_path):
    root = Path(c["root"]).resolve()
    require(Path(contract_path).resolve() == root / "execution_contract.json", "Contract outside owned root")
    operation = c["operation"]
    require(canonical(operation["files"]) == operation["files_sha256"], "Operation closure malformed")
    require(operation["files"]["operation/titantpp_time_head_full_refit_runtime.py"] == sha(__file__), "Worker runtime SHA changed")
    for name, expected in operation["files"].items():
        require(sha(confined(root, name)) == expected, "Operation file SHA changed: " + name)
    approval, start = read(root / "approval.json"), read(root / "start_permit.json")
    require(approval.get("approved") is True and approval.get("contract_sha256") == canonical(c)
        and approval.get("hosts") == ["5080"] and bool(approval.get("user_instruction")), "Missing user approval binding")
    require(start["contract_sha256"] == canonical(c) and start["approval_sha256"] == canonical(approval)
        and start["deadline_unix"] - start["issued_at_unix"] == c["limits"]["total_wall_seconds"]
        and start["issued_at_unix"] <= time.time() < start["deadline_unix"], "Expired or foreign campaign start permit")
    require(not (root / "failure.json").exists(), "Campaign failure requires manual review")
    parent = read(Path(c["parent_root"]) / "execution_contract.json")
    require(canonical(parent) == c["parent_contract_sha256"]
        and all(parent["source"][k] == c["source"][k] for k in ("files", "files_sha256", "git_revision")),
        "Parent scientific binding changed")
    expected_data = [d for d in parent["datasets"] if d["dataset_id"] in DATASETS]
    require(c["datasets"] == expected_data, "Original frozen dataset/statistics changed")
    parent_runtime = parent["hosts"]["5080"]
    require(all(c["runtime"][k] == parent_runtime[k] for k in ("python", "gpu_uuid", "runtime_expected", "source_root")),
        "Parent runtime identity changed")
    environment = {**parent_runtime["environment"], "MPLCONFIGDIR": str(root / "cache/matplotlib"), "XDG_CACHE_HOME": str(root / "cache")}
    require(c["runtime"]["environment"] == environment, "Approved cache-only environment differences changed")
    require(all(os.environ.get(k) == value for k, value in environment.items()), "Process environment differs from approved runtime")
    return start


def native(source_root):
    """Reject preloaded current-tree modules instead of silently mixing revisions."""
    root = Path(source_root).resolve()
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("models.", "paper.scripts.", "simple_lab_test.search.")) and getattr(module, "__file__", None):
            require(Path(module.__file__).resolve().is_relative_to(root), "Native module already loaded outside frozen source: " + name)
    sys.path.insert(0, str(root))
    engine = importlib.import_module("paper.scripts.run_titantpp_cnn_gru")
    engine.install_hooks()
    core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
    loader = importlib.import_module("paper.scripts.run_taxi_quantity_interface_ablation")
    runner = importlib.import_module("simple_lab_test.search.common.runner")
    for module in (engine, core, loader, runner):
        require(Path(module.__file__).resolve().is_relative_to(root), "Native module resolved outside frozen source")
    return engine, core, loader, runner


def mutable_context_clear(model):
    for module in model.modules():
        require(getattr(module, "_ctx_mem", None) is None, "Cross-forward contextual memory is forbidden")


def state_sha(model, digest, *, frozen_only=False):
    state = model.state_dict()
    if frozen_only:
        state = {k: v for k, v in state.items() if k not in PARAMETERS}
    return digest(state)


def freeze(model):
    model.eval()
    params = dict(model.named_parameters())
    require(all(k in params and tuple(params[k].shape) == PARAMETER_SHAPES[k] for k in PARAMETERS), "Native full130 head shapes required")
    require(tuple(name for name, _ in model.time_head_named_parameters()) == PARAMETERS
        and sum(params[k].numel() for k in PARAMETERS) == PARAMETER_COUNT, "Native full130 head API changed")
    for name, parameter in params.items():
        parameter.requires_grad_(name in PARAMETERS)
        parameter.grad = None
    require(set(name for name, p in model.named_parameters() if p.requires_grad) == set(PARAMETERS),
        "Unexpected trainable tensor")
    mutable_context_clear(model)


def parameter(model, name):
    require(name in PARAMETERS, "Parameter is outside the approved time head")
    return dict(model.named_parameters())[name]


def head_state(model):
    return {k: parameter(model, k).detach().cpu().clone() for k in PARAMETERS}


def restore_head_state(model, value):
    import torch
    require(set(value) == set(PARAMETERS), "Foreign time-head state")
    with torch.no_grad():
        for name in PARAMETERS:
            require(tuple(value[name].shape) == PARAMETER_SHAPES[name], "Time-head state shape changed")
            parameter(model, name).copy_(value[name].to(parameter(model, name).device))


def guard_frozen(model, digest, expected):
    require(all(not m.training for m in model.modules()), "Frozen model left evaluation mode")
    require(set(name for name, p in model.named_parameters() if p.requires_grad) == set(PARAMETERS), "Trainable parameters changed")
    mutable_context_clear(model)
    require(state_sha(model, digest, frozen_only=True) == expected, "Frozen parameter/buffer SHA changed")


def extract_batch(model, core, batch, device):
    """Exact native target boundary; no target quantity or gap is encoded."""
    import torch
    _, dts, mask, _, quantities = batch
    require(quantities is not None, "Missing raw quantities")
    dts, quantities, mask, lengths = core.right_pad_batch(dts.to(device), quantities.to(device), mask.to(device))
    ids = torch.arange(dts.size(0), device=dts.device)
    target, previous = lengths - 1, lengths - 2
    hidden_quantities = quantities.clone()
    hidden_quantities[ids, target] = 0.
    observed = mask.clone()
    observed[ids, target] = False
    with torch.no_grad():
        time_states, quantity_states = model.encode_task_states(dts, hidden_quantities, mask, memory_write_mask=observed)
        h = time_states[ids, previous]
        qh = quantity_states[ids, previous]
        require(torch.equal(h, qh), "CNNGRU shared state route changed")
        dt, qty = dts[ids, target].float(), quantities[ids, target].float()
        prediction = model.predict_quantity(qh)[1]
        nll = -model.log_observation_dt(h, dt)
        require(all(bool(torch.isfinite(v).all()) for v in (h, dt, qty, prediction, nll)), "Nonfinite native extraction")
    return {"hidden": h.detach().cpu().clone(), "dt": dt.cpu().clone(), "qty": qty.cpu().clone(),
        "prediction": prediction.cpu().clone(), "native_nll": nll.cpu().clone()}


def extract_cache(model, core, loader, device, digest, frozen_sha, pulse=lambda: None, *, max_batches=None):
    import torch
    batches = []
    first_batch = None
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        pulse()
        guard_frozen(model, digest, frozen_sha)
        record = extract_batch(model, core, batch, device)
        guard_frozen(model, digest, frozen_sha)
        if first_batch is None:
            first_batch = tuple(v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for v in batch)
        batches.append(record)
    require(bool(batches), "Empty admitted cache")
    return {key: torch.cat([v[key] for v in batches], dim=0) for key in batches[0]}, first_batch


def cache_nll(model, cache, indices, device):
    return -model.log_observation_dt(cache["hidden"][indices].to(device), cache["dt"][indices].to(device))


def cache_metrics(model, cache, device, *, threshold, pulse=lambda: None, check_quantity=True):
    import numpy as np
    import torch
    totals = {name: [0, 0., 0., 0.] for name in ("overall", "tail")}
    with torch.no_grad():
        for begin in range(0, len(cache["dt"]), 128):
            pulse()
            indices = slice(begin, begin + 128)
            h = cache["hidden"][indices].to(device)
            if check_quantity:
                require(torch.equal(model.predict_quantity(h)[1].cpu(), cache["prediction"][indices]), "Quantity output changed on the same cached batch")
            nll = cache_nll(model, cache, indices, device).cpu().numpy().astype(np.float64)
            qty = cache["qty"][indices].numpy().astype(np.float64)
            prediction = cache["prediction"][indices].numpy().astype(np.float64)
            require(np.isfinite(nll).all(), "Nonfinite full Validation NLL")
            error = prediction - qty
            for label, selected in (("overall", np.ones(qty.shape, dtype=bool)), ("tail", qty > threshold)):
                v = totals[label]
                v[0] += int(selected.sum())
                v[1] += float(np.abs(error[selected]).sum())
                v[2] += float(np.square(error[selected]).sum())
                v[3] += float(nll[selected].sum())
    def finish(v):
        n, absolute, squared, temporal = v
        return {"count": n, "qty_mae": absolute / n if n else None, "qty_rmse": math.sqrt(squared / n) if n else None,
            "qty_sse": squared, "time_nll": temporal / n if n else None}
    return {**finish(totals["overall"]), "tail": finish(totals["tail"]), "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False, "time_metric": "recorded_positive_integer_time_nll"}


def verify_baseline(actual, expected):
    for label in ("overall", "tail"):
        a, e = (actual, expected) if label == "overall" else (actual["tail"], expected["tail"])
        require(a["count"] == e["count"], "Original full Validation/tail population mismatch")
        for name in ("qty_rmse", "qty_mae", "time_nll"):
            require(math.isclose(a[name], e[name], rel_tol=1e-5, abs_tol=1e-5), "E0 original metric mismatch: " + label + "/" + name)


def spot_check(model, core, batch, device, reference, digest, frozen_sha):
    import torch
    got = extract_batch(model, core, batch, device)
    for key in ("hidden", "dt", "qty", "prediction"):
        require(torch.equal(got[key], reference[key][:len(got[key])]), "Native quantity/feature replay changed: " + key)
    guard_frozen(model, digest, frozen_sha)


def choose_best(history):
    finite = [r for r in history if math.isfinite(r["validation"]["time_nll"])]
    require(bool(finite), "No finite Validation selection")
    return min(finite, key=lambda r: r["validation"]["time_nll"])


def fit_cached(model, train, validation, *, core, first_batch, device, digest, frozen_sha,
               seed, threshold, fitting, pulse=lambda: None, on_epoch=lambda history, best: None):
    import torch
    require(fitting == FITTING, "Only the approved full 130-parameter time-head stage may run")
    guard_frozen(model, digest, frozen_sha)
    identity = head_state(model)
    e0 = cache_metrics(model, validation, device, threshold=threshold, pulse=pulse)
    history = [{"epoch": 0, "validation": e0, "train_time_nll": None, "steps": 0}]
    best_head_state, best_epoch = deepcopy(identity), 0
    on_epoch(history, best_head_state)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.Adam([parameter(model, k) for k in PARAMETERS], lr=0.001, weight_decay=0.)
    steps = 0
    for epoch in range(1, 41):
        epoch_started = time.monotonic()
        pulse()
        order = torch.randperm(len(train["dt"]), generator=generator)
        total, count = 0., 0
        for begin in range(0, len(order), 128):
            pulse()
            indices = order[begin:begin + 128]
            optimizer.zero_grad(set_to_none=True)
            losses = cache_nll(model, train, indices, device)
            require(bool(torch.isfinite(losses).all()), "Nonfinite Train refit NLL")
            losses.mean().backward()
            require(all(parameter(model, k).grad is not None and bool(torch.isfinite(parameter(model, k).grad).all()) for k in PARAMETERS),
                "Nonfinite/missing time-head gradients")
            torch.nn.utils.clip_grad_norm_([parameter(model, k) for k in PARAMETERS], 1., error_if_nonfinite=True)
            optimizer.step()
            require(all(bool(torch.isfinite(parameter(model, k)).all()) for k in PARAMETERS), "Nonfinite fitted time head")
            total += float(losses.detach().double().sum().cpu())
            count += len(indices)
            steps += 1
        guard_frozen(model, digest, frozen_sha)
        spot_check(model, core, first_batch, device, train, digest, frozen_sha)
        metrics = cache_metrics(model, validation, device, threshold=threshold, pulse=pulse)
        require(metrics["qty_rmse"] == e0["qty_rmse"] and metrics["qty_mae"] == e0["qty_mae"] and metrics["tail"]["qty_rmse"] == e0["tail"]["qty_rmse"],
            "Frozen quantity metrics changed")
        row = {"epoch": epoch, "validation": metrics, "train_time_nll": total / count, "steps": steps,
            "epoch_seconds": time.monotonic() - epoch_started,
            "completed_utc": datetime.now(timezone.utc).isoformat()}
        history.append(row)
        if metrics["time_nll"] < history[best_epoch]["validation"]["time_nll"]:
            best_epoch, best_head_state = epoch, head_state(model)
        require(choose_best(history)["epoch"] == best_epoch, "Strict earliest selector changed")
        on_epoch(history, best_head_state)
        if epoch - best_epoch >= 10:
            break
    last_head_state, last_metrics = head_state(model), history[-1]["validation"]
    restore_head_state(model, best_head_state)
    guard_frozen(model, digest, frozen_sha)
    selected = cache_metrics(model, validation, device, threshold=threshold, pulse=pulse)
    require(selected == history[best_epoch]["validation"], "Selected refit replay mismatch")
    require(selected["time_nll"] <= e0["time_nll"], "E0 fallback was lost")
    spot_check(model, core, first_batch, device, train, digest, frozen_sha)
    return {"history": history, "E0": e0, "selected": selected, "last": last_metrics,
        "best_epoch": best_epoch, "completed_epochs": history[-1]["epoch"], "identity_head_state": identity,
        "selected_head_state": best_head_state, "last_head_state": last_head_state, "optimizer_steps": steps}


class Resources:
    def __init__(self, c):
        self.c = c
        self.last_storage_check = -math.inf
    def __call__(self):
        import resource
        import torch
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        require(rss_bytes <= self.c["limits"]["max_cpu_rss_bytes"], "CPU RSS resource budget exceeded")
        if torch.cuda.is_initialized():
            total = torch.cuda.get_device_properties(0).total_memory
            require(torch.cuda.max_memory_allocated(0) <= total * self.c["limits"]["peak_device_fraction_max"],
                "Native GPU memory budget exceeded")
        now = time.monotonic()
        if now - self.last_storage_check >= 30:
            owned = Path(self.c["root"])
            size = sum(p.stat().st_size for p in owned.rglob("*") if p.is_file())
            require(size <= self.c["limits"]["max_owned_root_bytes"], "Owned storage budget exceeded")
            self.last_storage_check = now


def claim_output(output, allowed, *, job):
    require(not output.exists() or {p.name for p in output.iterdir()} <= allowed,
        "Fresh output required; automatic retry/resume forbidden")
    output.mkdir(parents=True, exist_ok=True)
    # The supervisor may create its permit and PID evidence before its child.
    # This exclusive child claim closes the retry/concurrent-worker boundary.
    with (output / "runtime_claim.json").open("x", encoding="utf-8") as stream:
        json.dump({"job": job, "pid": os.getpid(), "owner_pid": os.getppid(), "at_unix": time.time()}, stream)


class Permit:
    def __init__(self, c, job, path, owner_pid):
        require(path is not None and owner_pid is not None, "An explicit job training permit and owner PID are required")
        self.p = read(path)
        require(self.p.get("schema") == "time_head_full_refit_training_permit_v1" and self.p.get("contract_sha256") == canonical(c), "Foreign training permit")
        require(self.p.get("job") == job["id"] and self.p.get("owner_pid") == owner_pid == os.getppid(), "Permit/job/supervisor mismatch")
        q = self.p["qualification"]
        require(sha(q["path"]) == q["sha256"], "Qualification SHA changed")
        receipt = read(q["path"])
        representative = c["jobs"][0]
        require(receipt.get("job") == receipt.get("representative_job") == representative["id"]
            and receipt.get("checkpoint_sha256") == representative["checkpoint"]["sha256"]
            and receipt.get("source_files_sha256") == SOURCE_CLOSURE
            and receipt.get("device") == "cuda:0" and receipt.get("evaluation_scope") == "validation_only"
            and receipt.get("held_out_test_evaluated") is False and receipt.get("quantity_output_unchanged") is True
            and receipt.get("trainable_tensors") == list(PARAMETERS) and receipt.get("trainable_scalar_count") == PARAMETER_COUNT,
            "Representative qualification binding changed")
        require(receipt["runtime"]["gpu"]["uuid"] == c["runtime"]["gpu_uuid"], "Qualification GPU changed")
        for name, expected in c["runtime"]["runtime_expected"].items():
            require(receipt["runtime"].get(name) == expected, "Qualified runtime changed")
        require(receipt.get("schema") == "time_head_full_refit_native_qualification_v1"
            and receipt.get("status") == "passed" and receipt.get("contract_sha256") == canonical(c)
            and receipt.get("host") == "5080" and receipt.get("real_optimizer_updates") == 0, "Missing native qualification")
        require(self.p["issued_at_unix"] <= time.time() + 1, "Training permit issued in the future")
        require(self.p["deadline_unix"] > self.p["issued_at_unix"] and self.p["deadline_unix"] - self.p["issued_at_unix"] <= 3600,
            "Per-condition budget changed")
        self.resources = Resources(c)
        self.owner = owner_pid
        self.deadline = min(self.p["deadline_unix"], time.time() + 3600)
        self.monotonic_deadline = time.monotonic() + max(0., self.deadline - time.time())
        self()

    def __call__(self):
        require(os.getppid() == self.owner, "Owned supervisor exited")
        require(time.time() < self.deadline and time.monotonic() < self.monotonic_deadline, "Refit deadline reached")
        self.resources()


def require_frozen_working_directory(c):
    # The unchanged resolver's first branch resolves admitted relative inputs
    # against cwd, matching the original frozen campaign launch semantics.
    require(Path.cwd().resolve() == Path(c["source"]["root"]).resolve(),
        "Run native data preparation from the pinned frozen source working directory")


def load_job(c, job, data, device):
    require_frozen_working_directory(c)
    import torch
    engine, core, loader, runner = native(c["source"]["root"])
    numerical = importlib.import_module("paper.scripts.quantity_comparison_runtime")
    numerical.configure_runtime(device, threads=4)
    require(Path(sys.executable).resolve() == Path(c["runtime"]["python"]).resolve(), "Unqualified interpreter")
    generator = loader.set_seed(job["seed"])
    del generator
    torch.use_deterministic_algorithms(True, warn_only=False)
    runtime = numerical.runtime_identity(device)
    require(runtime["gpu"]["uuid"] == c["runtime"]["gpu_uuid"], "Wrong native GPU UUID")
    for name, expected in c["runtime"]["runtime_expected"].items():
        require(runtime.get(name) == expected, "Native runtime changed: " + name)
    payload = runner.torch_load_checkpoint(Path(job["checkpoint"]["path"]), map_location="cpu")
    require(engine.validate_checkpoint(payload, ARM) is True, "Wrong native checkpoint route")
    require(payload.get("backbone") == ARM and payload.get("seed") == job["seed"]
        and payload.get("epoch", payload.get("best_epoch")) == job["checkpoint"]["epoch"], "Selected source checkpoint identity mismatch")
    training = importlib.import_module("paper.scripts.count_aware_tpp_backbone.training")
    monitor = training.checkpoint_monitor_spec("validation_raw_quantity_rmse")
    require(payload.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
        and payload.get("checkpoint_monitor_history_key") == monitor["history_key"]
        and payload.get("checkpoint_selection") == monitor["selection"], "Source selector changed")
    actual = runner.canonical_state_dict_sha256(payload["model_state_dict"])
    require(actual == job["checkpoint"]["state_sha256"] == payload.get("model_state_sha256"), "Source model-state SHA changed")
    require(payload["interface_meta"]["time_head"]["observation_likelihood"] == data["model"]["time_observation_contract"], "Source observation law changed")
    require(torch.cuda.is_available() and device == "cuda:0", "Native CUDA qualification required")
    torch.cuda.set_per_process_memory_fraction(c["limits"]["peak_device_fraction_max"], device=0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, _ = engine.build_model(data, ARM)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    freeze(model)
    require(state_sha(model, runner.canonical_state_dict_sha256) == actual, "Loaded state changed")
    frame, receipt = engine.base.prepare_admitted_data(data, data_root=c.get("data_root", c["source"]["root"]))
    require(receipt["held_out_materialized"] is False and receipt["populations"] == data["inherited_data_identity"]["populations"], "Input population changed")
    train_loader = loader.make_loader(frame, target_split="train", **{k: data["loader"][k] for k in ("batch_size", "lookback_weeks", "max_seq_len")}, shuffle=False, generator=None)
    val_loader = loader.make_loader(frame, target_split="validation", **{k: data["loader"][k] for k in ("batch_size", "lookback_weeks", "max_seq_len")}, shuffle=False, generator=None)
    receipt = {**receipt, "refit_runtime": runtime}
    return model, core, runner, train_loader, val_loader, receipt


def _qualify(c, job, data, output, device):
    import torch
    model, core, runner, train_loader, val_loader, receipt = load_job(c, job, data, device)
    digest = runner.canonical_state_dict_sha256
    frozen_sha, whole_sha = state_sha(model, digest, frozen_only=True), state_sha(model, digest)
    started = time.monotonic()
    resources = Resources(c)
    def pulse():
        require(time.monotonic() - started < 1800, "Qualification budget reached")
        resources()
    train, first = extract_cache(model, core, train_loader, device, digest, frozen_sha, pulse, max_batches=2)
    spot_check(model, core, first, device, train, digest, frozen_sha)
    validation, _ = extract_cache(model, core, val_loader, device, digest, frozen_sha, pulse)
    require(len(validation["dt"]) == receipt["populations"]["validation"]["target_count"], "Qualification requires full Validation")
    with torch.no_grad():
        for begin in range(0, len(train["dt"]), 128):
            require(torch.equal(cache_nll(model, train, slice(begin, begin + 128), device).cpu(), train["native_nll"][begin:begin + 128]),
                "Cached kernel differs from native observation NLL")
    baseline = cache_metrics(model, validation, device, threshold=data["quantity_boundaries_all_train_rows"][-1], pulse=pulse)
    verify_baseline(baseline, job["baseline_metrics"])
    for begin in range(0, len(train["dt"]), 128):
        for k in PARAMETERS:
            parameter(model, k).grad = None
        loss = cache_nll(model, train, slice(begin, begin + 128), device).mean()
        loss.backward()
        require(all(parameter(model, k).grad is not None and bool(torch.isfinite(parameter(model, k).grad).all()) for k in PARAMETERS), "Native time-head gradient check failed")
    original = head_state(model)
    try:
        # Synthetic duration/features only; no real-label optimizer update.
        synthetic = {"hidden": torch.arange(128, dtype=model.v_t.weight.dtype).reshape(2, 64) / 128., "dt": torch.tensor([1., 2.])}
        optimizer = torch.optim.Adam([parameter(model, k) for k in PARAMETERS], lr=.001, weight_decay=0.)
        optimizer.zero_grad(set_to_none=True)
        loss = cache_nll(model, synthetic, slice(None), device).mean()
        loss.backward()
        require(all(bool(torch.isfinite(parameter(model, k).grad).all()) for k in PARAMETERS), "Synthetic gradient check failed")
        optimizer.step()
        spot_check(model, core, first, device, train, digest, frozen_sha)
    finally:
        restore_head_state(model, original)
        for k in PARAMETERS:
            parameter(model, k).grad = None
    require(state_sha(model, digest) == whole_sha, "Qualification did not restore the original state")
    result = {"status": "passed", "schema": "time_head_full_refit_native_qualification_v1", "host": "5080", "device": device,
        "contract_sha256": canonical(c), "job": job["id"], "representative_job": c["jobs"][0]["id"],
        "representative_scope": "shared_full130_time_head_runtime_each_fit_has_own_E0_gate",
        "source_files_sha256": SOURCE_CLOSURE, "runtime": receipt["refit_runtime"],
        "checkpoint_sha256": job["checkpoint"]["sha256"],
        "frozen_state_sha256": frozen_sha, "restored_whole_state_sha256": whole_sha,
        "trainable_tensors": list(PARAMETERS), "trainable_scalar_count": PARAMETER_COUNT, "real_optimizer_updates": 0,
        "synthetic_optimizer_updates": 1, "native_train_batches_checked": math.ceil(len(train["dt"]) / 128),
        "E0": baseline, "input_receipt": receipt, "quantity_output_unchanged": True, "no_grad_feature_extraction": True, "inference_mode_used": False,
        "quantity_repeated_forward_equal": True, "native_interval_kernel": True, "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False, "at_unix": time.time()}
    write(output / "receipt.json", result)
    return result


def qualify(c, job, data, output, device):
    require(job == c["jobs"][0], "Qualification must use the admitted representative first job")
    require(not output.exists() or {p.name for p in output.iterdir()} <= {"claim.json", "worker_process.json"},
        "Fresh qualification output required; automatic retry forbidden")
    if (output / "claim.json").exists():
        require(read(output / "claim.json").get("owner_pid") == os.getppid(), "Foreign qualification supervisor claim")
    claim_output(output, {"claim.json", "worker_process.json"}, job=job["id"])
    try:
        return _qualify(c, job, data, output, device)
    except BaseException as exc:
        write(output / "failure.json", {"status": "failed", "job": job["id"], "error": str(exc),
            "exception": type(exc).__name__, "automatic_retry_allowed": False, "at_unix": time.time()})
        raise


def execute(c, job, data, permit, device):
    import torch
    output = Path(job["output_dir"])
    claim_output(output, {"training_permit.json", "worker_process.json"}, job=job["id"])
    started = time.time()
    try:
        model, core, runner, train_loader, val_loader, receipt = load_job(c, job, data, device)
        digest = runner.canonical_state_dict_sha256
        frozen_sha = state_sha(model, digest, frozen_only=True)
        write(output / "input_receipt.json", receipt)
        train, first = extract_cache(model, core, train_loader, device, digest, frozen_sha, permit)
        validation, _ = extract_cache(model, core, val_loader, device, digest, frozen_sha, permit)
        require(len(train["dt"]) == receipt["populations"]["train"]["target_count"]
            and len(validation["dt"]) == receipt["populations"]["validation"]["target_count"], "Cache population changed")
        quantity_before = {s: digest({"prediction": cache["prediction"]}) for s, cache in (("train", train), ("validation", validation))}
        baseline = cache_metrics(model, validation, device, threshold=data["quantity_boundaries_all_train_rows"][-1], pulse=permit)
        verify_baseline(baseline, job["baseline_metrics"])
        write(output / "startup_gate.json", {"status": "passed", "job": job["id"], "contract_sha256": canonical(c),
            "original_checkpoint": job["checkpoint"], "E0": baseline, "input_receipt_sha256": sha(output / "input_receipt.json"),
            "runtime": receipt["refit_runtime"], "frozen_state_sha256": frozen_sha,
            "quantity_prediction_sha256": quantity_before, "evaluation_scope": "validation_only", "held_out_test_evaluated": False})
        for split, cache in (("train", train), ("validation", validation)):
            runner.atomic_torch_save({"scope": split, "source_checkpoint_sha256": job["checkpoint"]["sha256"],
                "frozen_state_sha256": frozen_sha, "cache": cache}, output / f"{split}_feature_cache.pt")
        def record(history, best):
            write(output / "history.json", {"history": history, "evaluation_scope": "validation_only", "held_out_test_evaluated": False})
            write(output / "status.json", {"status": "fitting", "job": job["id"], "epoch": history[-1]["epoch"],
                "best_epoch": choose_best(history)["epoch"], "trainable_scalar_count": PARAMETER_COUNT, "at_unix": time.time()})
        result = fit_cached(model, train, validation, core=core, first_batch=first, device=device, digest=digest,
            frozen_sha=frozen_sha, seed=job["seed"], threshold=data["quantity_boundaries_all_train_rows"][-1], fitting=c["fitting"],
            pulse=permit, on_epoch=record)
        for label in ("selected", "last"):
            runner.atomic_torch_save({"schema": SCHEMA, "job": job["id"], "source_checkpoint": job["checkpoint"],
                "contract_sha256": canonical(c), "frozen_state_sha256": frozen_sha, "head_state": result[label + "_head_state"],
                "epoch": result["best_epoch"] if label == "selected" else result["completed_epochs"],
                "evaluation_scope": "validation_only", "held_out_test_evaluated": False}, output / f"{label}_refit.pt")
        endpoint = {k: result[k] for k in ("E0", "selected", "last", "best_epoch", "completed_epochs", "optimizer_steps")}
        endpoint.update({"evaluation_scope": "validation_only", "held_out_test_evaluated": False,
            "frozen_state_sha256": frozen_sha, "quantity_predictions_exactly_equal": True, "selected_is_identity": result["best_epoch"] == 0,
            "trainable_scalar_count": PARAMETER_COUNT, "selected_head_values": {k: v.reshape(-1).tolist() for k, v in result["selected_head_state"].items()}})
        write(output / "endpoint_replays.json", endpoint)
        quantity_after = {}
        with torch.no_grad():
            for split, cache in (("train", train), ("validation", validation)):
                predicted = torch.cat([model.predict_quantity(cache["hidden"][i:i+128].to(device))[1].cpu()
                    for i in range(0, len(cache["dt"]), 128)])
                quantity_after[split] = digest({"prediction": predicted})
        require(quantity_before == quantity_after, "Full Train/Validation quantity bytehash changed")
        proof = {"schema": SCHEMA, "status": "complete", "job": job["id"], "contract_sha256": canonical(c),
            **endpoint, "quantity_output_unchanged": True, "quantity_prediction_sha256_before": quantity_before,
            "quantity_prediction_sha256_after": quantity_after, "frozen_state_sha256_before": frozen_sha,
            "frozen_state_sha256_after": state_sha(model, digest, frozen_only=True), "trainable_tensors": list(PARAMETERS),
            "original_checkpoint": job["checkpoint"], "source_files_sha256": SOURCE_CLOSURE}
        write(output / "result.json", proof)
        require(sha(job["checkpoint"]["path"]) == job["checkpoint"]["sha256"], "Original input checkpoint changed")
        guard_frozen(model, digest, frozen_sha)
        permit()
        write(output / "status.json", {"status": "complete", "job": job["id"], "best_epoch": result["best_epoch"], "at_unix": time.time()})
        files = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        terminal = {"status": "complete", "scientific_success": True, "contract_sha256": canonical(c), "job": job,
            "files": files, "elapsed_seconds": time.time() - started, "completed_at_unix": time.time(),
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False, "trainable_scalar_count": PARAMETER_COUNT,
            "frozen_state_sha256": frozen_sha, "original_checkpoint_sha_preserved": True}
        write(output / "terminal_manifest.json", terminal)
        return terminal
    except BaseException as exc:
        write(output / "failure.json", {"status": "failed", "job": job["id"], "error": str(exc),
            "exception": type(exc).__name__, "automatic_retry_allowed": False, "at_unix": time.time()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--qualification-output")
    parser.add_argument("--training-permit")
    parser.add_argument("--owner-pid", type=int)
    args = parser.parse_args(argv)
    require(not (args.dry_run and args.qualify), "Choose one mode")
    c = read(args.contract)
    datasets = validate_contract(c)
    job = next((j for j in c["jobs"] if j["id"] == args.job), None)
    require(job is not None, "Job not admitted")
    if args.dry_run:
        result = {"status": "validated", "contract_sha256": canonical(c), "job": job["id"], "data_loaded": False,
            "checkpoint_loaded": False, "device_initialized": False, "optimizer_updates": 0}
    elif args.qualify:
        verify_operational_authorization(c, args.contract)
        output = Path(args.qualification_output) if args.qualification_output else Path(c["root"]) / "qualification"
        require(output.resolve() == (Path(c["root"]) / "qualification").resolve(), "Qualification output escaped owned root")
        result = qualify(c, job, datasets[job["dataset"]], output, c.get("device", "cuda:0"))
    else:
        start = verify_operational_authorization(c, args.contract)
        permit = Permit(c, job, args.training_permit, args.owner_pid)
        require(permit.deadline <= start["deadline_unix"], "Job deadline exceeds campaign permit")
        result = execute(c, job, datasets[job["dataset"]], permit, c.get("device", "cuda:0"))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
