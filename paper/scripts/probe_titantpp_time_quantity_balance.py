"""Read-only MLP16 Train-batch time/quantity gradient diagnostics.

Every loss uses one training-mode forward graph. No optimizer, fitting, clipping,
checkpoint selection, Validation/Test loader, or network call is performed.
Fresh initialization is retained as a control: its zero quantity-head weights
can give graph-connected but numerically zero quantity-to-backbone gradients.
A separately SHA-bound existing selected checkpoint may also be inspected.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARM = "titantpp_history_mlp_width16"
SCHEMA = "titantpp_train_gradient_balance_v1"
CONTRACT = ROOT / "search_artifacts/titantpp_mlp_cap36_20261017_v2/execution_contract.json"
CONTRACT_SHA = "48bfbfeb75085af687b9055d9fe31f6cf1d7c6709ba3d2f37662360380133f4d"
CAPS = {"yellow_trip_hourly": 65., "raf_spare_parts": 10., "intermittent_frozen_5000": 20.}
SEEDS = (42, 52, 62)
QUANTITY_LOSSES = ("existing_log_mse", "cap_normalized_raw_mse", "cap_normalized_raw_mae")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def state_sha(model):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    return canonical_state_dict_sha256(model.state_dict())


def capture_rng():
    # CPU diagnostics must not initialize a CUDA device just to inspect RNG.
    return {"python": random.getstate(), "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
        "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def rng_sha(state):
    digest = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    digest.update(state["torch"].cpu().numpy().tobytes())
    if state["cuda"] is not None:
        for tensor in state["cuda"]:
            digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def preserve_rng():
    before = capture_rng()
    try:
        yield before
    finally:
        restore_rng(before)
        require(rng_sha(capture_rng()) == rng_sha(before), "Process RNG restoration failed")


def capture_context(model):
    """MLP16's unregistered episodic tensors are the attention _ctx_mem slots."""
    result = {}
    for name, module in model.named_modules():
        if hasattr(module, "_ctx_mem"):
            value = module._ctx_mem
            require(value is None or isinstance(value, torch.Tensor), "Unknown episodic context type")
            result[name] = None if value is None else value.detach().clone()
    return result


def context_equal(left, right):
    return left.keys() == right.keys() and all(
        (left[k] is None and right[k] is None) or
        (isinstance(left[k], torch.Tensor) and isinstance(right[k], torch.Tensor)
         and torch.equal(left[k], right[k])) for k in left)


def restore_context(model, snapshot):
    modules = dict(model.named_modules())
    for name, value in snapshot.items():
        modules[name]._ctx_mem = None if value is None else value.clone()


def _norm(gradients, indices):
    total = 0.
    for i in indices:
        gradient = gradients[i]
        if gradient is not None:
            require(bool(torch.isfinite(gradient).all()), "Nonfinite gradient")
            total += float(gradient.detach().to(dtype=torch.float64).square().sum().cpu())
    require(math.isfinite(total), "Nonfinite gradient squared norm")
    return math.sqrt(total)


def _dot(left, right, indices):
    value = 0.
    for i in indices:
        if left[i] is not None and right[i] is not None:
            value += float((left[i].detach().to(dtype=torch.float64)
                * right[i].detach().to(dtype=torch.float64)).sum().cpu())
    require(math.isfinite(value), "Nonfinite gradient dot product")
    return value


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def compare_gradients(names, parameters, time_gradients, quantity_gradients, time_loss, quantity_loss):
    require(len(names) == len(parameters) == len(time_gradients) == len(quantity_gradients),
        "Gradient inventory lengths differ")
    indices = list(range(len(names)))
    for gradients in (time_gradients, quantity_gradients):
        for parameter, gradient in zip(parameters, gradients):
            require(gradient is None or gradient.shape == parameter.shape, "Gradient shape mismatch")
            require(gradient is None or bool(torch.isfinite(gradient).all()), "Nonfinite gradient")
    tn = [_norm(time_gradients, [i]) for i in indices]
    qn = [_norm(quantity_gradients, [i]) for i in indices]
    groups = {
        "shared_graph": [i for i in indices if time_gradients[i] is not None and quantity_gradients[i] is not None],
        "time_only_graph": [i for i in indices if time_gradients[i] is not None and quantity_gradients[i] is None],
        "quantity_only_graph": [i for i in indices if time_gradients[i] is None and quantity_gradients[i] is not None],
        "inactive_none": [i for i in indices if time_gradients[i] is None and quantity_gradients[i] is None]}
    require(sum(map(len, groups.values())) == len(names), "Parameter classification incomplete")
    shared = groups["shared_graph"]
    groups.update(shared_nonzero_both=[i for i in shared if tn[i] > 0 and qn[i] > 0],
        shared_time_nonzero_quantity_zero=[i for i in shared if tn[i] > 0 and qn[i] == 0],
        shared_time_zero_quantity_nonzero=[i for i in shared if tn[i] == 0 and qn[i] > 0],
        shared_zero_both=[i for i in shared if tn[i] == qn[i] == 0])

    def describe(ix):
        t, q, dot = _norm(time_gradients, ix), _norm(quantity_gradients, ix), _dot(time_gradients, quantity_gradients, ix)
        cosine = max(-1., min(1., dot / (t * q))) if t > 0 and q > 0 else None
        joint2 = t * t + q * q + 2 * dot
        require(joint2 >= -1e-10 * max(1., t * t + q * q), "Invalid joint-gradient norm")
        joint = math.sqrt(max(0., joint2))
        return {"parameter_tensor_count": len(ix), "parameter_element_count": sum(parameters[i].numel() for i in ix),
            "parameter_names": [names[i] for i in ix], "time_gradient_norm": t,
            "quantity_gradient_norm": q, "time_to_quantity_gradient_norm_ratio": _safe_ratio(t, q),
            "quantity_to_time_gradient_norm_ratio": _safe_ratio(q, t), "gradient_dot_product": dot,
            "gradient_cosine": cosine, "cosine_status": "available" if cosine is not None else "undefined_zero_norm",
            "lambda_quantity_for_norm_equality": t / q if t > 0 and q > 0 else None,
            "norm_equality_status": "descriptive_only" if t > 0 and q > 0 else "undefined_zero_norm",
            "lambda1_joint_gradient_norm": joint}

    result = {"groups": {name: describe(ix) for name, ix in groups.items()}, "global": describe(indices),
        "loss_scalar": {"time": float(time_loss), "quantity": float(quantity_loss),
            "time_to_quantity_scalar_ratio": _safe_ratio(float(time_loss), float(quantity_loss))},
        "parameter_norms": [{"name": name, "shape": list(parameter.shape),
            "elements": parameter.numel(), "time_graph_present": time_gradients[i] is not None,
            "quantity_graph_present": quantity_gradients[i] is not None,
            "time_gradient_norm": tn[i], "quantity_gradient_norm": qn[i]}
            for i, (name, parameter) in enumerate(zip(names, parameters))]}
    joint = result["global"]["lambda1_joint_gradient_norm"]
    result["global"].update(clip_threshold=1., lambda1_joint_norm_exceeds_clip1=joint > 1.,
        descriptive_clip1_scale=min(1., 1. / (joint + 1e-6)), actual_clipping_applied=False)
    return result


def loss_components(value, cap):
    require(math.isfinite(float(cap)) and float(cap) > 0, "Cap must be finite and positive")
    required = ("time_loss", "quantity_train_loss", "log_qty_loss", "pred_qty", "true_qty")
    require(all(k in value for k in required), "Missing forward outputs")
    shape = value["true_qty"].shape
    require(len(shape) == 1 and shape[0] > 0, "Expected nonempty per-target output vectors")
    for key in required:
        require(value[key].shape == shape and bool(torch.isfinite(value[key]).all()),
            "Nonfinite or mismatched loss/output vector: " + key)
    require(bool(torch.equal(value["quantity_train_loss"], value["log_qty_loss"])), "Active quantity loss is not existing log-MSE")
    require(bool((value["true_qty"] >= 0).all() and (value["true_qty"] <= cap).all()), "Target is outside fixed Cap task")
    normalized = (value["pred_qty"] - value["true_qty"]) / float(cap)
    return {"time": value["time_loss"].mean(), "existing_log_mse": value["log_qty_loss"].mean(),
        "cap_normalized_raw_mse": normalized.square().mean(),
        "cap_normalized_raw_mae": normalized.abs().mean()}


def analyze_model_batch(model, dts, mask, quantities, *, cap, forward=None):
    """One training forward, four autograd calls; no .backward or .grad writes.

    Training dropout consumes RNG normally. Gradients consume no additional RNG.
    The caller's condition-level preserve_rng scope restores the entire stream.
    """
    if forward is None:
        from paper.scripts.count_aware_tpp_backbone.core import target_outputs
        forward = target_outputs
    before_state, before_context = state_sha(model), capture_context(model)
    modes = [(module, module.training) for module in model.modules()]
    entries = list(model.named_parameters())
    names, parameters = [x[0] for x in entries], [x[1] for x in entries]
    require(bool(entries), "No parameters to diagnose")
    flags = [p.requires_grad for p in parameters]
    before_grad = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    rng_before = rng_sha(capture_rng())
    try:
        model.train()
        for parameter in parameters:
            parameter.requires_grad_(True)
        value = forward(model, dts, mask, quantities, lambda_log_qty=1.)
        losses = loss_components(value, cap)
        rng_after_forward = rng_sha(capture_rng())
        gradients = {}
        for index, (name, loss) in enumerate(losses.items()):
            require(bool(torch.isfinite(loss)), "Nonfinite scalar loss: " + name)
            require(loss.requires_grad, "Loss has no gradient graph: " + name)
            gradients[name] = torch.autograd.grad(loss, parameters,
                allow_unused=True, retain_graph=index < len(losses) - 1, create_graph=False)
        require(rng_sha(capture_rng()) == rng_after_forward, "Gradient computation consumed RNG")
        comparisons = {name: compare_gradients(names, parameters, gradients["time"], gradients[name],
            losses["time"].detach().cpu(), losses[name].detach().cpu()) for name in QUANTITY_LOSSES}
        require(state_sha(model) == before_state, "Probe changed parameter/buffer tensor values")
        require(context_equal(before_context, capture_context(model)), "Probe changed episodic context")
        for parameter, previous in zip(parameters, before_grad):
            require((parameter.grad is None and previous is None) or
                (parameter.grad is not None and previous is not None and torch.equal(parameter.grad, previous)),
                "Probe populated or changed a parameter .grad")
        return {"comparisons": comparisons, "loss_means": {k: float(v.detach().cpu()) for k, v in losses.items()},
            "forward_count": 1, "autograd_grad_calls": 4, "training_mode": True,
            "dropout_rng_consumed_by_forward": rng_before != rng_after_forward,
            "rng_before_forward_sha256": rng_before, "rng_after_forward_sha256": rng_after_forward,
            "rng_unchanged_by_gradient_computation": True,
            "state_sha256_before": before_state, "state_sha256_after": state_sha(model),
            "parameter_buffer_values_unchanged": True, "parameter_grad_fields_unchanged": True,
            "episodic_context_unchanged": True,
            "episodic_context_fields": {k + "._ctx_mem": "None" if v is None else list(v.shape) for k, v in before_context.items()}}
    finally:
        restore_context(model, before_context)
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
        for module, flag in modes:
            module.training = flag


def validate_contract(contract, dataset, seed):
    from paper.scripts import run_titantpp_mlp_cap36 as cap36
    require(dataset in CAPS and seed in SEEDS, "Only approved Cap dataset/seeds are admitted")
    matches = [j for j in contract["jobs"] if (j["dataset"], j["arm"], j["seed"]) == (dataset, ARM, seed)]
    require(len(matches) == 1, "Expected exactly one existing MLP16 condition")
    job, data, _, _ = cap36.select_job(contract, matches[0]["host"], matches[0]["id"])
    require(data["cap_value"] == CAPS[dataset], "Fixed Cap value changed")
    require(data["optimizer"] == {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.}, "Original optimizer metadata changed")
    loader = data["loader"]
    require(loader["batch_size"] == 128 and loader["num_workers"] == 0
        and loader["drop_last"] is False and loader["train_shuffle"] is True, "Original Train loader changed")
    return job, data


def load_selected(model, path, *, expected_sha, expected_state_sha, dataset, seed, expected_epoch=None, data=None):
    """Only explicitly bound existing Cap36 MLP16 selection; never reselect."""
    from paper.scripts import run_titantpp_mlp_cap36 as cap36
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint
    path = Path(path).resolve()
    require(expected_sha and sha_file(path) == expected_sha, "Selected checkpoint byte SHA mismatch")
    cp = torch_load_checkpoint(path, map_location="cpu")
    require(cp.get("backbone") == ARM and cp.get("seed") == seed
        and cp.get("variant") == cap36.VARIANT and cp.get("checkpoint_monitor") == cap36.MONITOR
        and cp.get("evaluation_scope") == "validation_only" and cp.get("held_out_test_evaluated") is False,
        "Selected checkpoint model/seed/loss/scope/selection mismatch")
    require(type(cp.get("best_epoch")) is int and cp["best_epoch"] > 0, "Missing selected epoch")
    require(expected_epoch is None or cp["best_epoch"] == expected_epoch, "Selected epoch binding mismatch")
    resume = cp.get("resume_identity", {})
    args = resume.get("arguments", {})
    require(args.get("dataset_contract") == dataset, "Selected checkpoint dataset mismatch")
    if data is not None:
        require(args.get("data_sha256") == data["inherited_data_identity"]["data"]["sha256"], "Selected checkpoint input SHA mismatch")
        for key in ("time_head_mode", "time_scale", "time_sigma_floor", "lambda_log_qty", "lambda_tail"):
            require(args.get(key) == data["model"][key], "Selected checkpoint head/loss config mismatch: " + key)
    validate_checkpoint_route(cp, ARM)
    digest = canonical_state_dict_sha256(cp["model_state_dict"])
    require(digest == cp.get("model_state_sha256"), "Selected checkpoint internal tensor SHA mismatch")
    require(expected_state_sha is not None and digest == expected_state_sha, "Selected checkpoint bound tensor SHA mismatch")
    model.load_state_dict(cp["model_state_dict"], strict=True)
    require(state_sha(model) == digest and sha_file(path) == expected_sha, "Selected checkpoint load changed identity")
    return {"path": str(path), "sha256": expected_sha, "state_sha256": digest,
        "selected_epoch": cp["best_epoch"], "selection": cap36.MONITOR, "reselected": False}


class RecordingBatchSampler:
    def __init__(self, source):
        self.source, self.batches = source, []

    def __iter__(self):
        for batch in self.source:
            indices = list(map(int, batch))
            self.batches.append(indices)
            yield indices

    def __len__(self):
        return len(self.source)


def recording_loader(loader):
    """Record unchanged RandomSampler batches, including DataLoader base-seed draw."""
    from torch.utils.data import DataLoader
    sampler = RecordingBatchSampler(loader.batch_sampler)
    result = DataLoader(loader.dataset, batch_sampler=sampler, collate_fn=loader.collate_fn,
        generator=loader.generator, num_workers=0)
    return result, sampler


def batch_identity(dataset, indices, batch):
    _, dts, mask, parts, quantities = batch
    require(len(indices) == len(dts) == len(parts) == len(quantities), "Batch target count mismatch")
    rows = []
    for offset, index in enumerate(indices):
        part, previous = dataset.index[index]
        target = previous + 1
        require(dataset.split_lists[part][target] == "train", "Non-Train target in diagnostic")
        require(int(parts[offset]) == part, "Batch entity index mismatch")
        true_q = float(dataset.val_lists[part][target])
        # The original collator left-pads and always leaves the next target last.
        require(bool(mask[offset, -1]) and float(quantities[offset, -1]) == true_q,
            "Batch truth/index mismatch")
        rows.append({"target_index": index, "entity_id": str(dataset.parts[part]),
            "target_seq": int(dataset.seq_lists[part][target]), "raw_quantity": true_q})
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    return {"targets": rows, "target_ids_sha256": sha_json(rows), "batch_size": len(rows),
        "batch_tensors_sha256": canonical_state_dict_sha256({"dts": dts, "mask": mask,
            "part_ids": parts, "quantities": quantities}), "padded_shape": list(dts.shape)}


def run_probe(args):
    from paper.scripts import run_titantpp_mlp_cap36 as cap36
    from paper.scripts.quantity_comparison_runtime import configure_runtime, runtime_identity
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader, set_seed
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    import polars as pl
    require(1 <= args.num_batches <= 3, "Diagnostic is limited to the first one to three Train batches")
    require(args.initialization in ("fresh", "selected"), "Unknown initialization")
    require(args.initialization != "selected" or (args.checkpoint and args.checkpoint_sha and args.checkpoint_state_sha),
        "Selected mode requires checkpoint byte and tensor SHA bindings")
    require(args.initialization != "fresh" or not any((args.checkpoint, args.checkpoint_sha, args.checkpoint_state_sha, args.selected_epoch)),
        "Fresh mode cannot load a checkpoint")
    output = Path(args.output).resolve()
    require(not output.exists(), "Output already exists; refusing overwrite")
    contract_path = Path(args.contract).resolve()
    require(sha_file(contract_path) == args.contract_sha, "Cap36 contract byte SHA mismatch")
    contract = json.loads(contract_path.read_text())
    job, data = validate_contract(contract, args.dataset, args.seed)
    root = Path(args.input_root).resolve() if args.input_root else contract_path.parent / "input_snapshot"
    binding = data["inherited_data_identity"]["data"]
    input_path = (root / binding["path"]).resolve()
    require(input_path.is_relative_to(root) and input_path.name == "train_validation.parquet", "Input must be the pure Train/Validation snapshot")
    require(sha_file(input_path) == binding["sha256"], "Cap Train source SHA mismatch")
    configure_runtime(args.device, 4)
    runtime = runtime_identity(args.device)
    # Only Train records are selected from the pure Train/Validation snapshot.
    train_frame = pl.scan_parquet(input_path).filter(pl.col("chronological_split") == "train").collect()
    require(set(train_frame["chronological_split"].unique().to_list()) == {"train"}, "Non-Train frame")
    require(train_frame["demand_qty"].is_finite().all() and train_frame["demand_qty"].min() >= 0
        and train_frame["demand_qty"].max() <= data["cap_value"], "Invalid Cap Train quantities")
    frame = prepare_count_frame(train_frame.sort(["oper_part_no", "seq"]))
    source_before = sha_file(__file__)
    rng_before = rng_sha(capture_rng())
    with preserve_rng():
        generator = set_seed(args.seed)
        # set_seed's historical warn_only setting is tightened back to the pinned runtime.
        torch.use_deterministic_algorithms(True, warn_only=False)
        loader = make_loader(frame, target_split="train", shuffle=True, generator=generator,
            **{k: data["loader"][k] for k in ("batch_size", "lookback_weeks", "max_seq_len")})
        _, identity = cap36.population_identity(loader.dataset, "train")
        cap36.verify_population(identity, data["inherited_data_identity"]["populations"]["train"])
        model, metadata = cap36.build_model(data, ARM)
        require(getattr(model, "history_mlp_width", None) == 16, "Wrong MLP width")
        require(all(value is None for value in capture_context(model).values()), "Expected empty episodic context")
        fresh_state = state_sha(model)
        checkpoint = None
        if args.initialization == "selected":
            checkpoint = load_selected(model, args.checkpoint, expected_sha=args.checkpoint_sha,
                expected_state_sha=args.checkpoint_state_sha, expected_epoch=args.selected_epoch,
                dataset=args.dataset, seed=args.seed, data=data)
        model = model.to(args.device)
        initial_state = state_sha(model)
        head_zero = bool((model.quantity_head.weight.detach() == 0).all())
        generator_start = generator.get_state().clone()
        loader, sampler = recording_loader(loader)
        batches = []
        iterator = iter(loader)
        for number in range(args.num_batches):
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise ValueError("Insufficient Train batches") from error
            batch_ids = batch_identity(loader.dataset, sampler.batches[number], batch)
            _, dts, mask, _, quantities = batch
            diagnostic = analyze_model_batch(model, dts.to(args.device), mask.to(args.device),
                quantities.to(args.device), cap=data["cap_value"])
            batches.append({"batch_number_zero_based": number, **batch_ids, **diagnostic})
        require(state_sha(model) == initial_state, "Condition changed model state")
        require(all(value is None for value in capture_context(model).values()), "Condition changed episodic state")
        require(checkpoint is None or sha_file(checkpoint["path"]) == checkpoint["sha256"], "Checkpoint bytes changed during probe")
        generator_end = generator.get_state().clone()
    require(rng_sha(capture_rng()) == rng_before, "Entire diagnostic failed to restore RNG")
    require(sha_file(__file__) == source_before and sha_file(input_path) == binding["sha256"]
        and sha_file(contract_path) == args.contract_sha, "Input/source changed during probe")
    result = {"schema": SCHEMA, "status": "complete", "dataset": args.dataset, "seed": args.seed,
        "arm": ARM, "initialization": args.initialization, "checkpoint": checkpoint,
        "cap_value": data["cap_value"], "fresh_initial_state_sha256": fresh_state,
        "probed_state_sha256": initial_state, "quantity_head_weight_all_zero": head_zero,
        "model_metadata": metadata, "runtime": runtime, "training_mode": True,
        "batch_size": 128, "maximum_sequence_length": data["loader"]["max_seq_len"],
        "train_target_identity": identity, "batch_sampling": "first N batches of original Train shuffled DataLoader at the fixed seed; no optimizer step",
        "train_loader_generator_start_sha256": hashlib.sha256(generator_start.numpy().tobytes()).hexdigest(),
        "train_loader_generator_end_sha256": hashlib.sha256(generator_end.numpy().tobytes()).hexdigest(),
        "batch_count": len(batches), "batches": batches,
        "rng_before_probe_sha256": rng_before, "rng_after_probe_sha256": rng_sha(capture_rng()),
        "process_rng_restored": True, "model_parameters_buffers_context_unchanged": True,
        "contract_path": str(contract_path), "contract_raw_sha256": args.contract_sha,
        "contract_canonical_sha256": cap36.sha_json(contract), "train_input_path": str(input_path),
        "train_input_sha256": binding["sha256"], "source_sha256": source_before,
        "loss_definitions": {"time": "mean(existing recorded positive-integer duration negative log likelihood)",
            "existing_log_mse": "mean(existing quantity head log1p MSE)",
            "cap_normalized_raw_mse": "mean(((existing point prediction - true quantity)/fixed cap)^2)",
            "cap_normalized_raw_mae": "mean(abs(existing point prediction - true quantity)/fixed cap)"},
        "new_training": False, "optimizer_steps": 0, "actual_gradient_clipping": False,
        "validation_loader_created": False, "validation_records_selected": 0, "test_records_opened": 0,
        "network_calls": 0, "historical_cnn_accessed": False,
        "interpretation_limits": ["Fixed-weight local Train-batch gradient diagnostic, not training or loss selection.",
            "Graph presence includes zero-valued gradients; shared_nonzero_both separately identifies actual two-sided transmission.",
            "Fresh quantity head is zero initialized and may block quantity gradients to the shared backbone at initialization.",
            "Selected mode reads a previously fixed checkpoint and does not reselect it; fresh seeded dropout/sampling RNG is used rather than checkpoint historical RNG.",
            "Scalar loss ratios and gradient norm ratios are different quantities. Norm equality lambda values are descriptive and do not recommend objective weights.",
            "Training dropout consumes RNG inside the probe; autograd calls consume no extra RNG and the process RNG is restored at exit.",
            "MAE uses PyTorch abs subgradient, equal to zero at exact prediction equality.",
            "This probe neither establishes a causal performance explanation nor estimates Validation/Test improvements."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--contract-sha", default=CONTRACT_SHA)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--dataset", required=True, choices=tuple(CAPS))
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--num-batches", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--initialization", choices=("fresh", "selected"), default="fresh")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha")
    parser.add_argument("--checkpoint-state-sha")
    parser.add_argument("--selected-epoch", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    result = run_probe(parse_args(argv))
    print(json.dumps({k: result[k] for k in ("status", "dataset", "seed", "initialization", "batch_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
