"""Fresh Cap MLP16 quantity-only fits, with observed time inputs retained.

The existing time head is initialized normally from Train statistics, but is
not trained by time loss. Its NLL is diagnostic, not a jointly optimized score.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts import run_titantpp_time_weight_mlp18 as sealed_weight

SEALED_WORKER_SHA = "0e8e9af6e89595235bf6c052f6a762a4427027825487157239485a04625a5ebc"
_SEALED_PATH = Path(sealed_weight.__file__).resolve()
if hashlib.sha256(_SEALED_PATH.read_bytes()).hexdigest() != SEALED_WORKER_SHA:
    raise ValueError("Sealed time-weight worker source changed")
# Function globals must belong to a private namespace; never relabel the loaded
# alpha0.1/10 worker or change its source, constants, or functions.
_spec = importlib.util.spec_from_file_location(__name__ + "_private_adapter", _SEALED_PATH)
_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adapter)

ARMS, SEEDS, DATASETS = sealed_weight.ARMS, sealed_weight.SEEDS, sealed_weight.DATASETS
TIME_ALPHAS = (0.,)
SCHEMA = "titantpp_time_zero_mlp9_stage_v1"
CONTRACT_SCHEMA = "titantpp_time_zero_mlp9_v1"
VARIANT, MONITOR, INACTIVE_TAIL = sealed_weight.VARIANT, sealed_weight.MONITOR, sealed_weight.INACTIVE_TAIL
require, sha_file, sha_json, read = sealed_weight.require, sealed_weight.sha_file, sealed_weight.sha_json, sealed_weight.read
write_exclusive, verify_inventory = sealed_weight.write_exclusive, sealed_weight.verify_inventory
file_inventory, resolve_under = sealed_weight.file_inventory, sealed_weight.resolve_under
population_identity, verify_population = sealed_weight.population_identity, sealed_weight.verify_population
MetricAccumulator, compare_metrics, compare_prediction_files = sealed_weight.MetricAccumulator, sealed_weight.compare_metrics, sealed_weight.compare_prediction_files
build_model, fabricated_data = sealed_weight.build_model, sealed_weight.fabricated_data
inherited = _adapter.inherited
_ORIGINAL_IDENTITY_VALIDATOR = _adapter.validate_weight_binding
_ORIGINAL_ANNOTATE_ROW = _adapter.annotate_history_row
_ORIGINAL_ANNOTATE_PAYLOAD = _adapter.annotate_payload
_ORIGINAL_BASE_RECEIPT = _adapter.base_receipt
_ORIGINAL_VERIFY_TRAIN = _adapter.verify_train_receipt
_ORIGINAL_LOAD_SELECTED = _adapter._load_selected
_ORIGINAL_EVALUATE_SELECTED = _adapter.evaluate_selected
_TRACKER = None
TIME_HEAD_NAMES = ("v_t.weight", "b_t", "w_raw", "time_scale_weight.weight")


def weight_identity(alpha):
    require(type(alpha) in (float, int) and float(alpha) == 0., "Only quantity-only alpha0 is admitted")
    return {"schema": "titantpp_time_zero_identity_v1", "time_alpha": 0., "quantity_lambda": 1.,
        "objective_mode": "quantity_only", "quantity_only": True, "time_head_trained": False,
        "joint_formula": "existing quantity_train_loss tensor directly; no time-loss graph contribution",
        "quantity_variant": VARIANT, "checkpoint_monitor": MONITOR,
        "time_inputs_retained": True, "model_head_initialization_optimizer_budget_unchanged": True,
        "time_nll_role": "diagnostic from time head not optimized by time loss; shared representation may change"}


def make_weighted_target_outputs(original, alpha):
    weight_identity(alpha)
    def quantity_only(*args, **kwargs):
        import torch
        require(kwargs.get("lambda_log_qty") == 1., "Existing quantity weight must remain one")
        value = original(*args, **kwargs)
        require(torch.equal(value["quantity_train_loss"], value["log_qty_loss"]), "Existing quantity loss changed")
        result = dict(value)
        result["unweighted_joint_loss"] = value["joint_loss"]
        # In particular, do not use 0*time_loss: that would create zero .grad
        # tensors and allow AdamW decay/state updates in the time-only head.
        result["joint_loss"] = value["quantity_train_loss"]
        return result
    return quantity_only


def _time_head_sha(state, names=TIME_HEAD_NAMES):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(set(names).issubset(state), "Missing time-head tensor")
    return canonical_state_dict_sha256({name: state[name] for name in names})


class TimeHeadGuard:
    """Verify the actual fit's every AdamW step, not only fabricated tests."""
    def __init__(self, model):
        entries = tuple(model.time_head_named_parameters())
        require(tuple(name for name, _ in entries) == TIME_HEAD_NAMES, "Unexpected time-head inventory")
        all_parameters = dict(model.named_parameters())
        require(all(all_parameters.get(name) is p for name, p in entries), "Time-head parameter names changed")
        self.model, self.entries, self.optimizer = model, entries, None
        self.initial_sha = _time_head_sha(model.state_dict())
        self.steps = 0

    def check(self):
        require(all(parameter.grad is None for _, parameter in self.entries), "Time-only head received a gradient")
        require(_time_head_sha(self.model.state_dict()) == self.initial_sha, "Time-head tensor changed")
        if self.optimizer is not None:
            require(all(parameter not in self.optimizer.state for _, parameter in self.entries),
                "AdamW created time-head optimizer state")

    def bind_optimizer(self, optimizer):
        require(self.optimizer is None, "Duplicate optimizer binding")
        # The engine moves the model to its device after construction. Bind
        # the live Parameter objects afterward, even if a conversion replaced
        # Parameter identities while preserving the initial tensor values.
        live = tuple(self.model.time_head_named_parameters())
        require(tuple(name for name, _ in live) == TIME_HEAD_NAMES
            and _time_head_sha(self.model.state_dict()) == self.initial_sha,
            "Time-head identity/value changed before optimizer creation")
        self.entries = live
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        require(all(id(parameter) in optimizer_ids for _, parameter in live), "Optimizer omits the live time-head parameters")
        self.optimizer = optimizer
        original_step = optimizer.step
        self.original_step = original_step
        def checked_step(*args, **kwargs):
            self.check()
            value = original_step(*args, **kwargs)
            self.check()
            self.steps += 1
            return value
        optimizer.step = checked_step
        self.check()
        return optimizer

    def proof(self, payload=None):
        self.check()
        require(self.optimizer is not None and self.steps > 0, "No actual optimizer steps verified")
        if payload is not None:
            for key in ("model_state_dict", "best_state_dict"):
                if key in payload:
                    require(_time_head_sha(payload[key]) == self.initial_sha, "Checkpoint time-head tensor changed")
            if "optimizer_state_dict" in payload:
                saved = payload["optimizer_state_dict"]
                head_ids = {id(p) for _, p in self.entries}
                require(len(saved["param_groups"]) == len(self.optimizer.param_groups), "Optimizer groups changed")
                for live_group, saved_group in zip(self.optimizer.param_groups, saved["param_groups"]):
                    require(len(live_group["params"]) == len(saved_group["params"]), "Optimizer parameter count changed")
                    for parameter, state_index in zip(live_group["params"], saved_group["params"]):
                        if id(parameter) in head_ids:
                            require(state_index not in saved["state"], "Serialized time-head optimizer state exists")
        return {"schema": "time_head_quantity_only_verification_v1", "initial_time_head_sha256": self.initial_sha,
            "final_time_head_sha256": self.initial_sha, "checkpoint_time_head_sha256": self.initial_sha,
            "time_head_parameter_names": list(TIME_HEAD_NAMES),
            "time_head_parameter_count": sum(p.numel() for _, p in self.entries),
            "optimizer_steps_checked": self.steps, "all_time_head_gradients_none": True,
            "optimizer_state_absent": True, "time_head_parameters_unchanged": True,
            "verification_scope": "actual optimizer updates checked through artifact serialization; selected state checked when serialized",
            "step_count_is_selection_epoch": False,
            "time_head_initialization": "unchanged existing Train-statistic initialization",
            "time_nll_is_diagnostic": True}


def verify_time_head_proof(proof, state=None):
    require(isinstance(proof, dict) and proof.get("schema") == "time_head_quantity_only_verification_v1",
        "Missing actual fit time-head verification")
    digest = proof.get("initial_time_head_sha256")
    require(isinstance(digest, str) and len(digest) == 64
        and proof.get("final_time_head_sha256") == proof.get("checkpoint_time_head_sha256") == digest
        and proof.get("time_head_parameter_names") == list(TIME_HEAD_NAMES)
        and type(proof.get("optimizer_steps_checked")) is int and proof["optimizer_steps_checked"] > 0
        and all(proof.get(key) is True for key in ("all_time_head_gradients_none", "optimizer_state_absent", "time_head_parameters_unchanged")),
        "Time-head no-training proof mismatch")
    if state is not None:
        require(_time_head_sha(state) == digest, "Selected time-head state differs from verified initial state")
    return proof


def verify_checkpoint_fit_proof(checkpoint_proof, fit_proof):
    """A checkpoint may be serialized before the final fit-level receipt."""
    verify_time_head_proof(checkpoint_proof)
    verify_time_head_proof(fit_proof)
    require(checkpoint_proof["optimizer_steps_checked"] <= fit_proof["optimizer_steps_checked"],
        "Checkpoint proof extends beyond the completed fit")
    without_count = lambda value: {key: item for key, item in value.items() if key != "optimizer_steps_checked"}
    require(without_count(checkpoint_proof) == without_count(fit_proof),
        "Checkpoint/fit time-head state or verification policy mismatch")


def validate_weight_binding(payload, alpha):
    _ORIGINAL_IDENTITY_VALIDATOR(payload, alpha)
    require(payload.get("quantity_only") is True and payload.get("time_head_trained") is False,
        "Quantity-only metadata missing")
    return weight_identity(alpha)


def annotate_history_row(row, alpha):
    row = _ORIGINAL_ANNOTATE_ROW(row, alpha)
    row.update(quantity_only=True, time_head_trained=False, time_nll_is_diagnostic=True)
    return row


def annotate_payload(payload, alpha):
    payload.update(quantity_only=True, time_head_trained=False, time_nll_is_diagnostic=True)
    if _TRACKER is not None:
        payload["time_head_no_training_verification"] = _TRACKER.proof(payload)
    return _ORIGINAL_ANNOTATE_PAYLOAD(payload, alpha)


def training_args(contract, data, output, arm, *, time_alpha):
    require(arm in ARMS and time_alpha == 0., "Only MLP16 quantity-only fits are admitted")
    args = _adapter._BASE_TRAINING_ARGS(contract, data, output, arm)
    args.time_alpha, args.execution_role = 0., "fresh_time_zero_mlp9_validation"
    return args


def base_receipt(contract, host, job, stage):
    value = _ORIGINAL_BASE_RECEIPT(contract, host, job, stage)
    value.update(quantity_only=True, time_head_trained=False, time_nll_is_diagnostic=True)
    if stage == "train":
        require(_TRACKER is not None, "Actual training time-head guard is missing")
        value["time_head_no_training_verification"] = _TRACKER.proof()
    else:
        _, _, root, folder = select_job(contract, host, job["id"])
        reference = read(folder / "train_receipt.json")
        value["time_head_no_training_verification"] = verify_time_head_proof(reference.get("time_head_no_training_verification"))
    return value


def verify_train_receipt(contract, host, job, root, receipt):
    _ORIGINAL_VERIFY_TRAIN(contract, host, job, root, receipt)
    # The stage receipt binds its full interface through summary; unlike a
    # checkpoint/summary it does not duplicate interface_meta at the top level.
    require(receipt.get("quantity_only") is True and receipt.get("time_head_trained") is False,
        "Quantity-only stage receipt labels missing")
    proof = verify_time_head_proof(receipt.get("time_head_no_training_verification"))
    require(receipt["summary"].get("time_head_no_training_verification") == proof,
        "Summary/receipt time-head proof mismatch")


def _load_selected(data, job, train_receipt, root, device):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    proof = verify_time_head_proof(train_receipt.get("time_head_no_training_verification"))
    path = resolve_under(root, train_receipt["selected_checkpoint"])
    require(sha_file(path) == train_receipt["checkpoint_sha256"], "Selected checkpoint bytes changed")
    cp = torch_load_checkpoint(path, map_location="cpu")
    verify_time_head_proof(cp.get("time_head_no_training_verification"), cp["model_state_dict"])
    verify_checkpoint_fit_proof(cp["time_head_no_training_verification"], proof)
    return _ORIGINAL_LOAD_SELECTED(data, job, train_receipt, root, device)


def evaluate_selected(*args, **kwargs):
    return _ORIGINAL_EVALUATE_SELECTED(*args, **kwargs)


_adapter.SCHEMA, _adapter.CONTRACT_SCHEMA, _adapter.TIME_ALPHAS = SCHEMA, CONTRACT_SCHEMA, TIME_ALPHAS
_adapter.weight_identity = weight_identity
_adapter.make_weighted_target_outputs = make_weighted_target_outputs
_adapter.validate_weight_binding = validate_weight_binding
_adapter.annotate_history_row = annotate_history_row
_adapter.annotate_payload = annotate_payload
_adapter.training_args = lambda *args, **kwargs: training_args(*args, **kwargs)
_adapter.base_receipt = base_receipt
_adapter.verify_train_receipt = verify_train_receipt
_adapter._load_selected = _load_selected
_adapter.evaluate_selected = lambda *args, **kwargs: evaluate_selected(*args, **kwargs)


def select_job(contract, host, job_id):
    return _adapter.select_job(contract, host, job_id)


@contextmanager
def runtime_hooks(alpha=0.):
    global _TRACKER
    weight_identity(alpha)
    require(sealed_weight._ACTIVE_ALPHA is None and _TRACKER is None, "Concurrent/nested objective hooks are forbidden")
    require(sha_file(_SEALED_PATH) == SEALED_WORKER_SHA, "Sealed worker changed before entry")
    original_globals = (sealed_weight.SCHEMA, sealed_weight.CONTRACT_SCHEMA, sealed_weight.TIME_ALPHAS,
        sealed_weight.weight_identity, sealed_weight.make_weighted_target_outputs, sealed_weight.runtime_hooks)
    try:
        with _adapter.runtime_hooks(0.):
            from paper.scripts.count_aware_tpp_backbone import training
            original_build, original_optimizer = training.build_model, training.build_optimizer
            def build(*args, **kwargs):
                global _TRACKER
                model, metadata = original_build(*args, **kwargs)
                require(_TRACKER is None, "Duplicate actual fit model")
                _TRACKER = TimeHeadGuard(model)
                return model, metadata
            def optimizer(model, *args, **kwargs):
                require(_TRACKER is not None and _TRACKER.model is model, "Optimizer/model binding missing")
                return _TRACKER.bind_optimizer(original_optimizer(model, *args, **kwargs))
            training.build_model, training.build_optimizer = build, optimizer
            try:
                yield
            finally:
                if _TRACKER is not None and _TRACKER.optimizer is not None:
                    _TRACKER.optimizer.step = _TRACKER.original_step
                training.build_model, training.build_optimizer = original_build, original_optimizer
                _TRACKER = None
    finally:
        require(sha_file(_SEALED_PATH) == SEALED_WORKER_SHA, "Sealed worker source changed")
        require(original_globals == (sealed_weight.SCHEMA, sealed_weight.CONTRACT_SCHEMA, sealed_weight.TIME_ALPHAS,
            sealed_weight.weight_identity, sealed_weight.make_weighted_target_outputs, sealed_weight.runtime_hooks),
            "Original worker globals changed")


def run_stage(contract, host, job_id, stage):
    require(stage in ("train", "validation"), "New Test execution is forbidden")
    job, data, root, folder = select_job(contract, host, job_id)
    if stage == "train":
        require(not (folder / "runs").exists(), "Fresh fit only; no retry/resume")
    with runtime_hooks(0.):
        return inherited.run_stage(contract, host, job_id, stage)


def quantity_only_synthetic(device="cpu"):
    """Exact two-update comparison with a quantity-only oracle, fabricated only."""
    import torch
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.count_aware_tpp_backbone import core, training
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    configure_runtime(device, 4)
    devices = [] if str(device) == "cpu" else [torch.device(device)]
    with torch.random.fork_rng(devices=devices):
        inherited.install_hooks()
        torch.manual_seed(42); model, _ = build_model(fabricated_data(), ARMS[0])
        torch.manual_seed(42); oracle, _ = build_model(fabricated_data(), ARMS[0])
        model, oracle = model.to(device).train(), oracle.to(device).train()
        require(canonical_state_dict_sha256(model.state_dict()) == canonical_state_dict_sha256(oracle.state_dict()), "Initial model mismatch")
        guard, oracle_guard = TimeHeadGuard(model), TimeHeadGuard(oracle)
        optimizer = guard.bind_optimizer(torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01))
        oracle_optimizer = oracle_guard.bind_optimizer(torch.optim.AdamW(oracle.parameters(), lr=.001, weight_decay=.01))
        dts = torch.tensor([[1., 2., 1., 3.], [1., 1., 2., 1.]], device=device)
        q = torch.tensor([[1., 3., 2., 4.], [2., 4., 1., 3.]], device=device)
        mask = torch.ones_like(dts, dtype=torch.bool)
        initial_quantity = model.quantity_head.weight.detach().clone()
        shared_nonzero = False
        original = core.target_outputs
        wrapped = make_weighted_target_outputs(original, 0.)
        for _ in range(2):
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state_all() if devices else None
            oracle_optimizer.zero_grad(set_to_none=True)
            expected = original(oracle, dts, mask, q, lambda_log_qty=1.)
            expected["quantity_train_loss"].mean().backward()
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            optimizer.zero_grad(set_to_none=True)
            actual = wrapped(model, dts, mask, q, lambda_log_qty=1.)
            require(actual["joint_loss"] is actual["quantity_train_loss"], "Quantity loss was recombined with time graph")
            actual["joint_loss"].mean().backward()
            heads = set(TIME_HEAD_NAMES)
            for (name, parameter), (other_name, other) in zip(model.named_parameters(), oracle.named_parameters()):
                require(name == other_name, "Parameter order changed")
                if other.grad is None:
                    require(parameter.grad is None, "Quantity-only gradient presence differs")
                else:
                    require(parameter.grad is not None and torch.equal(parameter.grad, other.grad), "Quantity-only gradient values differ")
                    if name not in heads and not name.startswith("quantity_head."):
                        shared_nonzero |= bool((parameter.grad != 0).any())
            for target in (model, oracle):
                torch.nn.utils.clip_grad_norm_(target.parameters(), 1.)
            optimizer.step(); oracle_optimizer.step()
            require(canonical_state_dict_sha256(model.state_dict()) == canonical_state_dict_sha256(oracle.state_dict()), "Oracle AdamW updates differ")
        require(shared_nonzero and not torch.equal(initial_quantity, model.quantity_head.weight), "Shared/quantity parameters did not update")
        proof = guard.proof()
    return {"status": "passed", "device": str(device), "fabricated_optimizer_steps": 2,
        "quantity_only_gradient_and_adamw_oracle_exact": True, "shared_nonzero_gradient_observed": True,
        "quantity_head_updated": True, "time_head_no_training_verification": proof,
        "real_data_loaded": False, "held_out_test_evaluated": False}


def synthetic_check(contract=None, host=None, *, device=None):
    selected_device = device or ("cuda:0" if contract is not None else "cpu")
    result = _adapter.synthetic_check(contract, host, device=selected_device)
    result["quantity_only_oracle"] = quantity_only_synthetic(selected_device)
    result.update(quantity_only=True, time_head_trained=False, time_nll_is_diagnostic=True)
    result["checks"].update(quantity_only_gradient_oracle=True, time_head_grad_none_optimizer_state_absent=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--host", required=True, choices=("5080", "5090"))
    parser.add_argument("--job")
    parser.add_argument("--stage", required=True, choices=("train", "validation", "synthetic"))
    args = parser.parse_args(argv)
    contract = read(args.contract)
    if args.stage == "synthetic":
        result = synthetic_check(contract, args.host)
    else:
        require(args.job is not None, "Scientific job is required")
        result = run_stage(contract, args.host, args.job, args.stage)
    print(json.dumps({key: result[key] for key in ("status", "job_id", "stage", "host", "time_alpha", "quantity_only") if key in result}, sort_keys=True))


if __name__ == "__main__":
    main()
