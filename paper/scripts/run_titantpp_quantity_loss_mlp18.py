"""Quantity-only MLP16 controls with literal raw MSE or MAE.

The sealed alpha0 input, model, head, optimization and Validation selector are
preserved. Only the training quantity-loss tensor changes. No Test route exists.
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
from paper.scripts import run_titantpp_time_zero_mlp9 as sealed_zero

SEALED_ZERO_SHA = "2bcc15dec157ff8d0c6aa82198714ba2895e9c5d62d2b1c5d13b930a8ce9c86b"
_SEALED_PATH = Path(sealed_zero.__file__).resolve()
if hashlib.sha256(_SEALED_PATH.read_bytes()).hexdigest() != SEALED_ZERO_SHA:
    raise ValueError("Sealed alpha0 worker source changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_private_alpha0", _SEALED_PATH)
_zero = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_zero)

SCHEMA = "titantpp_quantity_loss_mlp18_stage_v1"
CONTRACT_SCHEMA = "titantpp_quantity_loss_mlp18_v1"
LOSS_MODES = ("raw_mse", "raw_mae")
TIME_ALPHAS = (0.,)
ARMS, SEEDS, DATASETS = _zero.ARMS, _zero.SEEDS, _zero.DATASETS
VARIANT, MONITOR, INACTIVE_TAIL = _zero.VARIANT, _zero.MONITOR, _zero.INACTIVE_TAIL
require, sha_file, sha_json, read = _zero.require, _zero.sha_file, _zero.sha_json, _zero.read
write_exclusive, verify_inventory = _zero.write_exclusive, _zero.verify_inventory
file_inventory, resolve_under = _zero.file_inventory, _zero.resolve_under
population_identity, verify_population = _zero.population_identity, _zero.verify_population
MetricAccumulator, compare_metrics = _zero.MetricAccumulator, _zero.compare_metrics
compare_prediction_files = _zero.compare_prediction_files
build_model, fabricated_data, inherited = _zero.build_model, _zero.fabricated_data, _zero.inherited
TimeHeadGuard, TIME_HEAD_NAMES = _zero.TimeHeadGuard, _zero.TIME_HEAD_NAMES
verify_time_head_proof, verify_checkpoint_fit_proof = _zero.verify_time_head_proof, _zero.verify_checkpoint_fit_proof
_BASE_SELECT_JOB, _BASE_TRAINING_ARGS = _zero.select_job, _zero.training_args
_BASE_ANNOTATE_ROW, _BASE_ANNOTATE_PAYLOAD = _zero.annotate_history_row, _zero.annotate_payload
_BASE_RECEIPT, _BASE_VERIFY_TRAIN = _zero.base_receipt, _zero.verify_train_receipt
_BASE_LOAD_SELECTED, _BASE_EVALUATE = _zero._load_selected, _zero.evaluate_selected
_ACTIVE_LOSS_MODE = None


def loss_identity(loss_mode):
    require(type(loss_mode) is str and loss_mode in LOSS_MODES, "Only literal raw_mse/raw_mae controls are admitted")
    return {"schema": "titantpp_quantity_loss_identity_v1", "loss_mode": loss_mode,
        "per_target_formula": "(pred_qty - true_qty)^2" if loss_mode == "raw_mse" else "abs(pred_qty - true_qty)",
        "reduction": "mean over batch targets", "cap_normalization": False,
        "objective_family": "literal_raw_quantity_point_regression",
        "quantity_head_family": "unchanged_softplus_log1p_location_then_expm1",
        "model_route_variant": VARIANT, "quantity_loss_normalization": "none",
        "prediction": "existing softplus(quantity_head) then expm1; same differentiable raw prediction as evaluation",
        "prediction_clipped_to_cap": False, "prediction_detached": False,
        "input_log1p_and_quantity_head_unchanged": True, "existing_log_mse_recorded_as_diagnostic": True,
        "time_alpha": 0., "time_loss_graph_excluded": True, "checkpoint_monitor": MONITOR}


def weight_identity(alpha, loss_mode=None):
    require(type(alpha) in (int, float) and alpha == 0., "Time objective must be excluded")
    mode = _ACTIVE_LOSS_MODE if loss_mode is None else loss_mode
    identity = loss_identity(mode)
    return {"schema": "titantpp_quantity_only_loss_weight_identity_v1", "time_alpha": 0.,
        "quantity_lambda": 1., "quantity_only": True, "time_head_trained": False,
        "joint_formula": "literal raw quantity loss tensor directly; no time-loss graph contribution",
        "quantity_variant": VARIANT, "checkpoint_monitor": MONITOR,
        "time_inputs_retained": True, "time_nll_role": "diagnostic; time-only head is not trained",
        "quantity_loss_identity": identity}


def raw_quantity_loss(prediction, truth, loss_mode):
    loss_identity(loss_mode)
    difference = prediction - truth
    return difference.square() if loss_mode == "raw_mse" else difference.abs()


def make_weighted_target_outputs(original, alpha):
    identity = weight_identity(alpha)
    mode = identity["quantity_loss_identity"]["loss_mode"]
    def quantity_only(*args, **kwargs):
        import torch
        require(kwargs.get("lambda_log_qty") == 1., "Quantity coefficient must stay one")
        value = original(*args, **kwargs)
        require(torch.equal(value["quantity_train_loss"], value["log_qty_loss"]), "Unexpected original quantity objective")
        prediction, truth = value["pred_qty"], value["true_qty"]
        require(prediction.shape == truth.shape == value["log_qty_loss"].shape, "Quantity loss shape changed")
        if torch.is_grad_enabled() and value["log_qty_loss"].requires_grad:
            require(prediction.requires_grad, "Raw prediction is detached from the quantity head")
        raw = raw_quantity_loss(prediction, truth, mode)
        result = dict(value)
        result["unweighted_joint_loss"] = value["joint_loss"]
        result["original_log_quantity_train_loss"] = value["quantity_train_loss"]
        result["quantity_train_loss"] = raw
        result["joint_loss"] = raw  # Never multiply the time graph by zero.
        return result
    return quantity_only


def validate_weight_binding(payload, alpha, loss_mode=None):
    mode = payload.get("loss_mode") if loss_mode is None else loss_mode
    identity, objective = weight_identity(alpha, mode), loss_identity(mode)
    require(payload.get("loss_mode") == mode and payload.get("time_alpha") == 0. and payload.get("quantity_only") is True
        and payload.get("time_head_trained") is False and payload.get("time_weight_identity") == identity
        and payload.get("quantity_loss_identity") == objective and payload.get("quantity_loss_normalization") == "none", "Quantity objective top-level binding mismatch")
    interface, resume = payload.get("interface_meta", {}), payload.get("resume_identity", {})
    require(interface.get("time_weight_identity") == identity and interface.get("quantity_loss_identity") == objective
        and interface.get("loss_mode") == mode and resume.get("time_weight_identity") == identity
        and resume.get("quantity_loss_identity") == objective and resume.get("arguments", {}).get("time_alpha") == 0.
        and resume.get("arguments", {}).get("loss_mode") == mode and resume.get("interface_meta") == interface,
        "Quantity objective interface/resume binding mismatch")
    return identity


def annotate_history_row(row, alpha):
    row.update(loss_mode=_ACTIVE_LOSS_MODE, quantity_loss_identity=loss_identity(_ACTIVE_LOSS_MODE), quantity_loss_normalization="none")
    return _BASE_ANNOTATE_ROW(row, alpha)


def annotate_payload(payload, alpha):
    payload.update(loss_mode=_ACTIVE_LOSS_MODE, quantity_loss_identity=loss_identity(_ACTIVE_LOSS_MODE), quantity_loss_normalization="none")
    return _BASE_ANNOTATE_PAYLOAD(payload, alpha)


def training_args(contract, data, output, arm, *, time_alpha):
    args = _BASE_TRAINING_ARGS(contract, data, output, arm, time_alpha=time_alpha)
    args.loss_mode, args.execution_role = _ACTIVE_LOSS_MODE, "fresh_quantity_loss_mlp18_validation"
    return args


def base_receipt(contract, host, job, stage):
    result = _BASE_RECEIPT(contract, host, job, stage)
    result.update(loss_mode=job["loss_mode"], quantity_loss_identity=loss_identity(job["loss_mode"]), quantity_loss_normalization="none",
        objective_family="literal_raw_quantity_point_regression", quantity_head_family="unchanged_softplus_log1p_location_then_expm1")
    return result


def verify_train_receipt(contract, host, job, root, receipt):
    require(receipt.get("loss_mode") == job["loss_mode"] and receipt.get("quantity_loss_identity") == loss_identity(job["loss_mode"]),
        "Training receipt quantity objective mismatch")
    _BASE_VERIFY_TRAIN(contract, host, job, root, receipt)


def _load_selected(data, job, train_receipt, root, device):
    require(job["time_alpha"] == 0. and train_receipt.get("loss_mode") == job["loss_mode"], "Selected objective mismatch")
    return _BASE_LOAD_SELECTED(data, job, train_receipt, root, device)


def evaluate_selected(*args, **kwargs):
    return _BASE_EVALUATE(*args, **kwargs)


def select_job(contract, host, job_id):
    result = _BASE_SELECT_JOB(contract, host, job_id)
    loss_identity(result[0].get("loss_mode"))
    return result


# Only private copies are adapted. The original alpha0 and time-weight worker
# modules, their source bytes, and their module globals remain untouched.
for module in (_zero, _zero._adapter):
    module.SCHEMA, module.CONTRACT_SCHEMA, module.TIME_ALPHAS = SCHEMA, CONTRACT_SCHEMA, TIME_ALPHAS
    module.weight_identity = weight_identity
    module.make_weighted_target_outputs = make_weighted_target_outputs
    module.validate_weight_binding = validate_weight_binding
    module.annotate_history_row = annotate_history_row
    module.annotate_payload = annotate_payload
    module.training_args = lambda *args, **kwargs: training_args(*args, **kwargs)
    module.base_receipt = base_receipt
    module.verify_train_receipt = verify_train_receipt
    module._load_selected = _load_selected
    module.evaluate_selected = lambda *args, **kwargs: evaluate_selected(*args, **kwargs)


@contextmanager
def runtime_hooks(loss_mode):
    global _ACTIVE_LOSS_MODE
    loss_identity(loss_mode)
    require(_ACTIVE_LOSS_MODE is None, "Nested/concurrent quantity-loss hooks are forbidden")
    require(sha_file(_SEALED_PATH) == SEALED_ZERO_SHA, "Sealed alpha0 source changed")
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts import run_multilag_detail_execution as base
    original_resume, original_interface = training._resume_identity, base.time_interface
    original_epoch = training.train_epoch_with_telemetry
    def resume_identity(*args, **kwargs):
        result = original_resume(*args, **kwargs)
        result["quantity_loss_identity"] = loss_identity(loss_mode)
        result["arguments"]["loss_mode"] = loss_mode
        return result
    def time_interface(*args, **kwargs):
        result = original_interface(*args, **kwargs)
        result.update(loss_mode=loss_mode, quantity_loss_identity=loss_identity(loss_mode))
        return result
    def train_epoch(*args, **kwargs):
        result = annotate_history_row(original_epoch(*args, **kwargs), 0.)
        print(f"[quantity-loss] mode={loss_mode} train_raw_loss={result['train_quantity_loss']:.8f} "
            f"diagnostic_untrained_time_nll={result['train_time_nll']:.8f}", flush=True)
        return result
    _ACTIVE_LOSS_MODE = loss_mode
    training._resume_identity, base.time_interface = resume_identity, time_interface
    try:
        with _zero.runtime_hooks(0.):
            inherited_epoch_hook = training.train_epoch_with_telemetry
            training.train_epoch_with_telemetry = train_epoch
            try:
                yield
            finally:
                training.train_epoch_with_telemetry = inherited_epoch_hook
    finally:
        training._resume_identity, base.time_interface = original_resume, original_interface
        _ACTIVE_LOSS_MODE = None
        require(sha_file(_SEALED_PATH) == SEALED_ZERO_SHA, "Sealed alpha0 source changed during execution")


def run_stage(contract, host, job_id, stage):
    require(stage in ("train", "validation"), "New Test execution is forbidden")
    job, _, _, folder = select_job(contract, host, job_id)
    if stage == "train":
        require(not (folder / "runs").exists(), "Fresh fit only; no retry/resume")
    with runtime_hooks(job["loss_mode"]):
        return inherited.run_stage(contract, host, job_id, stage)


def quantity_oracle(loss_mode, device="cpu"):
    """Two exact fabricated AdamW updates against a separately formed raw loss."""
    import torch
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.count_aware_tpp_backbone import core
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    loss_identity(loss_mode); configure_runtime(device, 4)
    devices = [] if str(device) == "cpu" else [torch.device(device)]
    with torch.random.fork_rng(devices=devices):
        inherited.install_hooks()
        torch.manual_seed(42); model, _ = build_model(fabricated_data(), ARMS[0])
        torch.manual_seed(42); oracle, _ = build_model(fabricated_data(), ARMS[0])
        model, oracle = model.to(device).train(), oracle.to(device).train()
        guard, oracle_guard = TimeHeadGuard(model), TimeHeadGuard(oracle)
        optimizer = guard.bind_optimizer(torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01))
        expected_optimizer = oracle_guard.bind_optimizer(torch.optim.AdamW(oracle.parameters(), lr=.001, weight_decay=.01))
        dts = torch.tensor([[1.,2.,1.,3.],[1.,1.,2.,1.]],device=device)
        q = torch.tensor([[1.,3.,2.,4.],[2.,4.,1.,3.]],device=device)
        mask = torch.ones_like(dts,dtype=torch.bool)
        original = core.target_outputs
        shared_nonzero = False
        with runtime_hooks(loss_mode):
            wrapped = make_weighted_target_outputs(original,0.)
            for _ in range(2):
                rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state_all() if devices else None
                expected_optimizer.zero_grad(set_to_none=True)
                ref = original(oracle,dts,mask,q,lambda_log_qty=1.)
                error = ref["pred_qty"] - ref["true_qty"]
                expected_loss = error.square() if loss_mode == "raw_mse" else error.abs()
                expected_loss.mean().backward()
                torch.set_rng_state(rng)
                if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
                optimizer.zero_grad(set_to_none=True)
                actual = wrapped(model,dts,mask,q,lambda_log_qty=1.)
                require(actual["joint_loss"] is actual["quantity_train_loss"],"Time graph was recombined")
                actual["joint_loss"].mean().backward()
                for (name,p),(other_name,other) in zip(model.named_parameters(),oracle.named_parameters()):
                    require(name==other_name,"Parameter order changed")
                    require((p.grad is None and other.grad is None) or
                        (p.grad is not None and other.grad is not None and torch.equal(p.grad,other.grad)),"Raw-loss gradient oracle differs")
                    if p.grad is not None and name not in TIME_HEAD_NAMES and not name.startswith("quantity_head."):
                        shared_nonzero |= bool((p.grad!=0).any())
                for target in (model,oracle):torch.nn.utils.clip_grad_norm_(target.parameters(),1.)
                optimizer.step();expected_optimizer.step()
                require(canonical_state_dict_sha256(model.state_dict())==canonical_state_dict_sha256(oracle.state_dict()),"Raw-loss AdamW oracle differs")
        require(shared_nonzero,"No raw quantity gradient reached shared parameters by update2")
        proof = guard.proof()
    return {"status":"passed","loss_mode":loss_mode,"fabricated_optimizer_steps":2,
        "literal_raw_gradient_and_adamw_oracle_exact":True,"shared_nonzero_gradient_observed":True,
        "time_head_no_training_verification":proof,"real_data_loaded":False,"held_out_test_evaluated":False}


def synthetic_condition(contract, host, loss_mode, *, device=None):
    selected_device = device or ("cuda:0" if contract is not None else "cpu")
    with runtime_hooks(loss_mode):
        original_arms = inherited.ARMS
        try:
            inherited.ARMS = ARMS
            result = inherited.synthetic_check(contract,host,device=selected_device)
        finally:
            inherited.ARMS = original_arms
    result.update(schema=SCHEMA, loss_mode=loss_mode, loss_modes=[loss_mode], time_alphas=[0.],
        quantity_only=True, time_head_trained=False, time_nll_is_diagnostic=True,
        quantity_oracle=quantity_oracle(loss_mode,selected_device))
    if str(selected_device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()
        # Include the separate two-model oracle in the measured process peak.
        for peak in result["cuda_memory_peaks"].values():
            peak["max_memory_reserved_bytes"] = max(peak["max_memory_reserved_bytes"], torch.cuda.max_memory_reserved())
            peak["max_memory_allocated_bytes"] = max(peak["max_memory_allocated_bytes"], torch.cuda.max_memory_allocated())
            peak["includes_quantity_oracle"] = True
    return result


def synthetic_check(contract=None, host=None, *, device=None):
    results = {mode:synthetic_condition(contract,host,mode,device=device) for mode in LOSS_MODES}
    first = next(iter(results.values()));peaks={}
    for arm in ARMS:
        entries=[r["cuda_memory_peaks"][arm] for r in results.values()]
        peak=dict(entries[0])
        for key in ("max_memory_reserved_bytes","max_memory_allocated_bytes"):
            if all(key in e for e in entries):peak[key]=max(e[key] for e in entries)
        peak["tested_loss_modes"]=list(LOSS_MODES);peaks[arm]=peak
    return {"schema":SCHEMA,"status":"passed","device":first["device"],"host":host,"arms":list(ARMS),
        "loss_modes":list(LOSS_MODES),"time_alphas":[0.],"per_loss":results,"cuda_memory_peaks":peaks,
        "parameter_count":first["parameter_count"],"initial_state_sha256":first["initial_state_sha256"],
        "quantity_only":True,"time_head_trained":False,"time_nll_is_diagnostic":True,
        "real_data_loaded":False,"held_out_test_evaluated":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--host",choices=("5080","5090"),required=True)
    parser.add_argument("--job")
    parser.add_argument("--stage",choices=("train","validation","synthetic"),required=True)
    args=parser.parse_args(argv);contract=read(args.contract)
    if args.stage=="synthetic":result=synthetic_check(contract,args.host)
    else:
        require(args.job is not None,"Scientific job is required")
        result=run_stage(contract,args.host,args.job,args.stage)
    print(json.dumps({k:result[k] for k in ("status","job_id","stage","host","loss_mode","real_data_loaded") if k in result},sort_keys=True))


if __name__=="__main__":main()
