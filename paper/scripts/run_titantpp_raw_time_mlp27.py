"""Validation-only MLP16 controls changing the observed time feature alone.

Raw nonnegative delta_t replaces log1p(delta_t). Quantity features, model
parameters/buffers, duration targets/head, three quantity losses and selection
are otherwise inherited unchanged. All shared hooks are process-local/scoped.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path

from paper.scripts import run_titantpp_quantity_loss_mlp18 as sealed_parent
from models.TPPs.CountAwareTitanHistoryWidth import CountAwareTitanHistoryWidth

_PARENT_PATH=Path(sealed_parent.__file__).resolve()
PARENT_SHA256="56baa0248c085cb30c6bf44811ee75a5a15504097119356f354af1de4699c023"
if hashlib.sha256(_PARENT_PATH.read_bytes()).hexdigest()!=PARENT_SHA256:
    raise ValueError("Sealed quantity-loss worker changed")
_spec=importlib.util.spec_from_file_location(__name__+"_private_parent",_PARENT_PATH)
_parent=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_parent)
SCHEMA="titantpp_raw_time_mlp27_stage_v1"
CONTRACT_SCHEMA="titantpp_raw_time_mlp27_v1"
LOSS_MODES=("log_mse","raw_mse","raw_mae")
TIME_ALPHAS=(0.,)
ARMS,SEEDS,DATASETS=_parent.ARMS,_parent.SEEDS,_parent.DATASETS
VARIANT,MONITOR,INACTIVE_TAIL=_parent.VARIANT,_parent.MONITOR,_parent.INACTIVE_TAIL
TIME_INPUT_TRANSFORM="raw_nonnegative_float32"
QUANTITY_INPUT_TRANSFORM="log1p_nonnegative_float32"
require,sha_file,sha_json,read=_parent.require,_parent.sha_file,_parent.sha_json,_parent.read
write_exclusive,verify_inventory=_parent.write_exclusive,_parent.verify_inventory
file_inventory,resolve_under=_parent.file_inventory,_parent.resolve_under
population_identity,verify_population=_parent.population_identity,_parent.verify_population
MetricAccumulator,compare_metrics,compare_prediction_files=_parent.MetricAccumulator,_parent.compare_metrics,_parent.compare_prediction_files
TimeHeadGuard,TIME_HEAD_NAMES=_parent.TimeHeadGuard,_parent.TIME_HEAD_NAMES
verify_time_head_proof,verify_checkpoint_fit_proof=_parent.verify_time_head_proof,_parent.verify_checkpoint_fit_proof
fabricated_data,inherited=_parent.fabricated_data,_parent.inherited
_BASE_BUILD_MODEL=_parent.build_model
_BASE_LOSS_IDENTITY=_parent.loss_identity
_BASE_WEIGHT_IDENTITY=_parent.weight_identity
_BASE_VALIDATE_BINDING=_parent.validate_weight_binding
_BASE_ANNOTATE_ROW,_BASE_ANNOTATE_PAYLOAD=_parent.annotate_history_row,_parent.annotate_payload
_BASE_TRAINING_ARGS,_BASE_RECEIPT=_parent.training_args,_parent.base_receipt
_BASE_VERIFY_TRAIN,_BASE_LOAD_SELECTED=_parent.verify_train_receipt,_parent._load_selected
_BASE_EVALUATE_SELECTED=_parent.evaluate_selected
_BASE_SELECT_JOB,_BASE_RUNTIME_HOOKS=_parent.select_job,_parent.runtime_hooks
_BASE_SYNTHETIC_CONDITION,_BASE_SYNTHETIC_CHECK=_parent.synthetic_condition,_parent.synthetic_check
_FEATURE_SCOPE_DEPTH=0


def feature_identity():
    return {"schema":"titantpp_raw_time_input_identity_v1","time_input_transform":TIME_INPUT_TRANSFORM,
        "quantity_input_transform":QUANTITY_INPUT_TRANSFORM,"time_input_normalization":"none",
        "time_input_upper_clipping":False,"time_head_target_and_scale_unchanged":True,
        "parameter_buffer_shapes_and_initialization_unchanged":True,"observed_mask_and_target_blocking_unchanged":True}


def feature_labels():
    return {"time_input_transform":TIME_INPUT_TRANSFORM,"quantity_input_transform":QUANTITY_INPUT_TRANSFORM,
        "input_feature_identity":feature_identity()}


class RawTimeHistoryMLP16(CountAwareTitanHistoryWidth):
    """Parameter-free override; inherited construction and state_dict are unchanged."""
    @staticmethod
    def continuous_features(dts,history_quantities,mask):
        import torch
        features=torch.stack([dts.float().clamp_min(0.),torch.log1p(history_quantities.float().clamp_min(0.))],dim=-1)
        return features*mask.unsqueeze(-1).to(dtype=features.dtype)


@contextmanager
def raw_feature_factory():
    """Bind the already-registered width factory to a subclass only in this scope."""
    global _FEATURE_SCOPE_DEPTH
    inherited.install_hooks()
    from paper.scripts import run_titantpp_history_width as width
    original=width.CountAwareTitanHistoryWidth
    if _FEATURE_SCOPE_DEPTH:
        require(original is RawTimeHistoryMLP16,"Raw-time factory scope changed")
        _FEATURE_SCOPE_DEPTH+=1
        try:yield
        finally:_FEATURE_SCOPE_DEPTH-=1
        return
    require(original is CountAwareTitanHistoryWidth,"Foreign width factory is already installed")
    _FEATURE_SCOPE_DEPTH=1;width.CountAwareTitanHistoryWidth=RawTimeHistoryMLP16
    try:yield
    finally:
        width.CountAwareTitanHistoryWidth=original
        _FEATURE_SCOPE_DEPTH=0


def build_model(data,arm):
    require(arm in ARMS,"Only MLP16 is admitted")
    with raw_feature_factory():model,metadata=_BASE_BUILD_MODEL(data,arm)
    require(type(model) is RawTimeHistoryMLP16,"Raw-time subclass was not constructed")
    return model,{**metadata,**feature_labels()}


def loss_identity(loss_mode):
    require(type(loss_mode) is str and loss_mode in LOSS_MODES,"Unexpected quantity loss")
    if loss_mode=="log_mse":
        result={"schema":"titantpp_raw_time_quantity_loss_identity_v1","loss_mode":loss_mode,
            "per_target_formula":"(softplus(quantity_head(h)) - log1p(true_qty))^2",
            "reduction":"mean over batch targets","cap_normalization":False,
            "objective_family":"existing_log1p_quantity_point_regression","quantity_loss_normalization":"none",
            "quantity_head_family":"unchanged_softplus_log1p_location_then_expm1","model_route_variant":VARIANT,
            "prediction_clipped_to_cap":False,"prediction_detached":False,"time_alpha":0.,
            "time_loss_graph_excluded":True,"checkpoint_monitor":MONITOR}
    else:
        result=_BASE_LOSS_IDENTITY(loss_mode)
        # Parent wording refers to BOTH inputs; only the quantity-input claim remains true.
        result["input_log1p_and_quantity_head_unchanged"]=False
    result.update(quantity_input_and_head_unchanged=True,**feature_labels())
    return result


def weight_identity(alpha,loss_mode=None):
    require(type(alpha) in (int,float) and alpha==0.,"Time graph must remain excluded")
    mode=_parent._ACTIVE_LOSS_MODE if loss_mode is None else loss_mode
    return {"schema":"titantpp_raw_time_quantity_only_identity_v1","time_alpha":0.,"quantity_lambda":1.,
        "quantity_only":True,"time_head_trained":False,"joint_formula":"selected quantity loss tensor directly; no time-loss graph contribution",
        "quantity_variant":VARIANT,"checkpoint_monitor":MONITOR,"time_inputs_retained":True,
        "time_nll_role":"diagnostic; unchanged time-only head is not trained","quantity_loss_identity":loss_identity(mode),**feature_labels()}


def make_weighted_target_outputs(original,alpha):
    mode=weight_identity(alpha)["quantity_loss_identity"]["loss_mode"]
    def quantity_only(*args,**kwargs):
        import torch
        require(kwargs.get("lambda_log_qty")==1.,"Quantity coefficient must remain one")
        value=original(*args,**kwargs)
        require(torch.equal(value["quantity_train_loss"],value["log_qty_loss"]),"Original log-MSE changed")
        prediction,truth=value["pred_qty"],value["true_qty"]
        require(prediction.shape==truth.shape==value["log_qty_loss"].shape,"Quantity target shape changed")
        if mode=="log_mse":loss=value["quantity_train_loss"]
        else:
            if torch.is_grad_enabled() and value["log_qty_loss"].requires_grad:
                require(prediction.requires_grad,"Raw quantity prediction detached")
            loss=(prediction-truth).square() if mode=="raw_mse" else (prediction-truth).abs()
        result=dict(value)
        result.update(unweighted_joint_loss=value["joint_loss"],original_log_quantity_train_loss=value["quantity_train_loss"],
            quantity_train_loss=loss,joint_loss=loss)
        return result
    return quantity_only


def validate_feature_binding(payload,*,full=False):
    require(all(payload.get(k)==v for k,v in feature_labels().items()),"Raw-time input binding missing or changed")
    if full:
        interface,resume=payload.get("interface_meta",{}),payload.get("resume_identity",{})
        require(all(interface.get(k)==v and resume.get(k)==v and resume.get("arguments",{}).get(k)==v for k,v in feature_labels().items()),
            "Raw-time interface/resume binding changed")
        if "encoder_config" in payload:
            require(all(payload["encoder_config"].get(k)==v for k,v in feature_labels().items()),"Raw-time encoder metadata changed")


def validate_weight_binding(payload,alpha,loss_mode=None):
    result=_BASE_VALIDATE_BINDING(payload,alpha,loss_mode)
    validate_feature_binding(payload,full=True)
    return result


def annotate_history_row(row,alpha):
    row.update(**feature_labels())
    return _BASE_ANNOTATE_ROW(row,alpha)


def annotate_payload(payload,alpha):
    payload.update(**feature_labels())
    return _BASE_ANNOTATE_PAYLOAD(payload,alpha)


def training_args(contract,data,output,arm,*,time_alpha):
    args=_BASE_TRAINING_ARGS(contract,data,output,arm,time_alpha=time_alpha)
    args.execution_role="fresh_raw_time_mlp27_validation"
    for key,value in feature_labels().items():setattr(args,key,value)
    return args


def base_receipt(contract,host,job,stage):
    result=_BASE_RECEIPT(contract,host,job,stage)
    result.update(**feature_labels(),objective_family=loss_identity(job["loss_mode"])["objective_family"])
    return result


def verify_train_receipt(contract,host,job,root,receipt):
    validate_feature_binding(receipt)
    return _BASE_VERIFY_TRAIN(contract,host,job,root,receipt)


def _load_selected(data,job,receipt,root,device):
    validate_feature_binding(receipt)
    return _BASE_LOAD_SELECTED(data,job,receipt,root,device)


def evaluate_selected(*args,**kwargs):
    return _BASE_EVALUATE_SELECTED(*args,**kwargs)


def select_job(contract,host,job_id):
    architecture=contract.get("architecture",{})
    require(architecture.get("time_input_transform")==TIME_INPUT_TRANSFORM
        and architecture.get("quantity_input_transform")==QUANTITY_INPUT_TRANSFORM,"Raw-time contract transform mismatch")
    return _BASE_SELECT_JOB(contract,host,job_id)


@contextmanager
def runtime_hooks(loss_mode):
    require(sha_file(_PARENT_PATH)==PARENT_SHA256,"Sealed parent source changed")
    with raw_feature_factory(),_BASE_RUNTIME_HOOKS(loss_mode):
        from paper.scripts.count_aware_tpp_backbone import training
        from paper.scripts import run_multilag_detail_execution as base
        original_interface,original_resume,original_build=base.time_interface,training._resume_identity,training.build_model
        def interface(*args,**kwargs):return {**original_interface(*args,**kwargs),**feature_labels()}
        def resume(*args,**kwargs):
            result=original_resume(*args,**kwargs);result.update(**feature_labels());result["arguments"].update(**feature_labels())
            return result
        def build(*args,**kwargs):
            model,metadata=original_build(*args,**kwargs)
            require(type(model) is RawTimeHistoryMLP16,"Training did not use raw-time subclass")
            return model,{**metadata,**feature_labels()}
        base.time_interface,training._resume_identity,training.build_model=interface,resume,build
        try:yield
        finally:
            base.time_interface,training._resume_identity,training.build_model=original_interface,original_resume,original_build
            require(sha_file(_PARENT_PATH)==PARENT_SHA256,"Sealed parent source changed during run")


def run_stage(contract,host,job_id,stage):
    require(stage in ("train","validation"),"New Test execution is forbidden")
    job,_,_,folder=select_job(contract,host,job_id)
    if stage=="train":require(not (folder/"runs").exists(),"Fresh fit only; no retry/resume")
    with runtime_hooks(job["loss_mode"]):return inherited.run_stage(contract,host,job_id,stage)


def quantity_oracle(loss_mode,device="cpu"):
    import torch
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.count_aware_tpp_backbone import core
    configure_runtime(device,4);devices=[] if str(device)=="cpu" else [torch.device(device)]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42);model,_=build_model(fabricated_data(),ARMS[0])
        torch.manual_seed(42);oracle,_=build_model(fabricated_data(),ARMS[0])
        model,oracle=model.to(device).train(),oracle.to(device).train()
        guard,ref_guard=TimeHeadGuard(model),TimeHeadGuard(oracle)
        optimizer=guard.bind_optimizer(torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01))
        ref_optimizer=ref_guard.bind_optimizer(torch.optim.AdamW(oracle.parameters(),lr=.001,weight_decay=.01))
        dts=torch.tensor([[1.,2.,16.,3.],[1.,32.,2.,1.]],device=device)
        q=torch.tensor([[1.,3.,2.,4.],[2.,4.,1.,3.]],device=device);mask=torch.ones_like(dts,dtype=torch.bool)
        original=core.target_outputs;shared_nonzero=False
        with runtime_hooks(loss_mode):
            wrapped=make_weighted_target_outputs(original,0.)
            for _ in range(2):
                rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all() if devices else None
                ref_optimizer.zero_grad(set_to_none=True)
                ref=original(oracle,dts,mask,q,lambda_log_qty=1.)
                loss=ref['log_qty_loss'] if loss_mode=='log_mse' else ((ref['pred_qty']-ref['true_qty']).square() if loss_mode=='raw_mse' else (ref['pred_qty']-ref['true_qty']).abs())
                loss.mean().backward();torch.set_rng_state(rng)
                if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
                optimizer.zero_grad(set_to_none=True);actual=wrapped(model,dts,mask,q,lambda_log_qty=1.)
                require(actual['joint_loss'] is actual['quantity_train_loss'],"Time graph recombined")
                actual['joint_loss'].mean().backward()
                for (name,p),(ref_name,ref_p) in zip(model.named_parameters(),oracle.named_parameters()):
                    require(name==ref_name and ((p.grad is None and ref_p.grad is None) or
                        (p.grad is not None and ref_p.grad is not None and torch.equal(p.grad,ref_p.grad))),"Quantity gradient oracle differs")
                    if p.grad is not None and name not in TIME_HEAD_NAMES and not name.startswith('quantity_head.'):
                        shared_nonzero=shared_nonzero or bool(p.grad.abs().sum()>0)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);torch.nn.utils.clip_grad_norm_(oracle.parameters(),1.)
                optimizer.step();ref_optimizer.step()
                require(all(torch.equal(p,ref_p) for p,ref_p in zip(model.state_dict().values(),oracle.state_dict().values())),"AdamW oracle differs")
        require(shared_nonzero,"No shared gradient observed")
        return {"status":"passed","loss_mode":loss_mode,"loss_formula_and_adamw_oracle_exact":True,
            "shared_nonzero_gradient_observed":True,"time_head_no_training_verification":guard.proof(),**feature_labels()}


def synthetic_condition(contract,host,loss_mode,*,device=None):
    result=_BASE_SYNTHETIC_CONDITION(contract,host,loss_mode,device=device)
    result.update(**feature_labels());return result


def initialization_oracle():
    """Fabricated seed42/52/62 construction proof; not research checkpoint inference."""
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    rows=[]
    with torch.random.fork_rng(devices=[]):
        for seed in SEEDS:
            torch.manual_seed(seed);reference,_=sealed_parent.build_model(fabricated_data(),ARMS[0]);old_rng=torch.get_rng_state().clone()
            torch.manual_seed(seed);model,_=build_model(fabricated_data(),ARMS[0]);new_rng=torch.get_rng_state().clone()
            old_sha=canonical_state_dict_sha256(reference.state_dict());new_sha=canonical_state_dict_sha256(model.state_dict())
            require(old_sha==new_sha and torch.equal(old_rng,new_rng),"Raw time changed initialization or RNG")
            require(model.time_head_contract()==reference.time_head_contract(),"Duration head changed")
            rows.append({"seed":seed,"parent_initial_state_sha256":old_sha,"raw_time_initial_state_sha256":new_sha,
                "post_construction_rng_sha256":hashlib.sha256(new_rng.numpy().tobytes()).hexdigest(),
                "parameter_count":sum(p.numel() for p in model.parameters()),"state_tensor_count":len(model.state_dict()),
                "parameter_buffer_and_rng_exact":True,"time_head_contract_equal":True})
    return {"status":"passed","scope":"fabricated same-config construction only","real_data_loaded":False,"seeds":rows}


def synthetic_check(contract=None,host=None,*,device=None):
    result=_BASE_SYNTHETIC_CHECK(contract,host,device=device)
    result.update(**feature_labels(),initialization_oracle=initialization_oracle());return result


# Adapt only private namespaces, never the imported sealed-parent module.
for _module in (_parent,_parent._zero,_parent._zero._adapter):
    _module.SCHEMA,_module.CONTRACT_SCHEMA,_module.LOSS_MODES=SCHEMA,CONTRACT_SCHEMA,LOSS_MODES
    for _name in ('loss_identity','weight_identity','make_weighted_target_outputs','validate_weight_binding',
        'annotate_history_row','annotate_payload','training_args','base_receipt','verify_train_receipt','_load_selected'):
        setattr(_module,_name,globals()[_name])
    _module.training_args=lambda *args,**kwargs:training_args(*args,**kwargs)
    _module.evaluate_selected=lambda *args,**kwargs:evaluate_selected(*args,**kwargs)
_parent.select_job=select_job
_parent.runtime_hooks=runtime_hooks
_parent.quantity_oracle=quantity_oracle
_parent.synthetic_condition=synthetic_condition


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--host',choices=('5080','5090'),required=True)
    parser.add_argument('--job');parser.add_argument('--stage',choices=('train','validation','synthetic'),required=True)
    args=parser.parse_args(argv);contract=read(args.contract)
    if args.stage=='synthetic':result=synthetic_check(contract,args.host)
    else:
        require(args.job is not None,'Scientific job is required');result=run_stage(contract,args.host,args.job,args.stage)
    print(json.dumps({k:result[k] for k in ('status','job_id','stage','host','loss_mode','real_data_loaded') if k in result},sort_keys=True))


if __name__=='__main__':main()
