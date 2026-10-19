"""Validation-only matched late observation/hidden paths on MLP16.

Two matched post-encoder paths differ only in their observed-input or hidden
source. A single completed-comparison condition is shared by every dataset and
arm. Existing source, queues and result files are never changed.
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
from models.TPPs.CountAwareTitanLateObservationPath import (
    CountAwareTitanLateObservationPath, ARMS, MODE_BY_ARM, metadata as late_metadata,
    identity as late_identity,
)

_PARENT_PATH=Path(sealed_parent.__file__).resolve()
PARENT_SHA256="56baa0248c085cb30c6bf44811ee75a5a15504097119356f354af1de4699c023"
if hashlib.sha256(_PARENT_PATH.read_bytes()).hexdigest()!=PARENT_SHA256:
    raise ValueError("Sealed quantity-loss worker changed")
_spec=importlib.util.spec_from_file_location(__name__+"_private_parent",_PARENT_PATH)
_parent=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_parent)
SCHEMA="titantpp_late_path18_stage_v1"
CONTRACT_SCHEMA="titantpp_late_path18_v1"
LOSS_MODES=("log_mse","raw_mse","raw_mae")
TIME_ALPHAS=(0.,)
SEEDS,DATASETS=_parent.SEEDS,_parent.DATASETS
VARIANT,MONITOR,INACTIVE_TAIL=_parent.VARIANT,_parent.MONITOR,_parent.INACTIVE_TAIL
TIME_INPUT_TRANSFORMS={"log1p":"log1p_nonnegative_float32","raw":"raw_nonnegative_float32"}
TIME_INPUTS=tuple(TIME_INPUT_TRANSFORMS)
_ACTIVE_TIME_INPUT="log1p"
_ACTIVE_ARM=None
_MODEL_SCOPE_DEPTH=0
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
    return {"schema":"titantpp_late_path_input_identity_v1",
        "time_input":_ACTIVE_TIME_INPUT,"time_input_transform":TIME_INPUT_TRANSFORMS[_ACTIVE_TIME_INPUT],
        "quantity_input_transform":QUANTITY_INPUT_TRANSFORM,"time_input_normalization":"none",
        "time_input_upper_clipping":False,"time_head_target_and_scale_unchanged":True,
        "observed_mask_and_target_blocking_unchanged":True}


def feature_labels():
    return {"time_input":_ACTIVE_TIME_INPUT,"time_input_transform":TIME_INPUT_TRANSFORMS[_ACTIVE_TIME_INPUT],
        "quantity_input_transform":QUANTITY_INPUT_TRANSFORM,"input_feature_identity":feature_identity()}


def candidate_labels(arm=None):
    arm=arm or _ACTIVE_ARM
    require(arm in ARMS,"Explicit late path arm is required")
    return {"candidate_name":arm,"late_path_mode":MODE_BY_ARM[arm],"late_path_parameter_count":1024,
        "parent_backbone":"titantpp_history_mlp_width16",**feature_labels()}


class RawTimeLateObservationPath(CountAwareTitanLateObservationPath):
    """Only observed delta_t features change; the selected candidate stays identified."""
    @staticmethod
    def continuous_features(dts,history_quantities,mask):
        import torch
        features=torch.stack([dts.float().clamp_min(0.),torch.log1p(history_quantities.float().clamp_min(0.))],dim=-1)
        return features*mask.unsqueeze(-1).to(dtype=features.dtype)


def _candidate_factory(backbone,**kwargs):
    require(backbone in ARMS,"Unexpected late path candidate")
    from models.TPPs import CountAwareFactory as factory
    require(kwargs.pop("titans_memory_gradient_clip",None) is None,"Memory gradient clipping changed")
    cls=RawTimeLateObservationPath if _ACTIVE_TIME_INPUT=="raw" else CountAwareTitanLateObservationPath
    model=cls(**kwargs,late_path_mode=MODE_BY_ARM[backbone])
    return factory.with_time_metadata(model,{**late_metadata(kwargs["hidden_dim"],mode=MODE_BY_ARM[backbone]),
        "max_len":kwargs["max_seq_len"],**feature_labels()})


def validate_candidate_route(payload,expected_backbone):
    """Validate the actual candidate identity without relabelling persisted objects."""
    import torch
    from models.TPPs.CountAwareTitanHistoryWidth import identity as width_identity
    from models.TPPs.CountAwareTitanCoreAblation import identity as core_identity
    require(expected_backbone in ARMS,"Candidate cannot be relabelled as an existing backbone")
    meta=payload.get("encoder_config",{});interface=payload.get("interface_meta",{});resume=payload.get("resume_identity",{})
    expected=late_metadata(meta.get("d_model"),mode=MODE_BY_ARM[expected_backbone])
    head=meta.get("time_head",{})
    require(payload.get("backbone")==expected_backbone and all(meta.get(k)==v for k,v in expected.items())
        and payload.get("evaluation_scope")=="validation_only" and payload.get("held_out_test_evaluated") is False
        and payload.get("variant")==VARIANT and resume.get("backbone")==expected_backbone
        and resume.get("interface_meta")==interface and resume.get("checkpoint_monitor")==MONITOR
        and resume.get("arguments",{}).get("model_role")==expected["model_role"]
        and head.get("mode")=="heteroscedastic_lognormal_duration"
        and isinstance(head.get("observation_likelihood"),dict)
        and head["observation_likelihood"].get("mode")=="positive_integer_round_clamp_v1"
        and head["observation_likelihood"]==interface.get("time_head",{}).get("observation_likelihood"),
        "Late path metadata, role, head, split or selector mismatch")
    validate_feature_binding(payload,full=True)
    states=[payload[k] for k in ("model_state_dict","best_state_dict") if k in payload]
    for state in states:
        for key,value in (("late_path_identity",late_identity(mode=MODE_BY_ARM[expected_backbone])),
            ("history_width_identity",width_identity(width=16)),("core_ablation_identity",core_identity(64,"mlp"))):
            require(isinstance(state.get(key),torch.Tensor) and torch.equal(state[key].cpu(),value),"Candidate state identity mismatch")
        for name,shape in (("late_path.input_projection.weight",(8,64)),("late_path.output_projection.weight",(64,8))):
            require(isinstance(state.get(name),torch.Tensor) and tuple(state[name].shape)==shape,"Candidate path tensor shape mismatch")
        for branch in range(8):
            for kind,shape in (("input",(16,128)),("output",(64,16))):
                value=state.get(f"multilag_detail.{kind}_projections.{branch}.weight")
                require(isinstance(value,torch.Tensor) and tuple(value.shape)==shape,"Parent history tensor shape mismatch")
    if not states:
        require(len(payload.get("checkpoint_state_sha256",""))==64 and payload.get("best_epoch",0)>=1,"Unbound candidate summary")
    return True


@contextmanager
def candidate_factory(time_input=None,arm=None):
    """Candidate-only process-local registration, fully restored after the scope."""
    global _ACTIVE_TIME_INPUT,_ACTIVE_ARM,_MODEL_SCOPE_DEPTH
    selected=_ACTIVE_TIME_INPUT if time_input is None else time_input
    require(selected in TIME_INPUTS and (arm is None or arm in ARMS),"Unknown common input/arm")
    if _MODEL_SCOPE_DEPTH:
        require(selected==_ACTIVE_TIME_INPUT and (arm is None or arm==_ACTIVE_ARM),"Nested candidate scope changed")
        _MODEL_SCOPE_DEPTH+=1
        try:yield
        finally:_MODEL_SCOPE_DEPTH-=1
        return
    inherited.install_hooks()
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import training,constants,observed_time
    from copy import copy
    previous=(_ACTIVE_TIME_INPUT,_ACTIVE_ARM)
    _ACTIVE_TIME_INPUT,_ACTIVE_ARM=selected,arm
    undo=[]
    def replace(module,name,value):
        undo.append((module,name,getattr(module,name)));setattr(module,name,value)
    original_build,original_route=factory.build_count_aware_model,factory.validate_checkpoint_route
    original_launch,original_role=observed_time.validate_launch,constants.validate_model_role_contract
    def build(backbone,**kwargs):
        return _candidate_factory(backbone,**kwargs) if backbone in ARMS else original_build(backbone,**kwargs)
    def route(payload,expected_backbone):
        identified=payload.get("backbone") in ARMS or "late_path_mode" in payload.get("encoder_config",{}) or any(
            "late_path_identity" in payload.get(k,{}) for k in ("model_state_dict","best_state_dict"))
        if expected_backbone in ARMS:return validate_candidate_route(payload,expected_backbone)
        require(not identified,"Late path candidate cannot be relabelled")
        return original_route(payload,expected_backbone)
    roles={a:late_metadata(mode=MODE_BY_ARM[a])["model_role"] for a in ARMS}
    def launch(args,backbones,variants):
        if any(a in ARMS for a in backbones) or getattr(args,"model_role",None) in roles.values():
            require(len(backbones)==1 and backbones[0] in ARMS and args.model_role==roles[backbones[0]],"Candidate launch role mismatch")
            proxy=copy(args);proxy.model_role="observed_time_history_mlp_width16_v1"
            return original_launch(proxy,("titantpp_history_mlp_width16",),variants)
        return original_launch(args,backbones,variants)
    def role_contract(**kwargs):
        arms=kwargs["backbones"]
        if any(a in ARMS for a in arms) or kwargs["model_role"] in roles.values():
            require(len(arms)==1 and arms[0] in ARMS and kwargs["model_role"]==roles[arms[0]],"Candidate model role mismatch")
            return original_role(**{**kwargs,"model_role":"observed_time_history_mlp_width16_v1","backbones":("titantpp_history_mlp_width16",)})
        return original_role(**kwargs)
    _MODEL_SCOPE_DEPTH=1
    try:
        for module,name,value in ((factory,"build_count_aware_model",build),(factory,"validate_checkpoint_route",route),
            (training,"build_count_aware_model",build),(training,"build_model",build),(training,"validate_checkpoint_route",route),
            (observed_time,"validate_launch",launch),(observed_time,"ROLES",(*observed_time.ROLES,*roles.values())),
            (training,"OBSERVED_TIME_ROLES",(*training.OBSERVED_TIME_ROLES,*roles.values())),
            (constants,"validate_model_role_contract",role_contract),
            (constants,"BACKBONE_LABELS",{**constants.BACKBONE_LABELS,**{a:a for a in ARMS}}),
            (training,"BACKBONE_LABELS",{**training.BACKBONE_LABELS,**{a:a for a in ARMS}}),
            (constants,"MODEL_ROLES",(*constants.MODEL_ROLES,*roles.values())),
            (constants,"SUPPORTED_BACKBONES",(*constants.SUPPORTED_BACKBONES,*ARMS)),
            (inherited,"ARMS",ARMS),(inherited,"role_for",lambda a:roles[a]),(inherited,"build_model",build_model)):
            replace(module,name,value)
        yield
    finally:
        for module,name,value in reversed(undo):setattr(module,name,value)
        _MODEL_SCOPE_DEPTH=0;_ACTIVE_TIME_INPUT,_ACTIVE_ARM=previous


def build_model(data,arm,*,time_input=None):
    require(arm in ARMS,"Only the two explicit late path candidates are admitted")
    with candidate_factory(time_input,arm):
        config={k:v for k,v in data["model"].items() if k not in ("backbone","lambda_log_qty","lambda_tail","time_head_lr_multiplier")}
        config.update(INACTIVE_TAIL)
        return _candidate_factory(arm,**config,train_log_mean=data["statistics"]["train_log_mean"],
            train_log_std=data["statistics"]["train_log_std"],max_seq_len=data["loader"]["max_seq_len"])


def loss_identity(loss_mode):
    require(type(loss_mode) is str and loss_mode in LOSS_MODES,"Unexpected quantity loss")
    if loss_mode=="log_mse":
        result={"schema":"titantpp_late_path_quantity_loss_identity_v1","loss_mode":loss_mode,
            "per_target_formula":"(softplus(quantity_head(h)) - log1p(true_qty))^2",
            "reduction":"mean over batch targets","cap_normalization":False,
            "objective_family":"existing_log1p_quantity_point_regression","quantity_loss_normalization":"none",
            "quantity_head_family":"unchanged_softplus_log1p_location_then_expm1","model_route_variant":VARIANT,
            "prediction_clipped_to_cap":False,"prediction_detached":False,"time_alpha":0.,
            "time_loss_graph_excluded":True,"checkpoint_monitor":MONITOR}
    else:
        result=_BASE_LOSS_IDENTITY(loss_mode)
        # Parent wording refers to BOTH inputs; only the quantity-input claim remains true.
        result["input_log1p_and_quantity_head_unchanged"]=(_ACTIVE_TIME_INPUT=="log1p")
    result.update(quantity_input_and_head_unchanged=True,**feature_labels())
    return result


def weight_identity(alpha,loss_mode=None):
    require(type(alpha) in (int,float) and alpha==0.,"Time graph must remain excluded")
    mode=_parent._ACTIVE_LOSS_MODE if loss_mode is None else loss_mode
    return {"schema":"titantpp_late_path_quantity_only_identity_v1","time_alpha":0.,"quantity_lambda":1.,
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
    require(all(payload.get(k)==v for k,v in feature_labels().items()),"Late path input binding missing or changed")
    if full:
        interface,resume=payload.get("interface_meta",{}),payload.get("resume_identity",{})
        require(all(interface.get(k)==v and resume.get(k)==v and resume.get("arguments",{}).get(k)==v for k,v in feature_labels().items()),
            "Late path interface/resume binding changed")
        if "encoder_config" in payload:
            require(all(payload["encoder_config"].get(k)==v for k,v in feature_labels().items()),"Late path encoder metadata changed")


def validate_weight_binding(payload,alpha,loss_mode=None):
    result=_BASE_VALIDATE_BINDING(payload,alpha,loss_mode)
    validate_feature_binding(payload,full=True)
    return result


def annotate_history_row(row,alpha):
    row.update(**candidate_labels())
    return _BASE_ANNOTATE_ROW(row,alpha)


def annotate_payload(payload,alpha):
    payload.update(**candidate_labels())
    return _BASE_ANNOTATE_PAYLOAD(payload,alpha)


def training_args(contract,data,output,arm,*,time_alpha):
    args=_BASE_TRAINING_ARGS(contract,data,output,arm,time_alpha=time_alpha)
    args.execution_role="fresh_late_path18_validation"
    for key,value in feature_labels().items():setattr(args,key,value)
    return args


def base_receipt(contract,host,job,stage):
    result=_BASE_RECEIPT(contract,host,job,stage)
    result.update(**candidate_labels(job["arm"]),objective_family=loss_identity(job["loss_mode"])["objective_family"])
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
    architecture=contract.get("architecture",{});condition=contract.get("common_condition",{})
    require(condition.get("time_input") in TIME_INPUTS and condition.get("loss_mode") in LOSS_MODES
        and type(condition.get("time_alpha")) in (int,float) and condition["time_alpha"]==0.,"Invalid shared condition")
    require(architecture.get("time_input_transform")==TIME_INPUT_TRANSFORMS[condition["time_input"]]
        and architecture.get("quantity_input_transform")==QUANTITY_INPUT_TRANSFORM
        and architecture.get("quantity_loss_mode")==condition["loss_mode"]
        and architecture.get("quantity_loss_modes")==[condition["loss_mode"]],"Shared architecture condition mismatch")
    with candidate_factory(condition["time_input"]):
        result=_BASE_SELECT_JOB(contract,host,job_id)
    require(result[0]["loss_mode"]==condition["loss_mode"] and result[0].get("time_input")==condition["time_input"],"Job differs from shared condition")
    return result


@contextmanager
def runtime_hooks(loss_mode,*,time_input=None,arm=None):
    require(sha_file(_PARENT_PATH)==PARENT_SHA256,"Sealed parent source changed")
    with candidate_factory(time_input,arm),_BASE_RUNTIME_HOOKS(loss_mode):
        from paper.scripts.count_aware_tpp_backbone import training
        from paper.scripts import run_multilag_detail_execution as base
        original_interface,original_resume,original_build=base.time_interface,training._resume_identity,training.build_model
        def interface(*args,**kwargs):return {**original_interface(*args,**kwargs),**feature_labels()}
        def resume(*args,**kwargs):
            result=original_resume(*args,**kwargs);result.update(**feature_labels());result["arguments"].update(**feature_labels())
            return result
        def build(*args,**kwargs):
            model,metadata=original_build(*args,**kwargs)
            cls=RawTimeLateObservationPath if _ACTIVE_TIME_INPUT=="raw" else CountAwareTitanLateObservationPath
            require(type(model) is cls and model.late_path_mode==MODE_BY_ARM[args[0]],"Training used wrong candidate/input")
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
    with runtime_hooks(job["loss_mode"],time_input=job["time_input"],arm=job["arm"]):return inherited.run_stage(contract,host,job_id,stage)


def quantity_oracle(loss_mode,device="cpu",*,arm=None,time_input=None):
    arm=arm or _ACTIVE_ARM or ARMS[0]
    import torch
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.count_aware_tpp_backbone import core
    configure_runtime(device,4);devices=[] if str(device)=="cpu" else [torch.device(device)]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42);model,_=build_model(fabricated_data(),arm,time_input=time_input)
        torch.manual_seed(42);oracle,_=build_model(fabricated_data(),arm,time_input=time_input)
        model,oracle=model.to(device).train(),oracle.to(device).train()
        guard,ref_guard=TimeHeadGuard(model),TimeHeadGuard(oracle)
        optimizer=guard.bind_optimizer(torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01))
        ref_optimizer=ref_guard.bind_optimizer(torch.optim.AdamW(oracle.parameters(),lr=.001,weight_decay=.01))
        dts=torch.tensor([[1.,2.,16.,3.],[1.,32.,2.,1.]],device=device)
        q=torch.tensor([[1.,3.,2.,4.],[2.,4.,1.,3.]],device=device);mask=torch.ones_like(dts,dtype=torch.bool)
        original=core.target_outputs;shared_nonzero=False
        with runtime_hooks(loss_mode,arm=arm,time_input=time_input):
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
        with candidate_factory(time_input,arm):labels=candidate_labels(arm)
        return {"status":"passed","loss_mode":loss_mode,"loss_formula_and_adamw_oracle_exact":True,
            "shared_nonzero_gradient_observed":True,"time_head_no_training_verification":guard.proof(),**labels}


def synthetic_condition(contract,host,loss_mode=None,*,arm=None,device=None):
    selected=(contract or {}).get("common_condition",{})
    loss_mode=loss_mode or selected.get("loss_mode","log_mse")
    time_input=selected.get("time_input",_ACTIVE_TIME_INPUT)
    require(loss_mode==selected.get("loss_mode",loss_mode),"Synthetic loss differs from common condition")
    arm=arm or ARMS[0]
    require(arm in ARMS,"Synthetic explicit arm is required")
    selected_device=device or ("cuda:0" if contract is not None else "cpu")
    with runtime_hooks(loss_mode,time_input=time_input,arm=arm):
        old_arms=inherited.ARMS
        try:
            inherited.ARMS=(arm,)
            result=inherited.synthetic_check(contract,host,device=selected_device)
        finally:inherited.ARMS=old_arms
        labels=candidate_labels(arm)
    oracle=quantity_oracle(loss_mode,selected_device,arm=arm,time_input=time_input)
    result.update(schema=SCHEMA,loss_mode=loss_mode,loss_modes=[loss_mode],time_alphas=[0.],
        quantity_only=True,time_head_trained=False,time_nll_is_diagnostic=True,quantity_oracle=oracle,**labels)
    if str(selected_device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()
        for peak in result["cuda_memory_peaks"].values():
            peak["max_memory_reserved_bytes"]=max(peak["max_memory_reserved_bytes"],torch.cuda.max_memory_reserved())
            peak["max_memory_allocated_bytes"]=max(peak["max_memory_allocated_bytes"],torch.cuda.max_memory_allocated())
            peak["includes_quantity_oracle"]=True
    return result


def initialization_oracle():
    """Both paths retain parent common tensors/head and caller construction RNG."""
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    rows=[]
    with torch.random.fork_rng(devices=[]):
        for seed in SEEDS:
            torch.manual_seed(seed);reference,_=sealed_parent.build_model(fabricated_data(),"titantpp_history_mlp_width16");rng=torch.get_rng_state().clone()
            ref=reference.state_dict()
            for arm in ARMS:
                torch.manual_seed(seed);model,_=build_model(fabricated_data(),arm)
                state=model.state_dict();require(torch.equal(rng,torch.get_rng_state()),"Late path changed caller RNG")
                require(all(torch.equal(value,state[name]) for name,value in ref.items()),"Common initial tensor changed")
                require(sum(p.numel() for p in model.parameters())-sum(p.numel() for p in reference.parameters())==1024,"Path capacity mismatch")
                require(model.time_head_contract()==reference.time_head_contract(),"Duration head changed")
                rows.append({"seed":seed,"arm":arm,"candidate_initial_state_sha256":canonical_state_dict_sha256(state),
                    "parent_initial_state_sha256":canonical_state_dict_sha256(ref),"extra_parameter_count":1024,
                    "common_parameter_buffer_and_rng_exact":True,"time_head_contract_equal":True})
    return {"status":"passed","scope":"fabricated same-config construction only","real_data_loaded":False,"seeds":rows}


def synthetic_check(contract=None,host=None,*,device=None):
    selected=(contract or {}).get("common_condition",{})
    mode=selected.get("loss_mode","log_mse");time_input=selected.get("time_input","log1p")
    results={arm:synthetic_condition(contract,host,mode,arm=arm,device=device) for arm in ARMS}
    first=next(iter(results.values()))
    with candidate_factory(time_input):labels=feature_labels()
    return {"schema":SCHEMA,"status":"passed","device":first["device"],"host":host,"arms":list(ARMS),
        "loss_modes":[mode],"time_alphas":[0.],"per_arm":results,
        "cuda_memory_peaks":{a:r["cuda_memory_peaks"][a] for a,r in results.items()},
        "parameter_count":{a:r["parameter_count"][a] for a,r in results.items()},
        "initial_state_sha256":{a:r["initial_state_sha256"][a] for a,r in results.items()},
        "quantity_only":True,"time_head_trained":False,"time_nll_is_diagnostic":True,
        "initialization_oracle":initialization_oracle(),"real_data_loaded":False,"held_out_test_evaluated":False,**labels}


# Adapt only private namespaces, never the imported sealed-parent module.
for _module in (_parent,_parent._zero,_parent._zero._adapter):
    _module.SCHEMA,_module.CONTRACT_SCHEMA,_module.LOSS_MODES,_module.ARMS=SCHEMA,CONTRACT_SCHEMA,LOSS_MODES,ARMS
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
