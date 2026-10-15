"""Local preparation adapter only. No data reads, remote operations, or fit CLI."""
import inspect

from models.TPPs.CountAwareTitanAdaptiveHistoryCNN import (
    CNN, ARMS, CountAwareTitanAdaptiveHistoryCNN, metadata, validate_checkpoint,
)
from paper.scripts import run_titantpp_cnn_gru_components as original


def build_model(data,arm):
    if arm==CNN:
        return original.build_model(data,arm)
    if arm not in ARMS:
        raise ValueError("Unknown preparation arm")
    original.install_hooks()
    from models.TPPs import CountAwareFactory as factory
    signature=inspect.getclosurevars(factory.build_count_aware_model).nonlocals.get("signature")
    if not isinstance(signature,inspect.Signature):
        raise ValueError("Original model factory signature unavailable")
    config={k:v for k,v in data["model"].items() if k not in
            ("backbone","lambda_log_qty","lambda_tail","time_head_lr_multiplier")}
    bound=signature.bind(CNN,**config,train_log_mean=data["statistics"]["train_log_mean"],
                         train_log_std=data["statistics"]["train_log_std"],max_seq_len=data["loader"]["max_seq_len"])
    bound.apply_defaults(); values=dict(bound.arguments)
    values.pop("backbone")
    if values.pop("titans_memory_gradient_clip") is not None:
        raise ValueError("Original memory clipping must remain unchanged")
    model=CountAwareTitanAdaptiveHistoryCNN(**values,history_arm=arm)
    return factory.with_time_metadata(model,{**metadata(arm=arm),"max_len":values["max_seq_len"]})


def run_fit(*args,**kwargs):
    raise PermissionError("Preparation only: new training requires the user's later decision and a native execution permit")
