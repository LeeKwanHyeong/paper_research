"""Explicit, lazy model selection. No global factory patching."""

from importlib import import_module


MODELS = {
    "phen": ("phen.models.model", "PHEN"),
    "rmtpp": ("phen.baselines.rmtpp", "RMTPP"),
    "thp": ("phen.baselines.thp", "THP"),
    "nhp": ("phen.baselines.nhp", "NHP"),
    "sahp": ("phen.baselines.sahp", "SAHP"),
    "s2p2_matched_head": ("phen.baselines.s2p2", "S2P2"),
    "attnhp_matched_head": ("phen.baselines.attnhp", "AttNHP"),
}


def build_model(name, *, max_seq_len, **kwargs):
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    module, symbol = MODELS[name]
    constructor = getattr(import_module(module), symbol)
    if name == "phen":
        kwargs["max_seq_len"] = max_seq_len
    return constructor(**kwargs)
