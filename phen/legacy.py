"""Explicit weight migration; never a training resume or optimizer migration."""


def translate_legacy_state(state):
    """Rename the full Shared15 PHEN weights, retaining strict load checks."""
    translated = {}
    for name, tensor in state.items():
        if name in {"core_ablation_identity", "shared_history_identity"}:
            continue
        if name.startswith("multilag_detail."):
            name = "history_correction." + name.removeprefix("multilag_detail.")
        elif name.startswith("lmm."):
            name = "prototype." + name.removeprefix("lmm.")
        if name in translated:
            raise ValueError("Duplicate migrated tensor: " + name)
        translated[name] = tensor
    return translated
