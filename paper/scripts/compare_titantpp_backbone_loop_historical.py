"""RAF6 comparison against immutable historical CNN/external references.

This adapter does not impersonate a fresh same-host CNN control. Its caller must
verify registry bytes/source SHA and native result/worker/gate evidence first.
No model loading, training, inference, or checkpoint selection occurs here.
"""
from __future__ import annotations

from paper.scripts.compare_titantpp_backbone_loop import (
    METRICS, SEEDS, SPLITS, _validated_group, require, stats, verify_registry,
)

CHANNEL = "titantpp_cnn_channel_nonlinear_v1"
JOINT = "titantpp_cnn_joint_nonlinear_v1"
RUNTIME_KEYS = {"host", "gpu_uuid", "python", "torch", "cudnn", "threads"}


def normalized_native_row(receipt, host):
    """Normalize a verified receipt while retaining its actual model and host."""
    runtime = receipt["runtime"]
    require(host == "5090" and receipt["arm"] in (CHANNEL, JOINT), "Foreign RAF6 result")
    return {
        "dataset": receipt["dataset"], "model": receipt["arm"],
        "split": receipt["split"], "seed": receipt["seed"], "status": "complete",
        "validation_selected_checkpoint_verified": True,
        "selected_epoch": receipt["selected_epoch"], "checkpoint_sha256": receipt["checkpoint_sha256"],
        "metrics": {m: receipt["metrics"][m] for m in METRICS}, "count": receipt["metrics"]["count"],
        "population": {k: receipt[k] for k in ("target_identity_sha256", "truth_sha256", "time_sha256")},
        "runtime": {"host": host, "gpu_uuid": runtime["gpu"]["uuid"], "python": runtime["python"],
                    "torch": runtime["torch"], "cudnn": runtime["cudnn"], "threads": runtime["threads"]},
    }


def compare_historical_rows(registry, candidate_rows, channel_rows=None):
    """Keep the user's four strict performance checks and attribution distinct."""
    require(registry.get("status") == "verified" and registry.get("seeds") == list(SEEDS),
            "Verified fixed three-seed registry required")
    candidate = _validated_group(candidate_rows, registry)
    candidate_arm = candidate_rows[0]["model"]
    require(candidate_arm in (CHANNEL, JOINT), "Only approved RAF6 candidates")
    channel = _validated_group(channel_rows, registry) if channel_rows is not None else None
    if channel is not None:
        require(candidate_arm == JOINT and channel_rows[0]["model"] == CHANNEL,
                "Attribution contrast must be joint minus channel")
    for group in (candidate,) + ((channel,) if channel is not None else ()):
        for row in group.values():
            require(set(row["runtime"]) == RUNTIME_KEYS
                    and all(v is not None for v in row["runtime"].values())
                    and row["runtime"]["host"] == "5090", "Native5090 Runtime identity required")
    history_rows = [r for r in registry["rows"] if r["model"] == "historical_CNN"]
    history = {(r["split"], r["seed"]): r for r in history_rows}
    require(len(history_rows) == len(history) == 6
            and set(history) == {(s, seed) for s in SPLITS for seed in SEEDS},
            "Historical CNN3 must exist on both splits")
    for seed in SEEDS:
        v, t = history[("validation", seed)], history[("test", seed)]
        require((v["selected_epoch"], v["checkpoint_sha256"]) ==
                (t["selected_epoch"], t["checkpoint_sha256"]), "Historical CNN checkpoint differs across splits")
    summaries, differences, user_checks, attribution_checks = {}, [], [], []
    for split in SPLITS:
        summaries[split] = {}
        for metric in METRICS:
            cs = stats([candidate[(split, s)]["metrics"][metric] for s in SEEDS])
            historical = registry["historical_cnn"][split][metric]
            require(historical == stats([history[(split, s)]["metrics"][metric] for s in SEEDS]),
                    "Historical CNN aggregate differs from pinned rows")
            frontier = registry["external_frontier"][split][metric]
            require(frontier["models"], "External frontier missing")
            user_pass = cs["mean"] < historical["mean"] and cs["mean"] < frontier["mean"]
            user_checks.append(user_pass)
            item = {"candidate": cs, "historical_CNN": historical, "external_frontier": frontier,
                    "user_performance_pass": user_pass}
            if channel is not None:
                control = stats([channel[(split, s)]["metrics"][metric] for s in SEEDS])
                item.update(channel_control=control, channel_attribution_pass=cs["mean"] < control["mean"])
                attribution_checks.append(item["channel_attribution_pass"])
            summaries[split][metric] = item
            for seed in SEEDS:
                key = (split, seed)
                refs = {"historical_CNN": history[key]["metrics"][metric]}
                for model in frontier["models"]:
                    rows = [r for r in registry["rows"] if
                            (r["model"], r["split"], r["seed"]) == (model, split, seed)]
                    require(len(rows) == 1, "External frontier seed reference missing/duplicate")
                    refs["external_frontier:" + model] = rows[0]["metrics"][metric]
                if channel is not None:
                    require(candidate[key]["runtime"] == channel[key]["runtime"],
                            "Joint/channel native Runtime mismatch")
                    refs["channel_control"] = channel[key]["metrics"][metric]
                value = candidate[key]["metrics"][metric]
                differences.append({"split": split, "seed": seed, "metric": metric, "candidate": value,
                    "differences": {name: {"reference": ref, "absolute_candidate_minus_reference": value-ref,
                        "relative_percent_candidate_minus_reference": 100*(value-ref)/ref if ref else None}
                        for name, ref in refs.items()}})
    return {"status": "complete", "dataset": registry["dataset"], "candidate_model": candidate_arm,
        "summaries": summaries, "seed_differences": differences,
        "user_performance_gate": all(user_checks), "user_performance_check_count": len(user_checks),
        "channel_attribution_gate": all(attribution_checks) if channel is not None else None,
        "same_host_control_gate": None, "same_host_control_performed": False,
        "same_host_control_reason": "User scope reuses historical CNN3; no CNN retraining or reevaluation.",
        "historical_cnn_role": "immutable_reference", "historical_cnn_retrained": False,
        "historical_cnn_reevaluated": False,
        "runtime_confound": "Historical CNN/external:5080 cuDNN91900; candidates:5090 cuDNN92000. Source/Runtime differences remain; joint/channel is the same-host attribution contrast.",
        "test_independent": False, "statistical_significance_established": False,
        "interpretation": "Four user performance criteria and channel attribution are distinct; adverse seeds are retained."}
