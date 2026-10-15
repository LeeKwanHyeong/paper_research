"""RAF location-pair comparison against immutable CNN, Channel and external references.

This adapter does not impersonate a fresh same-host CNN control. Its caller must
verify registry bytes/source SHA and native result/worker/gate evidence first.
No model loading, training, inference, or checkpoint selection occurs here.
"""
from __future__ import annotations

import math

from paper.scripts.compare_titantpp_backbone_loop import (
    METRICS, SEEDS, SPLITS, _validated_group, require, stats, verify_registry,
)

QUERY = "titantpp_cnn_query_nonlinear_v1"
VALUE = "titantpp_cnn_value_nonlinear_v1"
STORED_CHANNEL = "titantpp_cnn_channel_nonlinear_v1"
SELECTION_RULE = "passing_only:validation_qty_rmse_mean,validation_qty_mae_mean,arm_lexicographic"
NATIVE_RUNTIME = {"host":"5090", "gpu_uuid":"GPU-c9adc246-ef66-7906-bc07-6e152fa9952f",
                  "python":"3.12.13", "torch":"2.11.0+cu130", "cudnn":92000, "threads":4}
RUNTIME_KEYS = set(NATIVE_RUNTIME)


def normalized_native_row(receipt, host, *, stored=False):
    """Normalize a verified receipt while retaining its actual model and host."""
    require(receipt.get("status") == "complete" and receipt.get("native_runtime_checked") is True,
            "Native completed receipt required")
    runtime = receipt["runtime"]
    require(host == "5090" and receipt["arm"] in ((STORED_CHANNEL,) if stored else (QUERY, VALUE)), "Foreign RAF6 result")
    return {
        "dataset": receipt["dataset"], "model": receipt["arm"],
        "added_parameters": 3072 if stored else 1024,
        "split": receipt["split"], "seed": receipt["seed"], "status": "complete",
        "validation_selected_checkpoint_verified": True,
        "selected_epoch": receipt["selected_epoch"], "checkpoint_sha256": receipt["checkpoint_sha256"],
        "metrics": {m: receipt["metrics"][m] for m in METRICS}, "count": receipt["metrics"]["count"],
        "population": {k: receipt[k] for k in ("target_identity_sha256", "truth_sha256", "time_sha256")},
        "runtime": {"host": host, "gpu_uuid": runtime["gpu"]["uuid"], "python": runtime["python"],
                    "torch": runtime["torch"], "cudnn": runtime["cudnn"], "threads": runtime["threads"]},
    }


def compare_historical_rows(registry, candidate_rows, query_rows=None, stored_channel_rows=None):
    """Keep the user's four strict performance checks and attribution distinct."""
    require(registry.get("status") == "verified" and registry.get("seeds") == list(SEEDS)
            and registry.get("dataset") == "raf_spare_parts",
            "Verified fixed three-seed registry required")
    candidate = _validated_group(candidate_rows, registry)
    candidate_arm = candidate_rows[0]["model"]
    require(candidate_arm in (QUERY, VALUE), "Only approved RAF6 candidates")
    query = _validated_group(query_rows, registry) if query_rows is not None else None
    if query is not None:
        require(candidate_arm == VALUE and query_rows[0]["model"] == QUERY,
                "Attribution contrast must be value minus query")
    stored = _validated_group(stored_channel_rows, registry) if stored_channel_rows is not None else None
    if stored is not None:
        require(stored_channel_rows[0]["model"] == STORED_CHANNEL, "Stored reference must be original QKV Channel")
        for row in stored.values():
            require(row.get("reference_role") == "stored_loop1_channel_immutable_reference"
                    and row.get("retrained") is False and row.get("reevaluated") is False
                    and row.get("added_parameters") == 3072
                    and row.get("original_evidence_verified") is True,
                    "Stored Channel provenance/parameter identity missing")
    for group in (candidate,) + ((query,) if query is not None else ()):
        require(all(row.get("added_parameters") == 1024 for row in group.values()),
                "Candidate/control added parameter count must be 1024")
    for group in (candidate,) + ((query,) if query is not None else ()) + ((stored,) if stored is not None else ()):
        for row in group.values():
            require(row["runtime"] == NATIVE_RUNTIME, "Pinned native5090 Runtime identity mismatch")
    history_rows = [r for r in registry["rows"] if r["model"] == "historical_CNN"]
    history = {(r["split"], r["seed"]): r for r in history_rows}
    require(len(history_rows) == len(history) == 6
            and set(history) == {(s, seed) for s in SPLITS for seed in SEEDS},
            "Historical CNN3 must exist on both splits")
    for seed in SEEDS:
        v, t = history[("validation", seed)], history[("test", seed)]
        require((v["selected_epoch"], v["checkpoint_sha256"]) ==
                (t["selected_epoch"], t["checkpoint_sha256"]), "Historical CNN checkpoint differs across splits")
    external_models = sorted({r["model"] for r in registry["rows"] if r["model"] != "historical_CNN"})
    require(bool(external_models), "External references missing")
    external_stats = {}
    for split in SPLITS:
        for metric in METRICS:
            for model in external_models:
                rows = [r for r in registry["rows"] if r["model"] == model and r["split"] == split]
                require(len(rows) == 3 and sorted(r["seed"] for r in rows) == list(SEEDS),
                        "External frontier seed reference missing/duplicate")
                external_stats[(split, metric, model)] = stats([r["metrics"][metric] for r in rows])
            frontier = registry["external_frontier"][split][metric]
            best = min(external_stats[(split,metric,m)]["mean"] for m in external_models)
            winners = sorted(m for m in external_models if external_stats[(split,metric,m)]["mean"] == best)
            require(frontier["mean"] == best and sorted(frontier["models"]) == winners
                    and len(frontier["models"]) == len(set(frontier["models"]))
                    and frontier["sample_sd"] == {m:external_stats[(split,metric,m)]["sample_sd"] for m in winners},
                    "External metricwise frontier differs from pinned seed rows")
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
            if query is not None:
                control = stats([query[(split, s)]["metrics"][metric] for s in SEEDS])
                item.update(query_control=control, query_attribution_pass=cs["mean"] < control["mean"])
                attribution_checks.append(item["query_attribution_pass"])
            if stored is not None:
                stored_stats = stats([stored[(split, s)]["metrics"][metric] for s in SEEDS])
                item.update(stored_QKV_Channel=stored_stats,
                            stored_channel_auxiliary_pass=cs["mean"] < stored_stats["mean"])
            summaries[split][metric] = item
            for seed in SEEDS:
                key = (split, seed)
                refs = {"historical_CNN": history[key]["metrics"][metric]}
                for model in frontier["models"]:
                    rows = [r for r in registry["rows"] if
                            (r["model"], r["split"], r["seed"]) == (model, split, seed)]
                    require(len(rows) == 1, "External frontier seed reference missing/duplicate")
                    refs["external_frontier:" + model] = rows[0]["metrics"][metric]
                if query is not None:
                    require(candidate[key]["runtime"] == query[key]["runtime"],
                            "Value/query native Runtime mismatch")
                    refs["query_control"] = query[key]["metrics"][metric]
                if stored is not None:
                    require(candidate[key]["runtime"] == stored[key]["runtime"],
                            "Candidate/stored Channel native Runtime mismatch")
                    refs["stored_QKV_Channel_auxiliary"] = stored[key]["metrics"][metric]
                value = candidate[key]["metrics"][metric]
                differences.append({"split": split, "seed": seed, "metric": metric, "candidate": value,
                    "differences": {name: {"reference": ref, "absolute_candidate_minus_reference": value-ref,
                        "relative_percent_candidate_minus_reference": 100*(value-ref)/ref if ref else None}
                        for name, ref in refs.items()}})
    return {"status": "complete", "dataset": registry["dataset"], "candidate_model": candidate_arm,
        "campaign_new_fit_count":6, "candidate_fit_count":3, "historical_cnn_reference_count":3,
        "fresh_same_runtime_cnn_fit_count":0, "same_host_query_control_performed":query is not None,
        "stored_channel_reference_count":3 if stored is not None else 0,
        "summaries": summaries, "seed_differences": differences,
        "user_performance_gate": all(user_checks), "user_performance_check_count": len(user_checks),
        "value_minus_query_attribution_gate": all(attribution_checks) if query is not None else None,
        "same_host_control_gate": None, "same_host_control_performed": False,
        "same_host_control_reason": "User scope reuses historical CNN3; no CNN retraining or reevaluation.",
        "historical_cnn_role": "immutable_reference", "historical_cnn_retrained": False,
        "historical_cnn_reevaluated": False,
        "runtime_confound": "Historical CNN/external:5080 cuDNN91900; candidates:5090 cuDNN92000. Source/Runtime differences remain; value/query is the same-host attribution contrast.",
        "stored_channel_reused": stored is not None, "stored_channel_retrained": False,
        "stored_channel_reevaluated": False, "candidate_added_parameters": 1024,
        "stored_channel_added_parameters": 3072 if stored is not None else None,
        "stored_channel_confound": "Auxiliary contrast changes nonlinear location and added capacity (3072 to 1024); not a matched capacity attribution.",
        "test_independent": False, "statistical_significance_established": False,
        "interpretation": "Four user performance criteria and value-minus-query attribution are distinct; adverse seeds are retained."}


def select_passing_candidate(comparisons):
    """Test is an eligibility gate only; the fixed tie-break uses Validation only."""
    require(set(comparisons) == {QUERY, VALUE}, "Both approved location comparisons required")
    passing = []
    for arm, result in comparisons.items():
        require(result.get("status") == "complete" and result["candidate_model"] == arm and result["user_performance_check_count"] == 4,
                "Comparison identity/four-check count differs")
        checks = [result["summaries"][split][metric]["user_performance_pass"]
                  for split in SPLITS for metric in METRICS]
        require(all(type(v) is bool for v in checks)
                and type(result["user_performance_gate"]) is bool
                and result["user_performance_gate"] == all(checks), "Four-check gate inconsistent")
        recalculated = []
        for split in SPLITS:
            for metric in METRICS:
                item = result["summaries"][split][metric]
                means = [item[role]["mean"] for role in ("candidate","historical_CNN","external_frontier")]
                require(all(type(x) in (int,float) and math.isfinite(x) and x >= 0 for x in means),
                        "Invalid comparison mean")
                recalculated.append(means[0] < means[1] and means[0] < means[2])
        require(checks == recalculated, "User gate differs from metric means")
        if arm == VALUE:
            attribution = result["value_minus_query_attribution_gate"]
            require(type(attribution) is bool, "Value requires a complete query attribution comparison")
            if attribution is not None:
                per_metric = []
                for split in SPLITS:
                    for metric in METRICS:
                        item = result["summaries"][split][metric]
                        control_mean = item["query_control"]["mean"]
                        require(type(control_mean) in (int,float) and math.isfinite(control_mean) and control_mean >= 0,
                                "Invalid query control mean")
                        require(item["query_control"] == comparisons[QUERY]["summaries"][split][metric]["candidate"],
                                "Value query control differs from supplied Query comparison")
                        per_metric.append(item["candidate"]["mean"] < control_mean)
                require(attribution == all(per_metric), "Value attribution gate differs from metric means")
        if result["user_performance_gate"]:
            vals = result["summaries"]["validation"]
            passing.append((vals["qty_rmse"]["candidate"]["mean"],
                            vals["qty_mae"]["candidate"]["mean"], arm))
    chosen = min(passing)[2] if passing else None
    return {"selection_rule": SELECTION_RULE, "passing_candidates": sorted(v[2] for v in passing),
            "selected_candidate": chosen,
            "value_user_criteria_pass": comparisons[VALUE]["user_performance_gate"],
            "value_minus_query_attribution_gate": comparisons[VALUE]["value_minus_query_attribution_gate"],
            "value_hypothesis_supported": comparisons[VALUE]["user_performance_gate"] is True
                and comparisons[VALUE]["value_minus_query_attribution_gate"] is True,
            "query_only_pass_is_not_value_success": True,
            "test_used_for_tie_break": False}
