"""Read-only aggregate diagnostics for the frozen CNN+GRU54 history models.

No model forward is replaced, no parameter is changed, and no raw target
prediction row is persisted. Residual norms are vector magnitudes, not signed
raw-quantity effects. Module import itself needs only the standard library.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from contextlib import contextmanager
import math

FULL = "titantpp_cnn_gru54"
CONTROL_ARMS = (FULL + "_correction_disabled", FULL + "_current_only")


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ReadOnlyTelemetry:
    """Capture the actual forward tensors, then describe detached values.

    Frozen static_hard_lmm returns identical time and quantity task states.
    Its duration law calls F.linear directly, bypassing v_t module hooks. The
    actual quantity-head input is therefore the verified shared duration input;
    frozen _lognormal_time_terms is applied to its detached tensor *after* the
    original forward. This does not replace or rerun the encoder/head forward.
    CNN residual arithmetic similarly uses the frozen helper on detached actual
    QKV projections and actual observed context. The actual GRU correction is
    read directly from its module output, without any recurrent replay.
    """
    def __init__(self, model, arm, torch, residual_function):
        require(arm == FULL or arm in CONTROL_ARMS, "Foreign telemetry arm")
        require(model.memory_mode == "static_hard_lmm", "Shared task states must be the frozen static_hard_lmm route")
        require(model.time_head_mode == "heteroscedastic_lognormal_duration", "Frozen duration head differs")
        require(not model.training and all(not p.requires_grad for p in model.parameters()), "Telemetry requires eval and gradients disabled")
        self.model, self.arm, self.torch = model, arm, torch
        self.residual_function = residual_function
        self.enabled, self.values, self.handles = True, {}, []
        self.handles.append(model.quantity_head.register_forward_pre_hook(self._quantity_input))
        self.handles.append(model.quantity_head.register_forward_hook(self._quantity_output))
        if arm == FULL:
            attention = model.encoder.layers[0].attn
            self.handles.append(attention.register_forward_pre_hook(self._attention_context, with_kwargs=True))
            self.handles.append(attention.qkv.register_forward_hook(self._projection))
            self.handles.append(model.multilag_detail.register_forward_pre_hook(self._correction_input))
            self.handles.append(model.multilag_detail.register_forward_hook(self._correction_output))

    def _capture(self, name, value):
        if self.enabled:
            # Only detached aliases are retained until snapshot; no tensor writes.
            self.values[name] = value.detach()

    def _quantity_input(self, module, args):
        self._capture("shared_head_hidden", args[0])

    def _quantity_output(self, module, args, output):
        self._capture("quantity_head_logit", output.squeeze(-1))

    def _attention_context(self, module, args, kwargs):
        if self.enabled:
            self._capture("observed_sources", kwargs["observed_sources"])
            self._capture("observed_available", kwargs["observed_available"])

    def _projection(self, module, args, output):
        self._capture("qkv", output)

    def _correction_input(self, module, args):
        self._capture("gru_base", args[0])

    def _correction_output(self, module, args, output):
        self._capture("gru_correction", output)

    def begin(self):
        self.values.clear()

    @contextmanager
    def suspended(self):
        old = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = old

    def snapshot(self, history_lengths, recorded_gap):
        torch = self.torch
        require(not torch.is_grad_enabled(), "Diagnostic arithmetic must have gradients disabled")
        hidden = self.values["shared_head_hidden"]
        require(hidden.ndim == 2 and hidden.shape[0] == len(history_lengths), "Actual target hidden shape differs")
        # Exact frozen float64 law, observing parameters only. No head forward.
        location, scale, _ = self.model._lognormal_time_terms(hidden, recorded_gap)
        vectors = {"duration_location": location, "duration_scale": scale,
                   "quantity_head_logit": self.values["quantity_head_logit"],
                   "duration_standardized_log_residual": (torch.log(recorded_gap.double() / self.model.time_scale) - location) / scale,
                   "shared_head_hidden_norm": torch.linalg.vector_norm(hidden.double(), dim=-1)}
        if self.arm == FULL:
            positions = history_lengths.long() - 1
            ids = torch.arange(len(positions), device=positions.device)
            sources, available = self.values["observed_sources"], self.values["observed_available"]
            require(bool(available[ids, positions, 0].all()), "Final observed position unavailable")
            for name, base in zip(("q", "k", "v"), self.values["qkv"].chunk(3, dim=-1)):
                correction = self.residual_function(base, getattr(self.model.encoder.layers[0].attn, "causal_" + name + "_kernel").detach(), sources, available)
                base_norm = torch.linalg.vector_norm(base[ids, positions].double(), dim=-1)
                residual_norm = torch.linalg.vector_norm(correction[ids, positions].double(), dim=-1)
                vectors["cnn_" + name + "_base_norm"] = base_norm
                vectors["cnn_" + name + "_residual_norm"] = residual_norm
                # Explicit zero-denominator rule; no fabricated large/finite ratio.
                require(bool((base_norm > 0).all()), "Zero CNN base norm cannot support a residual/base ratio")
                vectors["cnn_" + name + "_residual_base_ratio"] = residual_norm / base_norm
            base = torch.linalg.vector_norm(self.values["gru_base"][ids, positions].double(), dim=-1)
            residual = torch.linalg.vector_norm(self.values["gru_correction"][ids, positions].double(), dim=-1)
            vectors.update(gru_base_norm=base, gru_correction_norm=residual)
            require(bool((base > 0).all()), "Zero GRU base norm cannot support a correction/base ratio")
            vectors["gru_correction_base_ratio"] = residual / base
        for name, value in vectors.items():
            require(value.ndim == 1 and value.shape[0] == hidden.shape[0] and bool(torch.isfinite(value).all()), "Nonfinite or malformed telemetry: " + name)
        return {name: value.detach().cpu().double().tolist() for name, value in vectors.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values.clear()

    def interpretation(self):
        return {"correction_norms": "observed_full_historical_route" if self.arm == FULL else "inapplicable_to_full_historical_route; control has different or bypassed correction",
                "control_norm_zero_fabricated": False,
                "residual_norm_is_signed_raw_quantity_effect": False,
                "time_parameters": "frozen_float64_terms_from_actual_shared_head_hidden",
                "no_forward_replacement": True, "no_encoder_or_GRU_replay": True}


class AggregateDiagnostics:
    """One-dimensional strata and specified joint strata, all target coverage."""
    def __init__(self, quantity_boundaries, history_boundaries, input_gap_boundaries, target_gap_boundaries, train_history_max):
        for boundaries in (quantity_boundaries, history_boundaries, input_gap_boundaries, target_gap_boundaries):
            require(all(math.isfinite(x) for x in boundaries) and list(boundaries) == sorted(set(boundaries)), "Diagnostic boundaries must be finite and increasing")
        self.quantity_boundaries = list(quantity_boundaries)
        self.history_boundaries = list(history_boundaries)
        self.input_gap_boundaries = list(input_gap_boundaries)
        self.target_gap_boundaries = list(target_gap_boundaries)
        self.groups = defaultdict(dict)
        self.train_history_max = int(train_history_max)
        require(self.train_history_max >= 1, "Frozen train history maximum invalid")
        self.processed = 0
        self.features = None

    def add(self, *, entity, calendar, last_input_calendar, history_length, unclipped_history_length,
            truncated, series_in_validation, true_quantity, predicted_quantity,
            time_nll, recorded_gap, last_input_gap, last_quantity, mean_quantity,
            features):
        base = {"true_quantity": float(true_quantity), "predicted_quantity": float(predicted_quantity),
                "signed_error": float(predicted_quantity - true_quantity),
                "squared_error": float((predicted_quantity - true_quantity) ** 2),
                "absolute_error": float(abs(predicted_quantity - true_quantity)),
                "time_nll": float(time_nll), "squared_time_nll": float(time_nll * time_nll), "recorded_target_gap": float(recorded_gap),
                "last_input_gap": float(last_input_gap), "last_quantity": float(last_quantity),
                "mean_quantity": float(mean_quantity), "history_length": float(history_length),
                "unclipped_history_length": float(unclipped_history_length)}
        require(all(math.isfinite(v) for v in (*base.values(), *features.values())), "Nonfinite target diagnostic value")
        feature_names = tuple(sorted(features))
        if self.features is None:
            self.features = feature_names
        require(self.features == feature_names, "Diagnostic feature set changed between targets")
        qbin = str(bisect_left(self.quantity_boundaries, true_quantity))
        hbin = str(bisect_left(self.history_boundaries, history_length))
        ibin = str(bisect_left(self.input_gap_boundaries, last_input_gap))
        gbin = str(bisect_left(self.target_gap_boundaries, recorded_gap))
        clip, membership = ("yes" if truncated else "no"), ("shared" if series_in_validation else "absent_validation")
        support = "within_train_max" if history_length <= self.train_history_max else "above_train_max"
        strata = [("all", "all"), ("series", str(entity)),
                  ("recorded_target_calendar", str(calendar)), ("last_observed_input_calendar", str(last_input_calendar)),
                  ("quantity_bin", qbin), ("history_bin", hbin), ("last_input_gap_bin", ibin),
                  ("recorded_target_gap_bin", gbin), ("left_truncated", clip), ("series_validation_membership", membership),
                  ("history_vs_train_max", support), ("quantity_by_history_vs_train_max", qbin + "/" + support),
                  ("quantity_by_history_by_truncation", qbin + "/" + hbin + "/" + clip),
                  ("quantity_by_series_validation_membership", qbin + "/" + membership)]
        if qbin == str(len(self.quantity_boundaries)):
            strata.append(("tail_by_series", str(entity)))
        for dimension, key in strata:
            cell = self.groups[dimension].setdefault(key, {"count": 0, "sums": {k: 0. for k in base}, "features": {}})
            cell["count"] += 1
            for name, value in base.items():
                cell["sums"][name] += value
            for name, value in features.items():
                a = cell["features"].setdefault(name, {"count": 0, "sum": 0., "sumsq": 0., "minimum": value, "maximum": value,
                    "sum_product_signed_error": 0., "sum_product_time_nll": 0.})
                a["count"] += 1
                a["sum"] += value
                a["sumsq"] += value * value
                a["minimum"], a["maximum"] = min(a["minimum"], value), max(a["maximum"], value)
                a["sum_product_signed_error"] += value * base["signed_error"]
                a["sum_product_time_nll"] += value * base["time_nll"]
        self.processed += 1

    def finish(self):
        require(self.processed > 0, "Empty diagnostics")
        result = {}
        for dimension, groups in self.groups.items():
            result[dimension] = []
            for key, raw in sorted(groups.items()):
                n, s = raw["count"], raw["sums"]
                cell = {"key": key, **raw, "qty_bias": s["signed_error"] / n,
                        "qty_mae": s["absolute_error"] / n, "qty_rmse": math.sqrt(s["squared_error"] / n),
                        "time_nll": s["time_nll"] / n}
                cell["features"] = {name: {**a, "mean": a["sum"] / n,
                    "population_variance": max(0., a["sumsq"] / n - (a["sum"] / n) ** 2),
                    "covariance_with_signed_error": a["sum_product_signed_error"] / n - (a["sum"] / n) * s["signed_error"] / n,
                    "covariance_with_time_nll": a["sum_product_time_nll"] / n - (a["sum"] / n) * s["time_nll"] / n}
                    for name, a in raw["features"].items()}
                result[dimension].append(cell)
            if dimension != "tail_by_series":
                require(sum(x["count"] for x in result[dimension]) == self.processed, "Diagnostic stratum coverage differs")
        return {"schema": "aggregate_association_telemetry_v1", "count": self.processed,
                "quantity_boundaries": self.quantity_boundaries, "input_history_boundaries": self.history_boundaries,
                "input_last_gap_boundaries": self.input_gap_boundaries, "target_gap_boundaries": self.target_gap_boundaries, "bin_rule": "first <= boundary; subsequent (previous,boundary]; last > boundary",
                "target_quantity_and_recorded_target_calendar_and_gap_bins_are_posthoc": True,
                "input_bins_are_forecast_available": True, "train_history_max": self.train_history_max,
                "association_is_causal_effect": False, "raw_prediction_rows_written": False, "strata": result}
