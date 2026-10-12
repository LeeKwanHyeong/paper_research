"""Pure, fit-local conditional ETA from an already validated saved snapshot.

No file, network, checkpoint or model is opened by this module. Epochs are
one-based, contiguous and complete; producer timing covers train, Validation
and checkpoint save. An ETA never establishes actual process termination.
"""
from __future__ import annotations

import math
import statistics

TIMING_SCOPE = "train_iteration_through_validation_and_checkpoint_save"
MINIMUM_EPOCHS = 40
MAXIMUM_EPOCHS = 300
PATIENCE = 40


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _completed_rows(payload, key, saved_epoch):
    if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
        raise ValueError(key + " rows missing")
    rows = payload[key]
    if len(rows) != saved_epoch or any(not isinstance(row, dict) for row in rows):
        raise ValueError(key + " row count differs from saved epoch; partial/current rows are not timing evidence")
    epochs = [row.get("epoch") for row in rows]
    if any(type(epoch) is not int for epoch in epochs) or sorted(epochs) != list(range(1, saved_epoch + 1)):
        raise ValueError(key + " epochs must be unique, contiguous and one-based")
    return sorted(rows, key=lambda row: row["epoch"])


def _early_stop_target_epoch(best_epoch):
    return min(MAXIMUM_EPOCHS, max(MINIMUM_EPOCHS, best_epoch + PATIENCE))


def estimate_fit_eta(history_payload: dict, timing_payload: dict, saved_epoch: int) -> dict:
    """Estimate only this fit, under unchanged best and unchanged recent pace.

    Nonfinite validation RMSE cannot be selected; ties retain the first finite
    minimum. Every supplied completed timing must be finite and positive. The
    median uses at most the last ten completed epochs, never a partial epoch.
    """
    result = {
        "status": "unconfirmed", "best_epoch": None,
        "saved_epoch": saved_epoch if type(saved_epoch) is int and 0 <= saved_epoch <= MAXIMUM_EPOCHS else None,
        "recent_epoch_count": 0, "median_epoch_seconds": None,
        "early_stop_target_epoch": None, "conditional_early_stop_seconds": None,
        "maximum_epoch_target": MAXIMUM_EPOCHS, "conditional_maximum_seconds": None,
        "condition": "각 fit의 현재 속도 유지; 조기 종료 시나리오는 best가 더 개선되지 않는 경우",
        "whole_queue_eta": None,
    }
    try:
        if type(saved_epoch) is not int or not 0 <= saved_epoch <= MAXIMUM_EPOCHS:
            raise ValueError("Saved epoch is invalid or outside the fixed300epoch contract")
        if saved_epoch == 0:
            raise ValueError("No completed checkpoint epoch has been saved")
        history = _completed_rows(history_payload, "history", saved_epoch)
        best_epoch, best_metric = None, math.inf
        for row in history:
            metric = row.get("val_qty_rmse")
            if _finite(metric) and metric >= 0 and metric < best_metric:
                best_epoch, best_metric = row["epoch"], metric
        if best_epoch is None:
            raise ValueError("No finite nonnegative full Validation quantity RMSE is available")
        result["best_epoch"] = best_epoch
        if not isinstance(timing_payload, dict) or timing_payload.get("scope") != TIMING_SCOPE:
            raise ValueError("Full epoch timing scope is missing or differs")
        timings = _completed_rows(timing_payload, "epochs", saved_epoch)
        for row in timings:
            seconds = row.get("elapsed_seconds")
            if not _finite(seconds) or seconds <= 0:
                raise ValueError("Completed epoch elapsed_seconds is missing, nonfinite or nonpositive")
            if "finished_at_unix" in row and not _finite(row["finished_at_unix"]):
                raise ValueError("An unfinished epoch row is not completion timing evidence")
        recent = timings[-10:]
        median = statistics.median(row["elapsed_seconds"] for row in recent)
        target = _early_stop_target_epoch(best_epoch)
        early_seconds = max(0, target - saved_epoch) * median
        maximum_seconds = max(0, MAXIMUM_EPOCHS - saved_epoch) * median
        if not _finite(median) or median <= 0 or not _finite(early_seconds) or not _finite(maximum_seconds):
            raise ValueError("Conditional ETA duration is not finite")
        result.update(status="estimated", recent_epoch_count=len(recent), median_epoch_seconds=median,
                      early_stop_target_epoch=target, conditional_early_stop_seconds=early_seconds,
                      conditional_maximum_seconds=maximum_seconds)
    except ValueError as exc:
        result["reason"] = str(exc)
    return result
