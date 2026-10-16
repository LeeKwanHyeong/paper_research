"""Fail-closed Validation-only summary for the fixed GRU residual-cap study.

The retrieval index deliberately binds locally recovered terminal artifacts rather
than accepting a hand-written metric summary.  It must contain exactly the nine
candidate job records specified by the frozen execution contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


KIND = "titantpp_gru_residual_cap_validation_summary_v2"
INDEX_KIND = "titantpp_gru_residual_cap_validation_retrieval_index_v1"
CANDIDATE = "titantpp_cnn_gru54_residual_cap025"
BASELINES = ("titantpp_cnn_gru54", "titantpp_cnn_gru54_cnn_only")
METRICS = (
    "whole_qty_rmse", "whole_qty_mae", "tail_qty_rmse",
    "tail_qty_mae", "whole_time_nll", "tail_time_nll",
)


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    ).encode()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def confined(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "Artifact path escapes retrieval root")
    require(not path.is_symlink(), "Symlinked artifact is not accepted")
    return path


def verified(root: Path, name: str, digest: str) -> Path:
    require(isinstance(name, str) and isinstance(digest, str) and len(digest) == 64,
            "Artifact binding is malformed")
    path = confined(root, name)
    require(path.is_file() and sha(path) == digest, "Artifact SHA differs: " + name)
    return path


def strict_selected_epoch(history: list[dict[str, Any]]) -> tuple[int, float]:
    best: tuple[int, float] | None = None
    previous = 0
    for row in history:
        epoch, value = row.get("epoch"), row.get("val_qty_rmse")
        require(type(epoch) is int and epoch > previous, "Saved history epochs are invalid")
        previous = epoch
        if type(value) not in (int, float) or not math.isfinite(value):
            continue
        if best is None or value < best[1]:
            best = (epoch, float(value))
    require(best is not None, "No finite full Validation quantity RMSE in history")
    return best


def selection_history(path: Path) -> list[dict[str, Any]]:
    value = read(path)
    value = value["history"] if isinstance(value, dict) and "history" in value else value
    require(isinstance(value, list) and all(isinstance(row, dict) for row in value),
            "Selection history must be a JSON row list")
    return value


def metrics(value: Any) -> dict[str, float]:
    require(isinstance(value, dict) and set(value) == set(METRICS),
            "Validation metric keys differ from the fixed six metrics")
    result: dict[str, float] = {}
    for key in METRICS:
        cell = value[key]
        require(type(cell) in (int, float) and math.isfinite(cell) and cell >= 0,
                "Validation metric is not finite and non-negative: " + key)
        result[key] = float(cell)
    return result


def baseline_metrics(value: Any, expected_count: int) -> dict[str, float]:
    """Read either frozen flat references or native nested Validation receipts.

    This only maps stored values; it never recomputes or selects a metric.
    The receipt SHA and the exact frozen six values are checked by the caller.
    """
    require(isinstance(value, dict), "Baseline metrics must be an object")
    if set(value) == set(METRICS):
        return metrics(value)
    native_keys = {
        "split", "count", "represented_n", "qty_sse", "qty_absolute_error_sum",
        "qty_signed_error_sum", "time_nll_sum", "qty_rmse", "qty_mae", "qty_bias",
        "time_nll", "quantity_boundaries", "quantity_cells", "tail",
        "tail_definition", "bias_definition",
    }
    tail_keys = {"count", "represented_n", "qty_sse", "qty_absolute_error_sum",
                 "qty_signed_error_sum", "time_nll_sum", "qty_rmse", "qty_mae",
                 "qty_bias", "time_nll"}
    require(set(value) == native_keys and isinstance(value.get("tail"), dict)
            and set(value["tail"]) == tail_keys, "Native baseline metric schema differs")
    require(value["split"] == "validation", "Native baseline is not Validation-only")
    require(type(value["count"]) is int and value["count"] == expected_count,
            "Native baseline metric target count differs")
    return endpoint_metrics(value)


def verify_baselines(contract: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows = contract.get("baseline_references")
    require(isinstance(rows, list) and canonical(rows) == contract.get("baseline_references_sha256"),
            "Frozen baseline reference closure differs")
    expected = {(job["dataset"], job["seed"], arm)
                for job in contract["jobs"] for arm in BASELINES}
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("dataset"), row.get("seed"), row.get("arm"))
        require(key in expected and key not in result, "Baseline reference is foreign or duplicated")
        for name in ("history_path", "validation_receipt_path"):
            require(isinstance(row.get(name), str), "Baseline artifact path is missing")
        history_path = Path(row["history_path"])
        replay_path = Path(row["validation_receipt_path"])
        require(history_path.is_file() and sha(history_path) == row.get("history_sha256"),
                "Frozen baseline history SHA differs")
        require(replay_path.is_file() and sha(replay_path) == row.get("validation_receipt_sha256"),
                "Frozen baseline Validation receipt SHA differs")
        selected_epoch, _ = strict_selected_epoch(selection_history(history_path))
        require(selected_epoch == row.get("selected_epoch"), "Frozen baseline selected epoch differs")
        replay = read(replay_path)
        require(replay.get("status") == "complete" and replay.get("split") == "validation"
                and replay.get("held_out_test_evaluated") is False,
                "Frozen baseline receipt is not Validation-only")
        require(replay.get("checkpoint_sha256") == row.get("checkpoint_sha256"),
                "Frozen baseline checkpoint binding differs")
        require(replay.get("state_sha256") == row.get("state_tensor_sha256"),
                "Frozen baseline state tensor binding differs")
        require(replay.get("selected_epoch") == row.get("selected_epoch"),
                "Frozen baseline receipt selected epoch differs")
        require(baseline_metrics(replay.get("metrics"), replay.get("expected_target_count")) == metrics(row.get("metrics")),
                "Frozen baseline Validation metrics differ")
        if set(replay["metrics"]) == set(METRICS):
            population = {"target_count": replay.get("expected_target_count"),
                          "target_identity_sha256": replay.get("target_identity_sha256"),
                          "target_quantity_sha256": replay.get("truth_sha256")}
        else:
            # The original training population and the richer evaluation-row
            # digest use different encodings. The frozen contract binds the former.
            require(replay.get("schema") in {
                "titantpp_components_inter8_frozen_evaluation_v1",
                "titantpp_components_inter62_frozen_evaluation_v1",
                "titantpp_components_taxi_raf24_frozen_evaluation_v1",
            }, "Unknown native baseline receipt schema")
            population = replay.get("original_population_identity")
            require(isinstance(population, dict)
                    and population.get("target_count") == replay.get("expected_target_count"),
                    "Native baseline original population missing or count differs")
        require(population == row["validation_population"], "Frozen baseline population differs")
        result[key] = row
    require(set(result) == expected and len(result) == 18, "Expected exactly 18 paired frozen baselines")
    return result


def terminal_has(terminal: dict[str, Any], digest: str, suffix: str) -> bool:
    files = terminal.get("files")
    if not isinstance(files, dict):
        return False
    return any(value == digest and isinstance(name, str) and name.endswith(suffix)
               for name, value in files.items())


def endpoint_metrics(selected: dict[str, Any]) -> dict[str, float]:
    """Map the immutable native Validation replay into the study's six metrics."""
    tail = selected.get("tail")
    require(isinstance(tail, dict), "Selected replay has no tail metrics")
    return metrics({
        "whole_qty_rmse": selected.get("qty_rmse"),
        "whole_qty_mae": selected.get("qty_mae"),
        "tail_qty_rmse": tail.get("qty_rmse"),
        "tail_qty_mae": tail.get("qty_mae"),
        "whole_time_nll": selected.get("time_nll"),
        "tail_time_nll": tail.get("time_nll"),
    })


def verify_candidate(root: Path, runtime_contract_sha: str, job: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    required = {
        "terminal_manifest_path", "terminal_manifest_sha256", "history_path", "history_sha256",
        "checkpoint_path", "checkpoint_sha256", "endpoint_replays_path",
        "endpoint_replays_sha256", "input_receipt_path", "input_receipt_sha256",
        "derived_validation_replay_path", "derived_validation_replay_sha256", "state_tensor_sha256",
    }
    require(isinstance(binding, dict) and set(binding) == required, "Candidate retrieval binding schema differs")
    terminal_path = verified(root, binding["terminal_manifest_path"], binding["terminal_manifest_sha256"])
    history_path = verified(root, binding["history_path"], binding["history_sha256"])
    checkpoint_path = verified(root, binding["checkpoint_path"], binding["checkpoint_sha256"])
    endpoint_path = verified(root, binding["endpoint_replays_path"], binding["endpoint_replays_sha256"])
    input_path = verified(root, binding["input_receipt_path"], binding["input_receipt_sha256"])
    derived_path = verified(root, binding["derived_validation_replay_path"], binding["derived_validation_replay_sha256"])
    terminal, endpoint, inputs, derived = read(terminal_path), read(endpoint_path), read(input_path), read(derived_path)
    require(terminal.get("status") == "complete" and terminal.get("scientific_success") is True
            and terminal.get("held_out_test_evaluated") is False
            and terminal.get("contract_sha256") == runtime_contract_sha
            and terminal.get("job") == job,
            "Candidate terminal manifest identity differs")
    require(terminal_has(terminal, binding["history_sha256"], "history.json")
            and terminal_has(terminal, binding["checkpoint_sha256"], "best_val_qty_rmse_model.pt")
            and terminal_has(terminal, binding["endpoint_replays_sha256"], "endpoint_replays.json")
            and terminal_has(terminal, binding["input_receipt_sha256"], "input_receipt.json"),
            "Candidate terminal evidence is incomplete")
    selected_epoch, selected_value = strict_selected_epoch(selection_history(history_path))
    require(endpoint.get("status") == "complete" and endpoint.get("held_out_test_evaluated") is False
            and endpoint.get("evaluation_scope") == "validation_only"
            and endpoint.get("job") == job,
            "Candidate endpoint replay is not complete Validation-only evidence")
    selected, last = endpoint.get("selected"), endpoint.get("last")
    best_epoch = endpoint.get("best_epoch")
    require(isinstance(selected, dict) and type(best_epoch) is int,
            "Candidate endpoint selected/best replay is missing")
    require(best_epoch == selected_epoch
            and selected.get("state_sha256") == binding["state_tensor_sha256"]
            and isinstance(selected.get("checkpoint_path"), str)
            and Path(selected["checkpoint_path"]).name == "best_val_qty_rmse_model.pt"
            and selected.get("evaluation_scope") == "validation_only"
            and selected.get("held_out_test_evaluated") is False,
            "Candidate endpoint selection binding differs")
    require(isinstance(last, dict) and last.get("evaluation_scope") == "validation_only"
            and last.get("held_out_test_evaluated") is False,
            "Candidate last-checkpoint replay is not Validation-only evidence")
    require(math.isclose(float(selected.get("qty_rmse")), selected_value, rel_tol=1e-10, abs_tol=1e-8),
            "Selected replay quantity RMSE differs from selected history row")
    require(inputs.get("held_out_materialized") is False and isinstance(inputs.get("populations"), dict),
            "Terminal-bound input receipt is not Validation-only")
    population = inputs["populations"].get("validation")
    require(isinstance(population, dict), "Input receipt validation population is missing")
    for key in ("target_count", "target_identity_sha256", "target_quantity_sha256"):
        require(population.get(key) is not None, "Candidate target fingerprint is missing: " + key)
    require(selected.get("count") == population["target_count"], "Selected replay target count differs")
    # The collector writes this local normalization; it is checked against immutable bytes here.
    require(derived.get("schema") == "titantpp_gru_residual_cap_validation_local_normalized_replay_v1"
            and derived.get("job") == job
            and derived.get("source_endpoint_replays_sha256") == binding["endpoint_replays_sha256"]
            and derived.get("history_sha256") == binding["history_sha256"]
            and derived.get("checkpoint_sha256") == binding["checkpoint_sha256"]
            and derived.get("state_tensor_sha256") == binding["state_tensor_sha256"]
            and derived.get("selected_epoch") == selected_epoch
            and derived.get("validation_population") == population
            and metrics(derived.get("metrics")) == endpoint_metrics(selected)
            and derived.get("held_out_test_evaluated") is False,
            "Derived local replay does not match terminal-bound Validation evidence")
    return {
        "job": job, "selected_epoch": selected_epoch, "checkpoint_sha256": binding["checkpoint_sha256"],
        "state_tensor_sha256": binding["state_tensor_sha256"], "history_sha256": binding["history_sha256"],
        "terminal_manifest_sha256": binding["terminal_manifest_sha256"],
        "endpoint_replays_sha256": binding["endpoint_replays_sha256"],
        "metrics": endpoint_metrics(selected), "validation_population": population,
        "retrieved_files": {"terminal_manifest": str(terminal_path.relative_to(root)),
                            "history": str(history_path.relative_to(root)),
                            "checkpoint": str(checkpoint_path.relative_to(root)),
                            "endpoint_replays": str(endpoint_path.relative_to(root)),
                            "input_receipt": str(input_path.relative_to(root)),
                            "derived_validation_replay": str(derived_path.relative_to(root))},
    }


def summarize(contract_path: Path, retrieval_root: Path, index_path: Path) -> dict[str, Any]:
    contract_path, retrieval_root, index_path = Path(contract_path), Path(retrieval_root), Path(index_path)
    contract = read(contract_path)
    require(contract.get("held_out_test_evaluated") is False, "This summary must not consume Test evidence")
    contract_raw_sha = sha(contract_path)
    runtime_contract_sha = canonical(contract)
    jobs = contract.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 9 and contract.get("planned_fit_count") == 9,
            "Frozen contract must contain exactly nine candidate fits")
    expected = {job["id"]: job for job in jobs}
    require(len(expected) == 9 and all(job.get("arm") == CANDIDATE for job in jobs),
            "Candidate job identities differ")
    baseline = verify_baselines(contract)
    index = read(index_path)
    require(index.get("schema") == INDEX_KIND and index.get("contract_raw_sha256") == contract_raw_sha,
            "Candidate retrieval index is not bound to this contract")
    bindings = index.get("jobs")
    require(isinstance(bindings, dict) and set(bindings) == set(expected),
            "Candidate retrieval index must contain exactly nine jobs")
    candidates = {job_id: verify_candidate(retrieval_root, runtime_contract_sha, job, bindings[job_id])
                  for job_id, job in expected.items()}

    per_dataset: dict[str, Any] = {}
    for dataset in sorted({job["dataset"] for job in jobs}):
        seeds = sorted(job["seed"] for job in jobs if job["dataset"] == dataset)
        require(seeds == [42, 52, 62], "Each dataset must contain all three seeds")
        arm_rows: dict[str, dict[int, dict[str, float]]] = {CANDIDATE: {}, **{arm: {} for arm in BASELINES}}
        population = None
        for seed in seeds:
            candidate = candidates[f"{dataset}__{seed}__{CANDIDATE}"]
            current_population = candidate["validation_population"]
            if population is None:
                population = current_population
            require(current_population == population, "Candidate Validation target differs across seeds")
            arm_rows[CANDIDATE][seed] = candidate["metrics"]
            for arm in BASELINES:
                reference = baseline[(dataset, seed, arm)]
                require(reference["validation_population"] == population,
                        "Candidate and frozen baseline Validation targets differ")
                arm_rows[arm][seed] = metrics(reference["metrics"])
        arms: dict[str, Any] = {}
        for arm, rows in arm_rows.items():
            arms[arm] = {"seed_metrics": {str(seed): rows[seed] for seed in seeds},
                         "mean": {metric: statistics.mean(rows[seed][metric] for seed in seeds) for metric in METRICS},
                         "sample_sd": {metric: statistics.stdev(rows[seed][metric] for seed in seeds) for metric in METRICS}}
        deltas: dict[str, Any] = {}
        for arm in BASELINES:
            by_seed = {str(seed): {metric: arm_rows[CANDIDATE][seed][metric] - arm_rows[arm][seed][metric]
                                   for metric in METRICS} for seed in seeds}
            deltas[arm] = {"candidate_minus_reference": by_seed,
                           "mean": {metric: statistics.mean(by_seed[str(seed)][metric] for seed in seeds)
                                    for metric in METRICS},
                           "unfavorable_or_tied_seed_metrics": [
                               {"seed": seed, "metric": metric, "delta": by_seed[str(seed)][metric]}
                               for seed in seeds for metric in METRICS if by_seed[str(seed)][metric] >= 0
                           ]}
        per_dataset[dataset] = {"seeds": seeds, "validation_population": population,
                                "arms": arms, "candidate_deltas": deltas}
    return {"schema": KIND, "status": "complete_validation_only", "contract_raw_sha256": contract_raw_sha,
            "contract_canonical_sha256": runtime_contract_sha,
            "retrieval_index_raw_sha256": sha(index_path), "candidate_fit_count": 9,
            "baseline_reference_count": 18, "held_out_test_evaluated": False,
            "selection_rule": "first_strict_finite_full_validation_raw_quantity_rmse_minimum",
            "metric_order": list(METRICS), "lower_is_better": True, "datasets": per_dataset,
            "interpretation_limits": [
                "Validation-only comparison does not establish Test generalization.",
                "The cap changes optimization trajectories as well as residual magnitude.",
                "CNN has fewer active parameters than FULL and CAPPED.",
            ]}


def markdown(summary: dict[str, Any]) -> str:
    labels = {
        "whole_qty_rmse": "전체 raw 수량 RMSE",
        "whole_qty_mae": "전체 raw 수량 MAE",
        "tail_qty_rmse": "큰 수량 RMSE",
        "tail_qty_mae": "큰 수량 MAE",
        "whole_time_nll": "전체 recorded positive-integer TimeNLL",
        "tail_time_nll": "큰 수량 recorded positive-integer TimeNLL",
    }
    arms = (CANDIDATE, *BASELINES)
    arm_labels = {CANDIDATE: "CAP0.25", BASELINES[0]: "FULL", BASELINES[1]: "CNN"}

    def signed(value: float) -> str:
        return "0" if value == 0 else f"{value:+.6g}"

    lines = ["# GRU 보정량 제한 후보의 3-seed Validation 비교", "",
             "이 결과는 Validation 전용이다. Test 일반화 개선이나 순위 반전의 원인을 입증하지 않는다.", "",
             "모든 값은 고정된 Validation 선택 checkpoint의 지표다. 평균±SD에서 SD는 3seed의 표본표준편차(ddof=1)이며, 여섯 지표 모두 낮을수록 좋다.", "",
             "CAP0.25는 GRU 보정량 제한 후보, FULL은 기존 CNN+GRU, CNN은 기존 CNN 단독 모델이다. 큰 수량의 구분은 고정된 Train 기준을 유지한다.", "",
             "차이는 CAP0.25−비교 모델이다. 음수는 개선, 양수는 악화, 0은 동률이다. 평균 차이와 seed별 예외를 함께 읽어야 한다.", ""]
    for dataset, value in summary["datasets"].items():
        lines.extend([f"## {dataset}", "", "### 여섯 지표의 3seed 평균±표본SD", "",
            "| 지표 | CAP0.25 | FULL | CNN |", "| --- | ---: | ---: | ---: |"])
        for metric in METRICS:
            cells = [f"{value['arms'][arm]['mean'][metric]:.6g}±{value['arms'][arm]['sample_sd'][metric]:.6g}"
                     for arm in arms]
            lines.append(f"| {labels[metric]} | " + " | ".join(cells) + " |")
        lines.append("")
        for reference in BASELINES:
            delta = value["candidate_deltas"][reference]
            name = arm_labels[reference]
            lines.extend([f"### CAP0.25−{name}: seed별 차이", "",
                "| 지표 | seed42 | seed52 | seed62 | 평균 차이 |",
                "| --- | ---: | ---: | ---: | ---: |"])
            for metric in METRICS:
                cells = [signed(delta['candidate_minus_reference'][str(seed)][metric])
                         for seed in value['seeds']]
                cells.append(signed(delta['mean'][metric]))
                lines.append(f"| {labels[metric]} | " + " | ".join(cells) + " |")
            lines.extend(["", f"**CAP0.25−{name}의 악화·동률 결과**", ""])
            exceptions = delta['unfavorable_or_tied_seed_metrics']
            if exceptions:
                lines.extend(["| seed | 지표 | 차이 | 판정 |", "| ---: | --- | ---: | --- |"])
                for row in exceptions:
                    verdict = "동률" if row['delta'] == 0 else "악화"
                    lines.append(f"| {row['seed']} | {labels[row['metric']]} | {signed(row['delta'])} | {verdict} |")
            else:
                lines.append("해당 3seed의 여섯 지표에서 악화·동률 결과 없음.")
            lines.append("")
    return "\n".join(lines) + "\n"


def write_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.contract, args.retrieval_root, args.index)
    write_exclusive(args.output_json, json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_exclusive(args.output_md, markdown(summary))
    print(json.dumps({"status": summary["status"], "candidate_fit_count": 9}, ensure_ascii=False))


if __name__ == "__main__":
    main()
