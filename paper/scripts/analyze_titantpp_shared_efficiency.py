"""Verify and summarize a retrieved native5090 Shared15/MAC cost bundle.

Only the Python standard library is used. This command reads frozen evidence;
it never imports a model, opens a checkpoint, or starts a remote/GPU process.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = REPO / "search_artifacts/titantpp_shared_efficiency_5090_20261020_v1"
DEFAULT_REPORT = REPO / "reports/titantpp_shared_efficiency_5090_20261020_v1"
SHARED = "titantpp_history_mlp_shared15_v1"
MAC = "titantpp_titans_mac"
DATASETS = {"raf_spare_parts": 84, "yellow_trip_hourly": 256, "intermittent_frozen_5000": 256}
LABELS = {"raf_spare_parts": "RAF", "yellow_trip_hourly": "Taxi", "intermittent_frozen_5000": "Intermittent"}
GPU_UUID = "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f"
OUTPUTS = ("analysis.json", "verification.json", "repetitions.csv", "per_batch.csv", "inputs.csv", "report.md")


class EvidenceError(ValueError):
    """Evidence is incomplete, inconsistent, or outside the approved scope."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read(path: Path) -> Any:
    def invalid(token: str) -> None:
        raise EvidenceError(f"Nonfinite JSON value in {path.name}: {token}")

    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=invalid, object_pairs_hook=_pairs)


def _file(root: Path, name: str) -> Path:
    path = Path(name)
    require(isinstance(name, str) and bool(name) and not path.is_absolute() and ".." not in path.parts,
            f"Unsafe evidence path: {name}")
    target = root / path
    require(target.resolve().is_relative_to(root.resolve()), f"Evidence escapes root: {name}")
    require(not any(part.is_symlink() for part in (target, *target.parents) if part != root.parent),
            f"Symlink evidence is not accepted: {name}")
    require(target.is_file(), f"Missing evidence: {name}")
    return target


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"Nonfinite/invalid {label}")
    require(value > 0 if positive else value >= 0, f"Out-of-range {label}")
    return float(value)


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    require(type(value) is int and (value > 0 if positive else value >= 0), f"Invalid integer {label}")
    return value


def _sd(values: list[float]) -> float:
    return st.stdev(values) if len(values) > 1 else 0.0


def _label(dataset: str, arm: str, mode: str, rep: int) -> str:
    return f"{dataset}__{arm}__{mode}__r{rep}"


def _contract(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(contract["models"] == [SHARED, MAC], "Only final Shared15 and frozen MAC are accepted")
    datasets = contract["datasets"]
    require(len(datasets) == 3 and {d["dataset_id"] for d in datasets} == set(DATASETS),
            "Exactly the three approved original datasets are required")
    indexed = {d["dataset_id"]: d for d in datasets}
    for name, dataset in indexed.items():
        require(dataset["loader"]["max_seq_len"] == DATASETS[name] and dataset["loader"]["batch_size"] == 128,
                f"Loader exposure changed: {name}")
    scope = contract["scope"]
    require(isinstance(scope, dict) and scope.get("train_only") is True and
            all(scope.get(key) is False for key in ("accuracy_evaluation", "checkpoint_writes", "full_epochs",
                                                   "full_fits", "held_out", "validation")), "Train-only cost scope required")
    protocol = contract["protocol"]
    expected = dict(seed=42, batch_size=128, repetitions=3, modes=["train", "eval"],
                    warmup_batches_per_worker=5, measured_batches_per_worker=32, workers=36,
                    maximum_optimizer_steps=666, maximum_evaluation_batches=666, parallel_workers=1,
                    automatic_retry=False)
    require(all(type(protocol.get(key)) is type(value) and protocol[key] == value for key, value in expected.items()),
            "Paired profiling exposure changed")
    require(contract["host"]["gpu_uuid"] == GPU_UUID, "Wrong native5090 GPU contract")
    native = contract["host"]["runtime_expected"]
    require(all(native.get(key) == value for key, value in
                dict(python="3.12.13", torch="2.11.0+cu130", cudnn=92000).items()), "Wrong native5090 runtime contract")
    require(len(contract["source_files"]) == 985, "Expected the approved 985-file source closure")
    require(canonical(contract["source_files"]) == contract["source_closure_sha256"], "Source closure identity mismatch")
    require({"benchmark.py", "supervise.py"}.issubset(contract["operational_files"]), "Both operational sources required")
    return protocol, indexed


def _verify_files(bundle: Path, retrieved: Path, manifest: dict[str, Any], contract: dict[str, Any]) -> set[str]:
    files = manifest["files"]
    require(isinstance(files, dict) and bool(files), "Empty terminal manifest")
    for name, digest in files.items():
        require(isinstance(digest, str) and len(digest) == 64 and sha(_file(retrieved, name)) == digest,
                f"Retrieved manifest SHA mismatch: {name}")
    required = {"contract.json", "approval.json", "benchmark.py", "status.json", "claim.json", "claim.exclusive",
                "preflight.json", "gpu_initial.json", "gpu_final.json"}
    require(required.issubset(files), "Terminal manifest omits required evidence")
    require(sha(_file(bundle, "contract.json")) == sha(_file(retrieved, "contract.json")), "Local/retrieved contract mismatch")
    require(read(_file(retrieved, "contract.json")) == contract, "Retrieved contract content mismatch")
    for name, digest in contract["source_files"].items():
        require(sha(_file(bundle / "source", name)) == digest, f"Frozen source SHA mismatch: {name}")
    for name, digest in contract["operational_files"].items():
        require(sha(_file(bundle, name)) == digest and sha(_file(retrieved, name)) == digest,
                f"Operational source SHA mismatch: {name}")
    for folder in ("results", "batches", "qualification", "child_exits"):
        actual = {str(p.relative_to(retrieved)) for p in (retrieved / folder).glob("*") if p.is_file()}
        require(actual and actual.issubset(files), f"Unmanifested or missing {folder} evidence")
    return set(files)


def _runtime(runtime: dict[str, Any], contract: dict[str, Any]) -> None:
    require(all(runtime.get(key) == value for key, value in contract["host"]["runtime_expected"].items()),
            "Native runtime differs from contract")
    require(runtime["gpu"]["uuid"] == contract["host"]["gpu_uuid"], "Worker GPU differs from contract")


def _verify_exits(retrieved: Path, manifest: dict[str, Any], status: dict[str, Any], expected: set[str]) -> dict[str, Any]:
    controller = status["pid"]
    require(manifest["controller_pid"] == controller, "Controller identity mismatch")
    external = read(_file(retrieved, "controller_exit.json"))
    require(external.get("actual_controller_exit_observed") is True and external.get("pid_absent_after_wait") is True,
            "Direct controller OS exit and PID absence must be observed")
    require(type(external.get("actual_os_returncode")) is int and external["actual_os_returncode"] == 0 and
            type(external.get("returncode")) is int and external["returncode"] == 0,
            "Controller OS exit was not zero")
    require(external.get("controller_pid") == external.get("pid") == controller, "External controller PID mismatch")
    qualification = {f"qualify__{arm}__{device}" for arm, device in itertools.product((SHARED, MAC), ("cpu", "cuda_0"))}
    labels = expected | qualification | {"prepare"}
    paths = sorted((retrieved / "child_exits").glob("*.json"))
    require({p.stem for p in paths} == labels and len(paths) == 41, "Expected exactly 41 child exit files")
    exits = {p.stem: read(p) for p in paths}
    require(len(manifest["actual_child_exits"]) == 41, "Incomplete terminal child exit records")
    listed = {record["label"]: record for record in manifest["actual_child_exits"]}
    require(listed == exits, "Terminal/file child exit records disagree")
    require(status["child_exit_count"] == 41 and status["worker_exit_count"] == 36, "Incomplete direct child exits")
    pids = set()
    for label, record in exits.items():
        require(record["label"] == label and record["owner_pid"] == controller, f"Child owner mismatch: {label}")
        require(type(record["returncode"]) is int and record["returncode"] == 0 and
                record["actual_child_exit_observed"] is True and record["pid_absent_after_wait"] is True and
                record["timed_out"] is False and record["error"] is None, f"Child exit unconfirmed: {label}")
        pid = _integer(record["pid"], "child PID", positive=True)
        require(pid not in pids and pid != controller, f"Reused child PID: {label}")
        pids.add(pid)
        require(record["log"] in manifest["files"], f"Missing child log binding: {label}")
        _number(record["wall_seconds"], "child wall time", positive=True)
        require(status["started_unix"] <= record["started_unix"] <= record["finished_unix"] <= status["finished_unix"],
                f"Child time outside controller: {label}")
    return {"children": exits, "controller": external}


def _verify_qualifications(retrieved: Path, contract: dict[str, Any], exits: dict[str, Any]) -> int:
    expected = {f"{arm}__{device}.json" for arm, device in itertools.product((SHARED, MAC), ("cpu", "cuda_0"))}
    require({p.name for p in (retrieved / "qualification").glob("*.json")} == expected, "Four synthetic qualifications required")
    evidence = {}
    steps = 0
    for arm, device in itertools.product((SHARED, MAC), ("cpu", "cuda:0")):
        name = f"{arm}__{device.replace(':', '_')}"
        path = _file(retrieved, f"qualification/{name}.json")
        value = read(path)
        require(value["status"] == "passed" and value["contract_sha256"] == canonical(contract) and
                value["arm"] == arm and value["device"] == device and
                value["real_data_loaded"] is False and value["accuracy_evaluated"] is False and
                value["checkpoint_saved"] is False, f"Invalid synthetic qualification: {name}")
        require(value["pid"] == exits[f"qualify__{name}"]["pid"], "Synthetic qualification PID mismatch")
        rows = value["rows"]
        require(len(rows) == 3 and {row["dataset"] for row in rows} == set(DATASETS), "Synthetic dataset coverage mismatch")
        for row in rows:
            require(all(row[key] is True for key in ("finite", "causality", "padding", "fresh_state")) and
                    row["optimizer_steps"] == 1, "Synthetic numerical/state checks failed")
            require(row["batch_size"] == (128 if device == "cuda:0" else 2) and
                    row["length"] == (DATASETS[row["dataset"]] if device == "cuda:0" else 8), "Synthetic exposure changed")
            steps += row["optimizer_steps"]
        if device == "cuda:0":
            _runtime(value["runtime"], contract)
        evidence[str(path.relative_to(retrieved))] = sha(path)
    preflight = read(_file(retrieved, "preflight.json"))
    require(preflight["status"] == "passed" and preflight["contract_sha256"] == canonical(contract) and
            preflight["source_closure"] == contract["source_closure_sha256"] and
            preflight["qualification_files"] == evidence and preflight["pid"] == exits["prepare"]["pid"],
            "Preparation does not bind all synthetic checks")
    return steps


def _batches(retrieved: Path, datasets: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require({p.stem for p in (retrieved / "batches").glob("*.json")} == set(datasets), "Batch dataset coverage mismatch")
    batches = {}
    summary = []
    for name, dataset in datasets.items():
        path = _file(retrieved, f"batches/{name}.json")
        value = read(path)
        require(value["split"] == "train" and value["seed"] == 42 and value["common_head_equal"] is True,
                f"Train/head initialization evidence missing: {name}")
        require(value["file_sha256"] == sha(_file(retrieved, f"batches/{name}.pt")), f"Frozen batch cache SHA mismatch: {name}")
        require(value["population"] == dataset["inherited_data_identity"]["populations"]["train"]["target_count"],
                f"Original train population mismatch: {name}")
        require(len(value["batches"]) == 37 and set(value["initial"]) == {SHARED, MAC}, "Incomplete frozen inputs")
        targets = []
        for number, batch in enumerate(value["batches"]):
            require(batch["batch"] == number and batch["phase"] == ("warmup" if number < 5 else "measured"),
                    f"Frozen batch ordering mismatch: {name}")
            require(len(batch["targets"]) == len(batch["valid_lengths"]) == 128, "Batch target exposure mismatch")
            require(batch["padded_length"] == dataset["loader"]["max_seq_len"], "Padded length differs from loader")
            require(all(type(length) is int and 1 <= length <= batch["padded_length"] for length in batch["valid_lengths"]),
                    "Invalid observed valid lengths")
            targets.extend(canonical(target) for target in batch["targets"])
        require(len(targets) == len(set(targets)) == 4736, "Sampled train targets must be distinct")
        for arm in (SHARED, MAC):
            _integer(value["initial"][arm]["parameters"], "initial parameter count", positive=True)
            require(isinstance(value["initial"][arm]["state_sha256"], str) and
                    len(value["initial"][arm]["state_sha256"]) == 64, "Missing initial tensor binding")
        measured = value["batches"][5:]
        lengths = [length for batch in measured for length in batch["valid_lengths"]]
        padded_events = sum(batch["padded_length"] * len(batch["valid_lengths"]) for batch in measured)
        summary.append(dict(dataset=name, split="train", train_target_population=value["population"],
                            unique_sampled_targets_including_warmup=len(targets), measured_targets_per_repeat=len(lengths),
                            padded_length=measured[0]["padded_length"], valid_length_including_target_min=min(lengths),
                            valid_length_including_target_mean=st.mean(lengths),
                            valid_length_including_target_median=st.median(lengths),
                            valid_length_including_target_max=max(lengths), padding_fraction=1 - sum(lengths) / padded_events))
        batches[name] = value
    return batches, summary


def _profiles(retrieved: Path, contract: dict[str, Any], datasets: dict[str, Any], status: dict[str, Any],
              expected: set[str], batches: dict[str, Any], exits: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    paths = sorted((retrieved / "results").glob("*.json"))
    require(len(paths) == 36 and {path.stem for path in paths} == expected, "Exactly 36 paired profiles are required")
    repetitions = []
    per_batch = []
    runtime = None
    for path in paths:
        value = read(path)
        dataset, arm, mode, rep = (value[key] for key in ("dataset", "arm", "mode", "rep"))
        require(type(rep) is int and path.stem == _label(dataset, arm, mode, rep) and path.stem in expected,
                f"Profile identity mismatch: {path.name}")
        require(value["status"] == "success" and value["accuracy_evaluated"] is False and
                value["checkpoint_saved"] is False and value["scientific_fit"] is False and value["seed"] == 42,
                f"Profile scope mismatch: {path.name}")
        require(value["contract_sha256"] == canonical(contract) and
                value["source_closure_sha256"] == contract["source_closure_sha256"], "Profile source/contract mismatch")
        require(value["input_file_sha256"] == datasets[dataset]["inherited_data_identity"]["data"]["sha256"],
                "Original input binding mismatch")
        child = exits[path.stem]
        require(value["pid"] == child["pid"] and value["owner_pid"] == status["pid"], "Worker PID/owner mismatch")
        require(child["started_unix"] <= value["started_unix"] <= value["finished_unix"] <= child["finished_unix"] and
                value["finished_unix"] <= status["deadline_unix"], "Profile ran outside its time boundary")
        require(value["optimizer_steps"] == (37 if mode == "train" else 0) and
                value["eval_batches"] == (37 if mode == "eval" else 0) and
                value["targets_total"] == 4736 and value["targets_measured"] == 4096, "Worker exposure mismatch")
        _runtime(value["runtime"], contract)
        if runtime is None:
            runtime = value["runtime"]
        require(value["runtime"] == runtime, "Native runtime drift across workers")
        frozen = batches[dataset]
        require(value["batch_manifest_sha256"] == sha(_file(retrieved, f"batches/{dataset}.json")) and
                value["initial_state_sha256"] == frozen["initial"][arm]["state_sha256"] and
                value["parameters"] == frozen["initial"][arm]["parameters"] and value["common_head_equal"] is True,
                "Batch/initial state/common head binding mismatch")
        rows = value["rows"]
        require(len(rows) == 37 and [row["batch"] for row in rows] == list(range(37)), "Missing/reordered batch rows")
        graphs = []
        for number, (row, batch) in enumerate(zip(rows, frozen["batches"])):
            require(row["phase"] == batch["phase"] and row["tensor_sha256"] == batch["tensor_sha256"] and
                    row["targets"] == 128 and row["valid_events"] == sum(batch["valid_lengths"]), "Per-batch input binding mismatch")
            for key in ("compute_seconds", "transfer_seconds", "transfer_compute_seconds"):
                _number(row[key], key, positive=True)
            require(math.isclose(row["compute_seconds"] + row["transfer_seconds"], row["transfer_compute_seconds"],
                                 rel_tol=1e-9, abs_tol=1e-9), "Transfer/compute timing does not add up")
            graphs.append(_integer(row["unique_compiled_graphs"], "compiled graphs"))
            per_batch.append(dict(dataset=dataset, arm=arm, mode=mode, rep=rep, **row))
        require(graphs == sorted(graphs), "Compiled graph counter decreased")
        require(value["compiled_graphs_after_warmup"] == graphs[4] and
                value["compiled_graphs_during_measured"] == graphs[-1] - graphs[4], "Compiled graph summary mismatch")
        require(value["cold_first_batch_seconds"] == rows[0]["transfer_compute_seconds"], "Cold batch timing mismatch")
        process_seconds = _number(value["process_seconds"], "process seconds", positive=True)
        require(process_seconds + 1e-6 >= sum(row["transfer_compute_seconds"] for row in rows), "Process time below timed batches")
        setup = _number(value["process_setup_seconds"], "process setup seconds")
        memory = {}
        for field in ("allocated", "reserved"):
            baseline = _integer(value["memory_baseline_bytes"][field], "memory baseline")
            low = _integer(value["memory_low_baseline_bytes"][field], "memory low baseline")
            peak = _integer(value["memory_peak_bytes"][field], "peak memory", positive=True)
            require(low <= baseline <= peak, "Memory low/baseline/peak ordering mismatch")
            memory[f"baseline_{field}_mib"] = baseline / 2**20
            memory[f"low_baseline_{field}_mib"] = low / 2**20
            memory[f"peak_{field}_mib"] = peak / 2**20
            memory[f"peak_minus_baseline_{field}_mib"] = (peak - baseline) / 2**20
        measured = rows[5:]
        ms = [row["compute_seconds"] * 1000 for row in measured]
        quartiles = st.quantiles(ms, n=4, method="inclusive")
        targets = sum(row["targets"] for row in measured)
        repetitions.append(dict(dataset=dataset, arm=arm, mode=mode, rep=rep, seed=42,
                                median_compute_ms=st.median(ms), q25_compute_ms=quartiles[0], q75_compute_ms=quartiles[2],
                                mean_compute_ms=st.mean(ms),
                                median_transfer_plus_compute_ms=st.median([row["transfer_compute_seconds"] * 1000 for row in measured]),
                                mean_transfer_plus_compute_ms=st.mean([row["transfer_compute_seconds"] * 1000 for row in measured]),
                                mean_transfer_ms=st.mean([row["transfer_seconds"] * 1000 for row in measured]),
                                compute_targets_per_second=targets / sum(row["compute_seconds"] for row in measured),
                                transfer_compute_targets_per_second=targets / sum(row["transfer_compute_seconds"] for row in measured),
                                first_batch_seconds=rows[0]["transfer_compute_seconds"],
                                warmup5_seconds=sum(row["transfer_compute_seconds"] for row in rows[:5]),
                                process_seconds=process_seconds, process_setup_seconds=setup, parameters=value["parameters"],
                                optimizer_steps=value["optimizer_steps"], eval_batches=value["eval_batches"],
                                graphs_after_warmup=graphs[4], new_graphs_in_measured32=graphs[-1] - graphs[4], **memory))
    return repetitions, per_batch, runtime


def aggregate(repetitions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair repetitions by identity; summarize repeat medians using sample SD."""
    comparisons = []
    adverse = []
    metrics = ("median_compute_ms", "mean_compute_ms", "median_transfer_plus_compute_ms", "mean_transfer_plus_compute_ms",
               "compute_targets_per_second", "transfer_compute_targets_per_second", "peak_allocated_mib",
               "peak_reserved_mib", "parameters")
    for dataset, mode in itertools.product(DATASETS, ("train", "eval")):
        arms = {arm: {row["rep"]: row for row in repetitions if
                      (row["dataset"], row["mode"], row["arm"]) == (dataset, mode, arm)} for arm in (SHARED, MAC)}
        require(all(set(values) == {1, 2, 3} for values in arms.values()), "Incomplete paired repetitions")
        ratios = [arms[MAC][rep]["median_compute_ms"] / arms[SHARED][rep]["median_compute_ms"] for rep in (1, 2, 3)]
        entry = dict(dataset=dataset, mode=mode, repetitions=3, paired_mac_over_shared_ratio_mean=st.mean(ratios),
                     paired_mac_over_shared_ratio_sd=_sd(ratios), paired_ratios=ratios)
        for name, arm in (("shared", SHARED), ("mac", MAC)):
            for metric in metrics:
                values = [arms[arm][rep][metric] for rep in (1, 2, 3)]
                entry[f"{name}_{metric}_mean"] = st.mean(values)
                entry[f"{name}_{metric}_sd"] = _sd(values)
        for metric in ("parameters", "peak_allocated_mib", "peak_reserved_mib"):
            entry[f"shared_vs_mac_{metric}_change_pct"] = 100 * (entry[f"shared_{metric}_mean"] / entry[f"mac_{metric}_mean"] - 1)
            if metric.startswith("peak_"):
                for rep in (1, 2, 3):
                    shared, mac = arms[SHARED][rep][metric], arms[MAC][rep][metric]
                    if shared > mac:
                        adverse.append(dict(dataset=dataset, mode=mode, metric=metric, rep=rep,
                                            shared_mib=shared, mac_mib=mac, shared_vs_mac_change_pct=100 * (shared / mac - 1)))
        comparisons.append(entry)
    return comparisons, adverse


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in rows) + "\n"


def _report(analysis: dict[str, Any], bundle: Path) -> str:
    comparisons, verification = analysis["comparison"], analysis["verification"]
    def estimate(row: dict[str, Any], arm: str, metric: str, precision: int = 3) -> str:
        return f"{row[f'{arm}_{metric}_mean']:.{precision}f} ± {row[f'{arm}_{metric}_sd']:.{precision}f}"
    latency, throughput, memory, parameters, inputs = [], [], [], [], []
    for row in comparisons:
        prefix = [LABELS[row["dataset"]], "학습 step" if row["mode"] == "train" else "평가 batch"]
        latency.append(prefix + [estimate(row, "shared", "median_compute_ms"), estimate(row, "mac", "median_compute_ms"),
                                 f"{row['paired_mac_over_shared_ratio_mean']:.3f} ± {row['paired_mac_over_shared_ratio_sd']:.3f}"])
        throughput.append(prefix + [estimate(row, "shared", "compute_targets_per_second", 1),
                                    estimate(row, "mac", "compute_targets_per_second", 1),
                                    estimate(row, "shared", "transfer_compute_targets_per_second", 1),
                                    estimate(row, "mac", "transfer_compute_targets_per_second", 1)])
        memory.append(prefix + [estimate(row, "shared", "peak_allocated_mib", 2), estimate(row, "mac", "peak_allocated_mib", 2),
                                f"{row['shared_vs_mac_peak_allocated_mib_change_pct']:+.2f}%",
                                estimate(row, "shared", "peak_reserved_mib", 2), estimate(row, "mac", "peak_reserved_mib", 2),
                                f"{row['shared_vs_mac_peak_reserved_mib_change_pct']:+.2f}%"])
        if row["mode"] == "train":
            parameters.append([LABELS[row["dataset"]], f"{row['shared_parameters_mean']:,.0f}",
                               f"{row['mac_parameters_mean']:,.0f}", f"{row['shared_vs_mac_parameters_change_pct']:+.2f}%"])
    for row in analysis["input_summary"]:
        inputs.append([LABELS[row["dataset"]], row["train_target_population"], row["padded_length"],
                       f"{row['valid_length_including_target_mean']:.2f}", row["valid_length_including_target_min"],
                       row["valid_length_including_target_max"], f"{100 * row['padding_fraction']:.2f}%"])
    adverse_rows = [[LABELS[row["dataset"]], row["mode"], row["rep"], row["metric"],
                     f"{row['shared_mib']:.2f}", f"{row['mac_mib']:.2f}", f"{row['shared_vs_mac_change_pct']:+.2f}%"]
                    for row in analysis["adverse_memory_repetitions"]]
    adverse_text = ("Shared15의 peak가 MAC보다 큰 반복을 모두 아래에 기록했다.\n\n" +
                    _table(["데이터", "경로", "반복", "메모리 지표", "Shared15 MiB", "MAC MiB", "상대 변화"], adverse_rows)
                    if adverse_rows else "이번 측정에서는 Shared15의 allocated/reserved peak가 MAC보다 큰 반복이 없었다.\n")
    return f"""# Shared15 / Titans-MAC 원본 Train 입력 비용 측정

상태: 완료. native5090의 동일 GPU에서 fresh seed42 모델을 새 프로세스로 만들어 36개 비용 측정을 실행했다. 원본 Train 표본만 사용했으며 Validation/Test 예측·성능은 평가하지 않았다. 이 문서의 수치는 회수한 원본 JSON과 CSV에서 계산했다.

## 지연시간

각 반복의 예열 5개를 제외한 32개 batch에서 계산 시간 중앙값을 구한 뒤, 반복 3회의 평균 ± 표본표준편차를 표시했다. batch 크기는 128이다. 속도비는 같은 반복 번호의 MAC 중앙값 / Shared15 중앙값 3개의 평균 ± 표본표준편차다. 1보다 크면 이번 입력에서 Shared15의 시간이 짧다. 반복 번호는 시간 측정 반복이며 학습 seed 3개를 뜻하지 않는다.

""" + _table(["데이터", "경로", "Shared15 ms/batch", "MAC ms/batch", "MAC/Shared15"], latency) + """
계산 시간은 CPU→GPU 전송을 제외하고 CUDA 동기화로 양쪽 경계를 구분한 host wall time이다. 학습 경로는 forward·공통 head·target loss·zero_grad·backward·gradient clip·AdamW step을 포함한다. 평가 경로는 eval/no-grad forward·head·target loss와 MAC의 온라인 메모리 갱신을 포함하므로 순수 inference 시간으로 해석하지 않는다. 반복별 산술평균·사분위수·첫 batch·예열·전체 프로세스 시간은 [repetitions.csv](repetitions.csv)에 있다.

## 처리량

각 반복에서 측정 target 4,096개를 해당 32개 batch 시간의 합으로 나눈 후 반복 평균 ± 표본표준편차를 계산했다. 연산 처리량은 compute만, 전송 포함 처리량은 CPU→GPU 전송+compute를 분모로 쓴다. 두 값 모두 loader·직렬화·로그·유한성 확인을 제외하므로 전체 데이터 파이프라인 처리량은 아니다.

""" + _table(["데이터", "경로", "Shared15 연산 targets/s", "MAC 연산 targets/s",
              "Shared15 전송 포함 targets/s", "MAC 전송 포함 targets/s"], throughput) + """
## 메모리와 parameter

예열 5개 뒤 peak를 reset하고 측정 32개 동안 얻은 allocated/reserved peak의 반복 평균 ± 표본표준편차다. allocated는 tensor에 할당된 메모리이며 reserved는 allocator cache를 포함한다. 상대 변화는 (Shared15 평균 / MAC 평균 − 1) × 100이며 양수는 Shared15가 더 큰 경우다. 예열 직후 baseline·측정 중 최소 baseline·peak 증분도 CSV에 남겼다.

""" + _table(["데이터", "경로", "Shared15 allocated MiB", "MAC allocated MiB", "allocated 상대 변화",
              "Shared15 reserved MiB", "MAC reserved MiB", "reserved 상대 변화"], memory) + "\n" + adverse_text + """
Parameter 수는 이번 native 모델 초기화 증적에서 읽었다. parameter 상대 변화와 실제 peak 메모리는 다른 지표다.

""" + _table(["데이터", "Shared15 parameter", "MAC parameter", "상대 변화"], parameters) + """
## 입력 범위

데이터마다 전체 Train target에서 seed42 순열의 서로 다른 4,736개를 뽑았다. 예열 640개와 측정 4,096개를 모든 모델·모드·반복이 공유한다. 유효 길이는 target을 포함한 mask 길이이며 아래 통계는 측정 32개 batch 기준이다.

""" + _table(["데이터", "전체 Train target", "padded 길이", "유효 길이 평균", "최소", "최대", "padding 비율"], inputs) + f"""
모델별 초기 state SHA와 batch tensor SHA를 모든 반복에서 확인했고, 준비 과정의 공통 head 초기 tensor 동일성 검사를 확인했다. 서로 다른 encoder의 parameter 수나 encoder tensor 동일성은 요구하지 않았다. 실제 Train 표본 optimizer step은 {verification['optimizer_steps_including_warmup']:,}개, 평가 batch는 {verification['evaluation_batches_including_warmup']:,}개이며 예열을 포함한다. 별도 합성 자격 검증은 CPU/CUDA × 두 모델 4개 프로세스에서 합계 {verification['synthetic_qualification_optimizer_steps']}개 합성 optimizer step을 수행했다.

## 증적과 해석 범위

- 원본 bundle: [{bundle.name}]({bundle})
- [terminal manifest]({bundle / 'retrieved/terminal_manifest.json'})의 {verification['verified_manifest_files']}개 파일 SHA와 local source {verification['verified_source_files']}개·운영 소스 {verification['verified_operational_files']}개 SHA를 확인했다.
- 비용 worker 36개, 합성 검증 4개, 준비 1개의 실제 자식 OS exit0와 controller OS exit0·PID 부재를 확인했다. 종료 시 GPU compute process가 없었다.
- 실행 시간은 준비를 포함해 {verification['elapsed_seconds']:.3f}초다. 측정 32개 안의 추가 compiled graph는 전체 {verification['new_graphs_in_measured32']}개다. 느린 batch와 첫 batch를 제거하지 않았다. 첫 batch에는 compilation·CUDA·allocator 초기화 비용이 함께 들어갈 수 있다.
- contract canonical SHA: `{verification['contract_sha256']}`
- source closure SHA: `{verification['source_closure_sha256']}`
- [기계판독 집계](analysis.json), [검증 결과](verification.json), [반복별 집계](repetitions.csv), [batch별 원시 시간](per_batch.csv), [입력 요약](inputs.csv)

이 결과는 fresh seed42로 초기화한 짧은 scratch 학습 step과 Train 입력 평가 경로의 비용이다. scientific fit, scientific checkpoint의 정확도, 수렴까지 총비용, 동일 정확도 효율, 실측 초/epoch 또는 모든 길이의 복잡도를 입증하지 않는다. 세 번의 시간 반복으로 학습 seed 변동성을 추정하지 않는다. 평가 경로 비용을 inference-only 성능으로 바꾸어 표현하지 않는다. 5080 과거 결과나 기존 scientific checkpoint 성능과 합치지 않는다.
"""


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(bundle: Path | str = DEFAULT_BUNDLE, report: Path | str = DEFAULT_REPORT) -> dict[str, Any]:
    """Verify complete evidence and exclusively create six local output files."""
    bundle, report = Path(bundle).resolve(), Path(report).resolve()
    require(bundle != report and not report.is_relative_to(bundle), "Report must be separate from evidence bundle")
    require(not any((report / name).exists() for name in (*OUTPUTS, ".analysis.exclusive")), "Analysis outputs already exist; overwrite forbidden")
    retrieved = bundle / "retrieved"
    manifest = read(_file(retrieved, "terminal_manifest.json"))
    contract = read(_file(bundle, "contract.json"))
    protocol, datasets = _contract(contract)
    files = _verify_files(bundle, retrieved, manifest, contract)
    approval = read(_file(retrieved, "approval.json"))
    require(approval["approved"] is True and approval["contract_sha256"] == canonical(contract), "Approval identity mismatch")
    status = read(_file(retrieved, "status.json"))
    require(manifest["status"] == status["status"] == "complete" and status["error"] is None,
            "Only fully complete successful evidence can be aggregated")
    require(all(manifest[key] is False for key in ("accuracy_evaluated", "checkpoint_saved", "automatic_retry")), "Terminal scope mismatch")
    require(status["contract_sha256"] == canonical(contract), "Status contract identity mismatch")
    for name in ("claim.json", "claim.exclusive"):
        claim = read(_file(retrieved, name))
        require(all(status[key] == claim[key] for key in ("pid", "started_unix", "deadline_unix", "contract_sha256", "max_seconds")),
                "Controller claim/status mismatch")
    elapsed = _number(status["elapsed_seconds"], "controller elapsed seconds", positive=True)
    budget = _number(contract.get("max_seconds", protocol["maximum_seconds_including_remote_preparation"]), "time budget", positive=True)
    require(budget <= 1800 and elapsed <= budget + 5 and status["max_seconds"] == budget and
            math.isclose(status["deadline_unix"] - status["started_unix"], budget, abs_tol=1e-5), "Bounded profiling time exceeded")
    require(status["gpu_final_empty"] is True and not read(_file(retrieved, "gpu_final.json"))["compute"].strip(),
            "Final GPU compute process remains or was not observed")
    require(not read(_file(retrieved, "gpu_initial.json"))["compute"].strip(), "Initial GPU compute process was present")
    expected = {_label(dataset, arm, mode, rep) for dataset, arm, mode, rep in
                itertools.product(datasets, (SHARED, MAC), ("train", "eval"), (1, 2, 3))}
    require(len(status["completed"]) == 36 and set(status["completed"]) == expected, "Incomplete/duplicate worker completion list")
    exit_evidence = _verify_exits(retrieved, manifest, status, expected)
    synthetic_steps = _verify_qualifications(retrieved, contract, exit_evidence["children"])
    batches, input_summary = _batches(retrieved, datasets)
    repetitions, per_batch, runtime = _profiles(retrieved, contract, datasets, status, expected, batches, exit_evidence["children"])
    comparisons, adverse = aggregate(repetitions)
    steps, evals = (sum(row[key] for row in repetitions) for key in ("optimizer_steps", "eval_batches"))
    require(steps == evals == 666 and len(comparisons) == 6, "Final paired exposure mismatch")
    verification = dict(status="passed", execution_status=status["status"], verified_manifest_files=len(files),
                        verified_source_files=len(contract["source_files"]), verified_operational_files=len(contract["operational_files"]),
                        complete_workers=36, actual_child_exit0_count=41, synthetic_qualification_passes=4,
                        synthetic_qualification_optimizer_steps=synthetic_steps, actual_controller_exit0=True,
                        controller_pid_absent=True, optimizer_steps_including_warmup=steps,
                        evaluation_batches_including_warmup=evals, measured_batch_rows=36 * 32,
                        all_batch_rows=len(per_batch), same_runtime=True, same_batch_sha_and_initial_state=True,
                        common_head_exact_runtime_assertion=True, gpu_no_compute_process_at_end=True,
                        new_graphs_in_measured32=sum(row["new_graphs_in_measured32"] for row in repetitions),
                        contract_sha256=canonical(contract), source_closure_sha256=contract["source_closure_sha256"],
                        terminal_manifest_sha256=sha(retrieved / "terminal_manifest.json"),
                        controller_exit_sha256=sha(retrieved / "controller_exit.json"), elapsed_seconds=elapsed,
                        local_measurement_source_changed=False, accuracy_evaluated=False, checkpoint_saved=False,
                        scientific_fit=False, timing_repetitions=3, scientific_training_seeds=0,
                        uncertainty="sample SD of three timing repeats of fresh seed42 models")
    analysis = dict(comparison=comparisons, repetitions=repetitions, input_summary=input_summary, runtime=runtime,
                    execution=status, verification=verification, adverse_memory_repetitions=adverse,
                    definitions=dict(latency="mean and sample SD of each repetition's measured32 batch median",
                                     speed_ratio="paired MAC/shared measured32 batch median, then mean/sample SD",
                                     compute_throughput="4096 targets / sum(compute seconds), then mean/sample SD",
                                     transfer_compute_throughput="4096 targets / sum(transfer+compute seconds), then mean/sample SD",
                                     relative_change="100 * (Shared15 repeat mean / MAC repeat mean - 1)",
                                     split="train only; eval names a cost path, not a Validation/Test split"))
    report_text = _report(analysis, bundle)
    report.mkdir(parents=True, exist_ok=True)
    with (report / ".analysis.exclusive").open("x", encoding="utf-8") as handle:
        json.dump(dict(bundle=str(bundle), contract_sha256=canonical(contract)), handle, ensure_ascii=False)
        handle.write("\n")
    for name, value in (("analysis.json", analysis), ("verification.json", verification)):
        with (report / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    for name, rows in (("repetitions.csv", repetitions), ("per_batch.csv", per_batch), ("inputs.csv", input_summary)):
        _csv(report / name, rows)
    with (report / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(report_text)
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    result = analyze(args.bundle, args.report)
    print(json.dumps(result["verification"], ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
