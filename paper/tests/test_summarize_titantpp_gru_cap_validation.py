"""Synthetic integrity tests for the cap-study Validation-only summary."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import copy
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "paper/scripts/summarize_titantpp_gru_cap_validation.py"
SPEC = importlib.util.spec_from_file_location("_cap_summary", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summary)


def dump(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return summary.sha(path)


class CapValidationSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="cap-summary-", dir="/private/tmp")
        self.root = Path(self.temp.name)
        self.retrieval = self.root / "retrieved"
        self.jobs = []
        self.baselines = []
        for dataset, host in (("yellow_trip_hourly", "5080"), ("raf_spare_parts", "5080"),
                              ("intermittent_frozen_5000", "5090")):
            for seed in (42, 52, 62):
                candidate_id = f"{dataset}__{seed}__{summary.CANDIDATE}"
                self.jobs.append({"id": candidate_id, "dataset": dataset, "seed": seed,
                                  "arm": summary.CANDIDATE, "host": host})
                for arm in summary.BASELINES:
                    self.baselines.append(self.make_baseline(dataset, seed, arm))
        self.contract = {
            "held_out_test_evaluated": False, "planned_fit_count": 9, "jobs": self.jobs,
            "baseline_references": self.baselines,
            "baseline_references_sha256": summary.canonical(self.baselines),
        }
        self.contract_path = self.root / "execution_contract.json"
        dump(self.contract_path, self.contract)
        self.index = {"schema": summary.INDEX_KIND,
                      "contract_raw_sha256": summary.sha(self.contract_path), "jobs": {}}
        for job in self.jobs:
            self.index["jobs"][job["id"]] = self.make_candidate(job)
        self.index_path = self.root / "retrieval_index.json"
        dump(self.index_path, self.index)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def history(self, epoch: int = 2) -> list[dict[str, float | int]]:
        return [{"epoch": 1, "val_qty_rmse": 4.0}, {"epoch": epoch, "val_qty_rmse": 3.0},
                {"epoch": epoch + 1, "val_qty_rmse": 3.0}]

    def population(self, dataset: str) -> dict[str, object]:
        return {"target_count": 10, "target_identity_sha256": "i-" + dataset,
                "target_quantity_sha256": "q-" + dataset}

    def metric(self, dataset: str, seed: int, arm: str) -> dict[str, float]:
        offset = {"yellow_trip_hourly": 10.0, "raf_spare_parts": 20.0,
                  "intermittent_frozen_5000": 30.0}[dataset] + seed / 1000
        adjustment = {summary.CANDIDATE: 0.0, summary.BASELINES[0]: 1.0,
                      summary.BASELINES[1]: 2.0}[arm]
        return {key: offset + adjustment + index / 10 for index, key in enumerate(summary.METRICS)}

    def make_baseline(self, dataset: str, seed: int, arm: str) -> dict[str, object]:
        base = self.root / "baseline" / dataset / str(seed) / arm
        hist = base / "history.json"
        history_sha = dump(hist, {"history": self.history()})
        checkpoint = base / "best_val_qty_rmse_model.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes((dataset + arm + str(seed)).encode())
        checkpoint_sha = summary.sha(checkpoint)
        population = self.population(dataset)
        replay = base / "validation_replay.json"
        replay_sha = dump(replay, {
            "status": "complete", "split": "validation", "held_out_test_evaluated": False,
            "checkpoint_sha256": checkpoint_sha, "state_sha256": "state-" + arm + str(seed),
            "selected_epoch": 2, "metrics": self.metric(dataset, seed, arm),
            "expected_target_count": population["target_count"],
            "target_identity_sha256": population["target_identity_sha256"],
            "truth_sha256": population["target_quantity_sha256"],
        })
        return {"id": f"{dataset}__{seed}__{arm}", "dataset": dataset, "seed": seed, "arm": arm,
                "history_path": str(hist), "history_sha256": history_sha,
                "validation_receipt_path": str(replay), "validation_receipt_sha256": replay_sha,
                "selected_epoch": 2, "checkpoint_sha256": checkpoint_sha,
                "state_tensor_sha256": "state-" + arm + str(seed),
                "validation_population": population, "metrics": self.metric(dataset, seed, arm)}

    def make_candidate(self, job: dict[str, object]) -> dict[str, str]:
        base = self.retrieval / str(job["id"])
        history = base / "history.json"
        history_sha = dump(history, {"history": self.history()})
        checkpoint = base / "best_val_qty_rmse_model.pt"
        checkpoint.write_bytes(str(job["id"]).encode())
        checkpoint_sha = summary.sha(checkpoint)
        population = self.population(str(job["dataset"]))
        input_receipt = base / "input_receipt.json"
        input_sha = dump(input_receipt, {"held_out_materialized": False,
            "populations": {"validation": population}})
        native_metrics = self.metric(str(job["dataset"]), int(job["seed"]), str(job["arm"]))
        native_metrics["whole_qty_rmse"] = 3.0
        endpoint = base / "endpoint_replays.json"
        endpoint_sha = dump(endpoint, {
            "status": "complete", "held_out_test_evaluated": False,
            "evaluation_scope": "validation_only", "job": job,
            "selected": {"checkpoint_path": "/remote/best_val_qty_rmse_model.pt",
                         "state_sha256": "candidate-state-" + str(job["id"]), "count": 10,
                         "qty_rmse": native_metrics["whole_qty_rmse"], "qty_mae": native_metrics["whole_qty_mae"],
                         "time_nll": native_metrics["whole_time_nll"],
                         "tail": {"qty_rmse": native_metrics["tail_qty_rmse"], "qty_mae": native_metrics["tail_qty_mae"],
                                  "time_nll": native_metrics["tail_time_nll"]},
                         "evaluation_scope": "validation_only", "held_out_test_evaluated": False},
            "last": {"evaluation_scope": "validation_only", "held_out_test_evaluated": False}, "best_epoch": 2,
        })
        derived = base / "derived_validation_replay.json"
        derived_sha = dump(derived, {"schema": "titantpp_gru_residual_cap_validation_local_normalized_replay_v1",
            "job": job, "source_endpoint_replays_sha256": endpoint_sha, "history_sha256": history_sha,
            "checkpoint_sha256": checkpoint_sha, "state_tensor_sha256": "candidate-state-" + str(job["id"]),
            "selected_epoch": 2, "validation_population": population, "metrics": native_metrics,
            "held_out_test_evaluated": False})
        terminal = base / "terminal_manifest.json"
        terminal_sha = dump(terminal, {"status": "complete", "scientific_success": True,
            "contract_sha256": summary.canonical(self.contract), "job": job,
            "held_out_test_evaluated": False, "evaluation_scope": "validation_only",
            "files": {"history.json": history_sha, "best_val_qty_rmse_model.pt": checkpoint_sha,
                      "endpoint_replays.json": endpoint_sha, "input_receipt.json": input_sha}})
        return {"terminal_manifest_path": str(terminal.relative_to(self.retrieval)),
                "terminal_manifest_sha256": terminal_sha, "history_path": str(history.relative_to(self.retrieval)),
                "history_sha256": history_sha, "checkpoint_path": str(checkpoint.relative_to(self.retrieval)),
                "checkpoint_sha256": checkpoint_sha, "endpoint_replays_path": str(endpoint.relative_to(self.retrieval)),
                "endpoint_replays_sha256": endpoint_sha,
                "input_receipt_path": str(input_receipt.relative_to(self.retrieval)), "input_receipt_sha256": input_sha,
                "derived_validation_replay_path": str(derived.relative_to(self.retrieval)),
                "derived_validation_replay_sha256": derived_sha,
                "state_tensor_sha256": "candidate-state-" + str(job["id"])}

    def refresh_index(self) -> None:
        dump(self.index_path, self.index)

    def test_exact_nine_candidates_and_eighteen_baselines_summarize(self) -> None:
        result = summary.summarize(self.contract_path, self.retrieval, self.index_path)
        self.assertEqual(result["status"], "complete_validation_only")
        self.assertEqual(result["candidate_fit_count"], 9)
        self.assertEqual(result["baseline_reference_count"], 18)
        self.assertFalse(result["held_out_test_evaluated"])
        taxi = result["datasets"]["yellow_trip_hourly"]
        self.assertEqual(taxi["seeds"], [42, 52, 62])
        self.assertLess(taxi["candidate_deltas"][summary.BASELINES[0]]["mean"]["whole_qty_rmse"], 0)
        self.assertEqual(taxi["arms"][summary.CANDIDATE]["sample_sd"]["whole_qty_rmse"], 0)

    def test_missing_candidate_or_extra_binding_is_rejected(self) -> None:
        self.index["jobs"].pop(self.jobs[0]["id"])
        self.refresh_index()
        with self.assertRaisesRegex(ValueError, "exactly nine jobs"):
            summary.summarize(self.contract_path, self.retrieval, self.index_path)

    def test_nonterminal_history_or_wrong_selected_epoch_is_rejected(self) -> None:
        job = self.jobs[0]
        binding = self.index["jobs"][job["id"]]
        history = self.retrieval / binding["history_path"]
        dump(history, {"history": [{"epoch": 1, "val_qty_rmse": 2.0}, {"epoch": 2, "val_qty_rmse": 3.0}]})
        binding["history_sha256"] = summary.sha(history)
        self.refresh_index()
        with self.assertRaisesRegex(ValueError, "terminal evidence|selection binding"):
            summary.summarize(self.contract_path, self.retrieval, self.index_path)

    def test_test_marked_candidate_endpoint_is_rejected(self) -> None:
        job = self.jobs[0]
        binding = self.index["jobs"][job["id"]]
        replay = self.retrieval / binding["endpoint_replays_path"]
        value = json.loads(replay.read_text())
        value["held_out_test_evaluated"] = True
        dump(replay, value)
        binding["endpoint_replays_sha256"] = summary.sha(replay)
        terminal = self.retrieval / binding["terminal_manifest_path"]
        terminal_value = json.loads(terminal.read_text())
        terminal_value["files"]["endpoint_replays.json"] = binding["endpoint_replays_sha256"]
        dump(terminal, terminal_value)
        binding["terminal_manifest_sha256"] = summary.sha(terminal)
        self.refresh_index()
        with self.assertRaisesRegex(ValueError, "Validation-only"):
            summary.summarize(self.contract_path, self.retrieval, self.index_path)

    def test_markdown_renders_all_six_metrics_and_three_seed_differences(self) -> None:
        result = summary.summarize(self.contract_path, self.retrieval, self.index_path)
        before = copy.deepcopy(result)
        rendered = summary.markdown(result)
        labels = ("전체 raw 수량 RMSE", "전체 raw 수량 MAE", "큰 수량 RMSE", "큰 수량 MAE",
                  "전체 recorded positive-integer TimeNLL", "큰 수량 recorded positive-integer TimeNLL")
        for dataset, value in result['datasets'].items():
            section = rendered.split('## ' + dataset + '\n', 1)[1].split('\n## ', 1)[0]
            for metric, label in zip(summary.METRICS, labels):
                cells = [f"{value['arms'][arm]['mean'][metric]:.6g}±{value['arms'][arm]['sample_sd'][metric]:.6g}"
                         for arm in (summary.CANDIDATE, *summary.BASELINES)]
                self.assertIn('| ' + label + ' | ' + ' | '.join(cells) + ' |', section)
                for reference, name in zip(summary.BASELINES, ('FULL', 'CNN')):
                    comparison = section.split('### CAP0.25−' + name + ': seed별 차이\n', 1)[1].split('\n### ', 1)[0]
                    delta = value['candidate_deltas'][reference]
                    differences = [delta['candidate_minus_reference'][str(seed)][metric] for seed in (42, 52, 62)]
                    differences.append(delta['mean'][metric])
                    formatted = ['0' if number == 0 else f'{number:+.6g}' for number in differences]
                    self.assertIn('| ' + label + ' | ' + ' | '.join(formatted) + ' |', comparison)
        self.assertEqual(rendered.count('| 지표 | seed42 | seed52 | seed62 | 평균 차이 |'), 6)
        self.assertIn('표본표준편차(ddof=1)', rendered)
        self.assertIn('Test 일반화 개선이나 순위 반전의 원인을 입증하지 않는다.', rendered)
        self.assertEqual(result, before, 'Rendering must not change computed results')

    def test_markdown_keeps_unfavorable_and_tied_seed_metric_cells_visible(self) -> None:
        result = summary.summarize(self.contract_path, self.retrieval, self.index_path)
        taxi = result['datasets']['yellow_trip_hourly']
        delta = taxi['candidate_deltas'][summary.BASELINES[0]]
        delta['candidate_minus_reference']['42']['whole_qty_rmse'] = 0.0
        delta['candidate_minus_reference']['52']['tail_qty_mae'] = 0.125
        delta['unfavorable_or_tied_seed_metrics'] = [
            {'seed': 42, 'metric': 'whole_qty_rmse', 'delta': 0.0},
            {'seed': 52, 'metric': 'tail_qty_mae', 'delta': 0.125},
        ]
        rendered = summary.markdown(result)
        taxi_text = rendered.split('## yellow_trip_hourly\n', 1)[1].split('\n## ', 1)[0]
        full_text = taxi_text.split('**CAP0.25−FULL의 악화·동률 결과**', 1)[1].split('\n### ', 1)[0]
        self.assertIn('| 42 | 전체 raw 수량 RMSE | 0 | 동률 |', full_text)
        self.assertIn('| 52 | 큰 수량 MAE | +0.125 | 악화 |', full_text)
        self.assertNotIn('결과 없음', full_text)
        cnn_text = taxi_text.split('**CAP0.25−CNN의 악화·동률 결과**', 1)[1]
        self.assertIn('해당 3seed의 여섯 지표에서 악화·동률 결과 없음.', cnn_text)


if __name__ == "__main__":
    unittest.main()
