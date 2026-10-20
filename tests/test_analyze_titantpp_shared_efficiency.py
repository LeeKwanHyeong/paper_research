"""Synthetic evidence tests; no torch, datasets, SSH, or GPUs are loaded."""
import copy
import csv
import itertools
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import unittest

from paper.scripts import analyze_titantpp_shared_efficiency as analyzer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def build_bundle(root):
    """Build complete but entirely synthetic native evidence with known ratios."""
    root.mkdir()
    source = root / "source"
    source.mkdir()
    sources = {}
    for index in range(985):
        name = f"frozen_{index:04}.py"
        (source / name).write_text(f"# synthetic frozen source {index}\n", encoding="utf-8")
        sources[name] = analyzer.sha(source / name)
    for name in ("benchmark.py", "supervise.py"):
        (root / name).write_text("# never executed\n", encoding="utf-8")
    runtime = dict(python="3.12.13", torch="2.11.0+cu130", cudnn=92000,
                   gpu=dict(uuid=analyzer.GPU_UUID, name="synthetic RTX 5090"))
    datasets = [dict(dataset_id=name, loader=dict(batch_size=128, max_seq_len=length),
                     inherited_data_identity=dict(data=dict(sha256="d" * 64),
                                                  populations=dict(train=dict(target_count=10000))))
                for name, length in analyzer.DATASETS.items()]
    contract = dict(models=[analyzer.SHARED, analyzer.MAC], datasets=datasets,
                    scope=dict(train_only=True, accuracy_evaluation=False, checkpoint_writes=False,
                               full_epochs=False, full_fits=False, held_out=False, validation=False),
                    protocol=dict(seed=42, batch_size=128, repetitions=3, modes=["train", "eval"],
                                  warmup_batches_per_worker=5, measured_batches_per_worker=32, workers=36,
                                  maximum_optimizer_steps=666, maximum_evaluation_batches=666,
                                  parallel_workers=1, automatic_retry=False,
                                  maximum_seconds_including_remote_preparation=1800),
                    host=dict(gpu_uuid=analyzer.GPU_UUID, runtime_expected={key: runtime[key] for key in ("python", "torch", "cudnn")}),
                    source_files=sources, source_closure_sha256=analyzer.canonical(sources), max_seconds=1800,
                    operational_files={name: analyzer.sha(root / name) for name in ("benchmark.py", "supervise.py")})
    write(root / "contract.json", contract)
    retrieved = root / "retrieved"
    retrieved.mkdir()
    for name in ("contract.json", "benchmark.py", "supervise.py"):
        shutil.copyfile(root / name, retrieved / name)
    write(retrieved / "approval.json", dict(approved=True, contract_sha256=analyzer.canonical(contract)))
    claim = dict(pid=900, started_unix=1000, deadline_unix=2800, max_seconds=1800,
                 contract_sha256=analyzer.canonical(contract))
    for name in ("claim.json", "claim.exclusive"):
        write(retrieved / name, claim)
    for name in ("gpu_initial.json", "gpu_final.json"):
        write(retrieved / name, dict(compute="", gpu="synthetic"))
    exits = []

    def add_exit(label):
        record = dict(label=label, pid=1000 + len(exits), owner_pid=900, returncode=0,
                      actual_child_exit_observed=True, pid_absent_after_wait=True,
                      timed_out=False, error=None, started_unix=1010, finished_unix=1020,
                      wall_seconds=10.0, log=f"logs/{label}.log", argv=["never-executed"])
        exits.append(record)
        write(retrieved / "child_exits" / f"{label}.json", record)
        path = retrieved / record["log"]
        path.parent.mkdir(exist_ok=True)
        path.write_text("synthetic successful child\n", encoding="utf-8")
        return record

    qualifications = {}
    for arm, device in itertools.product((analyzer.SHARED, analyzer.MAC), ("cpu", "cuda:0")):
        name = f"{arm}__{device.replace(':', '_')}"
        child = add_exit(f"qualify__{name}")
        rows = [dict(dataset=dataset["dataset_id"], batch_size=128 if device == "cuda:0" else 2,
                     length=dataset["loader"]["max_seq_len"] if device == "cuda:0" else 8,
                     optimizer_steps=1, finite=True, causality=True, padding=True, fresh_state=True)
                for dataset in datasets]
        path = retrieved / f"qualification/{name}.json"
        write(path, dict(status="passed", contract_sha256=analyzer.canonical(contract), arm=arm, device=device,
                         pid=child["pid"], runtime=runtime, rows=rows, real_data_loaded=False,
                         accuracy_evaluated=False, checkpoint_saved=False))
        qualifications[str(path.relative_to(retrieved))] = analyzer.sha(path)
    prepare = add_exit("prepare")
    write(retrieved / "preflight.json", dict(status="passed", pid=prepare["pid"],
                                             contract_sha256=analyzer.canonical(contract),
                                             source_closure=contract["source_closure_sha256"],
                                             qualification_files=qualifications))
    frozen = {}
    for dataset in datasets:
        name, length = dataset["dataset_id"], dataset["loader"]["max_seq_len"]
        cache = retrieved / f"batches/{name}.pt"
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(b"synthetic tensors are not deserialized")
        batch_rows = [dict(batch=number, phase="warmup" if number < 5 else "measured",
                           targets=[["part", number * 128 + index] for index in range(128)],
                           valid_lengths=[length // 2] * 128, padded_length=length,
                           tensor_sha256=analyzer.canonical([name, number])) for number in range(37)]
        frozen[name] = dict(split="train", population=10000, seed=42, common_head_equal=True,
                            file_sha256=analyzer.sha(cache), batches=batch_rows,
                            initial={arm: dict(state_sha256=analyzer.canonical([name, arm]),
                                               parameters=54321 if arm == analyzer.SHARED else 122310)
                                     for arm in (analyzer.SHARED, analyzer.MAC)})
        write(cache.with_suffix(".json"), frozen[name])
    labels = []
    for dataset, arm, mode, rep in itertools.product(analyzer.DATASETS, (analyzer.SHARED, analyzer.MAC), ("train", "eval"), (1, 2, 3)):
        label = analyzer._label(dataset, arm, mode, rep)
        labels.append(label)
        child = add_exit(label)
        rows = []
        for number in range(37):
            batch = frozen[dataset]["batches"][number]
            compute = .05 if number < 5 else .001 * rep * (1 if arm == analyzer.SHARED else rep + 1)
            rows.append(dict(batch=number, phase=batch["phase"], compute_seconds=compute,
                             transfer_seconds=.0005, transfer_compute_seconds=compute + .0005,
                             targets=128, valid_events=sum(batch["valid_lengths"]),
                             tensor_sha256=batch["tensor_sha256"], unique_compiled_graphs=1))
        allocated = (400 if arm == analyzer.SHARED else 100) if mode == "eval" else (100 if arm == analyzer.SHARED else 200)
        value = dict(status="success", dataset=dataset, arm=arm, mode=mode, rep=rep, seed=42,
                     pid=child["pid"], owner_pid=900, started_unix=1011, finished_unix=1019,
                     process_seconds=2., process_setup_seconds=.01, runtime=runtime,
                     contract_sha256=analyzer.canonical(contract), source_closure_sha256=contract["source_closure_sha256"],
                     input_file_sha256="d" * 64,
                     batch_manifest_sha256=analyzer.sha(retrieved / f"batches/{dataset}.json"),
                     parameters=frozen[dataset]["initial"][arm]["parameters"],
                     initial_state_sha256=frozen[dataset]["initial"][arm]["state_sha256"],
                     optimizer_steps=37 if mode == "train" else 0, eval_batches=37 if mode == "eval" else 0,
                     targets_total=4736, targets_measured=4096, common_head_equal=True,
                     memory_baseline_bytes=dict(allocated=10 * 2**20, reserved=20 * 2**20),
                     memory_low_baseline_bytes=dict(allocated=8 * 2**20, reserved=20 * 2**20),
                     memory_peak_bytes=dict(allocated=(allocated + rep) * 2**20, reserved=(allocated + rep + 50) * 2**20),
                     cold_first_batch_seconds=rows[0]["transfer_compute_seconds"],
                     compiled_graphs_after_warmup=1, compiled_graphs_during_measured=0, rows=rows,
                     accuracy_evaluated=False, checkpoint_saved=False, scientific_fit=False)
        write(retrieved / f"results/{label}.json", value)
    write(retrieved / "status.json", dict(status="complete", error=None, completed=labels,
                                          child_exit_count=41, worker_exit_count=36, gpu_final_empty=True,
                                          finished_unix=1100, elapsed_seconds=100., **claim))
    write(retrieved / "controller_exit.json", dict(controller_pid=900, pid=900, actual_os_returncode=0,
                                                   returncode=0, actual_controller_exit_observed=True,
                                                   pid_absent_after_wait=True))
    reseal(root, exits)


def reseal(root, exits=None):
    retrieved = root / "retrieved"
    if exits is None:
        exits = [analyzer.read(path) for path in sorted((retrieved / "child_exits").glob("*.json"))]
    files = {str(path.relative_to(retrieved)): analyzer.sha(path) for path in retrieved.rglob("*")
             if path.is_file() and path.name not in ("terminal_manifest.json", "controller_exit.json")}
    write(retrieved / "terminal_manifest.json", dict(status="complete", files=files, controller_pid=900,
                                                      controller_os_exit_observed=False,
                                                      actual_child_exits=exits, accuracy_evaluated=False,
                                                      checkpoint_saved=False, automatic_retry=False))


class AnalyzerEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pristine = tempfile.TemporaryDirectory()
        cls.template = Path(cls.pristine.name) / "bundle"
        build_bundle(cls.template)

    @classmethod
    def tearDownClass(cls):
        cls.pristine.cleanup()

    def setUp(self):
        self.sandbox = tempfile.TemporaryDirectory()
        self.addCleanup(self.sandbox.cleanup)
        self.bundle = Path(self.sandbox.name) / "bundle"
        self.report = Path(self.sandbox.name) / "report"
        shutil.copytree(self.template, self.bundle)
        self.retrieved = self.bundle / "retrieved"

    def mutate(self, name, callback, seal=True):
        path = self.retrieved / name
        value = analyzer.read(path)
        callback(value)
        write(path, value)
        if seal:
            reseal(self.bundle)

    def first_result(self):
        return sorted((self.retrieved / "results").glob("*.json"))[0]

    def assert_rejected(self, text):
        with self.assertRaisesRegex(analyzer.EvidenceError, text):
            analyzer.analyze(self.bundle, self.report)
        self.assertFalse(self.report.exists(), "Invalid evidence should not create report outputs")

    def test_complete_aggregation_and_all_output_files(self):
        self.report.mkdir()
        (self.report / "COLLECTION_RECEIPT.json").write_text("preserve me\n")
        result = analyzer.analyze(self.bundle, self.report)
        verification = result["verification"]
        self.assertEqual(verification["optimizer_steps_including_warmup"], 666)
        self.assertEqual(verification["evaluation_batches_including_warmup"], 666)
        self.assertEqual(verification["synthetic_qualification_optimizer_steps"], 12)
        self.assertEqual(verification["all_batch_rows"], 1332)
        self.assertEqual(verification["actual_child_exit0_count"], 41)
        self.assertEqual(verification["verified_source_files"], 985)
        self.assertEqual(len(result["comparison"]), 6)
        self.assertEqual(len(result["adverse_memory_repetitions"]), 18)
        for row in result["comparison"]:
            self.assertAlmostEqual(row["shared_median_compute_ms_mean"], 2.)
            self.assertAlmostEqual(row["shared_median_compute_ms_sd"], 1.)
            self.assertAlmostEqual(row["paired_mac_over_shared_ratio_mean"], 3.)
            self.assertAlmostEqual(row["paired_mac_over_shared_ratio_sd"], 1.)
            self.assertEqual(row["paired_ratios"], [2., 3., 4.])
            self.assertEqual(row["shared_parameters_mean"], 54321)
            self.assertAlmostEqual(row["shared_compute_targets_per_second_mean"], statistics.mean([128000., 64000., 128000. / 3]))
            self.assertAlmostEqual(row["shared_transfer_compute_targets_per_second_mean"], statistics.mean([128 / .0015, 128 / .0025, 128 / .0035]))
        inputs = {row["dataset"]: row for row in result["input_summary"]}
        self.assertEqual(inputs["raf_spare_parts"]["padded_length"], 84)
        self.assertEqual(inputs["raf_spare_parts"]["valid_length_including_target_mean"], 42)
        self.assertEqual(inputs["yellow_trip_hourly"]["padded_length"], 256)
        self.assertEqual(inputs["raf_spare_parts"]["padding_fraction"], .5)
        self.assertEqual({name for name in analyzer.OUTPUTS if (self.report / name).is_file()}, set(analyzer.OUTPUTS))
        for name, count in (("repetitions.csv", 36), ("per_batch.csv", 1332), ("inputs.csv", 3)):
            with (self.report / name).open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), count)
        text = (self.report / "report.md").read_text(encoding="utf-8")
        self.assertIn("54,321", text)
        self.assertNotIn("96,003", text)
        self.assertNotIn("2026-09-30", text)
        self.assertIn("학습 seed 3개를 뜻하지 않는다", text)
        self.assertIn("Shared15의 peak가 MAC보다 큰 반복을 모두", text)
        self.assertEqual((self.report / "COLLECTION_RECEIPT.json").read_text(), "preserve me\n")

    def test_existing_outputs_and_exclusive_claim_are_preserved(self):
        self.report.mkdir()
        for name in ("analysis.json", ".analysis.exclusive"):
            path = self.report / name
            path.write_text("original")
            with self.assertRaisesRegex(analyzer.EvidenceError, "overwrite forbidden"):
                analyzer.analyze(self.bundle, self.report)
            self.assertEqual(path.read_text(), "original")
            path.unlink()

    def test_failed_status_rejected(self):
        self.mutate("status.json", lambda value: value.update(status="failed_no_automatic_retry"))
        self.assert_rejected("fully complete")

    def test_actual_controller_exit_required(self):
        self.mutate("controller_exit.json", lambda value: value.update(actual_controller_exit_observed=False))
        self.assert_rejected("Direct controller OS exit")

    def test_child_pid_absence_required(self):
        self.mutate("child_exits/prepare.json", lambda value: value.update(pid_absent_after_wait=False))
        self.assert_rejected("Child exit unconfirmed")

    def test_missing_one_profile_rejected_even_after_resealing(self):
        self.first_result().unlink()
        reseal(self.bundle)
        self.assert_rejected("Exactly 36 paired profiles")

    def test_duplicate_worker_identity_rejected(self):
        path = self.first_result()
        self.mutate(str(path.relative_to(self.retrieved)), lambda value: value.update(rep=2))
        self.assert_rejected("Profile identity mismatch")

    def test_manifest_hash_change_rejected_before_aggregation(self):
        self.mutate(str(self.first_result().relative_to(self.retrieved)), lambda value: value.update(parameters=1), seal=False)
        self.assert_rejected("manifest SHA mismatch")

    def test_omitting_file_from_manifest_does_not_bypass_hash_checks(self):
        name = str(self.first_result().relative_to(self.retrieved))
        self.mutate("terminal_manifest.json", lambda value: value["files"].pop(name), seal=False)
        self.assert_rejected("Unmanifested or missing results")

    def test_frozen_source_hash_change_rejected(self):
        (self.bundle / "source/frozen_0000.py").write_text("changed")
        self.assert_rejected("Frozen source SHA mismatch")

    def test_batch_and_initial_binding_changes_rejected(self):
        name = str(self.first_result().relative_to(self.retrieved))
        for field in ("batch_manifest_sha256", "initial_state_sha256"):
            original = analyzer.read(self.retrieved / name)
            with self.subTest(field=field):
                self.mutate(name, lambda value: value.update({field: "a" * 64}))
                self.assert_rejected("Batch/initial state/common head")
                write(self.retrieved / name, original)
                reseal(self.bundle)

    def test_nonpositive_or_nonfinite_batch_times_rejected(self):
        name = str(self.first_result().relative_to(self.retrieved))
        original = analyzer.read(self.retrieved / name)
        for value in (0, -1, True):
            with self.subTest(value=value):
                self.mutate(name, lambda result: result["rows"][5].update(compute_seconds=value))
                self.assert_rejected("compute_seconds")
                write(self.retrieved / name, original)
                reseal(self.bundle)
        path = self.retrieved / name
        path.write_text(path.read_text().replace('"compute_seconds": 0.05', '"compute_seconds": NaN', 1))
        reseal(self.bundle)
        self.assert_rejected("Nonfinite JSON value")

    def test_eval_path_does_not_allow_validation_data(self):
        self.mutate("batches/raf_spare_parts.json", lambda value: value.update(split="validation"))
        self.assert_rejected("Train/head initialization")

    def test_qualification_failure_rejected(self):
        name = f"qualification/{analyzer.SHARED}__cpu.json"
        self.mutate(name, lambda value: value["rows"][0].update(causality=False))
        self.assert_rejected("Synthetic numerical/state checks")

    def test_final_gpu_must_be_empty(self):
        self.mutate("gpu_final.json", lambda value: value.update(compute="123, foreign process, 999"))
        self.assert_rejected("Final GPU compute process remains")

    def test_manifest_path_traversal_rejected(self):
        self.mutate("terminal_manifest.json", lambda value: value["files"].update({"../escape": "0" * 64}), seal=False)
        self.assert_rejected("Unsafe evidence path")

    def test_nonfinite_and_duplicate_json_keys_rejected(self):
        path = Path(self.sandbox.name) / "invalid.json"
        for text in ('{"x": NaN}', '{"x": Infinity}', '{"x": 1, "x": 2}'):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(analyzer.EvidenceError):
                analyzer.read(path)


class PairedAggregationTests(unittest.TestCase):
    def test_aggregation_pairs_by_repeat_not_input_position(self):
        rows = []
        metrics = ("median_compute_ms", "mean_compute_ms", "median_transfer_plus_compute_ms", "mean_transfer_plus_compute_ms",
                   "compute_targets_per_second", "transfer_compute_targets_per_second", "peak_allocated_mib", "peak_reserved_mib", "parameters")
        for dataset, arm, mode, rep in itertools.product(analyzer.DATASETS, (analyzer.SHARED, analyzer.MAC), ("train", "eval"), (3, 1, 2)):
            row = dict(dataset=dataset, arm=arm, mode=mode, rep=rep)
            row.update({key: rep if arm == analyzer.SHARED else rep * (rep + 1) for key in metrics})
            rows.append(row)
        comparisons, adverse = analyzer.aggregate(list(reversed(rows)))
        self.assertEqual(len(comparisons), 6)
        self.assertFalse(adverse)
        for row in comparisons:
            self.assertEqual(row["paired_ratios"], [2., 3., 4.])
            self.assertEqual(row["paired_mac_over_shared_ratio_sd"], 1.)


if __name__ == "__main__":
    unittest.main()
