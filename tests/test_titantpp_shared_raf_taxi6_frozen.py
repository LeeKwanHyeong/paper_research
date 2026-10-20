"""Fabricated CPU-only checks of frozen RAF3 evaluation; no real Test is read."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import torch

from paper.scripts import evaluate_titantpp_shared_raf_taxi6_frozen as evaluator
from paper.scripts import run_titantpp_shared_history_raw9_5090 as frozen


def jobs():
    return [{"id": f"{dataset}__{evaluator.ARM}__seed{seed}__losslog_mse",
             "dataset": dataset, "arm": evaluator.ARM, "seed": seed, "selected_epoch": epoch}
            for dataset, seeds in evaluator.SELECTED_EPOCHS.items() for seed, epoch in seeds.items()]


def contract_fixture():
    return {"schema": evaluator.SCHEMA, "host": "5080", "evaluation_authorized": True, "jobs": jobs()}


@pytest.mark.parametrize("seed", (42, 52, 62))
def test_exact_approved_raf_shared_jobs(seed):
    contract = contract_fixture()
    job = next(job for job in contract["jobs"] if job["seed"] == seed)
    assert evaluator.selected_job(contract, job["id"]) == job


@pytest.mark.parametrize("key,value", [("host", "5090"), ("schema", "other"), ("evaluation_authorized", False)])
def test_foreign_or_unapproved_contract_rejected(key, value):
    contract = contract_fixture()
    contract[key] = value
    with pytest.raises(ValueError):
        evaluator.selected_job(contract, contract["jobs"][0]["id"])


@pytest.mark.parametrize("key,value", [("selected_epoch", 10), ("arm", "titantpp_history_mlp_width16"),
                                      ("dataset", "yellow_trip_hourly"), ("seed", 32)])
def test_checkpoint_reselection_or_foreign_model_rejected(key, value):
    contract = contract_fixture()
    contract["jobs"][0][key] = value
    with pytest.raises(ValueError):
        evaluator.selected_job(contract, contract["jobs"][0]["id"])


def test_selected_subset_completion_does_not_require_or_assert_whole_raw9(tmp_path):
    receipt = {"schema": "titantpp_shared_raf_taxi6_selected_training_v1", "status": "complete_selected_subset",
        "selected_fit_count": 6, "training_validation_reference_count": 6,
        "actual_scientific_worker_exit0_count": 6, "all_selected_training_terminal_verified": True,
        "whole_raw9_complete": False, "held_out_test_evaluated": False}
    path = tmp_path / "selected_training.json"
    path.write_text(json.dumps(receipt))
    contract = {"selected_training_receipt": {"path": path.name, "sha256": evaluator.sha(path)}}
    assert evaluator.verify_training_completion(tmp_path, contract) == receipt
    for key, value in (("status", "partial"), ("selected_fit_count", 5), ("training_validation_reference_count", 5),
                       ("actual_scientific_worker_exit0_count", 5), ("whole_raw9_complete", True),
                       ("held_out_test_evaluated", True)):
        changed = {**receipt, key: value}
        path.write_text(json.dumps(changed))
        contract["selected_training_receipt"]["sha256"] = evaluator.sha(path)
        with pytest.raises(ValueError):
            evaluator.verify_training_completion(tmp_path, contract)


def test_artifact_hash_and_root_escape_rejected(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("{}")
    assert evaluator.bound_path(tmp_path, {"path": path.name, "sha256": evaluator.sha(path)}) == path
    with pytest.raises(ValueError):
        evaluator.bound_path(tmp_path, {"path": path.name, "sha256": "0"*64})
    with pytest.raises(ValueError):
        evaluator.bound_path(tmp_path, {"path": "../outside", "sha256": "0"*64})


@pytest.fixture(scope="module", params=evaluator.DATASETS)
def fabricated(request, tmp_path_factory):
    """Train the original joint engine only on four fabricated sequences."""
    from paper.scripts.count_aware_tpp_backbone import training, core
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    configure_runtime("cpu", 4)
    root = tmp_path_factory.mktemp("fabricated_frozen_raf")
    args, frame, data, interface, quantity = frozen.prepared._engine_fixture(root, evaluator.ARM)
    dataset = request.param
    args.dataset_contract = dataset
    data["model"]["time_observation_contract"]["unit"] = "month" if dataset == "raf_spare_parts" else "hour"
    args.time_alpha = 1.0
    for key, value in frozen.input_identity().items():
        setattr(args, key, value)
    interface.update(frozen.input_identity(), source_files_sha256="a"*64)
    data["loader"].update(batch_size=128, lookback_weeks=84)
    data["quantity_boundaries_all_train_rows"] = [2, 30, 60, evaluator.TAILS[dataset]]
    data["inherited_data_identity"] = {
        "data": {"path": str(args.data), "sha256": args.data_sha256},
        "split_manifest": {"path": str(args.split_manifest), "sha256": args.split_manifest_sha256},
        "populations": {}}
    for split in ("train", "validation"):
        loader = make_loader(frame, target_split=split, batch_size=128, lookback_weeks=84,
            max_seq_len=84, shuffle=False, generator=None)
        data["inherited_data_identity"]["populations"][split] = frozen.population_identity(loader.dataset, split)[1]
    with frozen._engine_hooks(evaluator.ARM) as tracked:
        summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
            interface_meta=interface, backbone=evaluator.ARM, quantity_variant=frozen.VARIANT, seed=42)
        proof = tracked["tracker"].proof()
    checkpoint = Path(summary["checkpoint_path"])
    receipt = {"selected_checkpoint": str(checkpoint), "selected_epoch": summary["best_epoch"],
        "checkpoint_sha256": frozen.sha_file(checkpoint), "state_sha256": summary["checkpoint_state_sha256"],
        "source_files_sha256": "a"*64, "time_head_training_verification": proof}
    raw = pl.read_parquet(args.data)
    test_rows = []
    for entity in raw["oper_part_no"].unique().sort().to_list():
        for seq, q in ((11, 10000.0), (12, 4.0)):
            test_rows.append({"oper_part_no": entity, "seq": seq, "delta_t": 2,
                "demand_qty": q, "chronological_split": "test"})
    full = pl.concat([raw, pl.DataFrame(test_rows)]).sort(["oper_part_no", "seq"])
    populations = {}
    for split in ("validation", "test"):
        loader = make_loader(core.prepare_count_frame(full), target_split=split, batch_size=128,
            lookback_weeks=84, max_seq_len=84, shuffle=False, generator=None)
        populations[split] = frozen.population_identity(loader.dataset, split)[1]
    job = {"id": f"{dataset}__{evaluator.ARM}__seed42__losslog_mse", "dataset": dataset, "arm": evaluator.ARM, "seed": 42}
    return root, raw, full, data, job, receipt, populations


def test_actual_joint_selected_checkpoint_validation_and_fabricated_test_six_metrics(fabricated):
    root, raw, full, data, job, receipt, populations = fabricated
    before = evaluator.sha(receipt["selected_checkpoint"])
    validation_file, test_file = root / "val_predictions.gz", root / "test_predictions.gz"
    validation = evaluator.evaluate_frame(data, job, receipt, root, raw, "validation", validation_file,
        populations["validation"], device="cpu")
    replay_file = root / "val_replay_predictions.gz"
    replay = evaluator.evaluate_frame(data, job, receipt, root, raw, "validation", replay_file,
        populations["validation"], device="cpu")
    frozen.compare_metrics(validation["metrics"], replay["metrics"])
    frozen.compare_prediction_files(validation_file, replay_file)
    test = evaluator.evaluate_frame(data, job, receipt, root, full, "test", test_file,
        populations["test"], device="cpu")
    assert validation["prediction_rows"] == test["prediction_rows"] == 8
    assert all(test["qualification"].values()) and test["parameters_unchanged"]
    assert test["metrics"]["tail"]["count"] == 4
    for group in ("overall", "tail"):
        for key in ("qty_rmse", "qty_mae", "time_nll"):
            assert np.isfinite(test["metrics"][group][key])
    for baseline in test["simple_quantity_baselines"].values():
        assert baseline["overall"]["time_nll"] is None
        assert baseline["tail"]["time_nll"] is None
        assert baseline["time_metric"] == "N/A_no_time_model"
    assert before == evaluator.sha(receipt["selected_checkpoint"])
    with gzip.open(test_file, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    q = np.asarray([row["raw_quantity"] for row in rows])
    p = np.asarray([row["predicted_raw_quantity"] for row in rows])
    assert max(q) == 10000  # No cap or raw upper-bound enforcement.
    assert test["metrics"]["overall"]["qty_rmse"] == pytest.approx(np.sqrt(np.mean((q-p)**2)))
    assert test["metrics"]["overall"]["qty_mae"] == pytest.approx(np.mean(np.abs(q-p)))


def test_original_summary_metadata_binding_and_tamper_rejection(fabricated):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    _, _, _, _, _, receipt, _ = fabricated
    path = Path(receipt["selected_checkpoint"])
    payload = torch_load_checkpoint(path, map_location="cpu")
    summary = frozen.read(path.parent / "summary.json")
    job = {"selected_epoch": receipt["selected_epoch"]}
    evaluator.validate_checkpoint_metadata(payload, summary, job)
    altered = deepcopy(summary)
    altered["interface_meta"]["time_alpha"] = 0
    with pytest.raises(ValueError):
        evaluator.validate_checkpoint_metadata(payload, altered, job)


def test_gate_is_checked_before_test_checkpoint_or_population_access(tmp_path):
    contract = contract_fixture()
    with patch.object(evaluator, "verify_contract", return_value=(contract, {})), \
         patch.object(evaluator, "verify_job_artifacts", side_effect=AssertionError("Checkpoint opened before gate")), \
         patch.object(pl, "scan_parquet", side_effect=AssertionError("Test opened before gate")):
        with pytest.raises(FileNotFoundError):
            evaluator.evaluate_stage(tmp_path, contract["jobs"][0]["id"], "test")


def gate_fixture(root):
    contract = contract_fixture()
    gate = {"schema": evaluator.GATE_SCHEMA, "status": "PASS", "contract_sha256": evaluator.canonical(contract),
        "job_count": 6, "validation_receipts": {}, "validation_worker_exits": {}}
    for job in contract["jobs"]:
        folder = root / "results" / job["id"]
        folder.mkdir(parents=True)
        receipt, exit_path = folder / "validation_receipt.json", folder / "validation_exit.json"
        receipt.write_text(json.dumps({"job_id": job["id"]}))
        exit_path.write_text(json.dumps({"job_id": job["id"], "split": "validation",
            "contract_sha256": evaluator.canonical(contract), "actual_os_returncode": 0,
            "timeout": False, "actual_child_exit_observed": True}))
        gate["validation_receipts"][job["id"]] = evaluator.sha(receipt)
        gate["validation_worker_exits"][job["id"]] = evaluator.sha(exit_path)
    (root / "validation_gate.json").write_text(json.dumps(gate))
    return contract, gate


def test_all_six_validation_receipts_and_actual_exit0_required(tmp_path):
    contract, gate = gate_fixture(tmp_path)
    with patch.object(evaluator, "verify_validation_receipt", side_effect=lambda *args: args[-1]):
        assert evaluator.verify_validation_gate(tmp_path, contract) == gate
        for key, value in (("actual_os_returncode", 1), ("timeout", True), ("actual_child_exit_observed", False)):
            job = contract["jobs"][0]
            path = tmp_path / "results" / job["id"] / "validation_exit.json"
            original = evaluator.read(path)
            changed = {**original, key: value}
            path.write_text(json.dumps(changed))
            gate["validation_worker_exits"][job["id"]] = evaluator.sha(path)
            (tmp_path / "validation_gate.json").write_text(json.dumps(gate))
            with pytest.raises(ValueError):
                evaluator.verify_validation_gate(tmp_path, contract)
            path.write_text(json.dumps(original))
            gate["validation_worker_exits"][job["id"]] = evaluator.sha(path)
            (tmp_path / "validation_gate.json").write_text(json.dumps(gate))
        gate["validation_receipts"].pop(job["id"])
        (tmp_path / "validation_gate.json").write_text(json.dumps(gate))
        with pytest.raises(ValueError):
            evaluator.verify_validation_gate(tmp_path, contract)


@pytest.mark.parametrize("split", ("train", "resume", "qualification", None))
def test_no_unsupported_inference_or_training_routes(tmp_path, split):
    with patch.object(evaluator, "verify_contract", side_effect=AssertionError("Unexpected artifact access")):
        with pytest.raises(ValueError):
            evaluator.evaluate_stage(tmp_path, "unused", split)


def test_real_cli_rejects_cpu_and_unknown_stage_before_artifact_access():
    for extra in (("--device", "cpu", "--split", "test"), ("--split", "train")):
        with patch.object(evaluator, "read", side_effect=AssertionError("Unexpected artifact access")):
            with pytest.raises(SystemExit):
                evaluator.main(("--root", "unused", "--job", "unused", *extra))


def test_validation_path_cannot_materialize_test(fabricated):
    _, _, full, _, _, _, _ = fabricated
    with pytest.raises(ValueError):
        evaluator.validate_frame(full, "validation")


@pytest.fixture(scope="module")
def fabricated_bindings(fabricated):
    """Build only synthetic artifact references for the production binding helpers."""
    root, raw, _, original_data, small_job, receipt, populations = fabricated
    data = deepcopy(original_data)
    data.update(dataset_id=small_job["dataset"],
        optimizer={"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.})
    data["model"].update(lambda_log_qty=1., lambda_tail=0., time_head_lr_multiplier=1.)
    data["loader"].update(num_workers=0, drop_last=False, train_shuffle=True, validation_shuffle=False)
    training = {"schema": frozen.CONTRACT_SCHEMA, "scope": "execution",
        "execution_authorization": {"real_data_training": True, "real_data_inference": True,
            "remote_execution": True, "test_evaluation": False},
        "common_condition": {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.},
        "data_variant": "original_uncapped", "quantity_capping_applied": False,
        "hosts": {"5090": {"root": str(root)}},
        "source": {"base_git_revision": "fabricated_preparation_only", "files_sha256": "a"*64},
        "architecture": {"fabricated": True}, "design_sha256": "b"*64,
        "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
            "batch_size": 128, "monitor": frozen.MONITOR},
        "datasets": [{**deepcopy(data), "dataset_id": name} for name in frozen.DATASETS],
        "jobs": [{"id": f"{name}__{evaluator.ARM}__seed{seed}__losslog_mse", "host": "5090",
            "dataset": name, "arm": evaluator.ARM, "seed": seed,
            "time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.}
            for name in frozen.DATASETS for seed in frozen.SEEDS]}
    checkpoint = Path(receipt["selected_checkpoint"])
    job = {**small_job, "selected_epoch": receipt["selected_epoch"]}
    prediction = root / "bound_original_validation.gz"
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    original = frozen.evaluate_selected(data, job, receipt, root, prepare_count_frame(raw),
        "validation", prediction, device="cpu")
    trained = {**frozen.base_receipt(training, "5090", job, "train"), **receipt,
        "validation_reference": original, "target_identity": original["target_identity"],
        "files": {str(prediction.relative_to(root)): evaluator.sha(prediction)}}
    train_path = root / "bound_train_receipt.json"
    train_path.write_text(json.dumps(trained))
    # A separate parent Validation stage has not run in this selected-subset test.
    for key, path in (("train_receipt", train_path), ("training_validation_reference", train_path),
                      ("checkpoint", checkpoint), ("selection_history", checkpoint.parent / "history.json"),
                      ("checkpoint_metadata", checkpoint.parent / "summary.json"), ("validation_predictions", prediction)):
        job[key] = {"path": str(path.relative_to(root)), "sha256": evaluator.sha(path)}
    exit_path = root / "selected_train_exit.json"
    exit_path.write_text(json.dumps({"pid": 456, "actual_os_returncode": 0, "timeout": False,
        "command": ["python", "-m", "original_worker", "--job", job["id"], "--stage", "train"]}))
    job["train_exit"] = {"path": exit_path.name, "sha256": evaluator.sha(exit_path)}
    contract = {"data": {job["dataset"]: {"populations": populations}}, "jobs": [job]}
    return root, data, training, contract, job, trained, original


def test_original_receipt_checkpoint_history_and_predictions_are_bound(fabricated_bindings):
    root, _, training, contract, job, trained, reference = fabricated_bindings
    data, relocated, actual_reference = evaluator.verify_job_artifacts(root, contract, training, job)
    assert data['dataset_id'] == job['dataset']
    assert actual_reference == reference
    assert relocated['checkpoint_sha256'] == trained['checkpoint_sha256']
    assert Path(relocated['selected_checkpoint']).is_file()
    tampered = deepcopy(job)
    tampered['selected_epoch'] += 1
    with pytest.raises(ValueError):
        evaluator.verify_job_artifacts(root, contract, training, tampered)
    assert evaluator.sha(Path(root) / job['train_receipt']['path']) == job['train_receipt']['sha256']


def test_validation_gate_checks_individual_predictions_not_just_metrics(fabricated_bindings):
    root, _, training, binding, job, trained, reference = fabricated_bindings
    contract = {**binding, 'schema': evaluator.SCHEMA, 'host': '5090', 'evaluation_authorized': True}
    prediction_path = root / 'new_native_validation_prediction.gz'
    original = root / job['validation_predictions']['path']
    prediction_path.write_bytes(original.read_bytes())
    result = {'schema': evaluator.SCHEMA, 'status': 'complete', 'split': 'validation',
        'job_id': job['id'], 'contract_sha256': evaluator.canonical(contract),
        'selected_epoch': job['selected_epoch'], 'checkpoint_sha256': job['checkpoint']['sha256'],
        'parameters_unchanged': True, 'context_unchanged': True, 'validation_replay_passed': True,
        'individual_predictions_match': True, 'native_runtime_checked': True,
        'held_out_test_evaluated': False, 'qualification': {key: True for key in evaluator.QUALIFICATION_KEYS},
        'source_files_sha256': trained['source_files_sha256'],
        'metrics': reference['metrics'], 'state_sha256': reference['state_sha256'],
        'target_identity': reference['target_identity'],
        'predictions': {'path': prediction_path.name, 'sha256': evaluator.sha(prediction_path)}}
    assert evaluator.verify_validation_receipt(root, contract, job, result) == result
    with gzip.open(prediction_path, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    rows[0]['predicted_raw_quantity'] += .1
    with gzip.open(prediction_path, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')
    result['predictions']['sha256'] = evaluator.sha(prediction_path)
    with pytest.raises(ValueError):
        evaluator.verify_validation_receipt(root, contract, job, result)



def test_all_six_synthetic_qualifications_required_before_validation(tmp_path):
    contract = contract_fixture()
    for job in contract['jobs']:
        path = tmp_path/'qualification'/(job['id']+'.json')
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({'schema': evaluator.SCHEMA, 'status': 'passed',
            'contract_sha256': evaluator.canonical(contract), 'job_id': job['id'], 'seed': job['seed'],
            'real_data_loaded': False, 'checkpoint_loaded': False, 'held_out_test_evaluated': False,
            'checks': {key: True for key in evaluator.QUALIFICATION_KEYS}}))
    evaluator.verify_synthetic_qualifications(tmp_path, contract)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        evaluator.verify_synthetic_qualifications(tmp_path, contract)


def test_synthetic_qualification_uses_no_dataset_or_checkpoint(tmp_path, fabricated):
    _, _, _, data, _, _, _ = fabricated
    contract = contract_fixture()
    job = contract['jobs'][0]
    with patch.object(evaluator, 'verify_contract', return_value=(contract, {})), \
         patch.object(evaluator, 'runtime_check', return_value={'synthetic_cpu_fixture': True}), \
         patch.object(frozen, 'select_job', return_value=(job, data, None, None)), \
         patch.object(pl, 'read_parquet', side_effect=AssertionError('Synthetic data read')), \
         patch.object(pl, 'scan_parquet', side_effect=AssertionError('Synthetic data read')), \
         patch('simple_lab_test.search.common.runner.torch_load_checkpoint', side_effect=AssertionError('Synthetic checkpoint read')):
        result = evaluator.synthetic_qualification(tmp_path, job['id'], device='cpu')
        assert result['status'] == 'passed' and all(result['checks'].values())
        assert not result['checkpoint_loaded'] and not result['real_data_loaded']
        assert not result['held_out_test_evaluated']
        assert (tmp_path/'qualification'/(job['id']+'.intent.json')).exists()
        with pytest.raises(ValueError):
            evaluator.synthetic_qualification(tmp_path, job['id'], device='cpu')


def test_real_stage_wrapper_reproduces_bound_validation_without_modifying_originals(fabricated_bindings, fabricated):
    root, data, training, binding, job, trained, original = fabricated_bindings
    _, raw, full, _, _, _, populations = fabricated
    training_data = root/'stage_training.parquet'
    full_data = root/'stage_full.parquet'
    raw.write_parquet(training_data)
    full.write_parquet(full_data)
    contract = {**contract_fixture(), 'data': {job['dataset']: {'populations': populations,
        'training': {'path': training_data.name, 'sha256': evaluator.sha(training_data)},
        'full': {'path': full_data.name, 'sha256': evaluator.sha(full_data)}}}}
    for candidate in contract['jobs']:
        path = root/'qualification'/(candidate['id']+'.json')
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({'schema': evaluator.SCHEMA, 'status': 'passed',
            'contract_sha256': evaluator.canonical(contract), 'job_id': candidate['id'], 'seed': candidate['seed'],
            'real_data_loaded': False, 'checkpoint_loaded': False, 'held_out_test_evaluated': False,
            'checks': {key: True for key in evaluator.QUALIFICATION_KEYS}}))
    original_shas = {key: evaluator.sha(root/job[key]['path']) for key in
        ('checkpoint', 'checkpoint_metadata', 'selection_history', 'validation_predictions', 'train_receipt', 'training_validation_reference')}
    with patch.object(evaluator, 'verify_contract', return_value=(contract, training)), \
         patch.object(evaluator, 'selected_job', return_value=job), \
         patch.object(evaluator, 'runtime_check', return_value={'synthetic_cpu_fixture': True}):
        result = evaluator.evaluate_stage(root, job['id'], 'validation', device='cpu')
        assert result['validation_replay_passed'] and result['individual_predictions_match']
        assert result['parameters_unchanged'] and result['context_unchanged']
        assert not result['held_out_test_evaluated']
        frozen.compare_metrics(result['metrics'], original['metrics'])
        with pytest.raises(ValueError):
            evaluator.evaluate_stage(root, job['id'], 'validation', device='cpu')
    assert all(evaluator.sha(root/job[key]['path']) == value for key, value in original_shas.items())


def test_reviewed_evaluator_is_unchanged_and_metrics_threshold_adaptation_is_exact():
    import hashlib
    source = Path(evaluator._REVIEWED_PATH).read_bytes()
    assert hashlib.sha256(source).hexdigest() == evaluator.REVIEWED_WORKER_SHA256
    assert evaluator.REVIEWED_WORKER_SHA256 == '2fdd0f56757c4330e3e140ca800bbfbb5d13060f8b663d7156a1e40a7e3225c5'
    assert len(evaluator._replacements) == 3
    assert evaluator.TAILS == {'raf_spare_parts': 200.0, 'yellow_trip_hourly': 3449.0}


def test_selected_exit_can_be_joined_only_to_matching_preserved_observation_and_launch(tmp_path):
    job = jobs()[0]
    actual = {'job_id': job['id'], 'stage': 'train', 'pid': 123, 'start_ticks': 456,
              'actual_os_returncode': 0, 'timeout': False}
    observation = tmp_path/'observed_status.json'
    observation.write_text(json.dumps({'status.json': {'actual_worker_exits': {job['id']: actual}}}))
    launch = tmp_path/'launch.jsonl'
    launch.write_text(json.dumps({'job_id': job['id'], 'stage': 'train', 'pid': 123, 'start_ticks': 456,
        'command': ['python', '-m', 'original', '--job', job['id'], '--stage', 'train']})+'\n')
    recorded = tmp_path/'joined_exit.json'
    record = {'actual_worker_exit': actual,
        'original_observation': {'path': observation.name, 'sha256': evaluator.sha(observation)},
        'launch_record': {'path': launch.name, 'sha256': evaluator.sha(launch)}}
    recorded.write_text(json.dumps(record))
    job['train_exit'] = {'path': recorded.name, 'sha256': evaluator.sha(recorded)}
    assert evaluator.verify_train_exit(tmp_path, job) == actual
    record['actual_worker_exit'] = {**actual, 'pid': 789}
    recorded.write_text(json.dumps(record))
    job['train_exit']['sha256'] = evaluator.sha(recorded)
    with pytest.raises(ValueError):
        evaluator.verify_train_exit(tmp_path, job)


@pytest.mark.parametrize('dataset', evaluator.DATASETS)
@pytest.mark.parametrize('seed', (42, 52, 62))
def test_every_selected_dataset_seed_epoch_is_fixed(dataset, seed):
    contract = contract_fixture()
    job = next(job for job in contract['jobs'] if job['dataset'] == dataset and job['seed'] == seed)
    assert evaluator.selected_job(contract, job['id'])['selected_epoch'] == evaluator.SELECTED_EPOCHS[dataset][seed]
    job['selected_epoch'] += 1
    with pytest.raises(ValueError):
        evaluator.selected_job(contract, job['id'])


def test_evaluation_runtime_is_separate_from_original_training_host():
    observed = {'python': '3.12.13', 'torch': '2.11.0+cu130', 'cuda': '13.0', 'cudnn': 91900,
                'threads': 4, 'interop_threads': 1, 'gpu': {'uuid': evaluator.GPU_UUID}}
    contract = {'native': {'runtime_expected': {key: observed[key] for key in
        ('python', 'torch', 'cuda', 'cudnn', 'threads', 'interop_threads')}}}
    with patch('paper.scripts.quantity_comparison_runtime.configure_runtime') as configure, \
         patch('paper.scripts.quantity_comparison_runtime.runtime_identity', return_value=observed):
        assert evaluator.runtime_check(contract, 'cuda:0') == observed
        configure.assert_called_once_with('cuda:0', 4)
        contract['native']['runtime_expected']['cudnn'] = 92000
        with pytest.raises(ValueError):
            evaluator.runtime_check(contract, 'cuda:0')
