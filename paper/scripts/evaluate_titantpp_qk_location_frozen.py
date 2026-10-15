#!/usr/bin/env python3
"""Sealed RAF6 component evaluation; no data read at import.

Adapted from the preserved Taxi/RAF24 evaluator. Only operational adaptation lives here. No training/source/model changes, remote
control, automatic retry or prediction export. Both real-data stages require an
explicit hash-bound operational permit, and Test additionally requires every
Validation receipt and actual worker exit0 in a global gate.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import re
import sys
import time
import traceback

KIND = "titantpp_qk_location_raf6_frozen_evaluation_v1"
COHORT = "qk_location_raf6_v1"
# Metadata only; no dataset or checkpoint is accessed at import.
_LINEAGE_PATH = Path(__file__).with_name("lineage.json")
_LINEAGE = json.loads(_LINEAGE_PATH.read_text()) if _LINEAGE_PATH.exists() else {}
SOURCE_CLOSURE = _LINEAGE.get("source_closure", "UNBOUND")
SOURCE_REVISION = _LINEAGE.get("source_revision", "UNBOUND")
TRAINING_CONTRACT = _LINEAGE.get("training_canonical", "UNBOUND")
GPU_UUIDS = {'5080': 'GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6',
            '5090': 'GPU-c9adc246-ef66-7906-bc07-6e152fa9952f'}
ARMS = ("titantpp_cnn_query_only_v1", "titantpp_cnn_key_only_v1")
LOCATION_BY_ARM = dict(zip(ARMS, ("q", "k")))
DATASETS = ("raf_spare_parts",)
TAILS = {"yellow_trip_hourly": 3449., "raf_spare_parts": 200., "intermittent_frozen_5000": 187.}
SEEDS = (42, 52, 62)
METRICS = ("qty_mae", "qty_rmse", "time_nll")
ROUTE = "paper.scripts.run_titantpp_qk_location_cnn"
PROJECT_PREFIXES = {"models", "paper", "data_loader", "simple_lab_test", "utils"}
TRUTH_FIELDS = ("target_id", "entity_id", "seq", "split", "recorded_date",
                "raw_quantity", "recorded_gap", "history_length", "last_quantity", "mean_quantity")
QUALIFICATION_KEYS = {"target_quantity_perturbation_preserves_prediction", "target_quantity_perturbation_preserves_time_loss",
    "target_gap_perturbation_preserves_quantity_prediction", "single_vs_batch_prediction_matches",
    "single_vs_batch_time_loss_matches", "parameters_unchanged", "finite_outputs"}

def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            value.update(block)
    return value.hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def canonical(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def confined(root, relative):
    root = Path(root).resolve()
    require(not Path(relative).is_absolute(), "Absolute artifact path is not permitted")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Artifact path escapes the frozen root")
    return path


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())



def verify_source(root, source):
    require(source.get("closure") == SOURCE_CLOSURE and source.get("revision") == SOURCE_REVISION,
            "Wrong frozen frozen_source lineage")
    require(len(source.get("files", {})) == 145, "Q-only/K-only source must contain exactly145 files")
    require(canonical(source["files"]) == source["closure"], "Frozen source manifest closure changed")
    base = confined(root, source["root"])
    for relative, expected in source["files"].items():
        path = confined(base, relative)
        require(sha(path) == expected, "Frozen scientific source changed: " + relative)
    return base


def verify_imports(source_root, source):
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in PROJECT_PREFIXES:
            continue
        file = getattr(module, "__file__", None)
        if file is None:
            continue
        path = Path(file).resolve()
        require(path.is_relative_to(source_root), "Nonfrozen project module imported: " + name)
        relative = path.relative_to(source_root).as_posix()
        require(relative in source["files"], "Unsealed project module imported: " + name)
        result[name] = relative
    return result



def validate_payload(payload, job, source, definition, metadata=None):
    require(job.get("cohort") == COHORT and job.get("arm") in ARMS,
            "Foreign component checkpoint cohort/route")
    require(payload.get("best_epoch") == job["selected_epoch"], "Selected checkpoint epoch mismatch")
    require(payload.get("backbone") == job["arm"] and payload.get("seed") == job["seed"],
            "Checkpoint arm or seed mismatch")
    require(payload.get("variant") == "count_only_log_regression"
            and payload.get("evaluation_scope") == "validation_only"
            and payload.get("held_out_test_evaluated") is False, "Checkpoint objective or original scope changed")
    require(payload.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
            and payload.get("checkpoint_monitor_history_key") == "val_qty_rmse"
            and payload.get("checkpoint_selection") == "best_validation_raw_quantity_rmse", "Checkpoint selector changed")
    require(payload.get("initial_state_sha256") == job["initial_state_sha256"], "Checkpoint initialization mismatch")
    require(payload.get("source_revision") == source["revision"]
            and payload.get("source_revision_history") == job["source_revision_history"], "Checkpoint source lineage mismatch")
    require(payload.get("selected_metric_value") == job["selected_metric_value"]
            and payload.get("model_state_sha256") == job["checkpoint_state_sha256"], "Selected checkpoint objective or tensor seal mismatch")
    head = payload.get("interface_meta", {}).get("time_head", {}) or payload.get("interface", {}).get("time_head", {})
    model = definition["model"]
    require(head.get("observation_likelihood") == model["time_observation_contract"]
            and head.get("time_scale") == model["time_scale"]
            and head.get("mode") == model["time_head_mode"], "Frozen recorded time interface changed")
    if metadata is not None:
        actual_metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
        require(canonical(actual_metadata) == canonical(metadata), "Frozen checkpoint complete metadata differs")


def validate_frame(frame, definition, split, np, pl):
    """Apply the frozen recorded-grid checks to the explicitly approved split."""
    required = {"oper_part_no", "seq", "delta_t", "demand_qty", "chronological_split", "demand_dt"}
    require(required <= set(frame.columns) and frame.height > 0, "Frozen input schema is incomplete")
    permitted = {"train", "validation"} if split == "validation" else {"train", "validation", "test"}
    require(set(frame["chronological_split"].unique().to_list()) <= permitted, "Unapproved split materialized")
    require(not frame.select(pl.struct(["oper_part_no", "seq"]).is_duplicated().any()).item(), "Duplicate target key")
    for key in ("demand_qty", "delta_t", "seq"):
        values = frame[key].to_numpy().astype(np.float64)
        require(np.isfinite(values).all(), "Nonfinite recorded " + key)
        if key != "seq":
            require((values >= 0).all(), "Negative recorded " + key)
        if key != "demand_qty":
            require(np.equal(values, np.round(values)).all(), "Recorded values must be integers: " + key)
    top = definition["model"]["time_observation_contract"]["top_code"]
    require(top is None or (frame["delta_t"].to_numpy() <= top).all(), "Recorded duration exceeds frozen top code")
    # Do not permit a later split to enter an earlier target's context.
    rank = frame["chronological_split"].replace_strict({"train": 0, "validation": 1, "test": 2})
    checked = frame.with_columns(rank.alias("_split_rank")).with_columns(
        pl.col("_split_rank").cum_max().over("oper_part_no").alias("_previous_max"))
    require(checked.filter(pl.col("_split_rank") < pl.col("_previous_max")).height == 0,
            "Frozen chronological splits are out of order")


def require_raf_training_slice(full, training, pl):
    require(set(training.columns) <= set(full.columns), "RAF full parent misses frozen export columns")
    subset = full.filter(pl.col("chronological_split").is_in(["train", "validation"])).select(training.columns)
    require(subset.equals(training), "RAF full parent Train/Validation differs from frozen export")


def population_identity(dataset, split, np):
    """Frozen hard_lmm identity formula, extended operationally to approved Test."""
    index = np.asarray(dataset.index, dtype=np.int64).reshape(-1, 2)
    part, position = index[:, 0], index[:, 1] + 1
    seq = np.fromiter((dataset.seq_lists[p][t] for p, t in zip(part, position)), dtype=np.int64)
    q = np.fromiter((dataset.val_lists[p][t] for p, t in zip(part, position)), dtype=np.float64)
    require(all(dataset.split_lists[p][t] == split for p, t in zip(part, position)), "Loader targets have the wrong split")
    value = hashlib.sha256(b"hard_lmm_target_identity_v1\0" + split.encode() + b"\0")
    value.update(json.dumps([str(p) for p in dataset.parts], ensure_ascii=False, separators=(",", ":")).encode())
    for array in (part, position, seq):
        value.update(array.astype("<i8").tobytes())
    return {"target_count": len(q), "target_identity_sha256": value.hexdigest(),
            "target_quantity_sha256": hashlib.sha256(b"hard_lmm_target_quantity_v1\0" + q.astype("<f8").tobytes()).hexdigest()}


def qualify(torch, model, target_outputs, dts, mask, quantities, output, top_code):
    altered_quantity = quantities.clone()
    altered_quantity[:, -1] += 12345.
    q = target_outputs(model, dts, mask, altered_quantity, lambda_log_qty=1.)
    altered_dt = dts.clone()
    altered_dt[:, -1] = (torch.where(dts[:, -1] > 1, torch.ones_like(dts[:, -1]), torch.full_like(dts[:, -1], 2.))
                         if top_code else dts[:, -1] + 123.)
    t = target_outputs(model, altered_dt, mask, quantities, lambda_log_qty=1.)
    single = target_outputs(model, dts[:1], mask[:1], quantities[:1], lambda_log_qty=1.)
    checks = {
        "target_quantity_perturbation_preserves_prediction": torch.allclose(output["pred_qty"], q["pred_qty"], rtol=1e-6, atol=1e-6),
        "target_quantity_perturbation_preserves_time_loss": torch.allclose(output["time_loss"], q["time_loss"], rtol=1e-6, atol=1e-6),
        "target_gap_perturbation_preserves_quantity_prediction": torch.allclose(output["pred_qty"], t["pred_qty"], rtol=1e-6, atol=1e-6),
        "single_vs_batch_prediction_matches": torch.allclose(output["pred_qty"][:1], single["pred_qty"], rtol=1e-5, atol=1e-5),
        "single_vs_batch_time_loss_matches": torch.allclose(output["time_loss"][:1], single["time_loss"], rtol=1e-5, atol=1e-5),
    }
    require(all(checks.values()), "Causality or single/batch qualification failed: " + str(checks))
    return checks


def record_executable(runtime):
    """Capture executable identity on the worker's own physical host."""
    return {**runtime, 'python_executable': str(Path(sys.executable).absolute()),
            'resolved_python_executable': str(Path(sys.executable).resolve())}


def validate_runtime_spec(expected, host=None):
    """Validate native fingerprint metadata with stdlib only, no defaults."""
    required = {'host', 'python', 'python_version', 'torch', 'cuda', 'cudnn',
                'numpy', 'polars', 'gpu_uuid', 'torch_num_threads', 'interop_threads',
                'dtype', 'deterministic_algorithms', 'TF32'}
    require(isinstance(expected, dict), 'Native Runtime fingerprint must be an object')
    missing = sorted(required - set(expected))
    require(not missing, 'Missing native Runtime fields: ' + ', '.join(missing))
    require(expected['host'] in GPU_UUIDS and (host is None or expected['host'] == host),
            'Native Runtime host identity differs')
    require(isinstance(expected['python'], str) and bool(expected['python'])
            and Path(expected['python']).is_absolute(), 'Native Python executable must be an absolute path')
    for key in ('python_version', 'torch', 'numpy', 'polars'):
        value = expected[key]
        pattern = r'\d+\.\d+\.\d+' if key == 'python_version' else r'\d+\.\d+\.\d+[A-Za-z0-9.+_-]*'
        require(isinstance(value, str) and re.fullmatch(pattern, value) is not None,
                'Invalid native version format: ' + key)
    require(isinstance(expected['cuda'], str) and re.fullmatch(r'\d+\.\d+', expected['cuda']) is not None,
            'Native CUDA version must be explicit major.minor')
    require(type(expected['cudnn']) is int
            and expected['cudnn'] == (91900 if expected['host'] == '5080' else 92000),
            'Native cuDNN version differs from approved host')
    require(expected['gpu_uuid'] == GPU_UUIDS[expected['host']], 'Native GPU UUID differs from approved host')
    require(type(expected['torch_num_threads']) is int and expected['torch_num_threads'] in (1, 4)
            and type(expected['interop_threads']) is int and expected['interop_threads'] == 1,
            'Native Runtime requires one or four intra-op threads and one inter-op thread')
    require(expected['dtype'] == 'float32' and expected['deterministic_algorithms'] is True
            and expected['TF32'] is False, 'Native evaluation precision flags differ')
    require(isinstance(expected.get('native', {}), dict), 'Native detailed fingerprint must be an object')
    return expected


def check_runtime(runtime, expected, device, *, current_process=True):
    validate_runtime_spec(expected)
    require(isinstance(runtime, dict), 'Observed Runtime fingerprint must be an object')
    required = {'python', 'torch', 'cuda', 'cudnn', 'numpy', 'polars', 'gpu', 'threads', 'interop_threads',
                'python_executable', 'resolved_python_executable', 'default_dtype', 'deterministic_algorithms',
                'cuda_matmul_tf32', 'cudnn_tf32'} | set(expected.get('native', {}))
    missing = sorted(required - set(runtime))
    require(not missing, 'Missing observed Runtime fields: ' + ', '.join(missing))
    require(runtime["python"] == expected['python_version'], "Pinned Python version mismatch")
    recorded = runtime.get('python_executable')
    resolved = runtime.get('resolved_python_executable')
    require(isinstance(recorded, str) and Path(recorded).is_absolute()
            and isinstance(resolved, str) and Path(resolved).is_absolute(), 'Recorded native executable identity missing')
    # Cross-host receipts must never resolve the other host's symlinks or compare
    # its native executable with this process. The live worker checks both.
    require(Path(recorded) == Path(expected['python']), 'Recorded Python executable differs from pinned native Runtime')
    if expected.get('resolved_python_executable') is not None:
        require(resolved == expected['resolved_python_executable'], 'Recorded resolved native executable differs')
    if current_process:
        require(Path(sys.executable).resolve() == Path(expected["python"]).resolve()
                and recorded == str(Path(sys.executable).absolute())
                and resolved == str(Path(sys.executable).resolve()), "Pinned current Python executable mismatch")
    for key in ("torch", "cuda", "cudnn", "numpy", "polars"):
        require(runtime[key] == expected[key], "Pinned numerical runtime mismatch: " + key)
    if device == "cuda":
        require(isinstance(runtime['gpu'], dict) and runtime['gpu'].get('uuid') == expected["gpu_uuid"],
                "Wrong physical GPU UUID")
    for key, value in expected.get("native", {}).items():
        require(runtime[key] == value, "Pinned native runtime mismatch: " + key)
    require(type(runtime['threads']) is int and type(runtime['interop_threads']) is int
            and runtime["threads"] == expected["torch_num_threads"] and runtime["interop_threads"] == expected["interop_threads"],
            "Pinned numerical thread counts differ")
    require(runtime['default_dtype'] == 'torch.float32' and runtime['deterministic_algorithms'] is True
            and runtime['cuda_matmul_tf32'] is False and runtime['cudnn_tf32'] is False,
            'Observed numerical precision flags differ')



def expected_jobs():
    return {(d, a, s) for d in DATASETS for a in ARMS for s in SEEDS}


def digest_valid(value):
    return isinstance(value, str) and len(value) == 64 and all(ch in '0123456789abcdef' for ch in value)


def strict_selected_epoch(history):
    """First finite strict minimum; no Test, last-state or secondary selector."""
    best = None
    previous_epoch = 0
    for row in history:
        epoch, value = row.get('epoch'), row.get('val_qty_rmse')
        require(type(epoch) is int and epoch > previous_epoch, 'Invalid or unordered saved Validation history')
        previous_epoch = epoch
        if type(value) not in (int, float) or not math.isfinite(value):
            continue
        if best is None or value < best[1]:
            best = (epoch, value)
    require(best is not None, 'No finite full Validation quantity RMSE')
    return best


def read_selection_history(path):
    """Read original CSV or JSON bytes without rewriting the saved history."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        require(rows and {'epoch', 'val_qty_rmse'} <= set(rows[0]), 'Selection CSV schema differs')
        return [{'epoch': int(row['epoch']), 'val_qty_rmse': float(row['val_qty_rmse'])}
                for row in rows]
    require(path.suffix.lower() == '.json', 'Selection history format must be original CSV or JSON')
    value = read(path)
    value = value['history'] if isinstance(value, dict) else value
    require(isinstance(value, list), 'Selection JSON history must be a list')
    return value


def verify_training_metadata(root, c):
    path = confined(root, c['training_contract_path'])
    require(sha(path) == c['training_contract_sha256'], 'Original training contract byte SHA changed')
    training = read(path)
    require(canonical(training) == TRAINING_CONTRACT, 'Original frozen_source training contract changed')
    definitions = {d['dataset_id']: d for d in training['datasets']}
    for dataset, spec in c['datasets'].items():
        require(canonical(spec['definition']) == canonical(definitions[dataset]),
                'Frozen dataset statistics/model/preprocessing differ from training contract')
    return training


def verify_evaluator_integrity(root, c):
    integrity = c.get('evaluator_integrity', {})
    files = integrity.get('files', {})
    require(files and integrity.get('closure') == canonical(files), 'Operational evaluator seal missing or changed')
    require(sha(__file__) in files.values(), 'Running evaluator bytes are not in the operational seal')
    for relative, expected in files.items():
        require(sha(confined(root, relative)) == expected, 'Operational evaluator file changed: ' + relative)
    return integrity


def verify_partial_metadata(root, c):
    approval_path = confined(root, c['partial_approval_path'])
    require(sha(approval_path) == c['partial_approval_sha256'], 'RAF6 approval SHA differs')
    a = read(approval_path)
    require(a['approved'] is True and a['new_training'] is False
            and a['checkpoint_reselection'] is False and a['expected_condition_count'] == 6
            and set(a['job_ids']) == {j['id'] for j in c['jobs']}
            and a['training_contract_sha256'] == TRAINING_CONTRACT,
            'RAF6 explicit approval scope differs')
    require(hashlib.sha256(a['user_instruction'].encode()).hexdigest() == a['user_instruction_sha256'],
            'RAF6 original approval text differs')
    allowed = {a['local_root'], *a['remote_roots'].values()}
    require(str(Path(root).resolve()) in allowed, 'Unapproved evaluation root')
    if c.get('binding_complete'):
        path = confined(root, c['training_receipt_path'])
        require(sha(path) == c['training_receipt_sha256'], 'Training collection receipt changed')
        r = read(path)
        require(r['status'] == 'complete' and r['new_fit_count'] == 6 and r['seed_count'] == 3
                and r['baseline_reference_count'] == 3 and r['held_out_test_evaluated'] is False
                and r['all_training_terminal_verified'] is True, 'Original six training/collection gate incomplete')
    return a




def verify_reused_reference_scope(c):
    """Validate immutable reference roles without reading model or result files."""
    references=c.get('baseline_references',[])
    require(len(references)==3 and {r.get('seed') for r in references}==set(SEEDS)
            and all(r.get('arm')=='titantpp_cnn_gru54_cnn_only'
                    and r.get('comparison_arm')=='titantpp_cnn'
                    and r.get('dataset')=='raf_spare_parts'
                    and r.get('role')=='historical_immutable_reference'
                    and r.get('retrained') is False and r.get('reevaluated') is False
                    and digest_valid(r.get('receipt_sha256'))
                    and digest_valid(r.get('checkpoint_sha256')) for r in references),
            'Historical CNN3 reference-only scope differs')
    require(not c.get('stored_channel_references') and not c.get('stored_qk_only_references'), 'Unapproved stored model references')
    require(c.get('candidate_selection_rule')=='passing_candidates:validation_rmse_then_mae_then_arm',
            'Prewritten passing-candidate selection rule differs')
    return True


def verify_contract(root, *, require_bound=True):
    path = Path(root) / 'contract.json'
    if not require_bound and not path.exists():
        path = Path(root) / 'draft_contract.json'
    c = read(path)
    require(c.get('contract_sha256') == canonical({k: v for k, v in c.items() if k != 'contract_sha256'}),
            'Contract canonical SHA mismatch')
    require(c.get('schema') == KIND and c.get('cohort') == COHORT
            and c.get('evaluation_scope') == 'test_qk_location_raf6_followup'
            and c.get('metrics_schema') == 'quantity_time_tail_v1', 'Foreign component evaluation contract')
    require(c.get('expected_seeds') == list(SEEDS) and c.get('expected_jobs')==6,
            'Frozen seed/condition contract differs')
    require(not require_bound or c.get('binding_complete') is True, 'Terminal checkpoint binding incomplete')
    jobs = c.get('jobs', [])
    actual = {(r.get('dataset'), r.get('arm'), r.get('seed')) for r in jobs}
    require(len(jobs) == len({r.get('id') for r in jobs}) == 6 and actual == expected_jobs(),
            'Contract must bind exactly RAF6; only the approved Q-only/K-only RAF arm')
    require(set(c.get('sources', {})) == {'components'}, 'Foreign component source map')
    source = c['sources']['components']
    require(len(source.get('files', {})) == 145, 'Q-only/K-only source145 required')
    require(c.get('absolute_host_deadline_unix') == 1792679647.915156, 'Inherited host deadline changed')
    require(source.get('closure') == SOURCE_CLOSURE and source.get('revision') == SOURCE_REVISION
            and canonical(source.get('files', {})) == SOURCE_CLOSURE, 'Wrong frozen frozen_source closure')
    require(set(c.get('datasets', {})) == set(DATASETS), 'Component dataset set differs')
    verify_reused_reference_scope(c)
    amendment=confined(root,'comparison_scope_amendment.json')
    require(sha(amendment)==c.get('comparison_scope_amendment_sha256'), 'Comparison scope amendment changed')
    verify_training_metadata(root, c)
    verify_evaluator_integrity(root, c)
    verify_partial_metadata(root, c)
    for name, expected in c.get('bound_files', {}).items():
        require(sha(confined(root, name)) == expected, 'Frozen evaluation artifact changed: ' + name)
    for job in jobs:
        require(job['id'] == f"{job['dataset']}__{job['seed']}__{job['arm']}"
                and job.get('cohort') == COHORT and job.get('study') == 'components', 'Foreign/unsafe component ID')
        host = '5090'
        require(job.get('evaluation_host') == job.get('training_host') == host, 'Native selected-model replay host differs')
        spec = c['datasets'][job['dataset']]
        definition = spec['definition']
        require(definition['dataset_id'] == job['dataset'], 'Dataset definition identity differs')
        require(float(spec['tail_threshold']) == TAILS[job['dataset']]
                == float(definition['quantity_boundaries_all_train_rows'][-1]), 'Frozen tail boundary changed')
        loader = definition['loader']
        require(loader['batch_size'] == 128 and loader['num_workers'] == 0
                and loader['drop_last'] is False and loader.get('shuffle', False) is False,
                'Frozen loader contract differs')
        expected_window = {'yellow_trip_hourly': (168, 256, 1., 'hour'),
                           'intermittent_frozen_5000': (520, 256, 3., 'week'),
                           'raf_spare_parts': (84, 84, 6., 'month')}[job['dataset']]
        lookback, max_len, time_scale, unit = expected_window
        model = definition['model']
        require(loader['lookback_weeks'] == lookback and loader['max_seq_len'] == max_len
                and model['hidden_dim'] == 64 and model['quantity_variant'] == 'count_only_log_regression'
                and model['time_head_mode'] == 'heteroscedastic_lognormal_duration'
                and model['time_scale'] == time_scale
                and model['time_observation_contract'] == {
                    'mode': 'positive_integer_round_clamp_v1', 'unit': unit, 'top_code': None},
                'Frozen quantity/time/window definition differs')
        require(job.get('frozen_source_closure') == SOURCE_CLOSURE, 'Checkpoint science closure differs')
        if not require_bound:
            continue
        require(job.get('binding_status') == 'bound_terminal_immutable_originals', 'Checkpoint originals not bound')
        require(type(job.get('selected_epoch')) is int and job['selected_epoch'] >= 1, 'Missing selected epoch')
        require(type(job.get('selected_metric_value')) in (int, float) and math.isfinite(job['selected_metric_value']),
                'Missing finite selected metric')
        for key in ('checkpoint_state_sha256', 'checkpoint_metadata_sha256', 'initial_state_sha256', 'selection_history_sha256'):
            require(digest_valid(job.get(key)), 'Missing checkpoint seal: ' + key)
        require(Path(job['checkpoint']['relative_path']).name == 'best_val_qty_rmse_model.pt'
                and digest_valid(job['checkpoint']['sha256']), 'Wrong selected binary')
        reference = job['validation_reference']
        require(reference.get('split') == 'validation', 'Selector/reference is not Validation')
        compare_validation_metrics(reference, reference)
        for split in ('validation', 'test'):
            population = spec[split + '_population']
            require(type(population.get('count')) is int and population['count'] > 0, 'Missing bound population count')
            for key in ('target_identity_sha256', 'truth_sha256', 'time_sha256'):
                require(digest_valid(population.get(key)), 'Missing population seal: ' + split + '/' + key)
    require(set(c.get('runtimes', {})) == {'5090'}, 'Native5090 runtime binding missing or foreign host present')
    for host, runtime in c['runtimes'].items():
        validate_runtime_spec(runtime, host)
        require(runtime['torch_num_threads'] == 4, 'RAF native training thread count changed')
    return c


def compare_validation_metrics(actual, reference):
    for group in (None, 'tail'):
        left = actual if group is None else actual[group]
        right = reference if group is None else reference[group]
        require(type(left.get('count')) is int and left['count'] == right['count'] >= 0,
                'Validation population count mismatch')
        for key in METRICS:
            if left['count'] == 0:
                require(left.get(key) is None and right.get(key) is None, 'Empty population metric must be N/A')
            else:
                require(type(left.get(key)) in (int, float) and type(right.get(key)) in (int, float)
                        and math.isfinite(left[key]) and math.isfinite(right[key])
                        and (key == 'time_nll' or left[key] >= 0 and right[key] >= 0)
                        and math.isclose(left[key], right[key], rel_tol=1e-5, abs_tol=1e-5),
                        'Validation replay mismatch: ' + str(group) + '/' + key)


def verify_permit(root, c, job, split, permit_path, permit_sha, *, now=None):
    # Must precede source/checkpoint hashing/imports and every dataset operation.
    approval = verify_partial_metadata(root, c)
    require(Path(root).resolve() == Path(approval['remote_roots'][job['evaluation_host']]).resolve(),
            'Inference may run only in the explicitly approved remote evaluation root')
    require(permit_path and digest_valid(permit_sha), 'Explicit operational evaluation permit required before data access')
    path = confined(root, permit_path)
    require(sha(path) == permit_sha, 'Operational permit byte SHA mismatch')
    p = read(path)
    require(p.get('schema') == 'titantpp_qk_location_raf6_evaluation_permit_v1'
            and p.get('authorized') is True and p.get('new_training') is False
            and p.get('frozen_selection') is True and p.get('stage') == split
            and p.get('contract_sha256') == c['contract_sha256']
            and p.get('evaluator_sha256') == sha(__file__)
            and p.get('host') == job['evaluation_host'], 'Foreign or insufficient operational permit')
    ids = p.get('job_ids', [])
    allowed = {j['id'] for j in c['jobs'] if j['evaluation_host'] == p['host']}
    require(len(ids) == len(set(ids)) and set(ids) <= allowed and job['id'] in ids, 'Permit scope differs')
    current = time.time() if now is None else now
    begin, deadline = p.get('not_before_unix'), p.get('deadline_unix')
    require(type(begin) in (int, float) and type(deadline) in (int, float)
            and math.isfinite(begin) and math.isfinite(deadline) and begin <= current < deadline
            and deadline - begin <= 43200 and deadline <= c['absolute_host_deadline_unix'],
            'Operational permit inactive/expired or host lease extended')
    return p


def verify_validation_receipt(receipt, c, job):
    verify_physical_counts(receipt)
    require(receipt.get('schema') == KIND and receipt.get('status') == 'complete'
            and receipt.get('split') == 'validation' and receipt.get('contract_sha256') == c['contract_sha256'],
            'Foreign or incomplete Validation receipt')
    for key in ('id', 'cohort', 'study', 'dataset', 'arm', 'seed', 'selected_epoch', 'initial_state_sha256'):
        require(receipt.get(key) == job[key], 'Validation receipt identity mismatch: ' + key)
    require(receipt.get('checkpoint_sha256') == job['checkpoint']['sha256']
            and receipt.get('state_sha256') == job['checkpoint_state_sha256']
            and receipt.get('source_closure') == SOURCE_CLOSURE, 'Validation checkpoint seal differs')
    require(receipt.get('retrained') is False and receipt.get('reselected') is False
            and receipt.get('raw_predictions_written') is False and receipt.get('full_population') is True
            and receipt.get('validation_match') == 'PASS' and receipt.get('native_runtime_checked') is True,
            'Validation receipt is not qualified native replay')
    require(set(receipt.get('qualification', {})) == QUALIFICATION_KEYS
            and all(value is True for value in receipt['qualification'].values()), 'Validation causal qualification failed')
    compare_validation_metrics(receipt['metrics'], job['validation_reference'])
    expected = c['datasets'][job['dataset']]['validation_population']
    require(receipt['metrics']['count'] == expected['count'], 'Validation bound population count differs')
    for key in ('target_identity_sha256', 'truth_sha256', 'time_sha256'):
        require(receipt.get(key) == expected[key], 'Validation bound population digest differs: ' + key)
    expected_runtime = c['runtimes'][job['evaluation_host']]
    check_runtime(receipt['runtime'], expected_runtime, 'cuda', current_process=False)
    check_runtime(receipt['runtime_after'], expected_runtime, 'cuda', current_process=False)
    require(receipt['runtime_after'] == receipt['runtime'], 'Recorded native Runtime changed during Validation')
    return receipt


def verify_global_validation_gate(root, c):
    gate = read(Path(root) / 'validation_gate.json')
    require(gate.get('schema') == 'titantpp_qk_location_raf6_validation_gate_v1'
            and gate.get('cohort') == COHORT, 'Foreign partial or whole36 gate cannot release RAF6 Test')
    require(gate.get('status') == 'PASS' and gate.get('contract_sha256') == c['contract_sha256']
            and gate.get('job_count') == 6, 'All RAF6 native Validation replays must pass before Test')
    ids = {j['id'] for j in c['jobs']}
    require(set(gate.get('validation_receipts', {})) == ids
            and set(gate.get('validation_worker_exits', {})) == ids, 'Global gate missing/foreign component IDs')
    created = gate.get('created_unix')
    require(type(created) in (int, float) and math.isfinite(created), 'Gate timestamp missing')
    for job in c['jobs']:
        path = confined(root, 'results/' + job['id'] + '/validation_replay.json')
        require(sha(path) == gate['validation_receipts'][job['id']], 'Validation receipt bytes changed')
        r = verify_validation_receipt(read(path), c, job)
        require(type(r.get('completed_unix')) in (int, float) and math.isfinite(r['completed_unix'])
                and r['completed_unix'] <= created, 'Replay completed after global gate')
        exit_path = confined(root, 'results/' + job['id'] + '/validation_worker_exit.json')
        require(sha(exit_path) == gate['validation_worker_exits'][job['id']], 'Worker exit bytes changed')
        actual = read(exit_path)
        require(actual.get('job_id') == job['id'] and actual.get('contract_sha256') == c['contract_sha256']
                and type(actual.get('exit_code')) is int and actual['exit_code'] == 0
                and actual.get('actual_child_exit_observed') is True
                and type(actual.get('completed_unix')) in (int, float)
                and math.isfinite(actual['completed_unix']) and actual['completed_unix'] <= created,
                'Validation actual worker exit0 not verified')
    return gate


def isolate_source(source_root):
    require(not any(name.split('.')[0] in PROJECT_PREFIXES for name in sys.modules),
            'Fresh isolated evaluator process is required')
    sys.dont_write_bytecode = True
    sys.path[:] = [str(source_root)] + [p for p in sys.path if p
        and not (Path(p).resolve() / 'paper').is_dir()]


def verify_population(actual, expected):
    require(actual['count'] == expected['count'], 'Population count differs')
    for key in ('target_identity_sha256', 'truth_sha256', 'time_sha256'):
        require(actual.get(key) == expected[key], 'Population digest differs: ' + key)


def verify_qk_local_structure(model):
    arm = getattr(model, 'qk_location_arm', None)
    require(arm in ARMS, 'Q/K-location model arm is missing')
    location = LOCATION_BY_ARM[arm]
    prefix = 'encoder.layers.0.attn.'
    expected = {prefix + 'causal_' + location + '_kernel'}
    parameters = dict(model.named_parameters())
    local = {name: value for name, value in parameters.items() if name.startswith(prefix)
             and any(token in name for token in ('causal_', 'difference_kernel', 'qk_only_kernel', 'qk_location_kernel'))}
    require(set(local) == expected and all(tuple(value.shape) == (3,64) for value in local.values())
            and sum(value.numel() for value in local.values()) == 192,
            'Q/K-location physical kernel identity differs')
    removed = {prefix+'causal_'+x+'_kernel' for x in ('q','k','v') if x != location}
    require(not removed.intersection(model.state_dict())
            and not any('difference_kernel' in name or 'qk_only_kernel' in name or 'qk_location_kernel' in name
                        for name in model.state_dict()), 'Unwanted local or dummy weights must be physically absent')
    return {'qk_location_arm':arm,'retained_projection':location,'active_local_cnn_parameters':192,
            'active_query_cnn_parameters':192 if location=='q' else 0,
            'active_key_cnn_parameters':192 if location=='k' else 0,'active_value_cnn_parameters':0,
            'qk_location_parameter_delta':-384,'original_value_kernel_absent':True,
            'value_operation':'unmixed_common_value_projection'}

def verify_physical_counts(receipt):
    c = receipt.get('physical_parameter_counts', {})
    arm=receipt.get('arm'); location=LOCATION_BY_ARM.get(arm)
    require(location is not None and c.get('qk_location_arm') == arm and c.get('retained_projection') == location
            and c.get('actual_registered_parameters') == 103775
            and c.get('actual_active_parameters_before_evaluation_freeze') == 79043
            and c.get('maximum_sequence_length') == 84
            and c.get('active_local_cnn_parameters') == 192 and c.get('active_value_cnn_parameters') == 0
            and c.get('active_query_cnn_parameters') == (192 if location=='q' else 0)
            and c.get('active_key_cnn_parameters') == (192 if location=='k' else 0)
            and c.get('qk_location_parameter_delta') == -384
            and c.get('original_value_kernel_absent') is True
            and c.get('value_operation') == 'unmixed_common_value_projection',
            'Q/K-location receipt physical parameter identity differs')
    return True

def worker(root, job_id, split, device, *, permit_path=None, permit_sha=None):
    root = Path(root).resolve()
    c = verify_contract(root)
    matching = [r for r in c["jobs"] if r["id"] == job_id]
    require(len(matching) == 1, "Unknown or duplicate scientific job")
    job = matching[0]
    require(split in ("validation", "test") and device == "cuda", "Real-data replay requires approved native CUDA Runtime")
    permit = verify_permit(root, c, job, split, permit_path, permit_sha)
    expected_runtime = c["runtimes"][job["evaluation_host"]]
    if split == "test":
        verify_global_validation_gate(root, c)  # Before any model, data or CUDA work.
    gate_sha = sha(root / "validation_gate.json") if split == "test" else None
    started_unix = time.time()
    deadline = min(permit["deadline_unix"], started_unix + 10800)
    output = root / "results" / job_id
    name = "validation_replay" if split == "validation" else "test_metrics"
    require(not (output / (name + ".json")).exists() and not (output / (name + "_failure.json")).exists(),
            "Existing receipt/failure prohibits automatic repeat")
    started = time.monotonic()
    write_exclusive(output / (name + "_claim.json"), {"job": job_id, "split": split,
        "pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv, "contract_sha256": c["contract_sha256"]})
    phase = "integrity"
    try:
        source = c["sources"]["components"]
        source_root = verify_source(root, source)
        checkpoint = confined(root, job["checkpoint"]["relative_path"])
        require(sha(checkpoint) == job["checkpoint"]["sha256"], "Checkpoint byte SHA mismatch")
        metadata_path = confined(root, job["checkpoint_metadata_path"])
        require(sha(metadata_path) == job["checkpoint_metadata_sha256"], "Checkpoint metadata byte SHA mismatch")
        expected_metadata = read(metadata_path)
        history_path = confined(root, job["selection_history_path"])
        require(sha(history_path) == job["selection_history_sha256"], "Saved selection history changed")
        history = read_selection_history(history_path)
        require(strict_selected_epoch(history) == (job["selected_epoch"], job["selected_metric_value"]),
                "Selected checkpoint is not first strict finite full Validation minimum")
        spec = c["datasets"][job["dataset"]]
        definition = spec["definition"]
        data_key = "training_data_path" if split == "validation" and spec.get("training_data_path") else "data_path"
        sha_key = "training_data_sha256" if data_key == "training_data_path" else "data_sha256"
        data_path = confined(root, spec[data_key])
        require(sha(data_path) == spec[sha_key], "Frozen input byte SHA mismatch")
        script_sha = sha(__file__)
        sys.dont_write_bytecode = True
        isolate_source(source_root)
        os.environ["MPLCONFIGDIR"] = str(output / "runtime_cache" / "matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(output / "runtime_cache")
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import numpy as np
        import polars as pl
        import torch
        numerical = importlib.import_module("paper.scripts.quantity_comparison_runtime")
        execution_device = "cuda:0" if device == "cuda" else "cpu"
        numerical.configure_runtime(execution_device, expected_runtime["torch_num_threads"])
        runtime = record_executable(numerical.runtime_identity(execution_device))
        check_runtime(runtime, expected_runtime, device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        engine = importlib.import_module(ROUTE)
        engine.install_hooks()
        factory = importlib.import_module("models.TPPs.CountAwareFactory")
        core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
        loader_module = importlib.import_module("paper.scripts.run_taxi_quantity_interface_ablation")
        runner = importlib.import_module("simple_lab_test.search.common.runner")
        metrics_module = importlib.import_module("paper.scripts.evaluate_titantpp_history_width")
        verify_imports(source_root, source)
        # Reject checkpoint metadata/tensor identity before any population load.
        payload = runner.torch_load_checkpoint(checkpoint, map_location="cpu")
        validate_payload(payload, job, source, definition, expected_metadata)
        factory.validate_checkpoint_route(payload, job["arm"])
        require(runner.canonical_state_dict_sha256(payload["model_state_dict"]) == job["checkpoint_state_sha256"],
                "Checkpoint actual tensor SHA mismatch")
        phase = "loading_population"
        lazy = pl.scan_parquet(data_path)
        if split == "validation":
            lazy = lazy.filter(pl.col("chronological_split").is_in(["train", "validation"]))
        frame = lazy.collect().sort(["oper_part_no", "seq"])
        validate_frame(frame, definition, split, np, pl)
        if job["dataset"] == "raf_spare_parts" and split == "test":
            training_path = confined(root, spec["training_data_path"])
            require(sha(training_path) == spec["training_data_sha256"], "RAF frozen export SHA mismatch")
            training = pl.read_parquet(training_path).sort(["oper_part_no", "seq"])
            require_raf_training_slice(frame, training, pl)
        # Audit train-only initialization without refitting or replacing stored scalars.
        train_q = frame.filter(pl.col("chronological_split") == "train")["demand_qty"].to_numpy().astype(np.float64)
        stats = definition["statistics"]
        digest = hashlib.sha256(b"quantity_comparison_all_train_row_quantity_v1\0" + train_q.astype("<f8").tobytes()).hexdigest()
        require(digest == stats["all_train_quantity_sha256"], "Frozen all-train quantity identity changed")
        for key, measured in (("train_log_mean", float(np.log1p(train_q).mean())),
                              ("train_log_std", float(np.log1p(train_q).std()))):
            require(math.isclose(measured, stats[key], rel_tol=0, abs_tol=1e-12), "Train-only initialization drift: " + key)
        frame = core.prepare_count_frame(frame)
        loader = loader_module.make_loader(frame, target_split=split, batch_size=128, shuffle=False, generator=None,
            lookback_weeks=definition["loader"]["lookback_weeks"], max_seq_len=definition["loader"]["max_seq_len"])
        dataset = loader.dataset
        identity = population_identity(dataset, split, np)
        expected_count = job["validation_reference"]["count"] if split == "validation" else spec["expected_test_count"]
        require(len(dataset) == expected_count > 0, "Frozen eligible target population count differs")
        if split == "validation":
            require(identity == definition["inherited_data_identity"]["populations"]["validation"],
                    "Original frozen Validation target identity differs")
        metadata = frame.select(["oper_part_no", "seq", "chronological_split", "demand_dt"])
        offsets, total = [], 0
        for sequences in dataset.seq_lists:
            offsets.append(total)
            total += len(sequences)
        require(total == frame.height, "Loader frame mapping differs")
        del frame
        config = {k: v for k, v in definition["model"].items()
                  if k not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
        model, _ = factory.build_count_aware_model(job["arm"], **config, train_log_mean=stats["train_log_mean"],
            train_log_std=stats["train_log_std"], max_seq_len=definition["loader"]["max_seq_len"])
        model.load_state_dict(payload["model_state_dict"], strict=True)
        parameter_counts = {
            'actual_registered_parameters': sum(p.numel() for p in model.parameters()),
            'actual_active_parameters_before_evaluation_freeze': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'maximum_sequence_length': definition['loader']['max_seq_len'],
        }
        require(parameter_counts['actual_registered_parameters'] == 103775
                and parameter_counts['actual_active_parameters_before_evaluation_freeze'] == 79043,
                'Q-only/K-only RAF parameter count differs')
        local = verify_qk_local_structure(model)
        parameter_counts.update(local)
        model.to(execution_device).eval().requires_grad_(False)
        before = runner.canonical_state_dict_sha256(model.state_dict())
        require(before == job["checkpoint_state_sha256"], "Loaded model tensor SHA mismatch")
        del payload
        bounds = definition["quantity_boundaries_all_train_rows"]
        metrics = metrics_module.QuantityMetrics(bounds)
        simple = {key: metrics_module.QuantityMetrics(bounds) for key in ("last_quantity", "mean_quantity")}
        target_digest, truth_digest = hashlib.sha256(), hashlib.sha256()
        time_digest = hashlib.sha256(b"titantpp_components_recorded_target_time_v1\0" + split.encode() + b"\0")
        qualification, processed, inference_seconds = {}, 0, 0.
        phase = "inference"
        with torch.no_grad():
            for batch_index, (_, dts, mask, part_indices, quantities) in enumerate(loader):
                require(time.time() < deadline, "Evaluation job/host deadline expired; automatic retry forbidden")
                dts, mask, quantities = (v.to(execution_device) for v in (dts, mask, quantities))
                if device == "cuda":
                    torch.cuda.synchronize()
                batch_start = time.monotonic()
                value = core.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
                if device == "cuda":
                    torch.cuda.synchronize()
                inference_seconds += time.monotonic() - batch_start
                if batch_index == 0:
                    qualification = qualify(torch, model, core.target_outputs, dts, mask, quantities, value,
                        definition["model"]["time_observation_contract"]["top_code"])
                for key in ("true_qty", "pred_qty", "time_loss"):
                    require(bool(torch.isfinite(value[key]).all()), "Nonfinite model output: " + key)
                history_mask = mask.cpu().clone()
                history_mask[:, -1] = False
                history_length = history_mask.sum(dim=1)
                require(bool((history_length >= 1).all()), "Target has no causal history")
                quantity_cpu = quantities.cpu().double()
                last = quantity_cpu[:, -2].numpy()
                mean = ((quantity_cpu * history_mask).sum(dim=1) / history_length).numpy()
                q, prediction, temporal = [value[key].cpu().numpy().astype(np.float64) for key in ("true_qty", "pred_qty", "time_loss")]
                metrics.add(q, prediction, temporal)
                for key, forecast in (("last_quantity", last), ("mean_quantity", mean)):
                    simple[key].add(q, forecast, np.zeros(q.shape, dtype=np.float64))
                gaps = dts[:, -1].cpu().tolist()
                time_digest.update(np.asarray(gaps, dtype="<f8").tobytes())
                lengths = value["history_length"].cpu().tolist()
                for offset in range(len(q)):
                    index = processed + offset
                    part, end = dataset.index[index]
                    require(int(part_indices[offset]) == part and int(history_length[offset]) == lengths[offset], "Loader order/history differs")
                    raw = metadata.row(offsets[part] + end + 1, named=True)
                    entity, sequence = str(dataset.parts[part]), int(dataset.seq_lists[part][end + 1])
                    require(raw["chronological_split"] == split and str(raw["oper_part_no"]) == entity
                            and int(raw["seq"]) == sequence and raw["demand_dt"] is not None, "Target metadata mapping differs")
                    date = str(raw["demand_dt"])
                    target_id = hashlib.sha256(json_bytes([job["dataset"], spec["data_sha256"], entity, sequence, split, date])).hexdigest()
                    row = {"target_id": target_id, "entity_id": entity, "seq": sequence, "split": split,
                        "recorded_date": date, "raw_quantity": float(q[offset]), "recorded_gap": float(gaps[offset]),
                        "history_length": int(lengths[offset]), "last_quantity": float(last[offset]), "mean_quantity": float(mean[offset])}
                    target_digest.update((target_id + "\n").encode())
                    truth_digest.update(json_bytes([row[key] for key in TRUTH_FIELDS]) + b"\n")
                processed += len(q)
        require(processed == expected_count, "Incomplete Test or Validation population")
        final_metrics = {"split": split, **metrics.finish()}
        sealed_population = {"count": processed, "target_identity_sha256": target_digest.hexdigest(),
                             "truth_sha256": truth_digest.hexdigest(), "time_sha256": time_digest.hexdigest()}
        verify_population(sealed_population, spec[split + "_population"])
        if split == "validation":
            compare_validation_metrics(final_metrics, job["validation_reference"])
        simple_results = {key.removesuffix("_quantity"): accumulator.finish() for key, accumulator in simple.items()}
        # Quantity-only baselines have no timing model: never report dummy zero TimeNLL.
        for result in simple_results.values():
            result["split"] = split
            for cell in [result, result["tail"], *result["quantity_cells"]]:
                cell.pop("time_nll", None)
                cell.pop("time_nll_sum", None)
        phase = "final_integrity"
        require(runner.canonical_state_dict_sha256(model.state_dict()) == before, "Model tensors changed during inference")
        require(sha(checkpoint) == job["checkpoint"]["sha256"] and sha(data_path) == spec[sha_key], "Checkpoint/input bytes changed during inference")
        require(sha(metadata_path) == job["checkpoint_metadata_sha256"], "Checkpoint metadata changed during inference")
        if spec.get("training_data_path"):
            require(sha(confined(root, spec["training_data_path"])) == spec["training_data_sha256"], "Frozen Train/Validation export changed")
        verify_source(root, source)
        require(verify_contract(root) == c and sha(__file__) == script_sha, "Contract/evaluator changed during inference")
        verify_permit(root, c, job, split, permit_path, permit_sha)
        require(time.time() < deadline, "Evaluation completed after job/host deadline")
        if split == "test":
            verify_global_validation_gate(root, c)
            require(sha(root / "validation_gate.json") == gate_sha, "Global Validation gate changed during Test")
        imports = verify_imports(source_root, source)
        runtime_after = record_executable(numerical.runtime_identity(execution_device))
        check_runtime(runtime_after, expected_runtime, device)
        require(runtime_after == runtime, 'Native Runtime identity changed during inference')
        elapsed = time.monotonic() - started
        receipt = {"status": "complete", "schema": KIND, "contract_sha256": c["contract_sha256"],
            **{key: job[key] for key in ("id", "cohort", "study", "dataset", "arm", "seed", "selected_epoch", "initial_state_sha256")},
            "job_id": job["id"], "frozen_source_closure": source["closure"],
            "split": split, "evaluation_scope": "validation_replay" if split == "validation" else "test_final_comparison",
            "permit_sha256": permit_sha, "native_runtime_checked": True,
            "physical_parameter_counts": parameter_counts,
            "checkpoint_sha256": job["checkpoint"]["sha256"], "state_sha256": before,
            "source_closure": source["closure"], "source_revision": source["revision"],
            "source_revision_history": job["source_revision_history"], "full_population": True,
            "expected_target_count": expected_count, "prediction_rows": processed, "metrics": final_metrics,
            "simple_quantity_baselines": simple_results, "original_population_identity": identity,
            "target_identity_sha256": target_digest.hexdigest(), "truth_sha256": truth_digest.hexdigest(), "time_sha256": time_digest.hexdigest(),
            "data_identity": {"raw_sha256": spec[sha_key], "path": spec[data_key]},
            "truth_digest_fields": TRUTH_FIELDS, "data_sha256": spec[sha_key], "loader": definition["loader"],
            "batch_size": 128, "shuffle": False, "num_workers": 0, "drop_last": False,
            "qualification": {**qualification, "parameters_unchanged": True, "finite_outputs": True},
            "validation_match": "PASS" if split == "validation" else None,
            "validation_replay_check": {"passed": split == "validation", "tolerance": {"relative": 1e-5, "absolute": 1e-5},
                "rule": "math.isclose", "count_exact": True, "six_metrics": split == "validation"},
            "retrained": False, "reselected": False, "test_independent": False,
            "held_out_test_evaluated": split == "test", "raw_predictions_written": False,
            "imported_frozen_modules": imports, "evaluator_sha256": script_sha,
            "runtime": runtime, "runtime_after": runtime_after,
            "pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv,
            "observed_unix": time.time(), "started_unix": started_unix, "completed_unix": time.time(),
            "test_started_unix": started_unix if split == "test" else None,
            "validation_gate_sha256": gate_sha, "elapsed_seconds": elapsed, "inference_seconds": inference_seconds,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0,
            "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
        write_exclusive(output / (name + ".json"), receipt)
        print(json.dumps({key: receipt[key] for key in ("status", "id", "split", "prediction_rows", "elapsed_seconds")}), flush=True)
        return 0
    except BaseException as error:
        write_exclusive(output / (name + "_failure.json"), {"status": "failed", "phase": phase,
            "id": job_id, "split": split, "error": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "contract_sha256": c["contract_sha256"],
            "elapsed_seconds": time.monotonic() - started, "receipt_written": False})
        raise



def qualify_synthetic(root):
    """The sealed RAF Q-only/K-only route on fabricated inputs, no checkpoint/data reads."""
    root = Path(root).resolve()
    c = verify_contract(root, require_bound=False)
    source = c['sources']['components']
    source_root = verify_source(root, source)
    os.environ['MPLCONFIGDIR'] = str(root / 'control' / 'synthetic_runtime_cache' / 'matplotlib')
    os.environ['XDG_CACHE_HOME'] = str(root / 'control' / 'synthetic_runtime_cache')
    isolate_source(source_root)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    engine = importlib.import_module(ROUTE)
    engine.install_hooks()
    core = importlib.import_module('paper.scripts.count_aware_tpp_backbone.core')
    factory = importlib.import_module('models.TPPs.CountAwareFactory')
    runner = importlib.import_module('simple_lab_test.search.common.runner')
    metrics_module = importlib.import_module('paper.scripts.evaluate_titantpp_history_width')
    import numpy as np
    verify_imports(source_root, source)
    outcomes = {}
    parameter_counts = {}
    for dataset, spec in c['datasets'].items():
        fixture = {'model': spec['definition']['model'],
                   'statistics': {'train_log_mean': 1.2, 'train_log_std': .8},
                   'loader': spec['definition']['loader']}
        parameter_counts[dataset] = {}
        for arm in ARMS:
            candidate, metadata = engine.build_model(fixture, arm)
            parameter_counts[dataset][arm] = {
                'actual_registered_parameters': sum(p.numel() for p in candidate.parameters()),
                'actual_active_parameters': sum(p.numel() for p in candidate.parameters() if p.requires_grad),
                'maximum_sequence_length': fixture['loader']['max_seq_len'],
                'legacy_declared_metadata_retained': metadata,
            }
            del candidate
    for arm in ARMS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            definition = engine.fabricated_data()
            model, metadata = engine.build_model(definition, arm)
            model.eval().requires_grad_(False)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.endswith(('causal_q_kernel', 'causal_k_kernel')):
                        parameter.copy_(torch.randn_like(parameter) * .03)
                verify_qk_local_structure(model)
                require(all(bool(parameter.abs().sum()) for name, parameter in model.named_parameters()
                    if name.endswith(('causal_q_kernel', 'causal_k_kernel'))), 'Q/K local paths were not exercised')
                state = {key: value.clone() for key, value in model.state_dict().items()}
                interface = {'time_head': model.time_head_contract()}
                payload = {'backbone': arm, 'encoder_config': metadata, 'model_state_dict': state,
                    'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False,
                    'variant': 'count_only_log_regression', 'interface_meta': interface,
                    'resume_identity': {'backbone': arm, 'interface_meta': interface,
                        'checkpoint_monitor': 'validation_raw_quantity_rmse',
                        'arguments': {'model_role': engine.role_for_arm(arm)}}}
                factory.validate_checkpoint_route(payload, arm)
                loaded, _ = engine.build_model(definition, arm)
                loaded.load_state_dict(state, strict=True)
                loaded.eval().requires_grad_(False)
                before = runner.canonical_state_dict_sha256(loaded.state_dict())
                dts = torch.tensor([[0., 0., 1., 2., 1., 3., 1., 2.], [1., 2., 1., 3., 1., 2., 1., 3.]])
                quantities = torch.tensor([[0., 0., 2., 4., 1., 3., 2., 5.], [1., 2., 3., 4., 3., 2., 1., 6.]])
                mask = torch.ones_like(dts, dtype=torch.bool)
                mask[0, :2] = False
                output = core.target_outputs(loaded, dts, mask, quantities, lambda_log_qty=1.)
                checks = qualify(torch, loaded, core.target_outputs, dts, mask, quantities, output, None)
                require(all(bool(torch.isfinite(value).all()) for value in output.values()), 'Nonfinite fabricated output')
                require(runner.canonical_state_dict_sha256(loaded.state_dict()) == before, 'Synthetic evaluation changed model state')
                for other in (*ARMS, "titantpp_cnn_gru54", "titantpp_cnn_gru54_cnn_only"):
                    if other == arm:
                        continue
                    rejected = False
                    try:
                        factory.validate_checkpoint_route(payload, other)
                    except ValueError:
                        rejected = True
                    require(rejected, 'Foreign component checkpoint route accepted')
                incomplete = dict(state)
                incomplete.pop(next(iter(incomplete)))
                rejected = False
                try:
                    loaded.load_state_dict(incomplete, strict=True)
                except (ValueError, RuntimeError):
                    rejected = True
                require(rejected, 'Incomplete checkpoint state accepted')
                outcomes[arm] = {'strict_state_loading': True, 'foreign_routes_rejected': True,
                    'qualification': {**checks, 'parameters_unchanged': True, 'finite_outputs': True}}
    metric_fixture = metrics_module.QuantityMetrics([0., 1., 2., 200.])
    metric_fixture.add(np.asarray([0., 200., 201.]), np.asarray([1., 198., 204.]), np.asarray([.1, .2, .3]))
    measured = metric_fixture.finish()
    expected = {'count': 3, 'qty_rmse': math.sqrt(14. / 3), 'qty_mae': 2., 'time_nll': .2,
                'tail': {'count': 1, 'qty_rmse': 3., 'qty_mae': 3., 'time_nll': .3}}
    compare_validation_metrics(measured, expected)
    verify_source(root, source)
    verify_imports(source_root, source)
    return {'schema': KIND, 'status': 'PASS', 'cohort': COHORT,
        'source_closure': SOURCE_CLOSURE, 'contract_sha256': c['contract_sha256'],
        'nonzero_qk_kernel_checked': True, 'original_value_kernel_absent': True, 'local_cnn_parameter_count':192, 'arms': outcomes, 'dataset_parameter_counts': parameter_counts,
        'six_metrics_hand_calculation_checked': True, 'tail_strict_boundary_checked': True,
        'real_data_loaded': False, 'held_out_test_evaluated': False,
        'checkpoint_reads': 0, 'optimizer_updates': 0, 'native_gpu_replay_verified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('check-contract', 'qualify-synthetic', 'worker'), required=True)
    parser.add_argument('--job')
    parser.add_argument('--split', choices=('validation', 'test'))
    parser.add_argument('--device', choices=('cuda',), default='cuda')
    parser.add_argument('--permit')
    parser.add_argument('--permit-sha')
    args = parser.parse_args(argv)
    if args.mode == 'check-contract':
        c = verify_contract(args.root, require_bound=False)
        print(json.dumps({'schema': KIND, 'status': 'STATIC_CONTRACT_CHECKED', 'job_count': 6,
            'contract_sha256': c['contract_sha256'], 'real_data_loaded': False,
            'execution_ready': False, 'binding_audit_performed': False}))
        return 0
    if args.mode == 'qualify-synthetic':
        print(json.dumps(qualify_synthetic(args.root), allow_nan=False))
        return 0
    require(args.job and args.split and args.permit and args.permit_sha,
            'Worker requires exact job/split and explicit hash-bound permit before data access')
    return worker(args.root, args.job, args.split, args.device,
                  permit_path=args.permit, permit_sha=args.permit_sha)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
