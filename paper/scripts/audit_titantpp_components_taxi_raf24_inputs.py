"""Native input fingerprints and selected-checkpoint metadata; prediction0."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

SCHEMA = 'titantpp_components_taxi_raf24_test_20261014_v1'
SOURCE = '15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53'
PARENT = 'e60badde6d9d902500aa67764f6834213ea1775797508f31d204ae67a5e63074'
DATASETS = ('yellow_trip_hourly', 'raf_spare_parts')
SEEDS = (42, 52, 62)
ARMS = ('titantpp_cnn_gru54_correction_disabled', 'titantpp_cnn_gru54_cnn_only',
        'titantpp_cnn_gru54_gru_only', 'titantpp_cnn_gru54')
DATA_FILES = {
    'yellow_trip_hourly': {'full': 'data/yellow_trip_hourly.parquet',
                         'training': 'data/yellow_trip_hourly.parquet',
                         'full_sha256': 'b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46',
                         'validation_count': 8268, 'test_count': 8327},
    'raf_spare_parts': {'full': 'data/raf_spare_parts.parquet',
                       'training': 'data/raf_spare_parts_train_validation.parquet',
                       'full_sha256': '439cc3bdd2f969f1121b149de5aa9ece4a7e1ce0c6da9e734a7687c54989780d',
                       'validation_count': 6690, 'test_count': 5226},
}
TRUTH_FIELDS = ('target_id', 'entity_id', 'seq', 'split', 'recorded_date',
                'raw_quantity', 'recorded_gap', 'history_length', 'last_quantity', 'mean_quantity')


def require(value, message):
    if not value:
        raise ValueError(message)


def jb(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def canon(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(jb(value) + b'\n')


def confined(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact relative path')
    path = Path(root) / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()), 'Artifact path escapes root')
    return path


def verify_imports(source_root, files):
    captured = {}
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] not in {'models', 'paper', 'data_loader', 'simple_lab_test', 'utils'}:
            continue
        path = getattr(module, '__file__', None)
        if path is None:
            paths = list(getattr(module, '__path__', ()))
            require(paths and all(Path(p).resolve().is_relative_to(source_root.resolve()) for p in paths),
                    'Unsealed namespace import: ' + name)
            continue
        require(Path(path).resolve().is_relative_to(source_root.resolve()), 'Unsealed project import: ' + name)
        relative = str(Path(path).resolve().relative_to(source_root.resolve()))
        require(relative in files and sha(path) == files[relative], 'Imported source SHA differs: ' + name)
        captured[name] = relative
    return captured


def fingerprints(loader, dataset_id, data_sha, metadata, np):
    dataset = loader.dataset
    offsets, total = [], 0
    for sequences in dataset.seq_lists:
        offsets.append(total)
        total += len(sequences)
    require(total == metadata.height, 'Loader/metadata mapping differs')
    split = loader.dataset.target_split if hasattr(loader.dataset, 'target_split') else None
    # The caller supplies the split explicitly through this attribute only for this audit.
    split = loader._audit_split
    target_digest, truth_digest = hashlib.sha256(), hashlib.sha256()
    time_digest = hashlib.sha256(b'titantpp_components_recorded_target_time_v1\0' + split.encode() + b'\0')
    original_time_digest = hashlib.sha256(b'frozen_lognormal_target_dt_v1\0')
    processed = 0
    for _, dts, mask, part_indices, quantities in loader:
        history_mask = mask.cpu().clone()
        history_mask[:, -1] = False
        lengths = history_mask.sum(dim=1)
        require(bool((lengths >= 1).all()), 'Missing causal history')
        q_cpu = quantities.cpu().double()
        q = q_cpu[:, -1].numpy().astype(np.float64)
        last = q_cpu[:, -2].numpy()
        mean = ((q_cpu * history_mask).sum(dim=1) / lengths).numpy()
        gaps = dts[:, -1].cpu().tolist()
        time_bytes = np.asarray(gaps, dtype='<f8').tobytes()
        time_digest.update(time_bytes)
        original_time_digest.update(time_bytes)
        for offset in range(len(q)):
            part, end = dataset.index[processed + offset]
            require(int(part_indices[offset]) == part, 'Loader order differs')
            raw = metadata.row(offsets[part] + end + 1, named=True)
            entity, sequence = str(dataset.parts[part]), int(dataset.seq_lists[part][end + 1])
            require(raw['chronological_split'] == split and str(raw['oper_part_no']) == entity
                    and int(raw['seq']) == sequence and raw['demand_dt'] is not None, 'Target metadata differs')
            date = str(raw['demand_dt'])
            target_id = hashlib.sha256(jb([dataset_id, data_sha, entity, sequence, split, date])).hexdigest()
            row = {'target_id': target_id, 'entity_id': entity, 'seq': sequence, 'split': split,
                   'recorded_date': date, 'raw_quantity': float(q[offset]), 'recorded_gap': float(gaps[offset]),
                   'history_length': int(lengths[offset]), 'last_quantity': float(last[offset]),
                   'mean_quantity': float(mean[offset])}
            target_digest.update((target_id + '\n').encode())
            truth_digest.update(jb([row[key] for key in TRUTH_FIELDS]) + b'\n')
        processed += len(q)
    require(processed == len(dataset), 'Population incomplete')
    return {'count': processed, 'target_identity_sha256': target_digest.hexdigest(),
            'truth_sha256': truth_digest.hexdigest(), 'time_sha256': time_digest.hexdigest(),
            'original_recorded_duration_sha256': original_time_digest.hexdigest()}


def validate_job_scope(training, approval):
    expected = {f'{dataset}__{seed}__{arm}' for dataset in DATASETS for seed in SEEDS for arm in ARMS}
    selected = sorted(j['id'] for j in training['jobs'] if j['host'] == '5080' and j['dataset'] in DATASETS)
    require(set(selected) == expected and len(selected) == 24 and selected == approval['job_ids'],
            'Exact Taxi/RAF24 job scope differs')
    return selected


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--approval-sha', required=True)
    parser.add_argument('--originals-manifest-sha', required=True)
    parser.add_argument('--audit-attempt', choices=('initial',), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit_dir = root / 'audit' / args.audit_attempt
    require(root == Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts') / (SCHEMA + '_5080'),
            'Foreign audit root')
    require(sha(root / 'partial_approval.json') == args.approval_sha, 'Approval SHA mismatch')
    approval = read(root / 'partial_approval.json')
    require(approval['approved'] is True and approval['partial_cohort_explicitly_authorized'] is True
            and approval['host'] == '5080' and approval['authorization_root'] == str(root)
            and approval['evaluation_scope'] == 'test_taxi_raf24_followup'
            and approval['cohort'] == 'components_taxi_raf24_source131'
            and approval['parent_contract_sha256'] == PARENT and approval['frozen_source_closure'] == SOURCE
            and approval['automatic_retry'] is False and approval['checkpoint_reselection'] is False
            and approval['new_training'] is False and approval['expected_condition_count'] == 24,
            'Partial input audit not authorized')
    require(hashlib.sha256(approval['user_instruction'].encode()).hexdigest() == approval['user_instruction_sha256'],
            'User instruction changed')
    c = read(root / 'provenance/training_contract.json')
    require(canon(c) == PARENT and c['source']['files_sha256'] == SOURCE, 'Training source changed')
    definitions = {d['dataset_id']: d for d in c['datasets'] if d['dataset_id'] in DATASETS}
    selected = validate_job_scope(c, approval)
    source_root = confined(root, 'source')
    for name, digest in c['source']['files'].items():
        require(sha(confined(source_root, name)) == digest, 'Source SHA mismatch: ' + name)
    sentinel = 'sample_data/.frozen_root_sentinel'
    require(sha(confined(source_root, sentinel)) == c['input_files'][sentinel], 'Frozen root sentinel differs')
    require(sha(root / 'originals_manifest.json') == args.originals_manifest_sha, 'Original manifest SHA mismatch')
    manifests = read(root / 'originals_manifest.json')['files']
    require(isinstance(manifests, dict) and len(manifests) == 360, 'Exactly360 immutable original files required')
    expected_originals = {}
    for job in (j for j in c['jobs'] if j['id'] in selected):
        prefix = 'originals/run/' + job['id'] + '/'
        terminal_name = prefix + 'terminal_manifest.json'
        worker_name = 'originals/worker_exits/' + job['id'] + '.json'
        require(terminal_name in manifests and worker_name in manifests, 'Original terminal/worker exit missing')
        terminal = read(confined(root, terminal_name))
        require(terminal['job'] == job and terminal['contract_sha256'] == PARENT
                and terminal['scientific_success'] is True and terminal['held_out_test_evaluated'] is False
                and terminal['status'] == 'complete' and terminal['evaluation_scope'] == 'validation_only',
                'Original terminal scientific identity differs')
        worker_exit = read(confined(root, worker_name))
        require(worker_exit['job'] == job and worker_exit['contract_sha256'] == PARENT
                and worker_exit['actual_child_exit_observed'] is True
                and type(worker_exit['returncode']) is int and worker_exit['returncode'] == 0,
                'Original actual scientific worker exit0 differs')
        expected_originals[terminal_name] = manifests[terminal_name]
        expected_originals[worker_name] = manifests[worker_name]
        for name, digest in terminal['files'].items():
            expected_originals[prefix + name] = digest
        selected_name = prefix + 'runs/' + job['arm'] + '/count_only_log_regression/seed_' + str(job['seed']) + '/best_val_qty_rmse_model.pt'
        require(selected_name in expected_originals, 'Selected checkpoint not bound by terminal')
    require(expected_originals == manifests, 'Original allowlist/terminal file SHA changed')
    for name, digest in manifests.items():
        require(sha(confined(root, name)) == digest, 'Original file changed: ' + name)
    data_bindings = {}
    for dataset_id, names in DATA_FILES.items():
        definition = definitions[dataset_id]
        full, training = confined(root, names['full']), confined(root, names['training'])
        require(sha(full) == names['full_sha256'], 'Frozen full data byte SHA changed: ' + dataset_id)
        require(sha(training) == definition['data_sha256'], 'Frozen training data byte SHA changed: ' + dataset_id)
        if dataset_id == 'raf_spare_parts':
            require(definition['parent_data_identity']['data']['sha256'] == names['full_sha256'],
                    'RAF original full parent differs')
        data_bindings[dataset_id] = {'data_path': names['full'], 'data_sha256': names['full_sha256'],
                                    'training_data_path': names['training'],
                                    'training_data_sha256': definition['data_sha256']}
    gpu = c['hosts']['5080']['gpu_uuid']
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True).stdout.splitlines()
    require(inventory.count(gpu) == 1, 'Fixed GPU missing')
    compute = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True).stdout
    require(not [row for row in compute.splitlines() if row.split(',')[0].strip() == gpu], 'Native GPU busy; no audit launch')
    write(audit_dir / 'input_claim.json', {'pid': os.getpid(), 'ppid': os.getppid(), 'argv': sys.argv,
                                         'started_unix': time.time(), 'new_training': False,
                                         'model_predictions': 0, 'automatic_retry': False})
    spec = c['hosts']['5080']
    require(Path(sys.executable).resolve() == Path(spec['python']).resolve(), 'Native executable differs before imports')
    require(not any(name.split('.')[0] in {'models', 'paper', 'data_loader', 'simple_lab_test', 'utils'} for name in sys.modules),
            'Audit requires a fresh project-module process')
    sys.dont_write_bytecode = True
    os.environ['MPLCONFIGDIR'] = str(audit_dir / 'runtime_cache/matplotlib')
    os.environ['XDG_CACHE_HOME'] = str(audit_dir / 'runtime_cache')
    sys.path[:] = [str(source_root)] + [p for p in sys.path if p and not (Path(p).resolve() / 'paper').is_dir()]
    import numpy as np
    import polars as pl
    import torch
    numerical = importlib.import_module('paper.scripts.quantity_comparison_runtime')
    numerical.configure_runtime('cuda:0', 1)
    observed = numerical.runtime_identity('cuda:0')
    observed.update(python_executable=str(Path(sys.executable).absolute()),
                    resolved_python_executable=str(Path(sys.executable).resolve()))
    require(Path(sys.executable).resolve() == Path(spec['python']).resolve(), 'Native executable differs')
    require(observed['gpu']['uuid'] == gpu and all(observed[k] == v for k, v in spec['runtime_expected'].items()),
            'Native Runtime fingerprint differs')
    runtime = {'host': '5080', 'python': spec['python'], 'python_version': observed['python'],
               **{key: observed[key] for key in ('torch', 'cuda', 'cudnn', 'numpy', 'polars')},
               'gpu_uuid': gpu, 'torch_num_threads': 1, 'interop_threads': 1, 'dtype': 'float32',
               'deterministic_algorithms': True, 'TF32': False,
               'resolved_python_executable': observed['resolved_python_executable']}
    runner = importlib.import_module('simple_lab_test.search.common.runner')
    verify_imports(source_root, c['source']['files'])
    metadata_refs = {}
    for job in (j for j in c['jobs'] if j['id'] in selected):
        rel = 'originals/run/' + job['id'] + '/runs/' + job['arm'] + '/count_only_log_regression/seed_' + str(job['seed']) + '/best_val_qty_rmse_model.pt'
        payload = runner.torch_load_checkpoint(root / rel, map_location='cpu')
        tensor_sha = runner.canonical_state_dict_sha256(payload['model_state_dict'])
        require(tensor_sha == payload['model_state_sha256'], 'Selected actual tensor SHA differs')
        metadata = {key: value for key, value in payload.items() if key != 'model_state_dict'}
        mpath = root / 'metadata' / (job['id'] + '.json')
        write(mpath, metadata)
        metadata_refs[job['id']] = {'metadata_path': str(mpath.relative_to(root)), 'metadata_sha256': sha(mpath),
                                   'checkpoint_state_sha256': tensor_sha, 'selected_epoch': payload['best_epoch'],
                                   'selected_metric_value': payload['selected_metric_value'],
                                   'initial_state_sha256': payload['initial_state_sha256']}
        del payload
    core = importlib.import_module('paper.scripts.count_aware_tpp_backbone.core')
    loader_module = importlib.import_module('paper.scripts.run_taxi_quantity_interface_ablation')
    verify_imports(source_root, c['source']['files'])
    populations, train_initialization_audit = {}, {}
    for dataset_id, names in DATA_FILES.items():
        definition = definitions[dataset_id]
        full_path, training_path = confined(root, names['full']), confined(root, names['training'])
        full = pl.read_parquet(full_path).sort(['oper_part_no', 'seq'])
        training = pl.read_parquet(training_path).sort(['oper_part_no', 'seq'])
        if dataset_id == 'raf_spare_parts':
            require_raf_training_slice(full, training, pl)
        populations[dataset_id] = {}
        train_frame = core.prepare_count_frame(training.filter(pl.col('chronological_split') == 'train'))
        train_loader = loader_module.make_loader(train_frame, target_split='train', batch_size=128, shuffle=False,
            generator=None, lookback_weeks=definition['loader']['lookback_weeks'],
            max_seq_len=definition['loader']['max_seq_len'])
        train_loader._audit_split = 'train'
        train_metadata = train_frame.select(['oper_part_no', 'seq', 'chronological_split', 'demand_dt'])
        train_fingerprints = fingerprints(train_loader, dataset_id, names['full_sha256'], train_metadata, np)
        train_identity = population_identity(train_loader.dataset, 'train', np)
        require(train_identity == definition['inherited_data_identity']['populations']['train'],
                'Frozen Train target identity differs: ' + dataset_id)
        require(train_fingerprints['original_recorded_duration_sha256']
                == definition['time_statistics']['expected_train_target_dt_sha256'],
                'Frozen Train recorded duration identity differs: ' + dataset_id)
        train_initialization_audit[dataset_id] = {**train_fingerprints, 'training_population_identity': train_identity,
                                                'statistics_replaced': False}
        for split in ('validation', 'test'):
            frame = (training.filter(pl.col('chronological_split').is_in(['train', 'validation']))
                     if split == 'validation' else full)
            validate_frame(frame, definition, split, np, pl)
            train_q = frame.filter(pl.col('chronological_split') == 'train')['demand_qty'].to_numpy().astype(np.float64)
            stats = definition['statistics']
            require(hashlib.sha256(b'quantity_comparison_all_train_row_quantity_v1\0' + train_q.astype('<f8').tobytes()).hexdigest()
                    == stats['all_train_quantity_sha256'], 'Train quantity fingerprint differs: ' + dataset_id)
            for key, value in (('train_log_mean', float(np.log1p(train_q).mean())),
                               ('train_log_std', float(np.log1p(train_q).std()))):
                require(math.isclose(value, stats[key], rel_tol=0, abs_tol=1e-12), 'Train statistics differ: ' + dataset_id)
            prepared = core.prepare_count_frame(frame)
            loader = loader_module.make_loader(prepared, target_split=split, batch_size=128, shuffle=False, generator=None,
                lookback_weeks=definition['loader']['lookback_weeks'], max_seq_len=definition['loader']['max_seq_len'])
            loader._audit_split = split
            metadata = prepared.select(['oper_part_no', 'seq', 'chronological_split', 'demand_dt'])
            fingerprints_value = fingerprints(loader, dataset_id, names['full_sha256'], metadata, np)
            identity = population_identity(loader.dataset, split, np)
            require(fingerprints_value['count'] == names[split + '_count'], 'Frozen target count differs: ' + dataset_id)
            if split == 'validation':
                require(identity == definition['inherited_data_identity']['populations']['validation'],
                        'Frozen Validation target identity differs: ' + dataset_id)
                require(fingerprints_value['original_recorded_duration_sha256'] == definition['time_statistics']['expected_validation_target_dt_sha256'],
                        'Frozen Validation recorded time identity differs: ' + dataset_id)
            populations[dataset_id][split] = {**fingerprints_value, 'training_population_identity': identity}
        require(sha(full_path) == names['full_sha256'] and sha(training_path) == definition['data_sha256'],
                'Input changed during audit: ' + dataset_id)
    for name, digest in manifests.items():
        require(sha(confined(root, name)) == digest, 'Original changed during audit')
    for name, digest in c['source']['files'].items():
        require(sha(confined(source_root, name)) == digest, 'Source changed during audit')
    require(sha(confined(source_root, sentinel)) == c['input_files'][sentinel], 'Frozen root sentinel changed during audit')
    runtime_after = numerical.runtime_identity('cuda:0')
    runtime_after.update(python_executable=str(Path(sys.executable).absolute()),
                         resolved_python_executable=str(Path(sys.executable).resolve()))
    require(runtime_after == observed, 'Runtime changed during input audit')
    imports = verify_imports(source_root, c['source']['files'])
    receipt = {'status': 'complete', 'model_predictions': 0, 'new_training': False, 'populations': populations,
               'metadata_refs': metadata_refs, 'runtime': runtime, 'runtime_observed': observed,
               'runtime_after': runtime_after, 'imported_frozen_modules': imports,
               'source131_closure': SOURCE, 'data_sha256': {k: v['data_sha256'] for k, v in data_bindings.items()},
               'data_bindings': data_bindings, 'train_initialization_audit': train_initialization_audit,
               'RAF_TrainValidation_parent_slice_exact': True,
               'checkpoint_count': 24, 'approval_sha256': args.approval_sha,
               'completed_unix': time.time()}
    write(audit_dir / 'input_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
