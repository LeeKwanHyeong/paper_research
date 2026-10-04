"""One immutable checkpoint, one complete frozen-loader population, one process.

Test requires an explicit identity-bound approval contract (see validate_contract).
Only the CLI worker imports frozen models. No training or checkpoint writes occur.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH_SIZE = 128
PART_ROWS = 8192
TRUTH_FIELDS = (
    'target_index', 'target_id', 'entity_id', 'seq', 'split', 'recorded_date',
    'time_bucket', 'site_id', 'raw_quantity', 'recorded_gap', 'history_length',
    'last_quantity', 'mean_quantity',
)
SCHEMA = {
    'target_index': 'Int64', 'target_id': 'String', 'entity_id': 'String',
    'seq': 'Int64', 'split': 'String', 'recorded_date': 'String',
    'time_bucket': 'String', 'site_id': 'String', 'raw_quantity': 'Float64',
    'predicted_raw_quantity': 'Float64', 'recorded_gap': 'Float64',
    'time_nll': 'Float64', 'history_length': 'Int64',
    'last_quantity': 'Float64', 'mean_quantity': 'Float64',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def resolve(root, path):
    return (root / path).resolve()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def select_condition(registry, dataset, model, seed):
    rows = [r for r in registry['rows'] if
            (r['dataset'], r['model'], r['seed']) == (dataset, model, seed)]
    require(len(rows) == 1, 'Condition must bind exactly one registry row')
    row = rows[0]
    bundle = registry['bundles'][row['evaluator_source_bundle']]
    specs = [d for d in bundle['datasets'] if d['dataset_id'] == dataset]
    require(len(specs) == 1, 'Condition must bind exactly one dataset specification')
    require(row['endpoint'] == 'selected', 'Only selected frozen checkpoints are supported')
    return row, bundle, specs[0]


def validate_contract(contract, *, split, dataset, model, seed, registry_sha,
                      manifest_sha, data_sha):
    """Approval and all three file identities must agree, before dataset access.

    {schema_version:1, approval:{test_inference_authorized:true,
    retraining_authorized:false, allowed_splits:[validation,test], datasets:[...],
    models:[...], seeds:[...]}, registry_sha256:..., dataset_manifest_sha256:...,
    dataset_sha256:{dataset:sha,...}}. Validation can omit the contract entirely.
    """
    if contract is None:
        require(split == 'validation', 'Test inference requires --contract')
        return
    require(contract.get('schema_version') == 1, 'Unsupported contract schema')
    approval = contract.get('approval', {})
    require(approval.get('retraining_authorized') is False, 'Contract must forbid retraining')
    if split == 'test':
        require(approval.get('test_inference_authorized') is True,
                'Contract does not authorize test inference')
    for field, item in (('allowed_splits', split), ('datasets', dataset),
                        ('models', model), ('seeds', seed)):
        require(item in approval.get(field, []), f'Condition outside approved {field}')
    require(contract.get('registry_sha256') == registry_sha, 'Registry contract SHA mismatch')
    require(contract.get('dataset_manifest_sha256') == manifest_sha,
            'Dataset manifest contract SHA mismatch')
    require(contract.get('dataset_sha256', {}).get(dataset) == data_sha,
            'Dataset contract SHA mismatch')


def validate_inputs(args):
    root = Path(args.root).resolve()
    registry_path = resolve(root, args.registry)
    manifest_path = resolve(root, args.dataset_manifest)
    registry, manifest = read_json(registry_path), read_json(manifest_path)
    row, bundle, spec = select_condition(registry, args.dataset, args.model, args.seed)
    datasets = [d for d in manifest['datasets'] if d['dataset'] == args.dataset]
    require(len(datasets) == 1, 'Dataset manifest must contain one matching identity')
    data = datasets[0]
    require(isinstance(data['sha256'], str) and len(data['sha256']) == 64,
            'Invalid dataset SHA')
    contract_path = resolve(root, args.contract) if args.contract else None
    registry_sha, manifest_sha = sha256_file(registry_path), sha256_file(manifest_path)
    validate_contract(read_json(contract_path) if contract_path else None,
                      split=args.split, dataset=args.dataset, model=args.model,
                      seed=args.seed, registry_sha=registry_sha,
                      manifest_sha=manifest_sha, data_sha=data['sha256'])
    bound_data_hashes = {spec.get(key, {}).get('data', {}).get('sha256')
                        for key in ('inherited_data_identity', 'parent_data_identity')}
    require(data['sha256'] in bound_data_hashes,
            'Dataset bytes are not a frozen inherited or parent dataset')
    require(spec['loader']['batch_size'] == BATCH_SIZE, 'Frozen batch size must be 128')
    return root, row, bundle, spec, data, {
        'registry_sha256': registry_sha, 'dataset_manifest_sha256': manifest_sha,
        'contract_sha256': sha256_file(contract_path) if contract_path else None,
    }


def target_metadata(dataset, data_sha, index, entity, seq, split, row):
    require(row['chronological_split'] == split, 'Wrong target split')
    require(row.get('demand_dt') is not None, 'Recorded target date is missing')
    require(str(row['oper_part_no']) == entity and int(row['seq']) == seq,
            'Frozen loader index does not match metadata row')
    date = str(row['demand_dt'])
    identity = [dataset, data_sha, entity, seq, split, date]
    return {
        'target_index': index, 'target_id': hashlib.sha256(json_bytes(identity)).hexdigest(),
        'entity_id': entity, 'seq': seq, 'split': split, 'recorded_date': date,
        'time_bucket': str(row['time_bucket']) if row.get('time_bucket') is not None else None,
        'site_id': entity.split('::', 1)[0] if dataset == 'intermittent_frozen_5000' else None,
    }


def row_digests(rows):
    targets, truth = hashlib.sha256(), hashlib.sha256()
    for row in rows:
        targets.update((row['target_id'] + '\n').encode('utf-8'))
        truth.update(json_bytes([row[k] for k in TRUTH_FIELDS]) + b'\n')
    return targets.hexdigest(), truth.hexdigest()


class PredictionWriter:
    """Bounded row buffer; each committed parquet has checked schema and hashes."""
    def __init__(self, output, part_rows=PART_ROWS):
        self.output = Path(output)
        (self.output / 'predictions').mkdir()
        self.part_rows = part_rows
        self.buffer = []
        self.parts = []
        self.rows = 0
        self.targets = hashlib.sha256()
        self.truth = hashlib.sha256()

    def add(self, rows):
        for row in rows:
            require(set(row) == set(SCHEMA), 'Prediction row schema mismatch')
            require(row['target_index'] == self.rows, 'Target indices must be contiguous')
            require(row['history_length'] >= 1, 'Missing causal history')
            for key, dtype in SCHEMA.items():
                if dtype == 'Float64':
                    require(math.isfinite(row[key]), f'Nonfinite {key}')
            require(all(row[key] >= 0 for key in (
                'raw_quantity', 'predicted_raw_quantity', 'last_quantity', 'mean_quantity')),
                'Quantity must be nonnegative')
            self.targets.update((row['target_id'] + '\n').encode('utf-8'))
            self.truth.update(json_bytes([row[k] for k in TRUTH_FIELDS]) + b'\n')
            self.buffer.append(row)
            self.rows += 1
            if len(self.buffer) == self.part_rows:
                self.flush()

    def flush(self):
        if not self.buffer:
            return
        import polars as pl
        relative = f'predictions/part-{len(self.parts):05d}.parquet'
        path = self.output / relative
        require(not path.exists(), 'Refusing to overwrite prediction part')
        frame = pl.DataFrame(self.buffer, schema={k: getattr(pl, v) for k, v in SCHEMA.items()})
        temporary = path.with_suffix('.parquet.tmp')
        frame.write_parquet(temporary, compression='zstd', statistics=True)
        loaded = pl.read_parquet(temporary)
        require(loaded.equals(frame), 'Parquet roundtrip altered prediction rows')
        require({k: str(v) for k, v in loaded.schema.items()} == SCHEMA,
                'Parquet schema mismatch')
        identities, truth = row_digests(self.buffer)
        os.replace(temporary, path)
        self.parts.append({
            'path': relative, 'sha256': sha256_file(path), 'rows': loaded.height,
            'target_index_start': self.buffer[0]['target_index'],
            'target_index_end': self.buffer[-1]['target_index'],
            'target_identity_sha256': identities, 'truth_sha256': truth, 'schema': SCHEMA,
        })
        self.buffer.clear()
        atomic_json(self.output / 'progress.json', {
            'status': 'running', 'prediction_rows': self.rows, 'parts': self.parts,
            'target_identity_sha256': self.targets.hexdigest(),
            'truth_sha256': self.truth.hexdigest(),
        })


def verify_source(root, bundle):
    source = resolve(root, bundle['source_root'])
    for relative, expected in bundle['source_files'].items():
        path = (source / relative).resolve()
        require(path.is_relative_to(source), 'Source manifest escapes source directory')
        require(sha256_file(path) == expected, f'Frozen source changed: {relative}')
    return source


def verify_imports(source, bundle):
    imported = {}
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] not in {'models', 'paper', 'data_loader', 'simple_lab_test'}:
            continue
        file = getattr(module, '__file__', None)
        if not file:
            continue
        path = Path(file).resolve()
        require(path.is_relative_to(source), f'Imported current/nonfrozen source: {name}')
        relative = path.relative_to(source).as_posix()
        require(relative in bundle['source_files'], f'Imported unhashed source: {relative}')
        imported[name] = relative
    return imported


def qualify(torch, model, target_outputs, dts, mask, quantities, value, top_code):
    altered_qty = quantities.clone()
    altered_qty[:, -1] += 12345.
    qtest = target_outputs(model, dts, mask, altered_qty, lambda_log_qty=1.)
    altered_dt = dts.clone()
    altered_dt[:, -1] = (torch.where(dts[:, -1] > 1, torch.ones_like(dts[:, -1]),
                                    torch.full_like(dts[:, -1], 2.))
                         if top_code else dts[:, -1] + 123.)
    ttest = target_outputs(model, altered_dt, mask, quantities, lambda_log_qty=1.)
    single = target_outputs(model, dts[:1], mask[:1], quantities[:1], lambda_log_qty=1.)
    checks = {
        'target_quantity_perturbation_preserves_prediction': torch.allclose(value['pred_qty'], qtest['pred_qty'], rtol=1e-6, atol=1e-6),
        'target_quantity_perturbation_preserves_time_loss': torch.allclose(value['time_loss'], qtest['time_loss'], rtol=1e-6, atol=1e-6),
        'target_gap_perturbation_preserves_quantity_prediction': torch.allclose(value['pred_qty'], ttest['pred_qty'], rtol=1e-6, atol=1e-6),
        'single_vs_batch_prediction_matches': torch.allclose(value['pred_qty'][:1], single['pred_qty'], rtol=1e-5, atol=1e-5),
        'single_vs_batch_time_loss_matches': torch.allclose(value['time_loss'][:1], single['time_loss'], rtol=1e-5, atol=1e-5),
    }
    require(all(checks.values()), f'Qualification failed: {checks}')
    return checks


def run_worker(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    writer = None
    phase = 'preflight'
    runner_sha = sha256_file(Path(__file__))
    def deadline():
        if args.deadline_seconds is not None and time.monotonic() - started >= args.deadline_seconds:
            raise TimeoutError(f'Deadline exceeded during {phase}')
    try:
        root, row, bundle, spec, data, identities = validate_inputs(args)
        source = verify_source(root, bundle)
        checkpoint = resolve(root, row['checkpoint_path'])
        data_path = resolve(root, data['path'])
        require(sha256_file(checkpoint) == row['checkpoint_file_sha256'], 'Checkpoint file SHA mismatch')
        require(sha256_file(data_path) == data['sha256'], 'Dataset file SHA mismatch')
        deadline()
        # Eliminate current checkout import fallback; inspect all imported project origins.
        sys.dont_write_bytecode = True
        sys.path[:] = [str(source)] + [p for p in sys.path if p and not Path(p).resolve().is_relative_to(root)]
        cache = output / 'runtime_cache'
        os.environ['MPLCONFIGDIR'] = str(cache / 'matplotlib')
        os.environ['XDG_CACHE_HOME'] = str(cache)
        if args.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import numpy as np
        import polars as pl
        import torch
        runtime_module = importlib.import_module('paper.scripts.quantity_comparison_runtime')
        runtime_device = 'cuda:0' if args.device == 'cuda' else 'cpu'
        runtime_module.configure_runtime(runtime_device, 4)
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        factory = importlib.import_module('models.TPPs.CountAwareFactory')
        core = importlib.import_module('paper.scripts.count_aware_tpp_backbone.core')
        loader_module = importlib.import_module('paper.scripts.run_taxi_quantity_interface_ablation')
        runner = importlib.import_module('simple_lab_test.search.common.runner')
        imported = verify_imports(source, bundle)
        phase = 'loading_population'
        lazy = pl.scan_parquet(data_path)
        if args.split == 'validation':
            lazy = lazy.filter(pl.col('chronological_split').is_in(['train', 'validation']))
        frame = lazy.collect().sort(['oper_part_no', 'seq'])
        require(frame.height > 0, 'Empty dataset')
        require(not frame.select(pl.struct(['oper_part_no', 'seq']).is_duplicated().any()).item(),
                'Duplicate entity/sequence key')
        require(set(frame['chronological_split'].unique()) <= {'train', 'validation', 'test'},
                'Unexpected split in frozen dataset')
        frame = core.prepare_count_frame(frame)
        loader = loader_module.make_loader(
            frame, target_split=args.split, batch_size=BATCH_SIZE, shuffle=False, generator=None,
            lookback_weeks=spec['loader']['lookback_weeks'], max_seq_len=spec['loader']['max_seq_len'])
        dataset = loader.dataset
        expected = len(dataset)
        require(expected > 0, 'No eligible targets')
        if 'populations' in data:
            require(expected == data['populations'][args.split]['target_count'],
                    'Population differs from frozen dataset manifest')
        if args.split == 'validation':
            original_count = row.get('validation_count')
            if original_count is None:
                original_count = spec['inherited_data_identity']['populations']['validation']['target_count']
            require(expected == original_count, 'Validation population differs from frozen registry')
        columns = ['oper_part_no', 'seq', 'chronological_split', 'demand_dt']
        if 'time_bucket' in frame.columns:
            columns.append('time_bucket')
        metadata = frame.select(columns)
        offsets, total = [], 0
        for seqs in dataset.seq_lists:
            offsets.append(total)
            total += len(seqs)
        require(total == frame.height, 'Loader population differs from frame')
        del frame
        deadline()
        if args.inventory_only:
            digest = hashlib.sha256()
            for index, (pi, end) in enumerate(dataset.index):
                entity, seq = str(dataset.parts[pi]), int(dataset.seq_lists[pi][end + 1])
                raw = metadata.row(offsets[pi] + end + 1, named=True)
                meta = target_metadata(args.dataset, data['sha256'], index, entity, seq, args.split, raw)
                digest.update((meta['target_id'] + '\n').encode('utf-8'))
                if index % PART_ROWS == 0:
                    deadline()
            require(sha256_file(data_path) == data['sha256'], 'Dataset changed during inventory')
            atomic_json(output / 'inventory.json', {
                'status': 'complete', 'dataset': args.dataset, 'split': args.split,
                'data_file_sha256': data['sha256'], 'expected_target_count': expected,
                'target_identity_sha256': digest.hexdigest(), 'loader': spec['loader'],
                'inference_calls': 0, **identities, 'elapsed_seconds': time.monotonic() - started,
            })
            return 0
        cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(cp.get('best_epoch') == row['selected_epoch'], 'Selected epoch mismatch')
        require(cp.get('backbone') == args.model, 'Checkpoint backbone route mismatch')
        factory.validate_checkpoint_route(cp, args.model)
        require(runner.canonical_state_dict_sha256(cp['model_state_dict']) == row['state_tensor_sha256'],
                'Checkpoint tensor SHA mismatch')
        config = {k: v for k, v in spec['model'].items()
                  if k not in ('backbone', 'lambda_log_qty', 'lambda_tail', 'time_head_lr_multiplier')}
        if args.model == 'deep_renewal_event_native_nb':
            # Exact native configuration from frozen run_pakdd_extension.build_model.
            config.update(quantity_variant='shifted_nb_quantity_nll',
                          time_head_mode='shifted_nb_duration')
        model, _ = factory.build_count_aware_model(
            args.model, **config, train_log_mean=spec['statistics']['train_log_mean'],
            train_log_std=spec['statistics']['train_log_std'], max_seq_len=spec['loader']['max_seq_len'])
        model.load_state_dict(cp['model_state_dict'], strict=True)
        model.to(args.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before = runner.canonical_state_dict_sha256(model.state_dict())
        require(before == row['state_tensor_sha256'], 'Loaded model tensor SHA mismatch')
        del cp
        writer = PredictionWriter(output)
        phase = 'inference'
        inference_seconds = 0.
        qualification = {}
        processed = 0
        with torch.no_grad():
            for batch_index, (_, dts, mask, part_indices, quantities) in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                deadline()
                dts, mask, quantities = (t.to(args.device) for t in (dts, mask, quantities))
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                batch_start = time.monotonic()
                value = core.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                inference_seconds += time.monotonic() - batch_start
                if batch_index == 0:
                    phase = 'qualification'
                    qualification = qualify(torch, model, core.target_outputs, dts, mask, quantities,
                                            value, spec['model']['time_observation_contract'].get('top_code'))
                    phase = 'inference'
                for key in ('true_qty', 'pred_qty', 'time_loss'):
                    require(bool(torch.isfinite(value[key]).all()), f'Nonfinite {key}')
                # Baselines use the identical frozen, truncated history, excluding target.
                quantities_cpu = quantities.cpu().double()
                history_mask = mask.cpu().clone()
                history_mask[:, -1] = False
                history_lengths = history_mask.sum(dim=1)
                require(bool((history_lengths >= 1).all()), 'Missing history')
                last = quantities_cpu[:, -2].tolist()
                means = ((quantities_cpu * history_mask).sum(dim=1) / history_lengths).tolist()
                vals = {key: value[key].cpu().tolist() for key in ('true_qty', 'pred_qty', 'time_loss', 'history_length')}
                gaps = dts[:, -1].cpu().tolist()
                records = []
                for i in range(len(gaps)):
                    index = processed + i
                    pi, end = dataset.index[index]
                    require(int(part_indices[i]) == pi, 'Loader batch order mismatch')
                    require(int(history_lengths[i]) == vals['history_length'][i], 'History length mismatch')
                    entity, seq = str(dataset.parts[pi]), int(dataset.seq_lists[pi][end + 1])
                    raw = metadata.row(offsets[pi] + end + 1, named=True)
                    meta = target_metadata(args.dataset, data['sha256'], index, entity, seq, args.split, raw)
                    records.append({**meta, 'raw_quantity': float(vals['true_qty'][i]),
                        'predicted_raw_quantity': float(vals['pred_qty'][i]), 'recorded_gap': float(gaps[i]),
                        'time_nll': float(vals['time_loss'][i]), 'history_length': int(vals['history_length'][i]),
                        'last_quantity': float(last[i]), 'mean_quantity': float(means[i])})
                writer.add(records)
                processed += len(records)
                deadline()
        phase = 'final_integrity'
        writer.flush()
        require(processed == min(expected, args.max_batches * BATCH_SIZE) if args.max_batches else processed == expected,
                'Incomplete or extra target rows')
        require(runner.canonical_state_dict_sha256(model.state_dict()) == before, 'Model state changed during evaluation')
        require(sha256_file(checkpoint) == row['checkpoint_file_sha256'], 'Checkpoint bytes changed during evaluation')
        require(sha256_file(data_path) == data['sha256'], 'Dataset bytes changed during evaluation')
        verify_source(root, bundle)
        imported = verify_imports(source, bundle)
        require(validate_inputs(args)[-1] == identities, 'Input identities changed during evaluation')
        require(sha256_file(Path(__file__)) == runner_sha, 'Evaluator source changed during evaluation')
        deadline()
        elapsed = time.monotonic() - started
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt = {
            'schema_version': 1, 'status': 'complete', 'dataset': args.dataset,
            'model': args.model, 'seed': args.seed, 'split': args.split,
            'evaluation_scope': 'validation_qualification' if args.max_batches else 'full_population',
            'full_population': args.max_batches is None, 'heldout_read': args.split == 'test',
            'retrained': False, 'max_batches': args.max_batches, 'batch_size': BATCH_SIZE,
            'expected_target_count': expected, 'prediction_rows': processed,
            'data_file_sha256': data['sha256'], 'data_path': str(data_path),
            **identities, 'checkpoint_path': row['checkpoint_path'],
            'checkpoint_file_sha256': row['checkpoint_file_sha256'],
            'state_tensor_sha256': before, 'selected_epoch': row['selected_epoch'],
            'evaluator_source_bundle': row['evaluator_source_bundle'],
            'source_closure_sha256': bundle['source_closure_sha256'],
            'imported_frozen_modules': imported, 'loader': spec['loader'],
            'runner_sha256': runner_sha, 'schema': SCHEMA,
            'target_id_contract': 'sha256(compact UTF-8 JSON [dataset,data_file_sha256,entity_id,seq,split,recorded_date])',
            'target_identity_sha256': writer.targets.hexdigest(),
            'truth_sha256': writer.truth.hexdigest(), 'truth_digest_fields': TRUTH_FIELDS,
            'parts': writer.parts, 'qualification': {**qualification, 'parameters_unchanged': True, 'finite_outputs': True},
            'runtime': {**runtime_module.runtime_identity(runtime_device), 'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'polars': pl.__version__, 'device': args.device,
                        'cuda_device': torch.cuda.get_device_name() if args.device == 'cuda' else None,
                        'cpu_peak_rss_bytes': int(max_rss if sys.platform == 'darwin' else max_rss * 1024),
                        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
                        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if args.device == 'cuda' else 0},
            'elapsed_seconds': elapsed, 'inference_seconds': inference_seconds,
            'inference_targets_per_second': processed / inference_seconds,
            'total_targets_per_second': processed / elapsed,
        }
        atomic_json(output / 'receipt.json', receipt)
        print(json.dumps({k: receipt[k] for k in ('status', 'dataset', 'model', 'seed', 'split', 'prediction_rows', 'elapsed_seconds')}), flush=True)
        return 0
    except BaseException as error:
        if writer is not None:
            try:
                writer.flush()
            except Exception:
                pass
        failure = {'status': 'failed', 'phase': phase, 'exception': type(error).__name__,
                   'message': str(error), 'traceback': traceback.format_exc(),
                   'elapsed_seconds': time.monotonic() - started,
                   'prediction_rows': writer.rows if writer else 0,
                   'parts': writer.parts if writer else [], 'receipt_written': False}
        atomic_json(output / 'failure.json', failure)
        raise


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--registry', required=True)
    ap.add_argument('--dataset-manifest', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--seed', required=True, type=int)
    ap.add_argument('--split', choices=['validation', 'test'], required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--contract')
    ap.add_argument('--inventory-only', action='store_true', help='Freeze target count/digest without model construction or inference')
    ap.add_argument('--max-batches', type=int, help='Bounded validation qualification only')
    ap.add_argument('--deadline-seconds', type=float)
    ap.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser().parse_args(argv)
    require(not (args.inventory_only and args.max_batches), 'Inventory always covers the full population')
    require(args.max_batches is None or (args.split == 'validation' and args.max_batches > 0),
            '--max-batches is available only for positive bounded validation qualification')
    require(args.deadline_seconds is None or args.deadline_seconds > 0,
            'Deadline must be positive')
    require(not args.output.exists(), 'Output already exists; use a new attempt directory')
    if args._worker:
        return run_worker(args)
    # No model imports in launcher; every condition gets a fresh module namespace.
    try:
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), *argv, '--_worker'],
                               timeout=args.deadline_seconds + 2 if args.deadline_seconds else None,
                               check=False)
        return child.returncode
    except subprocess.TimeoutExpired:
        args.output.mkdir(parents=True, exist_ok=True)
        if not (args.output / 'receipt.json').exists() and not (args.output / 'failure.json').exists():
            atomic_json(args.output / 'failure.json', {
                'status': 'failed', 'exception': 'TimeoutExpired',
                'message': 'Parent terminated worker at deadline; committed parts preserved',
                'receipt_written': False,
            })
        return 124


if __name__ == '__main__':
    raise SystemExit(main())
