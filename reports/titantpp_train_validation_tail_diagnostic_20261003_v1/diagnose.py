"""Frozen seed42 CPU diagnosis. Only train/validation; no optimizer or training.

Run each source bundle in a separate process. This file writes only its own
report directory, never into a checkpoint, source bundle, or original report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REG = ROOT / 'reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json'
MAN = ROOT / 'reports/titantpp_independent_evaluation_protocol_20261001_v1/dataset_manifest.json'
DATASETS = ['yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000', 'insta_market_basket']
MODELS = ['titantpp_history_mlp', 'titantpp_all_available_history_mlp']


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def freeze():
    assert not (OUT / 'contract.json').exists(), 'Do not overwrite frozen diagnosis contract'
    reg = read(REG)
    rows = [x for x in reg['rows'] if x['seed'] == 42 and x['dataset'] in DATASETS and x['model'] in MODELS]
    assert len(rows) == 8
    protected = subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()
    protected += ['paper/titantpp_pakdd_2027_draft/main.tex']
    contract = {
        'created_unix': time.time(), 'scope': 'local CPU frozen-weight train/validation diagnosis',
        'datasets': DATASETS, 'models': MODELS, 'seed': 42, 'endpoint': 'selected_only',
        'new_training': False, 'heldout_read': False, 'remote_calls': 0,
        'train_sampling': 'SRS without replacement within original TRAIN-derived quantity bins; maximum 2048 per bin',
        'train_sampling_seed': 20261003, 'validation_sampling_seed': 20261004,
        'max_per_bin': 2048, 'sample_weights': 'N_bin / n_bin; same targets across models',
        'validation': 'reuse complete MLP validation predictions; all-available inference on fixed stratified validation sample',
        'cpu_threads': 2, 'batch_size': 64,
        'registry_sha256': sha(REG), 'dataset_manifest_sha256': sha(MAN),
        'rows': rows, 'protected_sha256_before': {p: sha(ROOT / p) for p in protected},
        'interpretation': 'one-seed descriptive diagnosis; in-sample training fit is not generalization; target-defined tail bias alone is not miscalibration proof; no causal claim from this observational comparison',
    }
    write(OUT / 'contract.json', contract)


def one(ds, arm):
    started = time.monotonic()
    contract = read(OUT / 'contract.json')
    assert sha(REG) == contract['registry_sha256'] and sha(MAN) == contract['dataset_manifest_sha256']
    reg = read(REG)
    row = next(x for x in contract['rows'] if x['dataset'] == ds and x['model'] == arm)
    bundle = reg['bundles'][row['evaluator_source_bundle']]
    source = ROOT / bundle['source_root']
    spec = next(x for x in bundle['datasets'] if x['dataset_id'] == ds)
    for rel, expected in bundle['source_files'].items():
        assert sha(source / rel) == expected, rel
    sys.dont_write_bytecode = True
    sys.path[:] = [str(source)] + [p for p in sys.path if p and not Path(p).resolve().is_relative_to(ROOT)]
    os.environ.update(CUDA_VISIBLE_DEVICES='', POLARS_MAX_THREADS='2', MPLCONFIGDIR=str(OUT / 'cache'), XDG_CACHE_HOME=str(OUT / 'cache'))
    try:
        os.nice(10)
    except OSError:
        pass
    import numpy as np
    import polars as pl
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame, right_pad_batch, target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from data_loader.event_seq_data_module import collate_week_lookback
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    def population(data, split):
        idx = np.asarray(data.index, dtype=np.int64)
        pi, pos = idx[:, 0], idx[:, 1] + 1
        seq = np.fromiter((data.seq_lists[p][t] for p, t in zip(pi, pos)), dtype=np.int64)
        q = np.fromiter((data.val_lists[p][t] for p, t in zip(pi, pos)), dtype=np.float64)
        assert all(data.split_lists[p][t] == split for p, t in zip(pi, pos))
        h = hashlib.sha256(b'hard_lmm_target_identity_v1\0' + split.encode() + b'\0')
        h.update(json.dumps([str(p) for p in data.parts], ensure_ascii=False, separators=(',', ':')).encode())
        for a in [pi, pos, seq]:
            h.update(a.astype('<i8').tobytes())
        qh = hashlib.sha256(b'hard_lmm_target_quantity_v1\0' + q.astype('<f8').tobytes()).hexdigest()
        meta = {'target_count': len(q), 'target_identity_sha256': h.hexdigest(), 'target_quantity_sha256': qh}
        expected = spec['inherited_data_identity']['populations'][split]
        assert meta == expected, (split, meta, expected)
        return q, meta

    def sample(q, split):
        group = np.searchsorted(bounds, q, side='left')
        rng = np.random.default_rng(contract[f'{split}_sampling_seed'])
        ix, weights, meta = [], [], []
        for g in range(len(bounds) + 1):
            all_ix = np.flatnonzero(group == g)
            n = min(len(all_ix), contract['max_per_bin'])
            if not n:
                meta.append({'bin': g, 'population_n': 0, 'sample_n': 0})
                continue
            chosen = np.sort(rng.choice(all_ix, n, replace=False))
            ix.extend(chosen.tolist())
            weights.extend([len(all_ix) / n] * n)
            values = q[all_ix]
            meta.append({'bin': g, 'population_n': len(all_ix), 'sample_n': n,
                         'q_mean': float(values.mean()), 'q_p50': float(np.median(values)),
                         'q_p90': float(np.quantile(values, .9)), 'q_max': float(values.max())})
        order = np.argsort(ix)
        return np.array(ix, dtype=np.int64)[order], np.array(weights)[order], meta

    def aggregate(f):
        q = f['raw_quantity'].to_numpy()
        p = f['predicted_raw_quantity'].to_numpy()
        w = f['weight'].to_numpy()
        bins = np.searchsorted(bounds, q, side='left')
        err = p - q
        logerr = np.log1p(p) - np.log1p(q)
        total_sse = float(np.sum(w * err ** 2))
        result = []
        for g in [-1] + list(range(len(bounds) + 1)):
            keep = np.ones(len(q), dtype=bool) if g == -1 else bins == g
            if not keep.any():
                continue
            ww = w[keep]
            avg = lambda a: float(np.average(a[keep], weights=ww))
            result.append({'bin': g, 'n_evaluated': int(keep.sum()), 'represented_n': float(ww.sum()),
                'rmse': float(np.sqrt(avg(err ** 2))), 'mae': avg(abs(err)), 'bias': avg(err),
                'log_mse': avg(logerr ** 2), 'q_mean': avg(q), 'prediction_mean': avg(p),
                'sse_share': float(np.sum(ww * err[keep] ** 2) / total_sse),
                'underprediction_share': avg(p < q),
                'last_rmse': float(np.sqrt(avg((f['last_quantity'].to_numpy() - q) ** 2))),
                'mean_rmse': float(np.sqrt(avg((f['mean_quantity'].to_numpy() - q) ** 2))),
                'history_mean': avg(f['history_length'].to_numpy()),
                'time_nll': avg(f['time_nll'].to_numpy())})
        return result

    cpfile = ROOT / row['checkpoint_path']
    assert sha(cpfile) == row['checkpoint_file_sha256']
    cp = torch.load(cpfile, map_location='cpu', weights_only=False)
    validate_checkpoint_route(cp, arm)
    config = {k: v for k, v in spec['model'].items() if k not in ('backbone', 'lambda_log_qty', 'lambda_tail', 'time_head_lr_multiplier')}
    model, _ = build_count_aware_model(arm, **config, train_log_mean=spec['statistics']['train_log_mean'], train_log_std=spec['statistics']['train_log_std'], max_seq_len=spec['loader']['max_seq_len'])
    model.load_state_dict(cp['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    state = canonical_state_dict_sha256(model.state_dict())
    assert state == row['state_tensor_sha256'] and cp['best_epoch'] == row['selected_epoch']
    identity = next(x for x in read(MAN)['datasets'] if x['dataset'] == ds)['identities']
    identity = identity.get('deployed_data', identity['data'])
    datapath = ROOT / identity['path']
    assert sha(datapath) == identity['sha256']
    frame = pl.scan_parquet(datapath).filter(pl.col('chronological_split').is_in(['train', 'validation'])).collect().sort(['oper_part_no', 'seq'])
    assert set(frame['chronological_split']) <= {'train', 'validation'}
    frame = prepare_count_frame(frame)
    bounds = np.array(spec['quantity_boundaries_all_train_rows'], dtype=np.float64)
    job = OUT / 'runs' / f'{ds}__{arm}__seed42'
    job.mkdir(parents=True, exist_ok=True)
    assert not any(job.iterdir()), 'Do not overwrite an existing diagnostic result'
    evidence = {'dataset': ds, 'model': arm, 'seed': 42, 'selected_epoch': row['selected_epoch'],
        'source_root': str(source.relative_to(ROOT)), 'source_closure_sha256': bundle['source_closure_sha256'],
        'checkpoint_path': row['checkpoint_path'], 'checkpoint_sha256': sha(cpfile), 'quantity_bounds': bounds.tolist(),
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu', 'threads': 2},
        'new_training': False, 'heldout_read': False, 'splits': {}}
    validation_full = None
    if arm == MODELS[0]:
        cache = ROOT / 'reports/titantpp_legacy_evaluation_20261003_v1/runs/validation_full/attempt1' / f'{ds}__{arm}__seed42'
        receipt = read(cache / 'receipt.json')
        assert receipt['split'] == 'validation' and not receipt['heldout_read']
        assert receipt['state_tensor_sha256'] == state
        parts = []
        for part in receipt['parts']:
            p = cache / part['path']
            assert sha(p) == part['sha256']
            parts.append(pl.read_parquet(p))
        validation_full = pl.concat(parts).sort('target_index').with_columns(pl.lit(1.).alias('weight'))
        assert validation_full.height == row['validation_count'] and set(validation_full['split']) == {'validation'}
        evidence['cached_validation_receipt_sha256'] = sha(cache / 'receipt.json')
        evidence['full_validation_metrics'] = aggregate(validation_full)

    for split in ['train', 'validation']:
        loader = make_loader(frame, target_split=split, shuffle=False, generator=None,
            **{k: spec['loader'][k] for k in ['batch_size', 'lookback_weeks', 'max_seq_len']})
        data = loader.dataset
        q, pop = population(data, split)
        ix, w, bins_meta = sample(q, split)
        print(json.dumps({'phase': 'inference', 'dataset': ds, 'model': arm, 'split': split, 'population': len(q), 'sample': len(ix)}), flush=True)
        records, max_diff, verified_boundary = [], 0., False
        # Cached MLP validation needs only a 64-target numerical replay check;
        # its remaining predictions were already verified against checkpoint SHA.
        infer_ix = ix if validation_full is None or split == 'train' else ix[:64]
        with torch.inference_mode():
            for begin in range(0, len(infer_ix), contract['batch_size']):
                ids = infer_ix[begin:begin + contract['batch_size']]
                batch = collate_week_lookback([data[int(i)] for i in ids])
                _, dt, mask, _, qty = batch
                values = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
                assert np.array_equal(values['true_qty'].double().numpy(), q[ids])
                pred = values['pred_qty'].double().numpy()
                assert np.isfinite(pred).all() and (pred >= 0).all()
                rd, rq, rm, length = right_pad_batch(dt, qty, mask)
                bi = torch.arange(len(ids))
                last = rq[bi, length - 2].double().numpy()
                history_mask = rm.clone()
                history_mask[bi, length - 1] = False
                mean = ((rq * history_mask).sum(1) / history_mask.sum(1)).double().numpy()
                if begin == 0:
                    # Changing the appended target cannot change the quantity
                    # prediction; this checks the actual frozen forward route.
                    changed_qty, changed_dt = qty.clone(), dt.clone()
                    changed_qty[:, -1] = changed_qty[:, -1] + 12345.
                    # Respect the frozen top-code support (Instacart: 30).
                    changed_dt[:, -1] = torch.where(changed_dt[:, -1] == 1., 2., 1.)
                    counter = target_outputs(model, changed_dt, mask, changed_qty, lambda_log_qty=1.)
                    assert torch.allclose(values['pred_qty'], counter['pred_qty'], atol=1e-6, rtol=1e-6)
                    verified_boundary = True
                for k, idx in enumerate(ids):
                    pi, end = data.index[int(idx)]
                    if split == 'train':
                        assert all(x == 'train' for x in data.split_lists[pi][:end + 2])
                    if validation_full is not None and split == 'validation':
                        old = validation_full.row(int(idx), named=True)
                        assert str(data.parts[pi]) == old['entity_id'] and data.seq_lists[pi][end + 1] == old['seq']
                        max_diff = max(max_diff, abs(float(pred[k]) - old['predicted_raw_quantity']))
                        assert np.isclose(pred[k], old['predicted_raw_quantity'], atol=1e-4, rtol=1e-4)
                    records.append({'target_index': int(idx), 'entity_id': str(data.parts[pi]), 'seq': int(data.seq_lists[pi][end + 1]),
                        'split': split, 'raw_quantity': float(q[idx]), 'predicted_raw_quantity': float(pred[k]),
                        'last_quantity': float(last[k]), 'mean_quantity': float(mean[k]),
                        'history_length': int(values['history_length'][k]),
                        'time_nll': float(values['time_loss'][k]), 'weight': float(w[np.searchsorted(ix, idx)])})
                if begin and begin % (contract['batch_size'] * 16) == 0:
                    print(json.dumps({'phase': 'progress', 'dataset': ds, 'model': arm, 'split': split, 'done': begin + len(ids), 'total': len(infer_ix), 'elapsed_seconds': round(time.monotonic() - started)}), flush=True)
        if validation_full is not None and split == 'validation':
            assert np.array_equal(q, validation_full['raw_quantity'].to_numpy())
            f = validation_full[ix].with_columns(pl.Series('weight', w))
        else:
            f = pl.DataFrame(records)
        path = job / f'{split}_sample.parquet'
        f.write_parquet(path)
        evidence['splits'][split] = {'population': pop, 'strata': bins_meta, 'sample_n': len(ix),
            'sample_indices_sha256': hashlib.sha256(ix.astype('<i8').tobytes()).hexdigest(),
            'prediction_path': str(path.relative_to(OUT)), 'prediction_sha256': sha(path),
            'target_boundary_verified': verified_boundary, 'cached_replay_max_abs_difference': max_diff,
            'metrics': aggregate(f)}
        del loader, data
    assert canonical_state_dict_sha256(model.state_dict()) == state
    assert sha(cpfile) == row['checkpoint_file_sha256']
    for name, mod in list(sys.modules.items()):
        if name.split('.')[0] not in {'models', 'paper', 'data_loader', 'simple_lab_test', 'utils'}:
            continue
        f = getattr(mod, '__file__', None)
        if f:
            assert Path(f).resolve().is_relative_to(source), (name, f)
    evidence.update(parameters_unchanged=True, source_hashes_verified=True, population_hashes_verified=True,
                    elapsed_seconds=time.monotonic() - started, status='complete')
    write(job / 'receipt.json', evidence)
    print(json.dumps({'phase': 'complete', 'dataset': ds, 'model': arm, 'elapsed_seconds': evidence['elapsed_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    if args.freeze:
        return freeze()
    if args.dataset:
        return one(args.dataset, args.model)
    for ds in DATASETS:
        for arm in MODELS:
            subprocess.run([sys.executable, '-B', __file__, '--dataset', ds, '--model', arm], check=True,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))


if __name__ == '__main__':
    main()
