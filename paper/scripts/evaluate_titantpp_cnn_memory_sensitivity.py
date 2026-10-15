"""Native, read-only Train/Validation intervention on selected CNN checkpoints.

The only changed arithmetic is the exact hard-prototype residual multiplier.
Targets are used for scoring, never for choosing that multiplier or features.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

SCHEMA = 'titantpp_cnn_memory_sensitivity_v1'
ARM = 'titantpp_cnn_gru54_cnn_only'
ALPHAS = (1., .5, .25, 0.)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def checked(root, relative):
    path = (root / relative).resolve()
    need(not Path(relative).is_absolute() and path.is_relative_to(root), 'Unsafe bundle path')
    return path


def verify(root, contract_sha, job_id, split):
    need(split in ('train', 'validation'), 'Only Train/Validation is authorized')
    need(sha(root / 'contract.json') == contract_sha, 'Contract SHA differs')
    c = read(root / 'contract.json')
    need(c['schema'] == SCHEMA and c['allowed_splits'] == ['train', 'validation']
         and c['new_training'] is False and c['frozen_selection'] is True
         and c['max_parallel_per_host'] == 3, 'Diagnostic scope differs')
    need(c['alphas'] == list(ALPHAS), 'Unapproved attenuation grid')
    jobs = c['jobs']
    need(len(jobs) == len({j['id'] for j in jobs}) == 9, 'Expected nine unique CNN models')
    datasets = ('yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000')
    need({(j['dataset'], j['seed']) for j in jobs} == {(d, s) for d in datasets for s in (42, 52, 62)},
         'Dataset/seed cohort differs')
    for j in jobs:
        host = '5090' if j['dataset'] == 'intermittent_frozen_5000' and j['seed'] in (42, 52) else '5080'
        need(j['id'] == f"{j['dataset']}__{j['seed']}__{ARM}" and j['arm'] == ARM and j['host'] == host,
             'Model identity or native host differs')
    job = next(j for j in jobs if j['id'] == job_id)
    need(job['arm'] == ARM and job['seed'] in (42, 52, 62), 'Wrong model route/seed')
    need(time.time() < c['deadline_unix'], 'Fixed diagnostic deadline expired')
    need(not set(c['operation_files']) & set(c['input_files']), 'Operation/input paths overlap')
    for relative, digest in {**c['operation_files'], **c['input_files']}.items():
        need(sha(checked(root, relative)) == digest, 'Frozen bytes changed: ' + relative)
    need(sha(__file__) == c['operation_files']['operation/evaluate.py'], 'Evaluator bytes differ')
    return c, job


def observed_features(quantities, mask, np):
    """Loader uses left padding and the final slot is the withheld target."""
    q = quantities.detach().cpu().double().numpy()
    m = mask.detach().cpu().numpy().astype(bool).copy()
    m[:, -1] = False
    length = m.sum(1)
    need((length >= 1).all(), 'Empty observed context')
    logq = np.log1p(q)
    mean = np.where(m, logq, 0.).sum(1) / length
    variance = np.where(m, (logq - mean[:, None]) ** 2, 0.).sum(1) / length
    last = q[:, -2]
    change = np.where(length > 1, logq[:, -2] - logq[:, -3], 0.) if q.shape[1] >= 3 else np.zeros(len(q))
    return dict(history_length=length, last_quantity=last,
                mean_quantity=np.where(m, q, 0.).sum(1) / length,
                history_log_mean=mean, history_log_std=np.sqrt(variance),
                last_log_deviation=np.log1p(last) - mean, last_observed_log_change=change,
                previous_observation_available=length > 1)


def selected_telemetry(record, lengths, np):
    """core.target_outputs right-pads, then selects lengths-2 (observed)."""
    pos = lengths - 1
    rows = np.arange(len(pos))
    out = {}
    for key in ('encoded_norm', 'memory_norm', 'cosine_h_memory'):
        out[key] = record[key].numpy()[rows, pos]
    out['memory_to_encoded_norm'] = np.divide(out['memory_norm'], out['encoded_norm'],
        out=np.zeros(len(pos), dtype=float), where=out['encoded_norm'] > 0)
    out['encoded_norm_zero'] = out['encoded_norm'] == 0
    scores = record['topk_similarity'].numpy()[rows, pos]
    indices = record['prototype_indices'].numpy()
    selected = indices[rows, pos]
    previous = indices[rows, np.maximum(pos - 1, 0)]
    out['topk_similarity_mean'] = scores.mean(1)
    out['topk_similarity_std'] = scores.std(1)
    out['topk_similarity_spread'] = scores.max(1) - scores.min(1)
    out['previous_topk_overlap'] = np.asarray([len(set(a) & set(b)) / len(a) if n > 1 else 0.
                                              for a, b, n in zip(selected, previous, lengths)])
    for i in range(selected.shape[1]):
        out[f'prototype_{i}'] = selected[:, i]
        out[f'similarity_{i}'] = scores[:, i]
    return out


def load_observed_frame(path, split, pl):
    """Filter before materialization: a historically named export may include Test."""
    need(split in ('train', 'validation'), 'Only observed diagnostic splits are permitted')
    allowed = ['train'] if split == 'train' else ['train', 'validation']
    frame = pl.scan_parquet(path).filter(pl.col('chronological_split').is_in(allowed)).collect()
    need(frame.height > 0 and set(frame['chronological_split'].unique()) <= set(allowed),
         'Disallowed rows materialized')
    return frame.sort(['oper_part_no', 'seq'])


def worker(root, contract_sha, job_id, split):
    root = Path(root).resolve()
    c, job = verify(root, contract_sha, job_id, split)
    output = root / 'results' / job_id / split
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    write(output / 'claim.json', dict(job_id=job_id, split=split, pid=os.getpid(),
          contract_sha256=contract_sha, started_unix=started))
    try:
        utils = module(root / 'operation/row_utils.py', '_cnn_memory_row_utils')
        helper = module(root / job['helper'], '_cnn_memory_checkpoint_helper')
        source = c['source']
        source_root = checked(root, source['root'])
        spec = dict(source_root=source_root, source_files=source['files'])
        need(utils.canonical(source['files']) == source['closure'], 'Source closure differs')
        utils.verify_science(spec)
        original = job['original']
        definition = c['datasets'][job['dataset']]['definition']
        checkpoint = checked(root, original['checkpoint']['relative_path'])
        history = helper.read_selection_history(checked(root, original['selection_history_path']))
        need(helper.strict_selected_epoch(history) == (original['selected_epoch'], original['selected_metric_value']),
             'First strict finite Validation selection differs')
        utils.isolate(source_root)
        os.environ['MPLCONFIGDIR'] = str(output / 'cache/matplotlib')
        os.environ['XDG_CACHE_HOME'] = str(output / 'cache')
        import numpy as np
        import polars as pl
        import torch
        control = module(root / 'operation/control.py', '_cnn_memory_control')
        numerical = importlib.import_module('paper.scripts.quantity_comparison_runtime')
        numerical.configure_runtime('cuda:0', c['runtimes'][job['host']]['torch_num_threads'])
        runtime = numerical.runtime_identity('cuda:0')
        utils.check_runtime(runtime, c['runtimes'][job['host']])
        importlib.import_module(helper.ROUTE).install_hooks()
        factory = importlib.import_module('models.TPPs.CountAwareFactory')
        core = importlib.import_module('paper.scripts.count_aware_tpp_backbone.core')
        loaders = importlib.import_module('paper.scripts.run_taxi_quantity_interface_ablation')
        runner = importlib.import_module('simple_lab_test.search.common.runner')
        payload = runner.torch_load_checkpoint(checkpoint, map_location='cpu')
        helper.validate_payload(payload, original, source, definition,
                                read(checked(root, original['checkpoint_metadata_path'])))
        factory.validate_checkpoint_route(payload, ARM)
        state_sha = original['checkpoint_state_sha256']
        need(runner.canonical_state_dict_sha256(payload['model_state_dict']) == state_sha, 'Tensor SHA differs')
        data = c['datasets'][job['dataset']]
        frame = load_observed_frame(checked(root, data['path']), split, pl)
        helper.validate_frame(frame, definition, 'validation', np, pl)
        train_q = frame.filter(pl.col('chronological_split') == 'train')['demand_qty'].to_numpy().astype(np.float64)
        stats = definition['statistics']
        for key, value in [('train_log_mean', np.log1p(train_q).mean()), ('train_log_std', np.log1p(train_q).std())]:
            need(math.isclose(value, stats[key], rel_tol=0., abs_tol=1e-12), 'Train statistics differ')
        need(hashlib.sha256(b'quantity_comparison_all_train_row_quantity_v1\0' + train_q.astype('<f8').tobytes()).hexdigest()
             == stats['all_train_quantity_sha256'], 'Train quantity SHA differs')
        frame = core.prepare_count_frame(frame)
        loader = loaders.make_loader(frame, target_split=split, batch_size=128, shuffle=False, generator=None,
            lookback_weeks=definition['loader']['lookback_weeks'], max_seq_len=definition['loader']['max_seq_len'])
        population = helper.population_identity(loader.dataset, split, np)
        need(population == definition['inherited_data_identity']['populations'][split], 'Target population differs')
        config = {k: v for k, v in definition['model'].items()
                  if k not in ('backbone', 'lambda_log_qty', 'lambda_tail', 'time_head_lr_multiplier')}
        model, _ = factory.build_count_aware_model(ARM, **config, train_log_mean=stats['train_log_mean'],
            train_log_std=stats['train_log_std'], max_seq_len=definition['loader']['max_seq_len'])
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model.to('cuda:0').eval().requires_grad_(False)
        need(runner.canonical_state_dict_sha256(model.state_dict()) == state_sha, 'Loaded state differs')
        matchers = [m for m in model.modules() if type(m).__name__ == 'HardLocalMemoryMatcher']
        need(len(matchers) == 1 and matchers[0] is model.lmm, 'Expected exactly one original hard matcher')
        utils.verify_science(spec, imports=True)
        metrics = {str(a): utils.RawMetrics(definition['quantity_boundaries_all_train_rows'], np) for a in ALPHAS}
        writer = utils.PredictionWriter(output / 'predictions', pl)
        qualifications, processed, timing = {}, 0, {str(a): 0. for a in ALPHAS}
        time_sha = hashlib.sha256()
        with torch.inference_mode():
            for batch_index, (_, dts, mask, part_indices, quantities) in enumerate(loader):
                need(time.time() < c['deadline_unix'], 'Diagnostic deadline expired')
                dts, mask, quantities = (v.to('cuda:0') for v in (dts, mask, quantities))
                features = observed_features(quantities, mask, np)
                rows = []
                for i in range(len(dts)):
                    part, end = loader.dataset.index[processed + i]
                    need(int(part_indices[i]) == part, 'Loader order differs')
                    rows.append(dict(entity_id=str(loader.dataset.parts[part]), seq=int(loader.dataset.seq_lists[part][end + 1]),
                        dataset=job['dataset'], seed=job['seed'], split=split,
                        **{k: v[i].item() for k, v in features.items()}))
                baseline_trace = None
                for alpha in ALPHAS:
                    traces = []
                    torch.cuda.synchronize(); tick = time.monotonic()
                    with control.frozen_memory_attenuation(model.lmm, alpha, traces.append):
                        value = core.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
                    torch.cuda.synchronize(); timing[str(alpha)] += time.monotonic() - tick
                    need(len(traces) == 1, 'Expected one retrieval per prediction')
                    lengths = value['history_length'].cpu().numpy()
                    need(np.array_equal(lengths, features['history_length']), 'Observed telemetry position differs')
                    trace = selected_telemetry(traces[0], lengths, np)
                    if baseline_trace is None:
                        baseline_trace = trace
                    else:
                        need(all(np.array_equal(trace[k], baseline_trace[k]) for k in trace),
                             'Attenuation changed retrieval/encoder; intervention is not isolated')
                    q, p, t = (value[k].cpu().numpy().astype(np.float64) for k in ('true_qty', 'pred_qty', 'time_loss'))
                    need(all(np.isfinite(x).all() for x in (q, p, t)), 'Nonfinite result')
                    metrics[str(alpha)].add(q, p, t)
                    if batch_index == 0:
                        with control.frozen_memory_attenuation(model.lmm, alpha, lambda record: None):
                            qualifications[str(alpha)] = utils.qualify(torch, model, core.target_outputs, dts, mask, quantities,
                                value, definition['model']['time_observation_contract']['top_code'])
                    for i, row in enumerate(rows):
                        expected_q = loader.dataset.val_lists[loader.dataset.index[processed+i][0]][loader.dataset.index[processed+i][1]+1]
                        need(float(q[i]) == float(expected_q), 'Target order/quantity differs')
                        row.update(raw_quantity=float(q[i]), **{f'prediction_{alpha:g}': float(p[i]), f'time_nll_{alpha:g}': float(t[i])})
                for i, row in enumerate(rows):
                    row.update({k: v[i].item() for k, v in baseline_trace.items()})
                writer.add(rows)
                time_sha.update(dts[:, -1].cpu().numpy().astype('<f8').tobytes())
                processed += len(dts)
                if batch_index % 50 == 0:
                    print(json.dumps(dict(job=job_id, split=split, processed=processed,
                                          targets=population['target_count'])), flush=True)
        results = {a: metric.finish() for a, metric in metrics.items()}
        need(processed == population['target_count'], 'Incomplete target coverage')
        if split == 'validation':
            utils.compare_metrics(results['1.0'], original['validation_reference'])
        need(runner.canonical_state_dict_sha256(model.state_dict()) == state_sha, 'Full model parameters changed')
        need(sha(checkpoint) == original['checkpoint']['sha256'], 'Original checkpoint changed')
        utils.verify_science(spec, imports=True)
        manifest = writer.finish()
        receipt = dict(schema=SCHEMA, status='complete', job_id=job_id, split=split, host=job['host'],
            contract_sha256=contract_sha, started_unix=started, finished_unix=time.time(),
            checkpoint_sha256=original['checkpoint']['sha256'], state_sha256=state_sha,
            selected_epoch=original['selected_epoch'], selected_metric=original['selected_metric_value'],
            parameters_unchanged=True, no_training=True, held_out_test_evaluated=False,
            baseline_validation_reproduced=split == 'validation', population=population,
            recorded_gap_sha256=time_sha.hexdigest(), metrics=results, qualification=qualifications,
            prediction_manifest=manifest, inference_seconds_by_alpha=timing, runtime=runtime)
        write(output / 'receipt.json', receipt)
        print(json.dumps(dict(status='complete', job=job_id, split=split, receipt_sha256=sha(output / 'receipt.json'))), flush=True)
    except BaseException as error:
        write(output / 'failure.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
                                            job_id=job_id, split=split, automatic_retry=False))
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--contract-sha', required=True)
    p.add_argument('--job', required=True); p.add_argument('--split', choices=('train', 'validation'), required=True)
    a = p.parse_args()
    worker(a.root, a.contract_sha, a.job, a.split)


if __name__ == '__main__':
    main()
