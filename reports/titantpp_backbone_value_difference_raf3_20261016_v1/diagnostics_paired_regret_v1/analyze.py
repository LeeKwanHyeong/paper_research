"""Saved RAF CNN Train/Validation rows only; no model imports or inference."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

PROJECT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
BUNDLE = PROJECT / 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3'
CONTRACT_SHA = 'd51ab06365a7165e00cc18849b84caba4ee2dd8af6a36aee84a5c1d62e01fab3'
MANIFEST_SHA = '9b7044490727a42add4b9e05538fdf315137f0a956e25d8b1422f59a37430819'
PLAN_SHA = '0708d991be70c3792af95e6266005de2c9cea14d6851fc5aaa731351727ed8b3'
SOURCE_CLOSURE = '15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53'
SEEDS = (42, 52, 62)
SPLITS = ('train', 'validation')
FEATURES = {'last_observed_gap': 2, 'last_three_observed_span': 3}
KEYS = ('entity_id', 'target_quantity_bin', 'history_length', 'observed_change_magnitude_bin')
BOUNDARIES = (2., 30., 60., 200.)
INPUTS = {}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path, expected):
    path = Path(path)
    actual = sha(path)
    need(actual == expected, 'SHA differs: ' + str(path))
    INPUTS[str(path.relative_to(PROJECT))] = actual
    return path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def score(rows, predictions, weights=None):
    if not rows:
        return dict(rows=0, entities=0, weight_sum=0., ess=0., rmse=None, mae=None, bias=None,
                    weighted_sse=0., weighted_absolute_error_sum=0., weighted_error_sum=0.)
    w = np.ones(len(rows)) if weights is None else np.asarray(weights, dtype=np.float64)
    e = np.asarray([predictions[(r['entity_id'], r['seq'])] - r['raw_quantity'] for r in rows])
    mass = float(w.sum())
    sse, ae, signed = float(np.dot(w, e * e)), float(np.dot(w, np.abs(e))), float(np.dot(w, e))
    return dict(rows=len(rows), entities=len({r['entity_id'] for r in rows}), weight_sum=mass,
                ess=mass * mass / float(np.dot(w, w)), rmse=math.sqrt(sse / mass), mae=ae / mass,
                bias=signed / mass, weighted_sse=sse, weighted_absolute_error_sum=ae,
                weighted_error_sum=signed)


def matching(rows, feature, cuts):
    low, high = cuts
    groups = defaultdict(lambda: {'low': [], 'high': []})
    for row in rows:
        value = row[feature]
        if value is None:
            continue
        side = 'low' if value <= low else 'high' if value >= high else None
        if side is not None:
            groups[tuple(row[k] for k in KEYS)][side].append(row)
    matched = {side: [] for side in ('low', 'high')}
    weights = {side: [] for side in ('low', 'high')}
    audit = []
    for key, sides in sorted(groups.items()):
        if not all(sides.values()):
            continue
        mass = min(len(sides['low']), len(sides['high']))
        audit.append(dict(zip(KEYS, key), low_n=len(sides['low']), high_n=len(sides['high']),
                          matched_mass_per_group=mass))
        for side in matched:
            matched[side].extend(sides[side])
            weights[side].extend([mass / len(sides[side])] * len(sides[side]))
    return matched, weights, audit


def weighted_balance(rows, weights):
    result = {}
    for field in ('raw_quantity', 'absolute_observed_log_change', 'last_quantity', 'history_length'):
        data = []
        for side in ('low', 'high'):
            values, w = np.asarray([r[field] for r in rows[side]]), np.asarray(weights[side])
            if not len(values):
                data.append(None)
                continue
            mean = float(np.dot(values, w) / w.sum())
            variance = float(np.dot((values - mean) ** 2, w) / w.sum())
            data.append((mean, variance))
        if any(v is None for v in data):
            result[field] = dict(low_mean=None, high_mean=None, high_minus_low=None, smd=None)
        else:
            difference = data[1][0] - data[0][0]
            scale = math.sqrt((data[0][1] + data[1][1]) / 2)
            result[field] = dict(low_mean=data[0][0], high_mean=data[1][0], high_minus_low=difference,
                                 smd=difference / scale if scale else 0. if difference == 0 else None)
    return result


def build_rows(frame, split, expected):
    """Mirror the frozen loader's target order and past-only context, without importing it."""
    need(split in SPLITS, 'Forbidden split')
    allowed = ('train',) if split == 'train' else SPLITS
    frame = frame.filter(pl.col('chronological_split').is_in(allowed)).sort(['oper_part_no', 'seq'])
    groups = frame.partition_by('oper_part_no', maintain_order=True)
    parts = [str(g['oper_part_no'][0]) for g in groups]
    part_indices, positions, sequences, quantities, output = [], [], [], [], []
    raw_gap_checks = 0
    for p, group in enumerate(groups):
        seq = group['seq'].to_numpy()
        q = group['demand_qty'].to_numpy().astype(np.float32).astype(np.float64)
        gaps = group['delta_t'].to_numpy()
        dates = group['demand_dt'].to_numpy()
        months = dates // 10000 * 12 + (dates // 100) % 100
        labels = group['chronological_split'].to_list()
        need(np.all(np.diff(seq) > 0), 'Duplicate/reversed within-entity sequence')
        need(np.array_equal(np.diff(seq), gaps[1:]), 'Recorded past gaps differ from sequence coordinate')
        need(np.array_equal(np.diff(months), gaps[1:]), 'Recorded past gaps differ from calendar month intervals')
        raw_gap_checks += len(seq) - 1
        for target in range(1, len(seq)):
            if labels[target] != split:
                continue
            end = target - 1
            start = max(int(np.searchsorted(seq, seq[end] - 83, side='left')), target - 83)
            indices = np.arange(start, target)
            need(len(indices) >= 1 and int(indices.max()) < target, 'Future or target in features')
            hq, hs = q[indices], seq[indices]
            history_length = len(indices)
            logq = np.log1p(hq)
            change = float(logq[-1] - logq[-2]) if history_length >= 2 else 0.
            row = dict(entity_id=parts[p], seq=int(seq[target]), split=split,
                       raw_quantity=float(q[target]), history_length=history_length,
                       last_quantity=float(hq[-1]), mean_quantity=float(hq.mean()),
                       history_log_mean=float(logq.mean()), history_log_std=float(logq.std()),
                       last_log_deviation=float(logq[-1] - logq.mean()),
                       last_observed_log_change=change, previous_observation_available=history_length >= 2,
                       absolute_observed_log_change=abs(change),
                       last_observed_gap=float(gaps[end]) if history_length >= 2 else None,
                       last_three_observed_span=float(gaps[end] + gaps[end - 1]) if history_length >= 3 else None,
                       target_quantity_bin=sum(q[target] > b for b in BOUNDARIES),
                       observed_change_magnitude_bin=math.floor(abs(change) / .05),
                       last_observed_seq=int(hs[-1]), first_observed_seq=int(hs[0]),
                       target_gap_used_for_feature=False,
                       d1=float(logq[-1]-logq[-2]) if history_length>=2 else None,
                       d2=float(logq[-2]-logq[-3]) if history_length>=3 else None)
            if history_length >= 3:
                need(row['last_three_observed_span'] == float(hs[-1] - hs[-3]), 'Span differs')
            output.append(row)
            part_indices.append(p); positions.append(target); sequences.append(int(seq[target])); quantities.append(float(q[target]))
    digest = hashlib.sha256(b'hard_lmm_target_identity_v1\0' + split.encode() + b'\0')
    digest.update(json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode())
    for values in (part_indices, positions, sequences):
        digest.update(np.asarray(values, dtype='<i8').tobytes())
    population = dict(target_count=len(output), target_identity_sha256=digest.hexdigest(),
                      target_quantity_sha256=hashlib.sha256(b'hard_lmm_target_quantity_v1\0' + np.asarray(quantities, dtype='<f8').tobytes()).hexdigest())
    need(population == expected, 'Population or quantity identity differs: ' + split)
    return output, dict(population=population, calendar_and_sequence_gap_pairs_verified=raw_gap_checks,
                        features_use_only_indices_before_target=True, loader_source_imported=False)


def self_checks():
    rows = [dict(entity_id='x', seq=i, raw_quantity=1., target_quantity_bin=0, history_length=3,
                 observed_change_magnitude_bin=0, last_observed_gap=g) for i, g in enumerate([1., 1., 10.])]
    m, w, cells = matching(rows, 'last_observed_gap', (1., 10.))
    need(w == {'low': [.5, .5], 'high': [1.]} and len(cells) == 1, 'Synthetic weights failed')
    s = score(m['low'], {('x', 0): 2., ('x', 1): 4.}, w['low'])
    need(s['weight_sum'] == 1. and s['mae'] == 2. and s['rmse'] == math.sqrt(5.), 'Synthetic scores failed')
    return {'matched_cell_equal_mass': True, 'hand_computed_weighted_rmse_mae_bias': True}


PATTERNS = ('insufficient_history', 'stagnation', 'same_sign', 'reversal')
BASELINES = ('last_observed', 'history_mean')
FIELDS = ('history_length', 'last_quantity', 'raw_quantity', 'd1', 'd2', 'magnitude')


def features(row):
    h, d1, d2 = row['history_length'], row['d1'], row['d2']
    if h < 3:
        return dict(pattern='insufficient_history', magnitude=None)
    pattern = 'stagnation' if d1 == 0. or d2 == 0. else 'same_sign' if d1*d2 > 0. else 'reversal'
    return dict(pattern=pattern, magnitude=max(abs(d1), abs(d2)))


def bucket(value, cuts):
    if value is None:
        return 'unavailable'
    return 'low' if value <= cuts[0] else 'middle' if value <= cuts[1] else 'high'


def distributions(rows, weights):
    result = {}
    for field in FIELDS:
        usable = [(r[field], float(w)) for r, w in zip(rows, weights) if r[field] is not None]
        if not usable:
            result[field] = dict(valid_count=0, missing_count=len(rows), min=None, max=None, weighted_mean=None,
                                 weighted_q25=None, weighted_median=None, weighted_q75=None)
            continue
        values = np.array([x[0] for x in usable], dtype=float)
        w = np.array([x[1] for x in usable], dtype=float)
        idx = np.argsort(values, kind='stable'); vs, ws = values[idx], w[idx]
        cs = np.cumsum(ws)
        # Distribution summaries use weighted empirical inverse CDF, explicitly distinct from Train linear cuts.
        q = [float(vs[min(len(vs)-1, int(np.searchsorted(cs, level*cs[-1], side='left')))]) for level in (.25,.5,.75)]
        result[field] = dict(valid_count=len(usable), missing_count=len(rows)-len(usable), min=float(vs[0]), max=float(vs[-1]),
                             weighted_mean=float(np.dot(values,w)/w.sum()), weighted_q25=q[0], weighted_median=q[1], weighted_q75=q[2])
    return result


def metrics(rows, prediction, weights):
    if not rows:
        return dict(count=0, entities=0, weight_sum=0., ess=0., cnn=None, references={}, regret={})
    q = np.array([r['raw_quantity'] for r in rows], dtype=float)
    p = np.array([prediction[(r['entity_id'],r['seq'])] for r in rows], dtype=float)
    w = np.array(weights, dtype=float); mass=float(w.sum())
    need(np.all(np.isfinite(p)) and np.all(np.isfinite(q)) and np.all(w>0), 'Invalid score values')
    def errors(v):
        e = v-q
        ae = float(np.dot(w,np.abs(e))); se=float(np.dot(w,e*e)); signed=float(np.dot(w,e))
        return dict(mae=ae/mass, mse=se/mass, rmse=math.sqrt(se/mass), bias=signed/mass,
                    ae_sum=ae, se_sum=se, signed_sum=signed)
    cnn=errors(p); refs={}; regrets={}
    for name, field in (('last_observed','last_quantity'), ('history_mean','mean_quantity')):
        b=np.array([r[field] for r in rows],dtype=float); ref=errors(b); refs[name]=ref
        da=np.abs(p-q)-np.abs(b-q); ds=(p-q)**2-(b-q)**2
        regrets[name]=dict(ae_mean=float(np.dot(w,da)/mass),se_mean=float(np.dot(w,ds)/mass),
                           ae_sum=float(np.dot(w,da)),se_sum=float(np.dot(w,ds)),
                           ae_positive_fraction=float(np.dot(w,da>0)/mass),ae_negative_fraction=float(np.dot(w,da<0)/mass),
                           ae_tie_fraction=float(np.dot(w,da==0)/mass),se_positive_fraction=float(np.dot(w,ds>0)/mass),
                           se_negative_fraction=float(np.dot(w,ds<0)/mass),se_tie_fraction=float(np.dot(w,ds==0)/mass))
        for kind in ('ae','se'):
            need(math.isclose(regrets[name][kind+'_sum'],cnn[kind+'_sum']-ref[kind+'_sum'],rel_tol=1e-10,abs_tol=1e-7), 'Paired regret identity differs')
    return dict(count=len(rows),entities=len({r['entity_id'] for r in rows}),weight_sum=mass,
                ess=mass**2/float(np.dot(w,w)),cnn=cnn,references=refs,regret=regrets)


def diagnostic_self_checks():
    r=[dict(entity_id='a',seq=1,raw_quantity=2.,last_quantity=1.,mean_quantity=3.),
       dict(entity_id='a',seq=2,raw_quantity=4.,last_quantity=4.,mean_quantity=2.),
       dict(entity_id='b',seq=1,raw_quantity=1.,last_quantity=1.,mean_quantity=1.)]
    p={('a',1):4.,('a',2):5.,('b',1):2.}
    out=metrics(r,p,[.5,.5,1.])
    need(out['weight_sum']==2 and out['cnn']['mae']==1.25 and out['cnn']['mse']==1.75, 'Synthetic entity equal weights differ')
    need(out['regret']['last_observed']['ae_mean']==1. and out['regret']['last_observed']['se_mean']==1.5, 'Synthetic paired regret differs')
    for h,d1,d2,expected in [(2,1,None,'insufficient_history'),(3,0,2,'stagnation'),(3,2,3,'same_sign'),(3,2,-3,'reversal')]:
        need(features(dict(history_length=h,d1=d1,d2=d2))['pattern']==expected,'Synthetic pattern differs')
    need([bucket(x,(1,1)) for x in (None,0,1,2)]==['unavailable','low','low','high'],'Tied-cut partition differs')
    return dict(hand_computed_entity_weight=True,hand_computed_paired_regret=True,pattern_boundaries=True,tied_cut_partition=True)

def main():
    need(not (OUTPUT / 'RESULT.json').exists(), 'Result exists; no retry/overwrite')
    pin(OUTPUT / 'PLAN.json', PLAN_SHA)
    pin(PROJECT / 'reports/titantpp_backbone_location_raf6_20261015_v1/diagnostics_elapsed_v1/analyze.py', '13e8cd6a8f056b278a6cba299885735638e8401c27bed9616ee99acc87e27b8e')
    plan = read(OUTPUT / 'PLAN.json')
    contract = read(pin(BUNDLE / 'contract.json', CONTRACT_SHA))
    need(contract['allowed_splits'] == list(SPLITS), 'Contract is not Train/Validation-only')
    need(contract['source']['closure'] == canonical(contract['source']['files']) == SOURCE_CLOSURE, 'Source closure differs')
    for rel, expected in contract['source']['files'].items():
        pin(BUNDLE / 'source' / rel, expected)
    for rel in ('operation/evaluate.py', 'operation/helper_taxi_raf24.py'):
        pin(BUNDLE / rel, contract['operation_files'][rel])
    manifest = read(pin(BUNDLE / 'retrieval/manifest.json', MANIFEST_SHA))
    need(manifest['contract_sha256'] == CONTRACT_SHA, 'Retrieval contract differs')
    host_path = BUNDLE / 'retrieval/runtime/5080/receipt.json'
    host = read(pin(host_path, manifest['files']['runtime/5080/receipt.json']))
    need(host['status'] == 'complete', 'Original inference incomplete')
    dataset = contract['datasets']['raf_spare_parts']
    path = pin(BUNDLE / dataset['path'], dataset['sha256'])
    raw = pl.scan_parquet(path).filter(pl.col('chronological_split').is_in(SPLITS)).collect()
    need(raw.height == 37469 and set(raw['chronological_split']) == set(SPLITS), 'Unexpected permitted data scope')
    definition = dataset['definition']
    need(definition['loader']['lookback_weeks'] == definition['loader']['max_seq_len'] == 84, 'Loader contract differs')
    rows, reconstruction = {}, {}
    for split in SPLITS:
        rows[split], reconstruction[split] = build_rows(raw, split, definition['inherited_data_identity']['populations'][split])
    predictions, evidence = {}, []
    jobs = [j for j in contract['jobs'] if j['dataset'] == 'raf_spare_parts']
    need(sorted(j['seed'] for j in jobs) == list(SEEDS), 'Missing CNN seed')
    for job in jobs:
        original = job['original']
        ckpt = original['checkpoint']
        pin(BUNDLE / ckpt['relative_path'], ckpt['sha256'])
        for name in ('checkpoint_metadata', 'selection_history', 'worker_exit', 'terminal_manifest'):
            pin(BUNDLE / original[name + '_path'], original[name + '_sha256'])
        meta = read(BUNDLE / original['checkpoint_metadata_path'])
        history = read(BUNDLE / original['selection_history_path'])['history']
        finite = [h for h in history if math.isfinite(h['val_qty_rmse'])]
        chosen = min(finite, key=lambda h: h['val_qty_rmse'])
        need(chosen['epoch'] == meta['best_epoch'] == original['selected_epoch'], 'First strict minimum differs')
        need(meta['model_state_sha256'] == original['checkpoint_state_sha256'], 'State binding differs')
        need(meta['resume_identity']['interface_meta']['source_files_sha256'] == SOURCE_CLOSURE, 'Metadata source differs')
        need(meta['resume_identity']['arguments']['data_sha256'] == dataset['sha256'], 'Selected input differs')
        for split in SPLITS:
            rel = f"results/{job['id']}/{split}/receipt.json"
            rec = read(pin(BUNDLE / 'retrieval' / rel, manifest['files'][rel]))
            worker = next(w for w in host['workers'] if w['job_id'] == job['id'] and w['split'] == split)
            need(worker['actual_worker_exit_code'] == 0 and worker['receipt']['sha256'] == manifest['files'][rel], 'Native worker evidence differs')
            need(rec['status'] == 'complete' and rec['contract_sha256'] == CONTRACT_SHA and rec['split'] == split
                 and rec['host'] == '5080' and rec['no_training'] and rec['parameters_unchanged']
                 and not rec['held_out_test_evaluated'], 'Prediction provenance differs')
            need(rec['checkpoint_sha256'] == ckpt['sha256'] and rec['selected_epoch'] == original['selected_epoch']
                 and rec['state_sha256'] == original['checkpoint_state_sha256'], 'Prediction checkpoint differs')
            need(rec['population'] == reconstruction[split]['population'], 'Prediction target identity differs')
            need(split != 'validation' or rec['baseline_validation_reproduced'], 'Original replay gate failed')
            paths = []
            for part in rec['prediction_manifest']['parts']:
                p = BUNDLE / 'retrieval/results' / job['id'] / split / 'predictions' / part['path']
                pin(p, part['sha256']); need(p.stat().st_size == part['bytes'], 'Prediction bytes differ'); paths.append(p)
            frame = pl.read_parquet(paths)
            need(frame.height == len(rows[split]) and set(frame['seed']) == {job['seed']}
                 and set(frame['dataset']) == {'raf_spare_parts'} and set(frame['split']) == {split}, 'Prediction scope differs')
            need(frame.select(pl.struct('entity_id', 'seq').n_unique()).item() == frame.height, 'Duplicate prediction target')
            records = {(r['entity_id'], r['seq']): r for r in frame.iter_rows(named=True)}
            common = ('raw_quantity', 'history_length', 'last_quantity', 'mean_quantity', 'history_log_mean',
                      'history_log_std', 'last_log_deviation', 'last_observed_log_change', 'previous_observation_available')
            for row in rows[split]:
                observed = records[(row['entity_id'], row['seq'])]
                for field in common:
                    need(math.isclose(row[field], observed[field], rel_tol=1e-12, abs_tol=1e-12), 'History reconstruction differs: ' + field)
                need(math.isfinite(observed['prediction_1']), 'Nonfinite baseline prediction')
            predictions[job['seed'], split] = {k: r['prediction_1'] for k, r in records.items()}
            baseline = score(rows[split], predictions[job['seed'], split])
            expected = rec['metrics']['1.0']
            need(math.isclose(baseline['rmse'], expected['qty_rmse'], rel_tol=1e-12, abs_tol=1e-12)
                 and math.isclose(baseline['mae'], expected['qty_mae'], rel_tol=1e-12, abs_tol=1e-12), 'Stored scores differ')
            evidence.append(dict(seed=job['seed'], split=split, receipt=rel, receipt_sha256=manifest['files'][rel],
                                 selected_epoch=original['selected_epoch'], binary_sha256=ckpt['sha256'],
                                 state_sha256=original['checkpoint_state_sha256'], parts=rec['prediction_manifest']['parts'],
                                 population=rec['population'], runtime=rec['runtime'], baseline_scores=baseline))

    checks=diagnostic_self_checks()
    for split in SPLITS:
        need(len(rows[split]) == plan['expected_target_count'][split], 'Target count differs from plan')
        for row in rows[split]:
            row.update(features(row))
    eligible=[r['magnitude'] for r in rows['train'] if r['magnitude'] is not None]
    cuts=tuple(float(x) for x in np.quantile(eligible,[.25,.75],method='linear'))
    for split in SPLITS:
        for row in rows[split]: row['magnitude_bin']=bucket(row['magnitude'],cuts)
    cut_info=dict(definition=plan['magnitude'],q25=cuts[0],q75=cuts[1],eligible_train_targets=len(eligible),
                  tied_cuts=cuts[0]==cuts[1],uses_target_quantity=False)
    cells=[('whole',None,None)]+[('pattern',p,None) for p in PATTERNS]
    cells += [('pattern_x_magnitude',p,b) for p in PATTERNS for b in (('unavailable',) if p=='insufficient_history' else ('low','middle','high'))]
    support=[]; scores=[]; closures=[]
    for family,pattern,mag in cells:
        cell_id='|'.join(str(v) for v in (family,pattern,mag))
        selected={split:[r for r in rows[split] if (pattern is None or r['pattern']==pattern) and (mag is None or r['magnitude_bin']==mag)] for split in SPLITS}
        entity_sets={split:{r['entity_id'] for r in selected[split]} for split in SPLITS}
        common_entities=entity_sets['train'] & entity_sets['validation']
        common_entity_digest=hashlib.sha256(json.dumps(sorted(common_entities),separators=(',',':')).encode()).hexdigest()
        for split in SPLITS:
            full=selected[split]
            retained=[r for r in full if r['entity_id'] in common_entities]
            counts=defaultdict(int)
            for r in retained: counts[r['entity_id']]+=1
            common_weights=[1./counts[r['entity_id']] for r in retained]
            need(not common_weights or math.isclose(sum(common_weights),len(common_entities),rel_tol=1e-12,abs_tol=1e-9),'Entity mass differs')
            for weighting,subset,w in (('row',full,[1.]*len(full)),('common_entity_equal',retained,common_weights)):
                support.append(dict(cell_id=cell_id,family=family,pattern=pattern,magnitude_bin=mag,split=split,weighting=weighting,
                                    rows=len(subset),entities=len({r['entity_id'] for r in subset}),
                                    original_rows=len(full),original_entities=len(entity_sets[split]),excluded_rows=len(full)-len(subset),
                                    excluded_entities=len(entity_sets[split])-len({r['entity_id'] for r in subset}),
                                    common_entities=len(common_entities),common_entity_sha256=common_entity_digest,
                                    common_entity_ids=sorted(common_entities),
                                    distributions=distributions(subset,w),weight_sum=float(sum(w))))
                for seed in SEEDS:
                    scores.append(dict(cell_id=cell_id,family=family,pattern=pattern,magnitude_bin=mag,split=split,seed=seed,
                                       weighting=weighting,**metrics(subset,predictions[seed,split],w)))
    index={(r['cell_id'],r['split'],r['seed'],r['weighting']):r for r in scores}
    # Disjoint rowweighted partition closure; common-entity summaries intentionally have different retained populations.
    for split in SPLITS:
        for seed in SEEDS:
            whole=index[('whole|None|None',split,seed,'row')]
            parts=[index[(f'pattern|{p}|None',split,seed,'row')] for p in PATTERNS]
            need(sum(x['count'] for x in parts)==whole['count'],'Pattern count closure failed')
            for metric in ('ae_sum','se_sum','signed_sum'):
                need(math.isclose(sum(x['cnn'][metric] for x in parts if x['cnn']),whole['cnn'][metric],rel_tol=1e-10,abs_tol=1e-6),'Pattern error closure failed')
            for baseline in BASELINES:
                for metric in ('ae_sum','se_sum'):
                    need(math.isclose(sum(x['regret'][baseline][metric] for x in parts if x['cnn']),whole['regret'][baseline][metric],rel_tol=1e-10,abs_tol=1e-6),'Pattern regret closure failed')
            for pattern,part in zip(PATTERNS,parts):
                bins=[r for r in scores if r['family']=='pattern_x_magnitude' and r['pattern']==pattern and r['split']==split and r['seed']==seed and r['weighting']=='row']
                need(sum(x['count'] for x in bins)==part['count'],'Magnitude count closure failed')
                if part['cnn']:
                    for metric in ('ae_sum','se_sum','signed_sum'):
                        need(math.isclose(sum(x['cnn'][metric] for x in bins if x['cnn']),part['cnn'][metric],rel_tol=1e-10,abs_tol=1e-6),'Magnitude error closure failed')
                    for baseline in BASELINES:
                        for metric in ('ae_sum','se_sum'):
                            need(math.isclose(sum(x['regret'][baseline][metric] for x in bins if x['cnn']),part['regret'][baseline][metric],rel_tol=1e-10,abs_tol=1e-6),'Magnitude regret closure failed')
            closures.append(dict(split=split,seed=seed,pattern_count_error_regret=True,magnitude_count_error_regret=True))
    aggregates=[]; consistency=[]
    for family,pattern,mag in cells:
        cell_id='|'.join(str(v) for v in (family,pattern,mag))
        gate=[]
        for split in SPLITS:
            for weighting in ('row','common_entity_equal'):
                ss=[index[(cell_id,split,seed,weighting)] for seed in SEEDS]
                def stats(v):
                    return dict(mean=statistics.mean(v),sample_sd=statistics.stdev(v)) if all(x is not None for x in v) else dict(mean=None,sample_sd=None)
                group=dict(cell_id=cell_id,family=family,pattern=pattern,magnitude_bin=mag,split=split,weighting=weighting,
                           count=ss[0]['count'],entities=ss[0]['entities'],cnn={},references={},regret={})
                for metric in ('rmse','mae','bias','mse'):
                    group['cnn'][metric]=stats([r['cnn'][metric] if r['cnn'] else None for r in ss])
                for baseline in BASELINES:
                    group['references'][baseline]=ss[0]['references'].get(baseline)
                    group['regret'][baseline]={}
                    for metric in ('ae_mean','se_mean'):
                        values=[r['regret'][baseline][metric] if r['cnn'] else None for r in ss]
                        positive=[seed for seed,value in zip(SEEDS,values) if value is not None and value>0]
                        negative=[seed for seed,value in zip(SEEDS,values) if value is not None and value<0]
                        group['regret'][baseline][metric]=dict(**stats(values),positive_seeds=positive,negative_seeds=negative)
                        gate.append(dict(split=split,weighting=weighting,baseline=baseline,metric=metric,
                                         all_three_positive=len(positive)==3,values=dict(zip(map(str,SEEDS),values))))
                aggregates.append(group)
        consistency.append(dict(cell_id=cell_id,family=family,pattern=pattern,magnitude_bin=mag,
                                strict_descriptive_entry_signal=all(x['all_three_positive'] for x in gate),details=gate,
                                architecture_review_required_even_if_true=True))
    result=dict(schema='raf_cnn_stored_paired_regret_v1',created_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                plan_sha256=PLAN_SHA,allowed_splits=list(SPLITS),source_closure=SOURCE_CLOSURE,contract_sha256=CONTRACT_SHA,
                cuts=cut_info,reconstruction=reconstruction,evidence=evidence,support=support,per_seed=scores,three_seed=aggregates,
                consistency=consistency,additive_closure=closures,self_checks=checks,inputs=INPUTS,
                limitations=plan['limitations'],inference_count=0,remote_count=0,held_out_test_read=False,
                candidate_history_decomposition_available=False,
                interpretation='Descriptive error localization only; common-entity weights do not isolate history pattern causality or establish any new CNN mechanism.')
    write(OUTPUT/'RESULT.json',result)
    lines=['# RAF 기존 CNN의 같은 대상별 기준선 대비 오류 진단','',
           '저장된 Train·Validation 예측만 사용했다. 기존 CNN의 새 추론·재평가, 학습, 원격 호출, Test 열람은 없다.',
           '양수는 같은 대상에서 CNN의 오차가 단순 기준선보다 크다는 뜻이다. 최근 관측 수량 변화 d1·d2만으로 조건을 나눴다.',
           '새 후보의 개별 예측 행은 없으므로 이번 결과는 ValueDifference−CNN 조건별 분석이 아니다.','',
           f'Train 유효 대상 {len(eligible)}개에서 정한 변화 폭 max(|d1|,|d2|) Q25={cuts[0]:.9g}, Q75={cuts[1]:.9g}를 Validation에도 그대로 적용했다.','',
           '| 조건 | split | 행 수 | 개체 수 | CNN RMSE | CNN MAE | 직전 수량 대비 ΔMAE | 직전 수량 대비 ΔMSE | 이력 평균 대비 ΔMAE | 이력 평균 대비 ΔMSE |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in aggregates:
        if g['family'] not in ('whole','pattern') or g['weighting']!='row':continue
        f=lambda v:'N/A' if v is None else f'{v:+.6f}'
        lines.append('| '+ ' | '.join([g['pattern'] or '전체',g['split'],str(g['count']),str(g['entities']),
                      f(g['cnn']['rmse']['mean']),f(g['cnn']['mae']['mean']),
                      *[f(g['regret'][b][m]['mean']) for b in BASELINES for m in ('ae_mean','se_mean')]])+' |')
    lines += ['', '## Train·Validation 공통 개체의 같은 가중치 비교','',
              '조건별 두 split에 모두 있는 개체만 남겼다. 개체 e의 n개 행은 각각 1/n의 가중치를 받는다. 각 개체의 평균 오차를 같은 비중으로 평균하며, RMSE는 개체별 MSE 평균의 제곱근이다. 이력 길이를 정확히 매칭하지 않는다.',
              '세부 변화 폭 표는 해당 조건과 변화 폭 구간 모두에서 공통인 개체만 남기므로, 조건 전체 표와 모집단이 다르다. 제외 수·개체 목록·이력 길이와 수량 분포를 JSON에 보존했다.','',
              '| 조건 | split | 남은 행/제외 | 공통 개체 | CNN MAE | 직전 수량 대비 ΔMAE | 이력 평균 대비 ΔMAE |',
              '|---|---|---:|---:|---:|---:|---:|']
    for g in aggregates:
        if g['family'] not in ('whole','pattern') or g['weighting']!='common_entity_equal':continue
        sup=next(x for x in support if x['cell_id']==g['cell_id'] and x['split']==g['split'] and x['weighting']==g['weighting'])
        lines.append(f"| {g['pattern'] or '전체'} | {g['split']} | {g['count']}/{sup['excluded_rows']} | {g['entities']} | "+' | '.join(f(g[x][y]['mean']) if x=='cnn' else f(g[x][y]['ae_mean']['mean']) for x,y in [('cnn','mae'),('regret','last_observed'),('regret','history_mean')])+' |')
    passed=[x['cell_id'] for x in consistency if x['strict_descriptive_entry_signal']]
    lines += ['', '## 판정과 한계','',
              '사전 고정한 엄격한 기술적 진입 신호: '+ ('; '.join(passed) if passed else '해당 조건 없음')+'.',
              '이 신호는 두 단순 기준선 대비 AE와 SE가 모두 Train·Validation, 3seed, 행 가중·공통 개체 가중에서 악화한 경우다. 통과하더라도 새 구조의 효과나 원인을 입증하지 않으므로 과거 Adaptive·MultiLag·LevelHistory와의 구조 검토가 먼저다.',
              '직전 수량과 이력 평균은 정답 전에 관측된 값만 사용한다. 정답 수량은 오차 및 분포 확인에만 사용했다.',
              '공통 개체 가중은 개체별 관측 수 차이만 줄인다. 수량·이력 길이·시점 분포 차이와 선택 편향은 남는다. 같은 대상을 세 seed가 공유하므로 독립 표본을 세 배로 세지 않는다.',
              '전체·조건별·변화 폭별 모든 seed, 평균·표본SD, 부호 일관성, 두 기준선 오차와 AE/SE 차이, 가중 분포 및 합계 보존 검증은 RESULT.json에 보존했다.']
    with (OUTPUT/'RESULT.md').open('x') as stream: stream.write('\n'.join(lines)+'\n')
    # Rehash every frozen input after the analysis; never alter the originals.
    for rel,digest in INPUTS.items(): need(sha(PROJECT/rel)==digest,'Input changed during diagnostic: '+rel)
    receipt=dict(status='complete',schema='raf_cnn_stored_paired_regret_receipt_v1',allowed_splits=list(SPLITS),
                 baseline_model_count=3,existing_prediction_split_count=6,source_files_verified=len(contract['source']['files']),
                 inference_count=0,remote_count=0,held_out_test_read=False,inputs=INPUTS,
                 output_files={name:sha(OUTPUT/name) for name in ('PLAN.json','analyze.py','RESULT.json','RESULT.md')},
                 strict_descriptive_entry_cells=passed,all_inputs_unchanged_after=True,
                 checks=dict(self_checks=checks,additive_closure_count=len(closures),stored_baseline_scores_reproduced=True))
    write(OUTPUT/'RECEIPT.json',receipt)
    print(json.dumps(dict(status='complete',cuts=cut_info,strict_descriptive_entry_cells=passed,
                         receipt_sha256=sha(OUTPUT/'RECEIPT.json'),artifacts=receipt['output_files']),ensure_ascii=False))


if __name__ == '__main__':
    main()
