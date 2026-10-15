"""Pure, fixed-cell comparison of CURRENT candidate and stored CNN rows.

No neural-model import/inference, remote operation, threshold search or Test support.
The calling collector must verify receipt/source/checkpoint/input/row-shard SHA first.
"""
from collections import Counter
import math
import statistics

SPLITS = ('train', 'validation')
SEEDS = (42, 52, 62)
COMMON = ('raw_quantity', 'history_length', 'last_quantity', 'mean_quantity',
          'history_log_mean', 'history_log_std', 'last_log_deviation',
          'last_observed_log_change', 'previous_observation_available')
PATTERNS = ('insufficient_history', 'stagnation', 'same_sign', 'reversal')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(row):
    require(row['split'] in SPLITS and row['seed'] in SEEDS, 'Unapproved split/seed')
    require(row['dataset'] == 'raf_spare_parts', 'Unapproved dataset')
    return row['dataset'], row['seed'], row['split'], str(row['entity_id']), int(row['seq'])


def paired_rows(candidate, baseline, history, *, seed, split, expected_count):
    """One-to-one identity + history join; target_id hash domains are NOT conflated."""
    require(split in SPLITS and seed in SEEDS, 'Unapproved cohort')
    require(len(candidate) == len(baseline) == len(history) == expected_count, 'Count differs')
    cb, bb, hb = {}, {}, {}
    for rows, index in ((candidate, cb), (baseline, bb)):
        for row in rows:
            k = key(row)
            require(k[1:3] == (seed, split) and k not in index, 'Duplicate or mixed target')
            index[k] = row
    for row in history:
        k = ('raf_spare_parts', seed, split, str(row['entity_id']), int(row['seq']))
        require(row['split'] == split and k not in hb, 'Duplicate or mixed reconstructed history')
        hb[k] = row
    require(cb.keys() == bb.keys() == hb.keys(), 'Target sets differ')
    out = []
    for k, c in cb.items():
        b, h = bb[k], hb[k]
        for field in COMMON:
            for other in (b, h):
                require(math.isfinite(float(c[field])) and math.isfinite(float(other[field]))
                        and math.isclose(float(c[field]), float(other[field]), rel_tol=1e-12, abs_tol=1e-12),
                        'Observed history or target differs: ' + field)
        q, cp, bp, ct, bt = (float(c['raw_quantity']), float(c['prediction']),
                            float(b['prediction_1']), float(c['time_nll']), float(b['time_nll_1']))
        require(all(math.isfinite(x) for x in (q, cp, bp, ct, bt)), 'Nonfinite prediction')
        ce, be = cp-q, bp-q
        require(all(math.isfinite(x) for x in (ce, be, ce*ce, be*be)), 'Nonfinite derived error')
        for field, expected in (('signed_error', ce), ('absolute_error', abs(ce)), ('squared_error', ce*ce)):
            require(math.isclose(float(c[field]), expected, rel_tol=1e-12, abs_tol=1e-12), 'Stored error differs')
        out.append({**h, 'seed': seed, 'candidate_prediction': cp, 'cnn_prediction': bp,
                    'candidate_time_nll': ct, 'cnn_time_nll': bt,
                    'candidate_error': ce, 'cnn_error': be,
                    'delta_ae': abs(ce)-abs(be), 'delta_se': ce*ce-be*be,
                    'delta_signed': ce-be})
    return out


def fixed_features(row, cuts):
    require(len(cuts) == 2 and all(math.isfinite(v) for v in cuts) and cuts[0] <= cuts[1], 'Invalid frozen cuts')
    if row['history_length'] < 3:
        return 'insufficient_history', 'unavailable'
    d1, d2 = row['d1'], row['d2']
    require(d1 is not None and d2 is not None and all(math.isfinite(v) for v in (d1, d2)), 'Missing observed differences')
    pattern = 'stagnation' if d1 == 0 or d2 == 0 else 'same_sign' if d1*d2 > 0 else 'reversal'
    mag = max(abs(d1), abs(d2))
    return pattern, 'low' if mag <= cuts[0] else 'middle' if mag <= cuts[1] else 'high'


def cell_list():
    return [('whole', None, None)] + [('pattern', p, None) for p in PATTERNS] + [
        ('pattern_x_magnitude', p, b) for p in PATTERNS
        for b in (('unavailable',) if p == 'insufficient_history' else ('low', 'middle', 'high'))]


def scores(rows, weights):
    require(len(rows) == len(weights) and all(math.isfinite(w) and w > 0 for w in weights), 'Invalid weights')
    if not rows:
        return None
    mass = math.fsum(weights)
    result = {'count': len(rows), 'entities': len({str(r['entity_id']) for r in rows}), 'weight_sum': mass}
    for label in ('candidate', 'cnn'):
        e = [r[label+'_error'] for r in rows]
        require(all(math.isfinite(v) and math.isfinite(v*v) for v in e), 'Nonfinite aggregate error')
        ae, se, signed = [math.fsum(w*v for w, v in zip(weights, values)) for values in
                          ([abs(x) for x in e], [x*x for x in e], e)]
        result[label] = {'mae': ae/mass, 'mse': se/mass, 'rmse': math.sqrt(se/mass),
                         'bias': signed/mass, 'ae_sum': ae, 'se_sum': se, 'signed_sum': signed,
                         'time_nll': math.fsum(w*r[label+'_time_nll'] for w, r in zip(weights, rows))/mass}
    result['difference'] = {}
    for label, ref in (('ae', 'ae_sum'), ('se', 'se_sum'), ('signed', 'signed_sum')):
        total = math.fsum(w*r['delta_'+label] for w, r in zip(weights, rows))
        require(math.isclose(total, result['candidate'][ref]-result['cnn'][ref], rel_tol=1e-9, abs_tol=1e-7), 'Pair arithmetic differs')
        result['difference'][label+'_sum'] = total
        result['difference'][label+'_mean'] = total/mass
    result['difference']['mae'] = result['candidate']['mae']-result['cnn']['mae']
    result['difference']['rmse'] = result['candidate']['rmse']-result['cnn']['rmse']
    for metric in ('mae', 'rmse'):
        base = result['cnn'][metric]
        result['difference'][metric+'_relative_percent'] = 100*result['difference'][metric]/base if base else None
    result['delta_ae_positive_fraction'] = math.fsum(w*(r['delta_ae'] > 0) for r, w in zip(rows, weights))/mass
    return result


def summarize(rows_by_seed_split, cuts):
    """Frozen 15 cells, row weighting and split-common entity weighting; no adaptive cuts."""
    require(set(rows_by_seed_split) == {(s, p) for s in SEEDS for p in SPLITS}, 'Incomplete cohort')
    output, support, closure = [], [], []
    for family, pattern, magnitude in cell_list():
        cell = '|'.join(str(v) for v in (family, pattern, magnitude))
        chosen = {}
        for (seed, split), rows in rows_by_seed_split.items():
            chosen[seed, split] = [r for r in rows if (pattern is None or fixed_features(r, cuts)[0] == pattern)
                                  and (magnitude is None or fixed_features(r, cuts)[1] == magnitude)]
        for seed in SEEDS:
            common = {str(r['entity_id']) for r in chosen[seed, 'train']} & {str(r['entity_id']) for r in chosen[seed, 'validation']}
            for split in SPLITS:
                full = chosen[seed, split]
                kept = [r for r in full if str(r['entity_id']) in common]
                counts = Counter(str(r['entity_id']) for r in kept)
                weighted = [1/counts[str(r['entity_id'])] for r in kept]
                require(not kept or math.isclose(math.fsum(weighted), len(common)), 'Unequal entity mass')
                support.append({'cell': cell, 'seed': seed, 'split': split, 'original_rows': len(full),
                                'retained_rows': len(kept), 'excluded_rows': len(full)-len(kept),
                                'common_entities': len(common)})
                for weighting, subset, weights in [('row', full, [1.]*len(full)), ('common_entity_equal', kept, weighted)]:
                    output.append({'cell': cell, 'family': family, 'pattern': pattern, 'magnitude_bin': magnitude,
                                   'seed': seed, 'split': split, 'weighting': weighting, 'metrics': scores(subset, weights)})
    for seed in SEEDS:
        for split in SPLITS:
            chosen = [o for o in output if o['seed'] == seed and o['split'] == split and o['weighting'] == 'row']
            whole = next(o['metrics'] for o in chosen if o['family'] == 'whole')
            for family in ('pattern', 'pattern_x_magnitude'):
                members = [o['metrics'] for o in chosen if o['family'] == family and o['metrics'] is not None]
                require(sum(m['count'] for m in members) == whole['count'], 'Partition count differs')
                for metric in ('ae_sum', 'se_sum', 'signed_sum'):
                    actual = math.fsum(m['difference'][metric] for m in members)
                    require(math.isclose(actual, whole['difference'][metric], rel_tol=1e-9, abs_tol=1e-7), 'Partition error closure differs')
                closure.append({'seed': seed, 'split': split, 'partition': family, 'passed': True})
    means = []
    for family, pattern, magnitude in cell_list():
        cell = '|'.join(str(v) for v in (family, pattern, magnitude))
        for split in SPLITS:
            for weighting in ('row', 'common_entity_equal'):
                group = [o['metrics'] for o in output if o['cell'] == cell and o['split'] == split and o['weighting'] == weighting]
                if all(m is not None for m in group):
                    aggregate = {}
                    for kind in ('candidate', 'cnn', 'difference'):
                        aggregate[kind] = {}
                        for metric in group[0][kind]:
                            values = [m[kind][metric] for m in group]
                            aggregate[kind][metric] = {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)} if all(v is not None for v in values) else None
                    means.append({'cell': cell, 'split': split, 'weighting': weighting, 'metrics': aggregate})
    return {'per_seed': output, 'three_seed': means, 'support': support, 'additive_closure': closure,
            'held_out_test_evaluated': False, 'new_inference': False, 'causal_effect_established': False,
            'limitation': 'Row pairing isolates prediction differences on identical targets; it does not isolate Runtime, optimization or parameterization effects.'}
