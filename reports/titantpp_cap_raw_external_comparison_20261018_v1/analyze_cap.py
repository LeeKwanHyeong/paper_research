"""Read completed capped summaries; recompute mean/SD without inference."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / 'reports/titantpp_mlp_cap36_20261017_v2'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    plan = json.loads((OUT / 'ANALYSIS_PLAN.json').read_text())
    receipt_path = ROOT / 'search_artifacts/titantpp_mlp_cap36_20261017_v2/retrieval/final_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'complete' and receipt['candidate_test_count'] == 36
    assert receipt['validation_replay_count'] == 36 and receipt['seed_count'] == 3
    inputs = {str(receipt_path.relative_to(ROOT)): sha(receipt_path)}
    rows, bindings, targets, checks = [], {}, {}, []
    for split, file in [('validation', 'THREE_SEED_VALIDATION.json'), ('test', 'THREE_SEED_TEST.json')]:
        path = PARENT / file
        rel = str(path.relative_to(ROOT))
        assert sha(path) == plan['cap_inputs'][rel]
        assert receipt['report_files'][str(path)] == sha(path)
        inputs[rel] = sha(path)
        data = json.loads(path.read_text())
        assert data['split'] == split and data['seed_count'] == 3 and data['sample_sd_ddof'] == 1
        for ds in data['datasets']:
            targets[ds['dataset'], split] = ds['target_identity']
            for arm in ds['arms']:
                assert arm['arm'] in plan['arms']
                seeds = sorted(arm['seeds'], key=lambda r: r['seed'])
                assert [r['seed'] for r in seeds] == [42, 52, 62]
                values = {k: [] for k in ['qty_rmse', 'qty_mae', 'time_nll']}
                for r in seeds:
                    assert r['target_identity'] == ds['target_identity']
                    m = r['metrics']['overall']
                    assert m['count'] == ds['target_identity']['target_count']
                    common = dict(task='cap', dataset=ds['dataset'], model=arm['arm'], label=arm['label'],
                                  split=split, cap_value=ds['cap_value'], seed_count=3, count_per_seed=m['count'])
                    row = dict(**common, row_type='seed', seed=r['seed'], selected_epoch=r['selected_epoch'],
                               checkpoint_sha256=r['checkpoint_sha256'], state_sha256=r['state_sha256'],
                               target_identity_sha256=r['target_identity']['target_identity_sha256'],
                               target_quantity_sha256=r['target_identity']['target_quantity_sha256'], host=r['host'])
                    for k in values:
                        assert math.isfinite(m[k])
                        values[k].append(m[k]); row[k] = m[k]
                    rows.append(row)
                    key = ds['dataset'], arm['arm'], r['seed']
                    identity = (r['selected_epoch'], r['checkpoint_sha256'], r['state_sha256'])
                    if split == 'validation':
                        bindings[key] = identity
                    else:
                        assert bindings[key] == identity
                summary = dict(**common, row_type='summary')
                for k, v in values.items():
                    mean, sd = st.mean(v), st.stdev(v)
                    old = arm['groups']['overall']['metrics'][k]
                    assert math.isclose(mean, old['mean'], rel_tol=1e-12, abs_tol=1e-12)
                    assert math.isclose(sd, old['sample_sd'], rel_tol=1e-12, abs_tol=1e-12)
                    summary[k + '_mean'], summary[k + '_sample_sd'] = mean, sd
                    for r in seeds:
                        summary[k + '_seed' + str(r['seed'])] = r['metrics']['overall'][k]
                rows.append(summary)
                checks.append(dict(dataset=ds['dataset'], model=arm['arm'], split=split, metrics_recomputed=3))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / 'CAP_ROWS.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); w.writeheader(); w.writerows(rows)
    result = dict(status='PASS', input_files=inputs, seed_rows=72, summary_rows=24,
                  mean_sd_checks=144, checkpoint_validation_test_bindings=len(bindings),
                  checks=checks, target_identities=[dict(dataset=d, split=s, **v) for (d,s),v in targets.items()],
                  cap_external_results_used=False, new_training=False, new_inference=False, remote_server_calls=0)
    (OUT / 'CAP_RECEIPT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ['status','seed_rows','summary_rows','checkpoint_validation_test_bindings']}))


if __name__ == '__main__':
    main()
