"""Compare original, bias2 and full130 Validation at identical quantity sources."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import statistics

BUNDLE = Path(__file__).resolve().parents[1]
PROJECT = BUNDLE.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    c = read(BUNDLE/'execution_contract.json')
    receipt = read(BUNDLE/'analysis/completion_receipt.json')
    assert receipt['contract_sha256'] == canonical(c) and receipt['local_originals_SHA_verified'] is True
    assert sha(BUNDLE/'analysis/Validation_refit_comparison.json') == receipt['comparison_sha256']
    full = read(BUNDLE/'analysis/Validation_refit_comparison.json')
    prior_root = PROJECT/c['predecessor_bias2']['local_bundle']
    prior_c = read(prior_root/'execution_contract.json')
    assert canonical(prior_c) == c['predecessor_bias2']['canonical_sha256']
    prior_receipt = read(prior_root/'analysis/completion_receipt.json')
    assert prior_receipt['contract_sha256'] == canonical(prior_c)
    assert prior_receipt['local_originals_SHA_verified'] is True
    assert prior_receipt['archive_sha256'] == c['predecessor_bias2']['archive_sha256']
    assert sha(prior_root/'analysis/Validation_refit_comparison.json') == prior_receipt['comparison_sha256']
    bias = read(prior_root/'analysis/Validation_refit_comparison.json')
    for stage in (full, bias):
        assert stage['scope'] == 'Validation_development_selection' and stage['held_out_test_evaluated'] is False
        assert len(stage['rows']) == 6 and len({r['job']['canonical_original_id'] for r in stage['rows']}) == 6
        assert {(r['job']['dataset'],r['job']['seed']) for r in stage['rows']} == {
            (d,s) for d in ('yellow_trip_hourly','raf_spare_parts') for s in (42,52,62)}
    assert {r['job']['canonical_original_id'] for r in full['rows']} == {r['job']['canonical_original_id'] for r in bias['rows']}
    previous = {r['job']['canonical_original_id']: r for r in bias['rows']}
    rows, groups = [], defaultdict(list)
    for row in full['rows']:
        j, r = row['job'], row['result']
        old = previous[j['canonical_original_id']]; o = old['result']
        assert (j['dataset'],j['seed']) == (old['job']['dataset'],old['job']['seed'])
        assert r['job'] == j['id'] and o['job'] == old['job']['id']
        assert r['contract_sha256'] == canonical(c) and o['contract_sha256'] == canonical(prior_c)
        assert j['checkpoint'] == old['job']['checkpoint']
        assert r['original_checkpoint'] == o['original_checkpoint'] == j['checkpoint']
        assert r['held_out_test_evaluated'] is o['held_out_test_evaluated'] is False
        assert r['source_files_sha256'] == o['source_files_sha256'] == c['source']['files_sha256']
        assert r['trainable_scalar_count'] == 130 and o['trainable_scalar_count'] == 2
        assert r['E0'] == o['E0'], 'Different original native Validation; do not combine stages'
        assert r['quantity_prediction_sha256_before'] == r['quantity_prediction_sha256_after']
        assert r['quantity_prediction_sha256_before'] == o['quantity_prediction_sha256_before'] == o['quantity_prediction_sha256_after']
        assert r['frozen_state_sha256_before'] == r['frozen_state_sha256_after']
        assert row['all_owned_originals_SHA_verified'] and old['all_owned_originals_SHA_verified']
        d = {'dataset':j['dataset'], 'seed':j['seed'], 'canonical_original_id':j['canonical_original_id'],
            'original_checkpoint':j['checkpoint'], 'original_quantity_selected_epoch':j['checkpoint']['epoch'],
            'bias2_selected_epoch':o['best_epoch'], 'bias2_last_epoch':o['completed_epochs'],
            'full130_selected_epoch':r['best_epoch'], 'full130_last_epoch':r['completed_epochs'],
            'original_validation':r['E0'], 'bias2_validation':o['selected'], 'full130_validation':r['selected'],
            'full130_identity_selected':r['selected_is_identity'],
            'quantity_byteSHA_identical_across_all_stages':True,
            'full130_frozen_weight_buffer_SHA_unchanged':True}
        original, fitted, previous_nll = r['E0']['time_nll'], r['selected']['time_nll'], o['selected']['time_nll']
        d['full130_relative_change_percent'] = 100*(fitted/original-1)
        d['full130_vs_bias2_relative_change_percent'] = 100*(fitted/previous_nll-1)
        rows.append(d); groups[j['dataset']].append(d)
    aggregates = []
    for dataset, items in groups.items():
        assert sorted(r['seed'] for r in items) == [42,52,62]
        a = {'dataset':dataset, 'n_seeds':3,
            'full130_improved_seeds':[r['seed'] for r in items if r['full130_validation']['time_nll'] < r['original_validation']['time_nll']]}
        for stage in ('original','bias2','full130'):
            metrics = [r[stage+'_validation'] for r in items]
            for metric in ('time_nll','qty_rmse','qty_mae'):
                values = [m[metric] for m in metrics]
                a[stage+'_'+metric+'_mean'] = statistics.mean(values)
                a[stage+'_'+metric+'_sample_std'] = statistics.stdev(values)
            for metric in ('time_nll','qty_rmse','qty_mae'):
                values = [m['tail'][metric] for m in metrics]
                a[stage+'_tail_'+metric+'_mean'] = statistics.mean(values)
                a[stage+'_tail_'+metric+'_sample_std'] = statistics.stdev(values)
        a['full130_relative_mean_change_percent'] = 100*(a['full130_time_nll_mean']/a['original_time_nll_mean']-1)
        a['full130_vs_bias2_relative_mean_change_percent'] = 100*(a['full130_time_nll_mean']/a['bias2_time_nll_mean']-1)
        aggregates.append(a)
    assert len(rows) == 6 and len(aggregates) == 2
    output = {'scope':'Validation_development_selection','new_full130_refits':6,
        'same_original_quantity_checkpoints':True, 'full_Train_Validation_quantity_exactly_unchanged':True,
        'Train_used_for_fitting':True, 'held_out_Test_evaluated':False,
        'independent_calibration':False, 'S2P2_matching_refit':'not_performed',
        'CPUbinary_reinference_audit':'not_performed', 'rows':rows, 'aggregates':aggregates}
    path = BUNDLE/'analysis/Validation_three_stage_comparison.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'scope':output['scope'], 'aggregates':aggregates},ensure_ascii=False))


if __name__ == '__main__': main()
