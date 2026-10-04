"""Verify retrieved native Deep Renewal predictions; no model loading or inference."""
import csv
import json
import math

import numpy as np
import polars as pl

from pipeline import HERE, ROOT, check_prediction, identity, load_predictions, metrics, read, sha, write

TIME_DEFINITION = ('recorded_positive_integer_shifted_negative_binomial_mass_nll; '
                   'top-coded survival where configured')
METRICS = ('count', 'qty_mae', 'qty_rmse', 'qty_bias', 'time_nll',
           'qty_absolute_error_sum', 'qty_signed_error_sum', 'qty_sse', 'time_nll_sum')


def stratify(frame, quantity_boundaries, history_boundaries):
    threshold = quantity_boundaries[-1]
    groups = [('overall', frame, None, None),
              ('tail', frame.filter(pl.col('raw_quantity') > threshold), threshold, 'raw_quantity > threshold'),
              ('body', frame.filter(pl.col('raw_quantity') <= threshold), threshold, 'raw_quantity <= threshold')]
    for name, column, boundaries in [('quantity', 'raw_quantity', quantity_boundaries),
                                     ('history', 'history_length', history_boundaries)]:
        bins = np.searchsorted(boundaries, frame[column].to_numpy(), side='left')
        for index in range(len(boundaries)+1):
            groups.append((f'{name}_bin_{index}', frame.filter(pl.Series(bins == index)),
                           json.dumps(boundaries), 'searchsorted(side=left); equality in lower bin'))
    results=[]
    for name, subset, threshold, rule in groups:
        value=metrics(subset) if len(subset) else {k:0 if k=='count' else None for k in METRICS}
        results.append(dict(view=name,threshold=threshold,threshold_rule=rule,**value))
    full=results[0]
    for members in ([r for r in results if r['view'] in ('tail','body')],
                    [r for r in results if r['view'].startswith('quantity_bin_')],
                    [r for r in results if r['view'].startswith('history_bin_')]):
        assert sum(r['count'] for r in members)==full['count']
        for key in ('qty_absolute_error_sum','qty_signed_error_sum','qty_sse','time_nll_sum'):
            assert math.isclose(sum(r[key] or 0 for r in members),full[key],rel_tol=1e-10,abs_tol=1e-7),(key,members)
    return results


def main():
    registry=read(HERE/'evaluation_registry.json')
    contract=read(HERE/'execution_contract.json')
    completion=read(HERE/'inference_completion.json')
    gate=read(HERE/'validation_gate.json')
    retrieval=read(HERE/'retrieval_receipt.json')
    assert completion['status']=='complete' and completion['new_split_count']==24
    assert completion['contract_sha256']==sha(HERE/'execution_contract.json')
    assert gate['passed'] and gate['conditions']==12
    assert gate['contract_sha256']==sha(HERE/'execution_contract.json')
    assert retrieval['status']=='verified'
    expected={(identity(r),s) for r in registry['rows'] for s in ('validation','test')}
    assert {(r['condition'],r['split']) for r in completion['completed']}==expected
    assert len(completion['completed'])==24
    rows,evidence=[],[]
    for row in registry['rows']:
        assert sha(ROOT/row['checkpoint_path'])==row['checkpoint_file_sha256']
        spec=next(d for d in registry['bundles'][row['evaluator_source_bundle']]['datasets'] if d['dataset_id']==row['dataset'])
        for split in ('validation','test'):
            folder=HERE/'runs'/split/identity(row)
            verification=check_prediction(folder,row,split,contract)
            finished=next(r for r in completion['completed'] if (r['condition'],r['split'])==(identity(row),split))
            for key in ('receipt_sha256','target_identity_sha256','truth_sha256'):
                assert verification[key]==finished[key]
            assert verification['metrics']['count']==finished['metrics']['count']
            for key,value in verification['metrics'].items():
                if key!='count':assert math.isclose(value,finished['metrics'][key],rel_tol=1e-12,abs_tol=1e-9),(identity(row),split,key)
            receipt,frame=load_predictions(folder)
            assert receipt['source_closure_sha256']==row['source_closure_sha256']
            assert receipt['evaluator_source_bundle']==row['evaluator_source_bundle']
            assert frame['target_id'].n_unique()==frame.height
            assert frame['split'].unique().to_list()==[split]
            path=str((folder/'receipt.json').relative_to(ROOT))
            common=dict(dataset=row['dataset'],model=row['model'],seed=row['seed'],split=split,
                selected_epoch=row['selected_epoch'],source=path,source_sha256=sha(folder/'receipt.json'),
                receipt_path=path,receipt_sha256=sha(folder/'receipt.json'),
                time_nll_definition=TIME_DEFINITION,comparison_family='native_deep_renewal_research_only',
                provenance='reaggregated_SHA_verified_full_population_predictions',reused_predictions=False)
            values=stratify(frame,spec['quantity_boundaries_all_train_rows'],spec['history_boundaries'])
            rows.extend({**common,**v} for v in values)
            evidence.append({**common,'target_identity_sha256':receipt['target_identity_sha256'],
                'truth_sha256':receipt['truth_sha256'],'source_closure_sha256':receipt['source_closure_sha256'],
                'checkpoint_file_sha256':receipt['checkpoint_file_sha256'],
                'state_tensor_sha256':receipt['state_tensor_sha256'],'parts':receipt['parts']})
    assert len(evidence)==24
    for filename,subset in [('metrics.csv',rows),('overall.csv',[r for r in rows if r['view']=='overall']),
                            ('comparison_input.csv',[r for r in rows if r['view'] in ('overall','tail')])]:
        with (HERE/filename).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(subset[0]));writer.writeheader();writer.writerows(subset)
    write('analysis_receipt.json',dict(status='complete',scientific_conditions=12,split_count=24,
        new_split_count=24,reused_split_count=0,training=False,selection_changed=False,
        manuscript_comparison_included=False,contract_sha256=sha(HERE/'execution_contract.json'),
        metrics_sha256=sha(HERE/'metrics.csv'),overall_sha256=sha(HERE/'overall.csv'),
        comparison_input_sha256=sha(HERE/'comparison_input.csv'),metric_rows=len(rows),evidence=evidence,
        all_strata_reconstruct_overall=True,quantity_bins='searchsorted(side=left); equality belongs to lower bin',
        tail='raw_quantity > final fixed TRAIN quantity boundary',time_nll_definition=TIME_DEFINITION,
        scope='Retrospective frozen Validation/Test populations; not untouched independent evaluation',
        remaining_original_last_checkpoint_CPU_audit='Not performed: this scope binds selected checkpoints and selected predictions only.'))
    print(json.dumps(dict(status='complete',split_count=24,metric_rows=len(rows))))


if __name__=='__main__':main()
