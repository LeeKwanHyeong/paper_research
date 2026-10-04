"""Independent scope, evidence and aggregation checks; no inference or main edits."""
from pathlib import Path
from collections import defaultdict
import copy
import csv
import hashlib
import importlib.util
import json
import math
import sys

import pytest

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SPEC=importlib.util.spec_from_file_location('independent_summary_target',HERE/'summarize.py')
s=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(s)
SPLITS=('validation','test')
VIEWS=('overall','tail')
SEEDS=(42,52,62)
THRESHOLDS={'yellow_trip_hourly':3449,'intermittent_frozen_5000':187,'insta_market_basket':35,'raf_spare_parts':200}
COMMON_TIME='recorded_positive_integer_lognormal_round_clamp_mass_nll; top-coded survival where configured'
NATIVE_TIME='recorded_positive_integer_shifted_negative_binomial_mass_nll; top-coded survival where configured'
REFS=('last_observed_quantity','mean_quantity_in_same_observed_window')

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def records(path):return list(csv.DictReader(Path(path).open()))
def condition(row):return row['dataset'],row['model'],int(row['seed'])
def identifier(row):return row['dataset'],row['model'],row['seed'],row['split'],row['view']

def registries():
    a100_folder='a100_attempt2' if (HERE/'a100_attempt2/evaluation_registry.json').exists() else 'a100'
    paths=[ROOT/'reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json',
           ROOT/'reports/titantpp_b_test_evaluation_20261003_v1/evaluation_registry.json',
           *(HERE/folder/'evaluation_registry.json' for folder in ('width16','deep_renewal',a100_folder))]
    rows={};sources={}
    for path in paths:
        registry=read(path)
        for row in registry['rows']:
            key=condition(row);assert key not in rows
            rows[key]=row;sources[key]=(path,registry['bundles'][row['evaluator_source_bundle']])
    return rows,sources

def input_paths():
    if (HERE/'comparison_receipt.json').exists():
        receipt=read(HERE/'comparison_receipt.json')
        for row in receipt['inputs']:assert sha(ROOT/row['path'])==row['sha256']
        return [ROOT/row['path'] for row in receipt['inputs']]
    paths=[HERE/'existing/metrics_per_seed.csv',HERE/'width16/metrics.csv',HERE/'deep_renewal/comparison_input.csv']
    for name in ('comparison_input.csv','metrics.csv','metrics_per_seed.csv'):
        path=HERE/'a100'/name
        if path.exists():paths.append(path);break
    return paths

def inputs():
    return [s.normalize(row) for p in input_paths() for row in records(p)
            if (row.get('view') or 'overall') in ('overall','tail','all')]

def expected_keys():
    rows,_=registries()
    expected={(*key,split,view) for key in rows for split in SPLITS for view in VIEWS}
    expected.update((d,m,None,split,view) for d in THRESHOLDS for m in REFS for split in SPLITS for view in VIEWS)
    return expected

def test_registry_derived_scope_is_153_selected_conditions_with_no_new_seed():
    rows,_=registries()
    assert len(rows)==153
    assert all(r['endpoint']=='selected' for r in rows.values())
    a100={k for k in rows if k[1] in s.A100_MODELS}
    assert len(a100)==12 and {k[2] for k in a100}=={42}
    assert {k[0] for k in a100}==set(THRESHOLDS)-{'insta_market_basket'}
    assert len({k for k in rows if k[1]=='deep_renewal_event_native_nb'})==12
    assert len({k for k in rows if k[1]=='titantpp'})==9
    assert len(expected_keys())==644


def test_available_rows_exactly_match_declared_registry_coverage_and_tail_populations():
    rows=inputs();actual=[identifier(r) for r in rows]
    assert len(actual)==len(set(actual))
    missing=expected_keys()-set(actual)
    if len(input_paths())==3:
        assert len(missing)==48 and all(k[1] in s.A100_MODELS for k in missing)
    else:
        assert not missing
        assert s.validate_scope(rows)['exact_condition_split_view_coverage']
    assert not (set(actual)-expected_keys())
    populations=read(HERE/'existing/population_bindings.json')
    groups=defaultdict(set)
    for row in rows:
        groups[(row['dataset'],row['split'],row['view'])].add(row['count'])
        if row['view']=='overall':assert row['count']==populations[row['dataset']+'__'+row['split']]['prediction_rows']
        else:
            assert row['threshold']==THRESHOLDS[row['dataset']]
            assert row['threshold_rule']=='raw_quantity > frozen_train_threshold'
    assert all(len(counts)==1 for counts in groups.values())


def test_source_SHA_selected_state_and_common_population_bindings():
    registered,_=registries();populations=read(HERE/'existing/population_bindings.json')
    checked={};a100_frames={}
    for row in inputs():
        path=ROOT/(row.get('source') or row['receipt_path'])
        expected=row.get('source_sha256') or row['receipt_sha256']
        if path not in checked:checked[path]=sha(path)
        assert checked[path]==expected,(row['model'],str(path))
        artifact=read(path)
        if row['seed'] is not None:
            bound=registered[condition(row)]
            assert int(row['selected_epoch'])==bound['selected_epoch']
        if row.get('receipt_path'):
            pop=populations[row['dataset']+'__'+row['split']]
            assert artifact['full_population'] and artifact['status']=='complete'
            assert artifact['split']==row['split']
            for field in ('data_file_sha256','target_identity_sha256','truth_sha256','loader'):
                assert artifact[field]==pop[field],(condition(row) if row['seed'] is not None else row['model'],field)
            if row['seed'] is not None:
                for field in ('checkpoint_file_sha256','state_tensor_sha256','selected_epoch'):
                    assert artifact[field]==bound[field]
                assert (artifact['dataset'],artifact['model'],artifact['seed'])==condition(row)
            if row['model'] in s.A100_MODELS:
                import polars as pl
                if path not in a100_frames:
                    for part in artifact['parts']:assert sha(path.parent/part['path'])==part['sha256']
                    a100_frames[path]=pl.read_parquet([path.parent/part['path'] for part in artifact['parts']])
                frame=a100_frames[path]
                if row['view']=='tail':frame=frame.filter(pl.col('raw_quantity')>THRESHOLDS[row['dataset']])
                error=pl.col('predicted_raw_quantity')-pl.col('raw_quantity')
                values=frame.select(pl.len().alias('count'),error.abs().mean().alias('qty_mae'),
                    error.pow(2).mean().sqrt().alias('qty_rmse'),pl.col('time_nll').mean()).row(0,named=True)
                assert values['count']==row['count']
                for field in s.METRICS:assert math.isclose(values[field],row[field],rel_tol=1e-12,abs_tol=1e-9),(condition(row),row['view'],field)
        else:
            endpoint=artifact['selected']
            assert endpoint['state_sha256']==bound['state_tensor_sha256']
            assert endpoint['evaluation_scope']=='validation_only' and endpoint['held_out_test_evaluated'] is False
            if row['view']=='tail':endpoint=endpoint['tail']
            assert endpoint['count']==row['count']
            for field in s.METRICS:assert math.isclose(endpoint[field],row[field],rel_tol=1e-12,abs_tol=1e-9)


def test_time_scores_have_same_recorded_integer_support_with_explicit_native_family():
    registered,sources=registries()
    expected_units={'yellow_trip_hourly':'hour','intermittent_frozen_5000':'week','insta_market_basket':'day','raf_spare_parts':'month'}
    for row in inputs():
        if row['seed'] is None:
            assert row['time_nll'] is None and row['time_nll_definition']=='not_applicable_quantity_only'
            continue
        definition=NATIVE_TIME if row['model']=='deep_renewal_event_native_nb' else COMMON_TIME
        assert row['time_nll_definition']==definition
        _,bundle=sources[condition(row)]
        spec=next(d for d in bundle['datasets'] if d['dataset_id']==row['dataset'])
        observation=spec['model']['time_observation_contract']
        assert observation=={'mode':'positive_integer_round_clamp_v1','unit':expected_units[row['dataset']],
                             'top_code':30 if row['dataset']=='insta_market_basket' else None}
    assert read(HERE/'deep_renewal/evaluation_registry.json')['manuscript_comparison_included'] is False


def test_every_available_real_group_mean_and_sample_SD_matches_independent_arithmetic():
    rows=inputs();groups=defaultdict(list)
    for row in rows:groups[(row['dataset'],row['model'],row['split'],row['view'])].append(row)
    result=s.aggregate(rows)
    assert len(result)==len(groups)
    for row in result:
        values=groups[(row['dataset'],row['model'],row['split'],row['view'])]
        for metric in s.METRICS:
            x=[v[metric] for v in values]
            if x[0] is None:
                assert row[metric+'_mean'] is None and row[metric+'_sample_sd'] is None
                continue
            mean=math.fsum(x)/len(x)
            assert math.isclose(mean,row[metric+'_mean'],rel_tol=1e-12,abs_tol=1e-12)
            if len(x)==1:assert row[metric+'_sample_sd'] is None
            else:
                sd=math.sqrt(math.fsum((value-mean)**2 for value in x)/(len(x)-1))
                assert math.isclose(sd,row[metric+'_sample_sd'],rel_tol=1e-12,abs_tol=1e-12)
        if values[0]['seed'] is None:assert row['n_seeds']==0
    if (HERE/'comparison_receipt.json').exists():
        receipt=read(HERE/'comparison_receipt.json')
        for path,digest in receipt['outputs'].items():assert sha(HERE/path)==digest
        per_seed=records(HERE/'results/metrics_per_seed.csv')
        result_key=lambda r:(r['dataset'],r['model'],int(r['seed']) if r['seed'] else None,r['split'],r['view'])
        indexed={result_key(r):r for r in per_seed}
        assert len(per_seed)==len(indexed)==len(rows)==644
        for row in rows:
            actual=indexed[identifier(row)]
            for field,value in row.items():assert actual[field]==('' if value is None else str(value)),(identifier(row),field)
        saved=records(HERE/'results/metrics_aggregated.csv')
        saved_index={(r['dataset'],r['model'],r['split'],r['view']):r for r in saved}
        assert len(saved)==len(saved_index)==len(result)==268
        for row in result:
            actual=saved_index[(row['dataset'],row['model'],row['split'],row['view'])]
            for field,value in row.items():
                if isinstance(value,float):assert math.isclose(float(actual[field]),value,rel_tol=1e-12,abs_tol=1e-12)
                else:assert actual[field]==('' if value is None else str(value))
        # Validate all comparison deltas independently from the input per-seed values.
        for delta in records(HERE/'results/all_vs_width4.csv'):
            selected=[r for r in rows if (r['dataset'],r['split'],r['view'])==(delta['dataset'],delta['split'],delta['view'])]
            if delta['model'] in s.A100_MODELS:
                assert delta['comparison']=='paired_seed42'
                selected=[r for r in selected if r['seed']==42]
            else:assert delta['comparison']=='three_seed_mean'
            a=[r for r in selected if r['model']==delta['model']]
            b=[r for r in selected if r['model']=='titantpp_history_mlp']
            assert len(a)==len(b)==(1 if delta['model'] in s.A100_MODELS else 3)
            for metric in s.METRICS:
                av=math.fsum(r[metric] for r in a)/len(a);bv=math.fsum(r[metric] for r in b)/len(b)
                assert math.isclose(float(delta[metric+'_difference']),av-bv,rel_tol=1e-12,abs_tol=1e-12)
                if bv:assert math.isclose(float(delta[metric+'_relative_change_percent']),100*(av/bv-1),rel_tol=1e-12,abs_tol=1e-12)
                else:assert delta[metric+'_relative_change_percent']==''
        pairs=records(HERE/'results/all_vs_width4.csv')
        assert len(pairs)==220 and sum(r['comparison']=='paired_seed42' for r in pairs)==48
        widths=records(HERE/'results/width16_vs_width4.csv');assert len(widths)==8
        for width in widths:
            equivalent=next(r for r in pairs if (r['dataset'],r['split'],r['view'],r['model'])==
                            (width['dataset'],width['split'],'overall','titantpp_history_mlp_width16'))
            for metric in s.METRICS:
                assert width[metric+'_width16_minus_width4']==equivalent[metric+'_difference']
                assert width[metric+'_relative_change_percent']==equivalent[metric+'_relative_change_percent']
        singles=records(HERE/'results/width16_vs_width4_per_seed.csv')
        assert len(singles)==48
        seen=set()
        for delta in singles:
            k=(delta['dataset'],int(delta['seed']),delta['split'],delta['view']);assert k not in seen;seen.add(k)
            pair={r['model']:r for r in rows if (r['dataset'],r['seed'],r['split'],r['view'])==k
                  and r['model'] in ('titantpp_history_mlp','titantpp_history_mlp_width16')}
            base=pair['titantpp_history_mlp'];candidate=pair['titantpp_history_mlp_width16']
            assert int(delta['count'])==base['count']==candidate['count']
            for metric in s.METRICS:
                a,b=candidate[metric],base[metric]
                assert float(delta[metric+'_width4'])==b and float(delta[metric+'_width16'])==a
                assert math.isclose(float(delta[metric+'_difference']),a-b,rel_tol=1e-12,abs_tol=1e-12)
                if b:assert math.isclose(float(delta[metric+'_relative_change_percent']),100*(a/b-1),rel_tol=1e-12,abs_tol=1e-12)
                else:assert delta[metric+'_relative_change_percent']==''


def test_bad_runs_are_retained_and_seed_rmse_is_not_raw_pooled(tmp_path):
    example=next(r for r in inputs() if r['dataset']=='yellow_trip_hourly' and r['model']=='titantpp_history_mlp' and r['split']=='test' and r['view']=='overall')
    rows=[{**example,'seed':seed,'qty_rmse':rmse,'qty_mae':.5,'time_nll':1.} for seed,rmse in zip(SEEDS,(1.,2.,9.))]
    group=s.aggregate(rows)[0]
    assert group['qty_rmse_mean']==4.
    assert math.isclose(group['qty_rmse_sample_sd'],math.sqrt(19.))
    assert not math.isclose(group['qty_rmse_mean'],math.sqrt((1+4+81)/3))
    # A high-error candidate stays visible; output order is predetermined, not ranked by results.
    bad={**rows[0],'model':'titantpp_history_post_block','qty_rmse':999.}
    panel=tmp_path/'single_seed_panel.md'
    s.write_panels(panel,s.aggregate([rows[0],bad]),'test',singles=True)
    assert '999.000' in panel.read_text() and s.LABELS[bad['model']] in panel.read_text()
    if (HERE/'comparison_receipt.json').exists():
        inputs_by_dataset=defaultdict(list)
        for row in inputs():inputs_by_dataset[row['dataset']].append(row)
        receipt=read(HERE/'comparison_receipt.json');assert len(receipt['tables'])==8
        # Read displayed cells, rather than regenerating tables with the same renderer.
        for filename,digest in receipt['tables'].items():
            path=HERE/filename;assert sha(path)==digest
            split='validation' if filename.startswith('VALIDATION') else 'test'
            view='tail' if '_TAIL_' in filename else 'overall'
            singles='_SEED42_' in filename
            dataset=None;seen=set()
            for line in path.read_text().splitlines():
                if line.startswith('## '):dataset=next(d for d,label in s.DATASETS.items() if label==line[3:])
                if not line.startswith('| ') or line.startswith('| 모델'):continue
                fields=[v.strip() for v in line.strip('|').split('|')]
                model=next(m for m,label in s.LABELS.items() if label==fields[0])
                selected=[r for r in inputs_by_dataset[dataset] if r['model']==model and r['split']==split and r['view']==view]
                if singles:selected=[r for r in selected if r['seed']==42]
                assert selected
                if not singles:assert model not in s.A100_MODELS
                seen.add((dataset,model))
                n=0 if selected[0]['seed'] is None else len(selected)
                assert fields[1]==('—' if n==0 else str(n))
                for index,metric in enumerate(s.METRICS,2):
                    values=[r[metric] for r in selected]
                    if values[0] is None:assert fields[index]=='—';continue
                    mean=math.fsum(values)/len(values)
                    sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/(len(values)-1)) if len(values)>1 else None
                    digits=4 if metric=='time_nll' else 3
                    expected=f'{mean:.{digits}f}'+(f' ± {sd:.{digits}f}' if sd is not None else '')
                    assert fields[index]==expected,(filename,dataset,model,metric)
            expected={(r['dataset'],r['model']) for r in inputs() if r['split']==split and r['view']==view
                      and ((r['seed']==42) if singles else r['model'] not in s.A100_MODELS)}
            assert seen==expected,(filename,expected-seen,seen-expected)


def test_scope_rejects_omission_duplicate_population_change_and_tail_rule_change():
    rows=inputs()
    # Available data must not be falsely certified complete before the last campaign arrives.
    if len(input_paths())==3:
        with pytest.raises(AssertionError):s.validate_scope(rows)
    else:
        for modified in (rows[:-1],rows+[rows[0]]):
            with pytest.raises(AssertionError):s.validate_scope(modified)
        changed=copy.deepcopy(rows);changed[0]['count']+=1
        with pytest.raises(AssertionError):s.validate_scope(changed)
    tail=next(r for p in input_paths() for r in records(p) if r.get('view')=='tail')
    with pytest.raises(AssertionError):s.normalize({**tail,'threshold_rule':'raw_quantity >= threshold'})
    with pytest.raises(AssertionError):s.normalize({**tail,'threshold':str(float(tail['threshold'])+1)})
