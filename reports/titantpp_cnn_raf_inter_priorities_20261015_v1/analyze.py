"""Local descriptive audit of already saved CNN Train/Validation predictions."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OBS = 'reports/titantpp_cnn_history_adaptation_20261015_v1/analysis/ANALYSIS.json'
MATCH = 'reports/titantpp_cnn_history_adaptation_matched_20261015_v1/analysis/ANALYSIS.json'
MEM = 'reports/titantpp_cnn_memory_sensitivity_20261015_v3/analysis/SUMMARY.json'
DATASETS = {'raf_spare_parts': 200., 'intermittent_frozen_5000': 187.}
SPLITS = ('train', 'validation')
SEEDS = (42, 52, 62)
COLS = ['entity_id','seq','dataset','seed','split','history_length','last_quantity',
        'mean_quantity','history_log_std','last_log_deviation','last_observed_log_change',
        'previous_observation_available','raw_quantity','prediction_1','time_nll_1']
PINS = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(rel, expected):
    assert digest(ROOT / rel) == expected, f'SHA mismatch: {rel}'
    PINS[rel] = expected


def read(rel):
    return json.loads((ROOT / rel).read_text())


def stats(f, pred, total_n, total_sse):
    n = f.height
    if not n:
        return dict(count=0, entities=0, share=0., sse=0., sse_share=0.,
                    **dict.fromkeys(('rmse','mae','bias','over_rate','under_rate','time_nll')))
    e = f[pred] - f['raw_quantity']
    sse = float(e.pow(2).sum())
    return dict(count=n, entities=f['entity_id'].n_unique(), share=n/total_n,
                rmse=math.sqrt(sse/n), mae=float(e.abs().mean()), bias=float(e.mean()),
                over_rate=float((e>0).mean()), under_rate=float((e<0).mean()),
                sse=sse, sse_share=sse/total_sse if total_sse else None,
                time_nll=float(f['time_nll_1'].mean()) if pred=='prediction_1' else None)


def cuts_expr(name, cuts):
    return (pl.when(pl.col(name)<=cuts[0]).then(0).when(pl.col(name)<=cuts[1]).then(1)
            .when(pl.col(name)<=cuts[2]).then(2).otherwise(3))


def masks(f, cuts, tail):
    yield 'overall', 'all', pl.lit(True)
    yield 'target_tail_descriptive_only', 'tail', pl.col('raw_quantity')>tail
    yield 'target_tail_descriptive_only', 'non_tail', pl.col('raw_quantity')<=tail
    for direction in ('falling','flat','rising','unavailable'):
        yield 'observed_direction', direction, pl.col('direction')==direction
    for name, condition in [('1',pl.col('history_length')==1),('2-3',pl.col('history_length').is_between(2,3)),
                            ('4-8',pl.col('history_length').is_between(4,8)),('9-32',pl.col('history_length').is_between(9,32)),
                            ('33+',pl.col('history_length')>=33)]:
        yield 'history_length', name, condition
    for feature in ('last_quantity','history_log_std','last_log_deviation','absolute_observed_log_change'):
        for q in range(4):
            cond=cuts_expr(feature,cuts[feature])==q
            if feature=='absolute_observed_log_change': cond=cond & pl.col('previous_observation_available')
            yield feature, str(q), cond
    for direction in ('falling','flat','rising'):
        for level in ('upper','lower'):
            c=pl.col('last_quantity')>cuts['last_quantity'][2]
            if level=='lower': c=~c
            yield 'direction_x_level', direction+'_'+level, (pl.col('direction')==direction)&c
        for name,cond in [('last_above_mean',pl.col('last_quantity')>pl.col('mean_quantity')),
                          ('last_at_or_below_mean',pl.col('last_quantity')<=pl.col('mean_quantity'))]:
            yield 'direction_x_history_level', direction+'_'+name,(pl.col('direction')==direction)&cond


def aggregate(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r['dataset'],r['split'],r['kind'],r['group'],r['model'])].append(r)
    result=[]
    for key,values in groups.items():
        assert sorted(v['seed'] for v in values)==list(SEEDS)
        assert len({(v['count'],v['entities']) for v in values})==1
        metrics={}
        for m in ('rmse','mae','bias','sse_share','over_rate','under_rate','time_nll'):
            vv=[v[m] for v in values if v[m] is not None]
            assert len(vv) in (0,3)
            metrics[m]=dict(mean=statistics.mean(vv) if vv else None,
                            sample_sd=statistics.stdev(vv) if vv else None)
        result.append(dict(zip(('dataset','split','kind','group','model'),key),
                           count=values[0]['count'],entities=values[0]['entities'],share=values[0]['share'],metrics=metrics))
    return result


def main():
    assert not (OUT/'RESULTS.json').exists(), 'Refuse overwrite'
    pin(OBS,'b65c806f3b18e091b31ead1ab54fb0b3e3cc08a2074bcfcf5a82011beb42a926')
    pin(MATCH,'34883e3e23b8e28b6049a4a8bd3611dcd3008c517ad772c6666396e0ff0d3bba')
    receipt_path='reports/titantpp_cnn_memory_sensitivity_20261015_v3/FINAL_RECEIPT.json'
    pin(receipt_path,'cde075132699bc4100688753b85a8cbc3a8f5b1fe0523aff7b7c98a4ac4e07b2')
    receipt=read(receipt_path); pin(MEM,receipt['files'][MEM])
    obs,matched,memory=read(OBS),read(MATCH),read(MEM)
    assert obs['test_read'] is False and matched['test_read'] is False
    # Hash verification includes metadata/receipts, never model execution or data reads.
    for path,sha in obs['inputs'].items():
        assert '/test/' not in path.lower()
        pin(path,sha)
    all_rows=[]; concentrations=[]; fingerprints={}; validation_checks=[]; tail_detection=[]
    distributions=[]; target_transitions=[]; rows_read=0
    base=ROOT/'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/retrieval/results'
    for d,tail in DATASETS.items():
        for seed in SEEDS:
            cuts=next(r['cuts'] for r in obs['train_cuts'] if r['dataset']==d and r['seed']==seed)
            job=f'{d}__{seed}__titantpp_cnn_gru54_cnn_only'
            for split in SPLITS:
                evidence=next(e for e in obs['prediction_evidence'] if e['job_id']==job and e['split']==split)
                paths=[base/job/split/'predictions'/p['path'] for p in evidence['parts']]
                f=pl.scan_parquet(paths).select(COLS).collect()
                assert f.height==evidence['target_count']
                assert f['dataset'].unique().to_list()==[d] and f['seed'].unique().to_list()==[seed]
                assert f['split'].unique().to_list()==[split]
                assert f.select(pl.struct('entity_id','seq').n_unique()).item()==f.height
                assert f.select(pl.all_horizontal([pl.col(c).is_finite() for c in COLS if c not in
                    ('entity_id','dataset','split','previous_observation_available')]).all()).item()
                truth=f.select([c for c in COLS if c not in ('seed','prediction_1','time_nll_1')]).sort('entity_id','seq')
                key=(d,split)
                if key in fingerprints: assert truth.equals(fingerprints[key])
                else: fingerprints[key]=truth
                f=f.with_columns(pl.col('last_observed_log_change').abs().alias('absolute_observed_log_change'),
                    pl.when(~pl.col('previous_observation_available')).then(pl.lit('unavailable'))
                    .when(pl.col('last_observed_log_change')<0).then(pl.lit('falling'))
                    .when(pl.col('last_observed_log_change')>0).then(pl.lit('rising'))
                    .otherwise(pl.lit('flat')).alias('direction'))
                cnn_sse=float((f['prediction_1']-f['raw_quantity']).pow(2).sum())
                expected=next(r for r in obs['six_metrics'] if r['dataset']==d and r['seed']==seed and r['split']==split)
                for frame,prefix in ((f,''),(f.filter(pl.col('raw_quantity')>tail),'tail_')):
                    calculated=stats(frame,'prediction_1',f.height,cnn_sse)
                    for name,metric in [('rmse','qty_rmse'),('mae','qty_mae'),('time_nll','time_nll')]:
                        assert math.isclose(calculated[name],expected['metrics'][prefix+metric],rel_tol=1e-8,abs_tol=1e-8)
                validation_checks.append(dict(dataset=d,seed=seed,split=split,all_six_metrics_match=True,
                    checkpoint_sha256=expected['checkpoint_sha256'],selected_epoch=expected['selected_epoch'],host=expected['host']))
                for kind,group,cond in masks(f,cuts,tail):
                    selected=f.filter(cond)
                    for model,pred in [('CNN','prediction_1'),('last_observed','last_quantity'),('history_mean','mean_quantity')]:
                        model_sse=cnn_sse if model=='CNN' else float((f[pred]-f['raw_quantity']).pow(2).sum())
                        all_rows.append(dict(dataset=d,seed=seed,split=split,kind=kind,group=group,model=model,
                            **stats(selected,pred,f.height,model_sse)))
                entity=f.with_columns((pl.col('prediction_1')-pl.col('raw_quantity')).pow(2).alias('se')).group_by('entity_id').agg(pl.col('se').sum(),pl.len().alias('n')).sort('se',descending=True)
                for n in (10,max(1,math.ceil(.01*entity.height))):
                    concentrations.append(dict(dataset=d,seed=seed,split=split,top_entities=n,total_entities=entity.height,
                        share_sse=float(entity.head(n)['se'].sum()/cnn_sse),rows=int(entity.head(n)['n'].sum())))
                # Descriptive reach of fixed observed-history strata, not trained tail classifiers.
                for label,c in [('last_upper_train_quartile',pl.col('last_quantity')>cuts['last_quantity'][2]),
                                ('history_mean_above_tail',pl.col('mean_quantity')>tail),
                                ('last_above_tail',pl.col('last_quantity')>tail),
                                ('history_std_upper_train_quartile',pl.col('history_log_std')>cuts['history_log_std'][2])]:
                    sf=f.filter(c); tf=f.filter(pl.col('raw_quantity')>tail); hit=sf.filter(pl.col('raw_quantity')>tail)
                    tail_detection.append(dict(dataset=d,seed=seed,split=split,group=label,count=sf.height,
                        population_share=sf.height/f.height,tail_count=tf.height,tail_hits=hit.height,
                        tail_recall=hit.height/tf.height if tf.height else None,
                        tail_precision=hit.height/sf.height if sf.height else None))
                if seed==42:
                    distributions.append(dict(dataset=d,split=split,count=f.height,entities=f['entity_id'].n_unique(),
                        history_length_quantiles=[f['history_length'].quantile(q,interpolation='linear') for q in (0,.25,.5,.75,1)],
                        raw_quantity_quantiles=[f['raw_quantity'].quantile(q,interpolation='linear') for q in (0,.5,.95,.99,1)],
                        last_quantity_train_cuts=cuts['last_quantity'],observed_change_train_cuts=cuts['absolute_observed_log_change']))
                for od in ('falling','flat','rising'):
                    sub=f.filter(pl.col('direction')==od)
                    for future,cond in [('falling',pl.col('raw_quantity')<pl.col('last_quantity')),
                                        ('flat',pl.col('raw_quantity')==pl.col('last_quantity')),
                                        ('rising',pl.col('raw_quantity')>pl.col('last_quantity'))]:
                        sf=sub.filter(cond)
                        target_transitions.append(dict(dataset=d,seed=seed,split=split,observed=od,
                            future_target_descriptive_only=future,within_observed_share=sf.height/sub.height if sub.height else None,
                            **stats(sf,'prediction_1',f.height,cnn_sse)))
                rows_read+=f.height
    decomposition=[]
    for d in DATASETS:
        for seed in SEEDS:
            def row(split,group):
                return next(r for r in all_rows if (r['dataset'],r['seed'],r['split'],r['model'],r['kind'],r['group'])==(d,seed,split,'CNN','observed_direction',group))
            composition=within=0.
            for direction in ('falling','flat','rising','unavailable'):
                a,b=row('train',direction),row('validation',direction)
                ma=a['sse']/a['count'] if a['count'] else 0.
                mb=b['sse']/b['count'] if b['count'] else 0.
                composition+=(b['share']-a['share'])*ma
                within+=b['share']*(mb-ma)
            total={s:next(r['rmse']**2 for r in all_rows if (r['dataset'],r['seed'],r['split'],r['model'],r['kind'])==(d,seed,s,'CNN','overall')) for s in SPLITS}
            delta=total['validation']-total['train']
            assert math.isclose(delta,composition+within,rel_tol=1e-8,abs_tol=1e-8)
            decomposition.append(dict(dataset=d,seed=seed,delta_mse=delta,composition=composition,within_direction=within))
    # Partition checks are meaningful irrespective of metric direction.
    for d in DATASETS:
        for seed in SEEDS:
            for split in SPLITS:
                rr=[r for r in all_rows if (r['dataset'],r['seed'],r['split'],r['model'])==(d,seed,split,'CNN')]
                overall=next(r for r in rr if r['kind']=='overall')
                for kind in ('target_tail_descriptive_only','observed_direction','history_length','last_quantity','history_log_std','last_log_deviation'):
                    vv=[r for r in rr if r['kind']==kind]
                    assert sum(r['count'] for r in vv)==overall['count']
                    assert math.isclose(sum(r['sse'] for r in vv),overall['sse'],rel_tol=1e-10)
    for path,sha in PINS.items(): assert digest(ROOT/path)==sha
    result=dict(schema='cnn_raf_inter_priorities_v1',created_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        status='complete',verification_status='ANALYZED',scope='saved Train/Validation predictions only',
        new_training=False,new_inference=False,remote_operations=0,test_read=False,rows_read=rows_read,
        unique_target_rows=rows_read//3,seed_count=3,inputs=PINS,analyzer_sha256=digest(__file__),
        plan_sha256=digest(OUT/'ANALYSIS_PLAN.md'),baseline_checks=validation_checks,
        rows=all_rows,three_seed=aggregate(all_rows),entity_error_concentration=concentrations,
        observed_tail_reach=tail_detection,distributions=distributions,
        target_transitions_descriptive_only=target_transitions,mse_decomposition=decomposition,
        matched_evidence=[r for r in matched['rows'] if r['dataset'] in DATASETS],
        memory_sensitivity=[r for r in memory['three_seed_summaries'] if r['dataset'] in DATASETS],
        limitations=['post-Test-informed exploration; no Test read this turn','selected checkpoint has seen Train and was selected on Validation',
            'three seeds repeat the same targets; no independent population replication','no temporal-gap features in saved rows',
            'matching leaves unmeasured confounders and incomplete support','no new architecture effect established'])
    with (OUT/'RESULTS.json').open('x') as stream: json.dump(result,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
    print(json.dumps(dict(status='complete',rows_read=rows_read,unique_targets=rows_read//3,verified_files=len(PINS),result_sha256=digest(OUT/'RESULTS.json'))))


if __name__=='__main__': main()
