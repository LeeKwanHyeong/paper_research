"""Reproducible local diagnostics from immutable Train/Validation evidence only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import cached_head_diagnostics as head
import history_context_diagnostics as context


def historical_row(dataset, seed, history, complete, observed_utc, source):
    rows=history['history'] if isinstance(history,dict) else history
    best=min(rows,key=lambda r:r['val_qty_rmse'])
    tbest=min(rows,key=lambda r:r['val_time_nll'])
    def compact(r):
        return {k:r[k] for k in ('epoch','val_qty_rmse','val_qty_mae','val_time_nll','train_time_nll')}
    return {'dataset':dataset,'seed':seed,'complete':complete,'actual_observed_utc':observed_utc,
        'source':source,'saved_epoch':rows[-1]['epoch'],'quantity_selected':compact(best),
        'time_minimum_descriptive':compact(tbest),'last':compact(rows[-1]),
        'time_minimum_quantity_RMSE_penalty_percent':100*(tbest['val_qty_rmse']/best['val_qty_rmse']-1),
        'early5_training_time_median':float(np.median([r['train_time_nll'] for r in rows[:5]])),
        'late5_training_time_median':float(np.median([r['train_time_nll'] for r in rows[-5:]])),
        'early5_validation_time_median':float(np.median([r['val_time_nll'] for r in rows[:5]])),
        'late5_validation_time_median':float(np.median([r['val_time_nll'] for r in rows[-5:]])),
        'note':'Training losses are online batch averages, not fixed-endpoint scores. Time-minimum is descriptive; original checkpoint selection is unchanged.'}


def original_patterns(bundle):
    bundle=Path(bundle)
    registry=head.read(bundle/'reuse_seed42_registry.json')
    output=[];sources=[]
    for anchor in registry['rows']:
        path=Path(anchor['history']['path'])
        context.require(context.sha(path)==anchor['history']['sha256'],'Reused history SHA changed')
        payload=head.read(path)
        # Reused registry is consulted solely for training history lineage.
        output.append(historical_row(anchor['dataset'],42,payload,True,None,
            {'path':str(path),'sha256':context.sha(path),'status':'verified preserved original'}))
    for host in ('5080','5090'):
        pointer=head.read(bundle/'hourly_monitor'/('latest_'+host+'.json'))
        path=Path(pointer['snapshot']);snap=head.read(path)
        context.require(snap['source_SHA_verified'] is True and snap['operation_SHA_verified'] is True,'Unverified original source')
        src={'path':str(path),'sha256':context.sha(path),'actual_observed_utc':snap['actual_observed_utc']}
        sources.append(src)
        for row in snap['rows']:
            job=row['job']
            output.append(historical_row(job['dataset'],job['seed'],row['history'],
                bool(row.get('terminal',{} ) and row['terminal'].get('scientific_success')),
                snap['actual_observed_utc'],src))
    context.require(len(output)==9 and len({(r['dataset'],r['seed']) for r in output})==9,'Original condition duplication')
    return {'rows':output,'sources':sources,'scope':'original Train/Validation history only',
        'all_final':all(r['complete'] for r in output),'Test_read':False}


def refresh_original_link(diagnostic_root, original=None):
    """Read saved local evidence once; no SSH, inference, fitting, or retry."""
    root=Path(diagnostic_root).resolve()
    original=Path(original or head.DEFAULT_SOURCE.parent)
    prior=head.read(root/'context_results.json')
    context.require(prior['Test_read'] is False and prior['optimizer_steps']==0 and prior['GPU_used'] is False,
                    'Foreign diagnostic scope')
    patterns=original_patterns(original)
    complete=original/'analysis/completion_receipt.json'
    link={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'pending_original_terminal_and_retrieval',
          'scope':'Original Validation-only terminal linkage','Test_read':False,'remote_calls':0,
          'new_inference':False,'original_history_patterns':patterns}
    if complete.exists():
        receipt=head.read(complete);contract=head.read(original/'execution_contract.json')
        context.require(receipt['contract_sha256']==head.canonical(contract)==
                        '845a891db284f2cf88aa2ab17bee47716ea8dc11e6268646a71c772073ab7130',
                        'Original completion contract changed')
        context.require(receipt['status']=='nine_conditions_Validation_originals_complete'
                        and receipt['held_out_test_evaluated'] is False and receipt['canonical_conditions']==9,
                        'Original terminal/retrieval gate not passed')
        context.require(set(receipt['outputs'])=={'Validation_comparison.json','Validation_per_seed.csv','README.md'},
                        'Missing original completion output')
        context.require(set(receipt['originals'])=={'5080','5090'},'Missing original retrieval host')
        for name,digest in receipt['outputs'].items():
            context.require(name in ('Validation_comparison.json','Validation_per_seed.csv','README.md'),
                            'Foreign completion output')
            context.require(context.sha(original/'analysis'/name)==digest,'Original summary SHA changed')
        for host,r in receipt['originals'].items():
            context.require(host in ('5080','5090') and r['status']=='passed'
                            and r['contract_sha256']==receipt['contract_sha256'],'Original retrieval not passed')
            context.require(context.sha(original/'retrieved'/host/'training_originals.tar.gz')==r['archive_sha256'],
                            'Original archive SHA changed')
            cache=head.read(original/'hourly_monitor'/('terminal_cache_'+host+'.json'))
            count=4 if host=='5080' else 2
            expected={j['id'] for j in contract['jobs'] if j['host']==host}
            context.require(cache['contract_sha256']==receipt['contract_sha256']
                            and cache['analysis']['counts']['complete']==count
                            and not cache['analysis']['owned_fit_pids'],'Missing actual server terminal gate')
            context.require(all(cache['analysis']['counts'][k]==0 for k in ('failed','running','waiting','uncertain'))
                            and not cache['analysis']['gpu_pids']
                            and cache['analysis']['held_out_test_evaluated'] is False,
                            'Nonterminal or foreign server cache')
            context.require({m['id'] for m in r['manifests']}==expected
                            and all(m['all_binary_small_SHA_verified'] is True for m in r['manifests']),
                            'Missing original manifest verification')
            snap=head.read(cache['snapshot'])
            context.require(snap['contract_sha256']==receipt['contract_sha256']
                            and snap['source_SHA_verified'] is True and snap['operation_SHA_verified'] is True
                            and snap['gpu_uuid_verified'] is True and not snap['processes'] and not snap['gpu_pids']
                            and snap['supervisor_exit']['returncode']==0,'Unverified actual terminal snapshot')
            context.require({s['job']['id'] for s in snap['rows']}==expected,'Missing original terminal condition')
            for s in snap['rows']:
                m=next(m for m in r['manifests'] if m['id']==s['job']['id'])
                context.require(s['SHA_verified'] is True and s['terminal']['scientific_success'] is True
                                and s['terminal']['held_out_test_evaluated'] is False
                                and s['endpoint']['held_out_test_evaluated'] is False
                                and s['endpoint']['evaluation_scope']=='validation_only'
                                and m['terminal_manifest_sha256']==s['manifest_sha256'],
                                'Original terminal/manifest evidence changed')
        summary=head.read(original/'analysis/Validation_comparison.json')
        context.require(summary['scope']=='Validation_only' and summary['held_out_test_evaluated'] is False
                        and len(summary['rows'])==9 and patterns['all_final'],'Incomplete or foreign summary')
        keys={(r['dataset'],r['seed']) for r in summary['rows']}
        context.require(len(keys)==9 and keys=={(r['dataset'],r['seed']) for r in patterns['rows']},
                        'Original summary duplicate/missing conditions')
        for r in summary['rows']:
            hist=next(h for h in patterns['rows'] if (h['dataset'],h['seed'])==(r['dataset'],r['seed']))
            context.require(r['best_epoch']==hist['quantity_selected']['epoch'],'Original selector changed')
            anchor=contract['anchors42'][r['dataset']] if r['seed']==42 else None
            if anchor:
                endpoint=original/'source'/anchor['endpoint']
                context.require(context.sha(endpoint)==anchor['endpoint_sha256']==r['endpoint_sha256'],
                                'Reused endpoint SHA changed')
                ep=head.read(endpoint)
                context.require(ep['evaluation_scope']=='validation_only' and ep['held_out_test_evaluated'] is False,
                                'Foreign reused endpoint scope')
            for metric in ('qty_rmse','qty_mae','time_nll'):
                if anchor:
                    context.require(r['metrics'][metric]==anchor['metrics'][metric]==ep['selected'][metric],
                                    'Reused summary/endpoint metric mismatch')
                    context.require(hist['quantity_selected']['val_'+metric]==anchor['history_metrics'][metric],
                                    'Reused original history metric mismatch')
                else:
                    context.require(abs(r['metrics'][metric]-hist['quantity_selected']['val_'+metric])<1e-8,
                                    'Summary/history metric mismatch')
        link.update(status='verified_original9_final_Validation_and_binary_SHA_linked',
            completion_receipt_sha256=context.sha(complete),
            validation_summary_sha256=context.sha(original/'analysis/Validation_comparison.json'),
            grouped_3seed=summary['grouped_3seed'],CPU_binary_reinference_audit='not_performed')
    (root/'Intermittent_final_link.json').write_text(json.dumps(link,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# Intermittent 최종 결과 연결','',
        '**현재 상태 — '+('완료' if link['status'].startswith('verified') else '외부 작업 대기')+'**','',
        '| Seed | 완료 검증 | 저장/best epoch | Validation RMSE | MAE | TimeNLL |',
        '|---|---|---|---:|---:|---:|']
    for row in sorted((r for r in patterns['rows'] if r['dataset']=='intermittent_frozen_5000'),key=lambda r:r['seed']):
        q=row['quantity_selected'];lines.append(f"|{row['seed']}|{'완료' if row['complete'] else '진행 중: 최종 아님'}|{row['saved_epoch']}/{q['epoch']}|{q['val_qty_rmse']:.6f}|{q['val_qty_mae']:.6f}|{q['val_time_nll']:.6f}|")
    if link['status'].startswith('verified'):
        lines+=['','원래9조건의 실제 종료·선택/마지막 Validation·새 원본 archive SHA 회수 검증과 연결했습니다. 세 데이터의 3seed Validation 평균·표본SD는 연결 JSON의 grouped_3seed에 있습니다.']
    else:
        lines+=['','세 seed 최종 결론과 원본 회수 완료 판정은 대기합니다. 위 값은 보존된 실제 관측이며 새 SSH 관측이 아닙니다.']
    for src in patterns['sources']:
        if '/5090/' in src['path']:lines+=['','5090 실제 관측 UTC: '+src['actual_observed_utc']+' (KST=UTC+9).']
    lines+=['','이번 링크는 기존 학습 결과의 로컬 읽기 전용 연결입니다. Intermittent 시간 출력부 재적합·새 Test·독립 평가·CPU binary encoder 재추론은 수행하지 않았습니다.']
    (root/'INTERMITTENT_STATUS.md').write_text('\n'.join(lines)+'\n')
    return link


def run(output, bundle=head.DEFAULT_BUNDLE, source=head.DEFAULT_SOURCE, original=None):
    torch.set_num_threads(4)
    audit=head.audit_bundle(bundle,source)
    native=head.load_native_class(source)
    contexts={d['dataset_id']:context.load_context(source,d) for d in audit['contract']['datasets']}
    jobs=[]
    for job in audit['contract']['jobs']:
        inputs=head.load_job_inputs(audit,job);data=inputs['data']
        losses={s:{} for s in ('train','validation')}
        for stage,cp in [('E0',None),('last',inputs['last_checkpoint'])]:
            view=head.head_view(inputs['source_checkpoint'],data['model'],cp['head_state'] if cp else None,native)
            for split,cache in inputs['caches'].items():
                losses[split][stage]=head.native_nll(cache,view).numpy()
        jobs.append({'job':job['id'],'dataset':job['dataset'],'seed':job['seed'],
            'source_checkpoint_sha256':job['checkpoint']['sha256'],
            **context.context_analysis(contexts[job['dataset']],inputs,losses,data['quantity_boundaries_all_train_rows'])})
    original=original or Path(source).parent
    result={'schema':'titantpp_time_generalization_context_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Train/Validation-only cached arithmetic and admitted history cohorts',
        'encoder_reexecuted':False,'GPU_used':False,'optimizer_steps':0,'Test_read':False,
        'row_or_prediction_arrays_exported':False,'causal_experiment':False,'provenance':audit['provenance'],
        'scripts_sha256':{Path(p).name:context.sha(p) for p in (head.__file__,context.__file__,__file__)},
        'jobs':jobs,'original_history_patterns':original_patterns(original),
        'limits':['Within-stratum gaps are descriptive, not proof of encoder or gradient causality.',
                  'Original Intermittent seed62 remains provisional until a verified terminal is available.',
                  'A rejected last refit head is a diagnostic endpoint; selected Taxi heads remain original E0.',
                  'Counterfactual or time-minimum history summaries do not reselect checkpoints.']}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    p.add_argument('--refresh-original-link',type=Path)
    args=p.parse_args()
    if args.refresh_original_link:
        context.require(args.output is None,'Choose one local operation')
        r=refresh_original_link(args.refresh_original_link)
        print(json.dumps({'status':r['status'],'remote_calls':0,'Test_read':False}))
    else:
        context.require(args.output is not None,'Output required')
        r=run(args.output)
        print(json.dumps({'output':str(args.output),'jobs':len(r['jobs']),'original_all_final':r['original_history_patterns']['all_final']}))
