"""Read-only remote hourly observations for the explicitly approved component study."""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from paper.scripts import launch_titantpp_cnn_gru_components_v5 as launcher
from paper.scripts.titantpp_component_hourly_eta import estimate_fit_eta
from paper.scripts import titantpp_components_terminal_verifier as terminal_verifier

seed_terminal_cache = terminal_verifier.seed_terminal_cache

PROJECT=Path('/Users/igwanhyeong/PycharmProjects/paper_research')
CANONICAL='e60badde6d9d902500aa67764f6834213ea1775797508f31d204ae67a5e63074'
SOURCE='15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53'
HOSTS=('5080','5090')
TERMINAL='training_complete_remote_verified'


def kst(value):
    return datetime.fromtimestamp(value,ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')


def read_snapshot(path,c,host):
    snapshot=launcher.read(path)
    files=launcher.decode_snapshot(snapshot,c,host)
    return snapshot,files


def latest_snapshot(bundle,host):
    candidates=[]
    for path in (bundle/'control/status'/host).glob('*.json'):
        value=launcher.read(path)
        if 'status.json' in value.get('files',{}):
            candidates.append((value.get('observed_unix',0),path))
    launcher.require(bool(candidates),'No previously saved training snapshot for '+host)
    return max(candidates,key=lambda x:x[0])[1]


def verified_terminal(bundle, output, c, host):
    return terminal_verifier.verified_terminal(bundle, output, c, host)


def read_status_once(bundle, c, host, remote=None):
    """One bounded read-only remote call; preserve metadata on evidence errors."""
    approval = launcher.authorization(bundle, c, host)
    qualification_cache = bundle/'control/qualification_cache'/host/'snapshot.json'
    if qualification_cache.exists() and not (bundle/'control/intents'/('train_'+host+'.json')).exists():
        snapshot = launcher.read(qualification_cache)
        binding = launcher.qualification_binding(snapshot, c, host)
        return qualification_cache, snapshot, {'status':'qualification_passed_cached',
            'qualification_raw_sha256':binding['raw_sha256'], 'jobs':[], 'completed':0,
            'total':sum(j['host']==host for j in c['jobs']), 'observed_unix':snapshot['observed_unix']}, 0
    token = uuid.uuid4().hex
    payload = {'contract':c, 'contract_sha256':launcher.canonical(c), 'host':host, 'approval':approval}
    try:
        snapshot = launcher.one_call(bundle, remote or launcher.SSHRemote(), host, 'status', payload,
            label='status_v2_'+token)
    except Exception as error:
        return None, None, {'host':host, 'status':'query_failed_or_identity_unconfirmed',
            'reason':type(error).__name__+': '+str(error), 'training_failure_inferred':False,
            'user_action_required':True, 'jobs':[], 'completed':None,
            'total':sum(j['host']==host for j in c['jobs']), 'whole_queue_eta':'unconfirmed'}, 1
    path = bundle/'control/status'/host/(token+'.json')
    launcher.write_exclusive(path, snapshot)
    try:
        launcher.decode_snapshot(snapshot, c, host)
        terminal_verifier.qualification_bridge(bundle, c, host, snapshot)
        report = terminal_verifier.training_status(snapshot, c, host)
    except (KeyError, ValueError, TypeError) as error:
        report = {'host':host, 'status':'training_identity_unconfirmed',
            'reason':type(error).__name__+': '+str(error), 'training_failure_inferred':False,
            'identity_verified':False, 'user_action_required':True, 'jobs':[], 'completed':None,
            'total':sum(j['host']==host for j in c['jobs']), 'whole_queue_eta':'unconfirmed'}
    return path, snapshot, report, 1


def observe_host(bundle, output, c, host, local_only=False, remote=None):
    launcher.authorization(bundle, c, host)
    cache = output/'terminal_cache'/f'{host}.json'
    terminal = verified_terminal(bundle, output, c, host)
    if terminal:
        path, snapshot, report = terminal
        calls=0;origin='verified_terminal_cache_original_observation'
    elif local_only:
        path=latest_snapshot(bundle,host);snapshot,_=read_snapshot(path,c,host)
        terminal_verifier.qualification_bridge(bundle,c,host,snapshot)
        report=terminal_verifier.training_status(snapshot,c,host)
        calls=0;origin='existing_snapshot_original_observation'
    else:
        path,snapshot,report,calls=read_status_once(bundle,c,host,remote)
        origin='single_read_only_remote_status_call'
        if snapshot is None:
            return {**report,'remote_calls':calls,'observation_source':origin,
                'immutable_local_original_retrieval':'not_performed_by_monitor'}
        if report['status']=='training_identity_unconfirmed':
            return {**report,'remote_calls':calls,'snapshot_path':str(path),
                'snapshot_raw_sha256':launcher.sha(path),'observation_source':origin,
                'observed_unix':snapshot.get('observed_unix'),
                'observed_KST':kst(snapshot['observed_unix']) if terminal_verifier._finite(snapshot.get('observed_unix')) else None,
                'immutable_local_original_retrieval':'not_performed_by_monitor'}
    if report['status']==TERMINAL and not cache.exists():
        try:
            seed_terminal_cache(bundle,output,host,path)
        except (KeyError,ValueError,OSError,TypeError) as error:
            report={**report,'status':'training_identity_unconfirmed',
                'reason':'Terminal cache evidence unresolved: '+type(error).__name__+': '+str(error),
                'training_failure_inferred':False,'user_action_required':True}
    files=launcher.decode_snapshot(snapshot,c,host)
    rows=[]
    for item in report.get('jobs',[]):
        row=dict(item);job=item['job'];folder='run/'+job['id']+'/runs/'+job['arm']+'/count_only_log_regression/seed_'+str(job['seed'])
        row['eta']={'status':'unconfirmed','reason':'Not a verified live saved epoch or not yet started'}
        if item['status']=='complete':
            row['eta']={'status':'complete','conditional_early_stop_seconds':0,'conditional_maximum_seconds':0}
        elif report['status']=='training_running' and item['status']=='running' and item.get('saved_epoch'):
            try:
                history=json.loads(files[folder+'/history.json'])
                timing=json.loads(files[folder+'/epoch_timing.json'])
                eta=estimate_fit_eta(history,timing,item['saved_epoch'])
                if eta['status']=='estimated':
                    launcher.require(eta['best_epoch']==item['best_epoch'],'ETA selector/history best differ')
                    eta['conditional_early_stop_KST']=kst(snapshot['observed_unix']+eta['conditional_early_stop_seconds'])
                    eta['conditional_maximum_KST']=kst(snapshot['observed_unix']+eta['conditional_maximum_seconds'])
                row['eta']=eta
            except (KeyError,ValueError,TypeError,json.JSONDecodeError) as error:
                row['eta']={'status':'unconfirmed','reason':str(error)}
        rows.append(row)
    return {'completed':report.get('completed'),'total':sum(j['host']==host for j in c['jobs']),
        **report,'host':host,'jobs':rows,'remote_calls':calls,'snapshot_path':str(path),
        'snapshot_raw_sha256':launcher.sha(path),'observation_source':origin,
        'observed_KST':kst(snapshot['observed_unix']),'whole_queue_eta':'unconfirmed',
        'immutable_local_original_retrieval':'not_performed_by_monitor',
        'user_action_required':report['status'] not in ('training_running',TERMINAL)}


def verify_monitor(output,expected):
    path=output/'monitor_integrity.json'
    launcher.require(launcher.sha(path)==expected,'Pinned monitor integrity rawSHA changed; SSH0')
    seal=launcher.read(path)
    launcher.require(seal['contract_sha256']==CANONICAL,'Monitor contract identity changed; SSH0')
    launcher.require(seal['files_sha256']==launcher.canonical(seal['files']),'Monitor code closure changed; SSH0')
    for name,digest in seal['files'].items():
        launcher.require(launcher.sha(launcher.safe(PROJECT,name))==digest,'Monitor code changed; SSH0')
    approval=output/'approval.json'
    launcher.require(launcher.sha(approval)==seal['approval_raw_sha256'],'Hourly approval rawSHA changed; SSH0')
    a=launcher.read(approval)
    launcher.require(a['approved'] is True and a['hosts']==list(HOSTS) and a['period_seconds']==3600
        and a['contract_sha256']==CANONICAL and a['remote_read_only'] is True
        and a['held_out_test_evaluated'] is False,'Hourly observation approval drift; SSH0')


def duration(seconds):
    if seconds is None:return '미확정'
    minutes=int(round(seconds/60))
    hours,minute=divmod(minutes,60)
    return (f'{hours}시간 {minute}분' if hours else f'{minute}분')


def markdown(result):
    arms={'titantpp_cnn_gru54_correction_disabled':'보정 없음','titantpp_cnn_gru54_cnn_only':'CNN만',
        'titantpp_cnn_gru54_gru_only':'GRU만','titantpp_cnn_gru54':'CNN+GRU'}
    data={'yellow_trip_hourly':'Taxi','raf_spare_parts':'RAF','intermittent_frozen_5000':'Intermittent'}
    lines=['# CNN·GRU36 시간별 checkpoint·조건부 ETA', '',
        '**승인된 학습을 실제 저장 증적으로 관측한다 — 진행 중**','',
        '| GPU / 관측 KST | 상태 | 완료 / 전체 |','|---|---|---:|']
    for host in HOSTS:
        report=result['hosts'][host]
        lines.append(f"| {host} / {report.get('observed_KST','조회 미확정')} | {report['status']} | {report.get('completed','미확정')} / {report['total']} |")
    lines+=['','| GPU | dataset · seed · 구조 | 저장 / 이력상 best | best 미갱신·현재 속도 유지 | 300 epoch 실행 |','|---|---|---:|---:|---:|']
    for host in HOSTS:
        for row in result['hosts'][host].get('jobs',[]):
            if row['status'] not in ('complete','running','running_before_first_checkpoint','checkpoint_snapshot_unconfirmed'):continue
            job=row['job'];eta=row['eta'];saved=row.get('saved_epoch');best=row.get('best_epoch')
            state='저장 전' if saved is None else f'{saved} / {best}'
            if row['status']=='complete':state+=' (완료)'
            lines.append(f"| {host} | {data[job['dataset']]} · {job['seed']} · {arms[job['arm']]} | {state} | {duration(eta.get('conditional_early_stop_seconds'))} | {duration(eta.get('conditional_maximum_seconds'))} |")
    lines+=['','- ETA는 각 fit 자체의 최근 최대10개 완료 epoch 유한 양수 elapsed_seconds 중앙값을 사용한다. best가 갱신되지 않을 때의 목표 epoch는 min(300,max(40,best+40))이다. 표의 잔여시간과 완료시각은 원관측시점 기준의 조건부 시나리오다.',
        '- 미시작 arm·다른 fit/GPU의 속도를 복사하지 않는다. best 재갱신·부하/속도 변화에 따라 달라지며, 전체36개 큐 ETA는 미확정이다. lease/deadline은 ETA가 아니다.',
        '- best는 저장된 fullValidation history의 최초 strict finite 최소 raw수량RMSE epoch다. 완료 행은 최종 manifest에 묶인 history와 선택/최종 binary의 원격 SHA 검증을 사용한다. 로컬 binary 원본 회수는 별도다.'
        ,'- 종료 캐시는 해당 root의 과거 종료 관측이다. 원관측 KST를 유지하며 현재 전역 GPU 유휴 상태를 뜻하지 않는다.',
        '- 이 모니터는 원격 상태만 읽는다. 학습·launch·qualification·retry/resume·kill·claim삭제·Test·원본회수는 수행하지 않는다. 원본archive/manifest의 로컬회수와 최종Validation 집계는 별도다.']
    if result['all_training_terminal_verified']:
        lines+=['','**36개 원격 종료를 확인한다 — 완료 / 원본 회수와 분석은 다음 작업**','',
            '- host별24/12 실제worker·supervised dispatcher exit0·마지막phasecomplete·GPU/PID부재·원격모든binary manifest SHA가 검증됐다. 로컬 immutable 원본 회수와 최종Validation 집계는 별도 미완료다.']
    else:
        lines+=['','**학습 종료 뒤 원본과 여섯 Validation 지표를 확정한다 — 다음 작업**','',
            '- 모든36 ID의 실제종료·원본archive/manifest·선택/최종binary 로컬SHA를 확정한 뒤 평균/표본SD·모든seed/대조를 판정하고 Results·Methods·주장을 정합한다.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,default=PROJECT/'search_artifacts/titantpp_cnn_gru_components_20261012_v5')
    p.add_argument('--output',type=Path,default=PROJECT/'reports/titantpp_cnn_gru_components_20261012_v5/hourly_monitor_v2')
    p.add_argument('--monitor-integrity-sha',required=True)
    p.add_argument('--local-snapshot-only',action='store_true')
    args=p.parse_args();bundle=args.bundle.resolve();output=args.output.resolve()
    report_root=PROJECT/'reports/titantpp_cnn_gru_components_20261012_v5'
    launcher.require(output.is_relative_to(report_root) and output!=report_root
        and not output.is_relative_to(report_root/'hourly_monitor'),'Unexpected or old sealed monitor output; SSH0')
    verify_monitor(output,args.monitor_integrity_sha)
    c,proof=launcher.check_bundle(bundle)
    launcher.require(proof['contract_sha256']==CANONICAL and c['source']['files_sha256']==SOURCE,'Frozen component identity changed; SSH0')
    launcher.require(len({j['id'] for j in c['jobs']})==36 and len(c['jobs'])==36,'36 unique jobs required; SSH0')
    for host in HOSTS:launcher.authorization(bundle,c,host)
    with (output/'observer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        token=uuid.uuid4().hex
        launcher.write_exclusive(output/'rounds'/f'{token}_intent.json',{'pid':__import__('os').getpid(),
            'started_unix':time.time(),'hosts':list(HOSTS),'per_live_host_status_calls':1,'remote_mutations':0})
        # Independent host status reads write disjoint immutable host receipts.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={host:pool.submit(observe_host,bundle,output,c,host,args.local_snapshot_only) for host in HOSTS}
            reports={}
            for host,future in futures.items():
                try:reports[host]=future.result()
                except Exception as error:
                    reports[host]={'host':host,'status':'query_or_evidence_unconfirmed','reason':type(error).__name__+': '+str(error),
                        'training_failure_inferred':False,'user_action_required':True,'completed':None,
                        'total':sum(j['host']==host for j in c['jobs']),'jobs':[],'whole_queue_eta':'unconfirmed','remote_calls':0}
        result={'schema':'titantpp_components_hourly_monitor_v2','contract_sha256':CANONICAL,
            'source_files_sha256':SOURCE,'monitor_integrity_raw_sha256':args.monitor_integrity_sha,
            'report_generated_KST':kst(time.time()),'hosts':reports,
            'all_training_terminal_verified':all(reports[h]['status']==TERMINAL for h in HOSTS),
            'whole_queue_eta':'unconfirmed','held_out_test_evaluated':False,'remote_mutations':0,
            'originals_retrieval':'not_performed_by_monitor','pipeline_wrapper_OS_exit_directly_observed':False}
        path=output/'rounds'/f'{token}.json';launcher.write_exclusive(path,result)
        md=output/'rounds'/f'{token}.md'
        with md.open('x') as stream:stream.write(markdown(result))
        latest={'round_path':str(path),'round_raw_sha256':launcher.sha(path),'markdown_path':str(md),
            'markdown_raw_sha256':launcher.sha(md),'contract_sha256':CANONICAL}
        temp=output/(token+'.latest.tmp');launcher.write_exclusive(temp,latest);temp.replace(output/'latest.json')
        print(json.dumps({'round_path':str(path),'markdown_path':str(md),'all_training_terminal_verified':result['all_training_terminal_verified'],
            'hosts':{h:{k:v for k,v in reports[h].items() if k not in ('jobs','gpu_pids')} for h in HOSTS}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
