"""Local hourly read-only report for the isolated five-way retrieval pilot.

Only a launched host's sealed operation monitor is invoked once. Pending hosts
remain local-only. This helper cannot qualify, activate, finalize, train or retry.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

NAME='titantpp_contextual_retrieval_pilot_dual_20261007_v1'
PROJECT=Path(__file__).resolve().parents[2]
BUNDLE=PROJECT/'search_artifacts'/NAME
HOSTS=('5090','5080')
STATES=('complete','failed','running','waiting','uncertain')
LABELS={'complete':'완료','failed':'실패','running':'진행','waiting':'대기','uncertain':'미확정'}
KST=dt.timezone(dt.timedelta(hours=9))


def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def raw_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def safe(root,name):
    root=Path(root).resolve();relative=Path(name)
    if not isinstance(name,str) or not name or relative.is_absolute() or '..' in relative.parts or '\\' in name or '\x00' in name:
        raise ValueError('Unsafe sealed file path')
    target=root/relative
    if not target.resolve().is_relative_to(root) or any(part.is_symlink() for part in (target,*target.parents) if part.is_relative_to(root)):
        raise ValueError('Sealed file escapes root or is a symlink')
    return target


def preflight(bundle):
    """Check bytes using stdlib only, before any sealed Python can execute."""
    c=read(safe(bundle,'execution_contract.json'));current=read(safe(bundle,'current.json'))
    if c is None or current is None or c.get('schema')!=NAME or current.get('schema')!=NAME:
        raise ValueError('Foreign/missing pilot contract/current')
    if current.get('canonical_sha256')!=canonical(c) or current.get('hosts')!={h:v['root'] for h,v in c['hosts'].items()}:
        raise ValueError('Current contract/host binding changed before import')
    for closure_name,pointer,root in (('source','source_closure_sha256',bundle/'source'),('operation','operation_closure_sha256',bundle)):
        closure=c[closure_name]
        if canonical(closure['files'])!=closure['files_sha256'] or current.get(pointer)!=closure['files_sha256']:
            raise ValueError('Current '+closure_name+' closure changed before import')
        for name,digest in closure['files'].items():
            if raw_sha(safe(root,name))!=digest:raise ValueError('Sealed '+closure_name+' file changed before import: '+name)
    expected_operations={str(path.relative_to(bundle)) for path in (bundle/'operation').glob('*.py')}
    if set(c['operation']['files'])!=expected_operations or not {'operation/campaign.py','operation/monitor.py','operation/qualify.py'}<=expected_operations:
        raise ValueError('Operation Python coverage changed before import')
    integrity_path=safe(bundle,'control/integrity_receipt.json');integrity=read(integrity_path)
    if integrity is None or integrity.get('schema')!='titantpp_contextual_retrieval_control_integrity_v1' or integrity.get('contract_sha256')!=canonical(c) or integrity.get('automatic_retry') is not False:
        raise ValueError('Control integrity binding changed before import')
    files=integrity['files'];closure=canonical(files)
    if integrity.get('files_sha256')!=closure or current.get('control_integrity_receipt_sha256')!=raw_sha(integrity_path) or current.get('control_closure_sha256')!=closure:
        raise ValueError('Control current pointer changed before import')
    expected_controls={str(path.relative_to(bundle)) for path in (bundle/'control').glob('*.py')}
    if set(files)!=expected_controls or not {'control/finalize_once.py','control/test_finalize_offline.py'}<=expected_controls:
        raise ValueError('Control Python coverage changed before import')
    for name,digest in files.items():
        if not name.startswith('control/') or raw_sha(safe(bundle,name))!=digest:
            raise ValueError('Control file changed before import: '+name)
    return c


def load_monitor(bundle):
    preflight(bundle)
    operation=bundle/'operation';sys.path.insert(0,str(operation))
    spec=importlib.util.spec_from_file_location('_contextual_pilot_hourly_monitor',operation/'monitor.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if Path(module.campaign.__file__).resolve()!=operation/'campaign.py':
        raise ValueError('Foreign cached campaign import')
    return module


def actual_kst(value):
    if value is None:return None
    measured=dt.datetime.fromisoformat(value.replace('Z','+00:00'))
    if measured.tzinfo is None:raise ValueError('Actual observation has no timezone')
    return measured.astimezone(KST).strftime('%Y-%m-%d %H:%M:%S KST')


def cost_summary(bundle,host,verifier=None):
    try:
        path=bundle/'qualification'/host/'receipt.json'
        receipt=read(path)
        if receipt is None:return {'technical_status':'pending','cost_eligible_arms':None,'cost_ineligible_arms':None}
        if verifier is not None:verifier(host,receipt)
        permit=read(bundle/('training_permit_'+host+'.json'))
        if permit is not None and (permit.get('host')!=host or permit.get('qualification_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest() or permit.get('qualifications',{}).get(host)!=receipt):
            raise ValueError('Local qualification/own training permit binding changed')
        eligible=receipt.get('cost_candidate_eligibility') or {}
        return {'technical_status':receipt.get('technical_qualification',receipt.get('status','unknown')),
            'cost_eligible_arms':[a for a,v in eligible.items() if v.get('eligible') is True],
            'cost_ineligible_arms':[a for a,v in eligible.items() if v.get('eligible') is False],
            'cost_gate_is_training_launch_gate':False,'scope':'same_GPU_synthetic_native_profiles_only'}
    except (ValueError,OSError,KeyError,TypeError) as exc:
        return {'technical_status':'unverified','cost_eligible_arms':None,'cost_ineligible_arms':None,'reason':str(exc)}


def empty_host(c,host,state,reason,*,query_failed=False):
    jobs=[j for j in c['jobs'] if j['host']==host]
    return {'host':host,'actual_observed_utc':None,'actual_observed_kst':None,
        'counts':{s:len(jobs) if s==state else 0 for s in STATES},
        'rows':[{'job':j,'id':j['id'],'dataset':j['dataset'],'arm':j['arm'],'seed':j['seed'],
                 'state':state,'reason':reason,'actual_saved_epoch':None,'best_epoch':None,'eta':None} for j in jobs],
        'query_failed':query_failed,'training_failure':False,'terminal_cache_reused':False,
        'remote_monitor_invoked':False,'reason':reason,'automatic_retry':False}


def host_once(c,bundle,host,runner,qualification_verifier=None):
    if not (bundle/'deployment'/('launch_receipt_'+host+'.json')).exists():
        reason='native qualification 및 실행 허가 대기'
        if host=='5080' and (bundle/'capacity/pending_5080.json').exists():
            reason='기존54 종료·원본 회수 확인 대기'
        result=empty_host(c,host,'waiting',reason)
        result['qualification']=cost_summary(bundle,host,qualification_verifier)
        return result
    command=[sys.executable,str(bundle/'operation/monitor.py'),'--bundle',str(bundle),'--host',host]
    try:
        response=runner(command,capture_output=True,text=True,timeout=240)
        if response.returncode:raise RuntimeError(response.stderr[-4000:] or 'Operation monitor exited '+str(response.returncode))
        result=json.loads(response.stdout)
        if result.get('query_failed'):
            failed=empty_host(c,host,'uncertain',result.get('error') or '원격 조회 실패',query_failed=True)
            failed['remote_monitor_invoked']=True;failed['qualification']=cost_summary(bundle,host,qualification_verifier)
            return failed
        expected=[j for j in c['jobs'] if j['host']==host]
        if result['host']!=host or [r['job'] for r in result['rows']]!=expected:
            raise ValueError('Foreign monitor host/job coverage')
        if set(result['counts'])!=set(STATES) or sum(result['counts'].values())!=5:
            raise ValueError('Monitor five-condition counts differ')
        if {s:sum(r['state']==s for r in result['rows']) for s in STATES}!=result['counts']:
            raise ValueError('Monitor row states/counts differ')
        result['actual_observed_kst']=actual_kst(result['actual_observed_utc'])
        result['query_failed']=False;result['training_failure']=bool(result['counts']['failed'])
        result['remote_monitor_invoked']=True;result['qualification']=cost_summary(bundle,host,qualification_verifier)
        return result
    except Exception as exc:
        result=empty_host(c,host,'uncertain',str(exc),query_failed=True)
        result['remote_monitor_invoked']=True;result['qualification']=cost_summary(bundle,host,qualification_verifier)
        return result


def number(value):
    return '—' if value is None else str(value)


def duration(value):
    return '—' if value is None else f'{value/3600:.2f}시간'


def markdown(report):
    lines=[f"# 검색5방식 파일럿 관측 — {report['reported_kst']}",'',
        'Taxi seed42의 과학적 설정은 5개이며, 두 GPU에서 같은 설정을 각각 새로 학습하는 총 10회 실행입니다. GPU 반복을 별도 seed나 독립 평가로 세지 않습니다.','']
    for host in HOSTS:
        item=report['hosts'][host];counts=item['counts']
        lines.extend([f"**{host} — 실제 관측 {item.get('actual_observed_kst') or '없음(로컬 대기/조회 실패)'}**",
            ' · '.join(f'{LABELS[s]} {counts[s]}' for s in STATES)])
        if item.get('terminal_cache_reused'):lines.append('검증된 완료 캐시를 사용했으며 실제 관측 시각을 보존했습니다. 추가 SSH 조회는 없습니다.')
        if item.get('query_failed'):lines.append('조회 실패로 상태가 미확정입니다. 학습 실패로 집계하지 않습니다.')
        elif item.get('reason'):lines.append(item['reason'])
        qualification=item.get('qualification') or {}
        technical=qualification.get('technical_status','pending')
        costs=qualification.get('cost_ineligible_arms')
        lines.append(f"native 기술 검증: {technical}; 비용 기준: {'대기' if costs is None else ('전부 적격' if not costs else '초과 '+', '.join(a.removeprefix('titantpp_retrieval_') for a in costs))}. 비용 기준은 학습 시작 허가와 분리합니다.")
        lines.extend(['','| 데이터 | 방식 | seed | 상태 | 저장 epoch | 최선 epoch | 최근10 epoch 기준 조건부 잔여 | 최대300 잔여 |',
            '|---|---|---:|---|---:|---:|---:|---:|'])
        for row in item['rows']:
            eta=row.get('eta') or {}
            lines.append(f"| {row['dataset']} | {row['arm'].removeprefix('titantpp_retrieval_')} | {row['seed']} | {LABELS[row['state']]} | {number(row.get('actual_saved_epoch'))} | {number(row.get('best_epoch'))} | {duration(eta.get('conditional_remaining_seconds'))} | {duration(eta.get('max300_remaining_seconds'))} |")
        lines.append('')
    counts=report['execution_counts']
    lines.extend(['**전체10회 실행 집계**',' · '.join(f'{LABELS[s]} {counts[s]}' for s in STATES),
        'ETA는 해당 실행의 최근 완료 epoch 최대10개를 사용합니다. 이후 개선이 없고 GPU 속도가 유지된다는 조건이며, 체크포인트 재평가·원본 회수 시간은 제외합니다.',
        '이 관측 도우미는 실행 활성화·최종 원본 회수·학습·평가·자동 재시도를 수행하지 않습니다.',''])
    return '\n'.join(lines)


def run(bundle=BUNDLE,hosts=HOSTS,*,runner=subprocess.run,monitor_loader=load_monitor,report_root=None):
    bundle=Path(bundle).resolve()
    if bundle.name!=NAME:raise ValueError('Foreign pilot bundle name')
    if not hosts or any(h not in HOSTS for h in hosts) or len(set(hosts))!=len(hosts):raise ValueError('Foreign/duplicate selected host')
    monitor=monitor_loader(bundle)
    c=monitor.load_bundle(bundle) # Verify source/operation/current before any pending-state read.
    if c['schema']!=NAME:raise ValueError('Foreign pilot schema')
    now=dt.datetime.now(dt.timezone.utc)
    report={'schema':'titantpp_contextual_retrieval_hourly_report_v1','contract_sha256':monitor.digest(c),
        'reported_utc':now.isoformat(),'reported_kst':now.astimezone(KST).strftime('%Y-%m-%d %H:%M:%S KST'),
        'canonical_configs':5,'execution_conditions':10,'seed':42,'GPU_blocks':['5090','5080'],
        'selected_hosts':list(hosts),'hosts':{},'automatic_retry':False,'activation_performed':False,
        'finalizer_invoked':False,'held_out_test_evaluated':False}
    qualification_verifier=(lambda h,q:monitor.campaign.verify_qualification(c,h,q)) if hasattr(monitor,'campaign') else None
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        futures={h:pool.submit(host_once,c,bundle,h,runner,qualification_verifier) for h in hosts}
        for host in hosts:report['hosts'][host]=futures[host].result()
    for host in HOSTS:
        if host not in report['hosts']:
            item=empty_host(c,host,'uncertain','본 관측에서 선택하지 않은 GPU')
            item['query_failed']=False;item['qualification']=cost_summary(bundle,host,qualification_verifier)
            report['hosts'][host]=item
    report['execution_counts']={s:sum(report['hosts'][h]['counts'][s] for h in HOSTS) for s in STATES}
    root=(Path(report_root) if report_root else PROJECT/'reports'/NAME/'hourly').resolve()
    stamp=now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+str(time.time_ns())
    json_path=root/(stamp+'.json');md_path=root/(stamp+'.md')
    write(json_path,report)
    with md_path.open('x') as stream:stream.write(markdown(report))
    latest={'report':str(json_path.resolve()),'markdown':str(md_path.resolve()),
        'report_sha256':raw_sha(json_path),'markdown_sha256':raw_sha(md_path),'reported_utc':report['reported_utc']}
    temporary=root/('latest.'+str(time.time_ns())+'.tmp');write(temporary,latest);temporary.replace(root/'latest.json')
    return report,md_path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,default=BUNDLE)
    p.add_argument('--host',choices=HOSTS)
    a=p.parse_args();report,path=run(a.bundle,(a.host,) if a.host else HOSTS)
    print(markdown(report));print('로컬 기록: '+str(path.resolve()))


if __name__=='__main__':main()
