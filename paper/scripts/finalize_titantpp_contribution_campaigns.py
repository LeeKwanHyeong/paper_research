"""One-time immutable originals recovery and selected-epoch Validation report."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import tarfile
import time
import sys
_PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(_PROJECT_ROOT))
from paper.scripts import titantpp_contribution_runtime as op
from paper.scripts.deploy_titantpp_contribution_campaigns import bundle,local_contract,remote,prefix,dependency_ready
from paper.scripts.prepare_titantpp_contribution_campaigns import PROJECT,REPORT,write

def preserved_local_files(b):
    out={}
    for pattern in ('originals/*/preservation/manifest.json','originals/*/preservation/archive_receipt.json','retrieval/archive_*','analysis/selected_conditions.json','analysis/VALIDATION_REPORT.md'):
        for p in sorted(b.glob(pattern)):
            if p.is_file():out[str(p.relative_to(b))]=op.sha_file(p)
    for folder in sorted((b/'originals').glob('*')):
        mp=folder/'preservation/manifest.json'
        if mp.exists():
            for name,digest in op.read(mp)['files'].items():
                assert op.sha_file(folder/name)==digest,'Local original drift '+name
                out[str((folder/name).relative_to(b))]=digest
    return out

def terminal_cache(b,c,host):
    m=b/'hourly_monitor';p=m/f'terminal_cache_{host}.json'
    if not p.exists():return None
    assert op.sha_file(p)==op.read(m/f'terminal_pointer_{host}.json')['sha256']
    r=op.read(p)
    assert r['contract_sha256']==op.sha_json(c) and r['verified_terminal'] is True
    assert r['counts']['complete']==sum(j['host']==host for j in c['jobs'])
    assert not r['processes'] and not r['gpu_processes'].strip() and r['supervisor_exit']['returncode']==0
    return r

def retrieve_host(b,c,host):
    write(b/'retrieval'/f'claim_{host}.json',{'contract_sha256':op.sha_json(c),'host':host,'created_unix':time.time(),'automatic_retry':False})
    code=prefix(c,host,idle=True)+'''import tarfile
assert json.loads((b/'supervisor_exit.json').read_text())['returncode']==0 and json.loads((b/'status.json').read_text())['status']=='complete'
ps=subprocess.run(['ps','-eo','args='],capture_output=True,text=True,check=True).stdout
assert not any(str(b) in x for x in ps.splitlines()),'Owned process still present'
pres=b/'preservation';pres.mkdir(exist_ok=False)
with (pres/'claim.json').open('x') as f:json.dump({'contract_sha256':canonical(c),'host':%r,'created_unix':time.time()},f)
files={}
for p in sorted(b.rglob('*')):
 if not p.is_file() or p.is_relative_to(pres) or p.is_relative_to(b/'cache'):continue
 files[str(p.relative_to(b))]=sha(p)
for job in c['jobs']:
 if job['host']!=%r:continue
 folder=b/'run'/job['id'];run=folder/'runs'/job['arm']/'count_only_log_regression'/('seed_'+str(job['seed']))
 endpoints=json.loads((run/'endpoint_replays.json').read_text())
 terminal=json.loads((folder/'terminal_manifest.json').read_text())
 for label in ('selected','last'):
  checkpoint=Path(endpoints[label]['checkpoint_path']);assert checkpoint.is_relative_to(run) and checkpoint.is_file(),'Required binary missing or foreign'
  assert str(checkpoint.relative_to(folder)) in terminal['files'],'Binary absent from terminal manifest'
  assert files[str(checkpoint.relative_to(b))]==terminal['files'][str(checkpoint.relative_to(folder))],'Binary rawSHA drift'
manifest={'contract_sha256':canonical(c),'host':%r,'files':files,'selected_last_binary_included':True,'held_out_test_evaluated':False}
with (pres/'manifest.json').open('x') as f:json.dump(manifest,f,sort_keys=True,indent=2)
archive=pres/'originals.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
 for name in files:tf.add(b/name,arcname=name,recursive=False)
 tf.add(pres/'manifest.json',arcname='preservation/manifest.json',recursive=False)
receipt={'contract_sha256':canonical(c),'host':%r,'archive_sha256':sha(archive),'manifest_sha256':sha(pres/'manifest.json'),'files':len(files),'observed_unix':time.time()}
with (pres/'archive_receipt.json').open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2)
print(json.dumps(receipt))
'''%(host,host,host,host)
    receipt=remote(c,host,code,'preserve_originals',timeout=600);write(b/'retrieval'/f'archive_receipt_{host}.json',receipt)
    archive=b/'retrieval'/f'archive_{host}.tar.gz'
    r=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',host+':'+c['hosts'][host]['root']+'/preservation/originals.tar.gz',str(archive)],capture_output=True,text=True,timeout=1800)
    assert r.returncode==0,'Original transfer failed; no automatic retry'
    assert op.sha_file(archive)==receipt['archive_sha256'],'Archive rawSHA mismatch'
    dest=b/'originals'/host;dest.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            p=Path(member.name);assert member.isfile() and not p.is_absolute() and '..' not in p.parts,'Unsafe original archive'
        tf.extractall(dest,filter='data')
    assert op.sha_file(dest/'preservation/manifest.json')==receipt['manifest_sha256']
    for p,v in op.read(dest/'preservation/manifest.json')['files'].items():assert op.sha_file(dest/p)==v,'Original rawSHA mismatch '+p
    write(dest/'preservation/archive_receipt.json',{k:v for k,v in receipt.items() if k!='local_received_unix'})
    return receipt

def selected_records(b,c):
    records=[]
    caches={h:terminal_cache(b,c,h) for h in c['hosts']}
    for j in c['jobs']:
        folder=b/'originals'/j['host']/'run'/j['id'];m=op.read(folder/'terminal_manifest.json')
        assert op.sha_file(folder/'terminal_manifest.json')==caches[j['host']]['jobs'][j['id']]['terminal_manifest_sha256'],'Observed terminal changed before recovery'
        assert m['job']==j and m['contract_sha256']==op.sha_json(c) and m['scientific_success'] is True
        for name,digest in m['files'].items():assert op.sha_file(folder/name)==digest
        run=folder/'runs'/j['arm']/op.VARIANT/('seed_'+str(j['seed']))
        endpoints=op.read(run/'endpoint_replays.json');history=op.read(run/'history.json')['history']
        binary={}
        remote_folder=Path(c['hosts'][j['host']]['root'])/'run'/j['id']
        remote_run=remote_folder/'runs'/j['arm']/op.VARIANT/('seed_'+str(j['seed']))
        for label in ('selected','last'):
            checkpoint=Path(endpoints[label]['checkpoint_path']);assert checkpoint.is_relative_to(remote_run),'Foreign binary path'
            name=str(checkpoint.relative_to(remote_folder));assert name in m['files'],'Missing selected/last binary in terminal manifest'
            assert op.sha_file(folder/name)==m['files'][name],'Required binary rawSHA drift'
            binary[label]={'relative_path':name,'raw_sha256':m['files'][name]}
        finite=[x for x in history if math.isfinite(x['val_qty_rmse'])];best=min(finite,key=lambda x:x['val_qty_rmse'])
        assert best['epoch']==endpoints['best_epoch'],'Selection changed'
        assert math.isclose(best['val_qty_rmse'],endpoints['selected']['qty_rmse'],rel_tol=1e-10,abs_tol=1e-8)
        data=next(d for d in c['datasets'] if d['dataset_id']==j['dataset']);op.verify_full_validation(endpoints['selected'],data);op.verify_full_validation(endpoints['last'],data)
        initial=op.read(folder/'initialization.json')[j['arm']]
        assert initial==endpoints['initial_state_sha256'],'Initial binding changed'
        records.append({**j,'best_epoch':endpoints['best_epoch'],'last_epoch':endpoints['completed_epochs'],
            'initial_state_sha256':initial,'selected':endpoints['selected'],'last':endpoints['last'],
            'terminal_manifest_sha256':op.sha_file(folder/'terminal_manifest.json'),'endpoint_raw_sha256':op.sha_file(run/'endpoint_replays.json'),
            'train_exposure':op.read(run/'exposure.json'),'binary_originals':binary})
    return records

def validate_pair_exposure(a,b):
    x,y=a['train_exposure'],b['train_exposure'];n=min(len(x['train']),len(y['train']))
    assert x['train'][:n]==y['train'][:n],'Cross-stage paired train order drift'
    assert len({r['batch_order_sha256'] for rows in (x['validation'],y['validation']) for r in rows})==1,'Cross-stage full Validation order drift'
    assert (a['host'],a['dataset'],a['seed'])==(b['host'],b['dataset'],b['seed']),'Cross-stage paired owner changed'

def validate_cross_stage(b,c,records,baselines,previous):
    assert dependency_ready(c),'Unverified prior stage originals/report'
    prior=op.read(previous/'execution_contract.json')
    assert op.sha_json(prior)==c['prior_stage']['contract_sha256']
    assert c['datasets']==prior['datasets'] and c['input_files']==prior['input_files'],'Cross-stage input contract drift'
    assert c['training']==prior['training'] and c['source']['parent_files']==prior['source']['parent_files'],'Cross-stage shared science drift'
    for r in records:
        baseline=next(x for x in baselines if (x['dataset'],x['seed'])==(r['dataset'],r['seed']))
        validate_pair_exposure(r,baseline);h=r['host'];d=r['dataset'];seed=str(r['seed'])
        q=op.read(b/'originals'/h/'qualification/receipt.json');old=op.read(previous/'originals'/h/'qualification/receipt.json')
        assert q['runtime']==old['runtime'],'Cross-stage native Runtime drift'
        assert c['hosts'][h]['python']==prior['hosts'][h]['python'] and c['hosts'][h]['gpu_uuid']==prior['hosts'][h]['gpu_uuid']
        assert q['initialization'][d][seed]['titantpp_cnn_gru54']==old['initialization'][d][seed]['titantpp_cnn_gru54']==baseline['initial_state_sha256'],'Cross-stage common initialization drift'
        assert q['initialization'][d][seed][r['arm']]==r['initial_state_sha256'],'Retrieved fit initialization drift'
        left=dict(q['inputs'][d]['input_receipt']);right=dict(old['inputs'][d]['input_receipt'])
        left.pop('resolved_inputs',None);right.pop('resolved_inputs',None)
        assert left==right,'Cross-stage Train/Validation statistics or population drift'

def report(b,c,records):
    stage=c['stage'];lines=['# '+stage+' 최종 Validation 비교','', '모든 수치는 각 조건의 최초 strict finite 최소 전체 수량 RMSE epoch에서 함께 집계했다. Test·새 독립 평가 성과가 아니다.','',
        '|dataset|arm|seed|GPU|선택/종료epoch|전체 RMSE|전체 MAE|전체 TimeNLL|큰 수량 RMSE|큰 수량 MAE|큰 수량 TimeNLL|','|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    allrecords=list(records)
    if stage=='retrieval':
        previous=PROJECT/c['prior_stage']['bundle'];refs=op.read(previous/'analysis/selected_conditions.json')['conditions']
        baselines=[r for r in refs if r['arm']=='titantpp_cnn_gru54'];assert len(baselines)==9
        validate_cross_stage(b,c,records,baselines,previous)
        allrecords+=baselines
    for r in sorted(allrecords,key=lambda x:(x['dataset'],x['arm'],x['seed'])):
        e=r['selected'];values=[e[k] for k in ('qty_rmse','qty_mae','time_nll')]+[e['tail'][k] for k in ('qty_rmse','qty_mae','time_nll')]
        lines.append('|'+ '|'.join([r['dataset'],r['arm'],str(r['seed']),r['host'],f"{r['best_epoch']}/{r['last_epoch']}",*[f'{x:.6g}' for x in values]])+'|')
    summary=[]
    for d,a in sorted({(r['dataset'],r['arm']) for r in allrecords}):
        group=[r for r in allrecords if (r['dataset'],r['arm'])==(d,a)];assert sorted(r['seed'] for r in group)==list(op.SEEDS)
        metrics={}
        for prefix in ('overall','tail'):
            for k in ('qty_rmse','qty_mae','time_nll'):
                v=[r['selected'][k] if prefix=='overall' else r['selected']['tail'][k] for r in group]
                metrics[prefix+'_'+k]={'mean':statistics.mean(v),'sample_sd':statistics.stdev(v),'seeds':list(op.SEEDS)}
        summary.append({'dataset':d,'arm':a,'metrics':metrics})
    lines+=['','3seed 평균 ± 표본 표준편차:', '', '|dataset|arm|전체 RMSE|전체 MAE|전체 TimeNLL|큰 수량 RMSE|큰 수량 MAE|큰 수량 TimeNLL|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append('|'+ '|'.join([r['dataset'],r['arm'],*[f"{m['mean']:.6g} ± {m['sample_sd']:.6g}" for m in r['metrics'].values()]])+'|')
    decision={'stage':stage,'development_support':None,'independent_confirmation':False}
    if stage in ('history','retrieval'):
        candidate='titantpp_cnn_gru54' if stage=='history' else 'titantpp_retrieval_fixed'
        controls=list(c['arms'][1:]) if stage=='history' else ['titantpp_cnn_gru54','titantpp_retrieval_weighted','titantpp_retrieval_generic512']
        details=[];supported=[]
        for d in sorted({r['dataset'] for r in allrecords}):
            wins=[]
            for seed in op.SEEDS:
                cand=next(r for r in allrecords if (r['dataset'],r['seed'],r['arm'])==(d,seed,candidate))
                comps={a:next(r for r in allrecords if (r['dataset'],r['seed'],r['arm'])==(d,seed,a)) for a in controls}
                comparisons={a:{'overall_RMSE_strict_improvement':cand['selected']['qty_rmse']<r['selected']['qty_rmse'],
                    'tail_RMSE_delta':cand['selected']['tail']['qty_rmse']-r['selected']['tail']['qty_rmse'],
                    'overall_MAE_delta':cand['selected']['qty_mae']-r['selected']['qty_mae'],'overall_TimeNLL_delta':cand['selected']['time_nll']-r['selected']['time_nll'],
                    'tail_MAE_delta':cand['selected']['tail']['qty_mae']-r['selected']['tail']['qty_mae'],'tail_TimeNLL_delta':cand['selected']['tail']['time_nll']-r['selected']['tail']['time_nll']} for a,r in comps.items()}
                wins.append(all(x['overall_RMSE_strict_improvement'] for x in comparisons.values()));details.append({'dataset':d,'seed':seed,'comparisons':comparisons})
            if all(wins):supported.append(d)
        decision.update(same_datasets_all3seed_strict_RMSE_wins=supported,accuracy_development_support=len(supported)>=2,paired_results=details)
        cost=[]
        for h in c['hosts']:
            q=op.read(b/'originals'/h/'qualification/receipt.json')
            for d,lengths in q['measurements'].items():
                for length,samples in lengths.items():
                    ref=samples['titantpp_cnn_gru54']
                    for arm,m in samples.items():
                        ratios={key:m[key]/ref[key] for key in ('inference_median_seconds','training_step_median_seconds','peak_allocated_bytes')}
                        eligible=ratios['inference_median_seconds']<=1.2 and ratios['training_step_median_seconds']<=1.3 and ratios['peak_allocated_bytes']<=1.2
                        cost.append({'host':h,'dataset':d,'length':int(length),'arm':arm,'ratios_vs_mean':ratios,'candidate_cost_eligible':eligible})
        cost_datasets=[]
        for d in sorted({r['dataset'] for r in allrecords}):
            cells=[x for x in cost if x['dataset']==d and x['arm']=='titantpp_retrieval_fixed']
            if stage=='history' or cells and all(x['candidate_cost_eligible'] for x in cells):cost_datasets.append(d)
        joint=sorted(set(supported)&set(cost_datasets))
        decision.update(native_cost=cost,cost_eligible_datasets=cost_datasets,accuracy_and_cost_supported_datasets=joint,
            development_support=len(joint)>=2)
        lines+=['',f"고정 기준의 개발 단계 지지: {decision['development_support']}. 같은 데이터의 모든3seed에서 필요한 대조를 모두 이긴 데이터: {supported}.",
            '모든 불리한 RMSE/MAE/TimeNLL과 native 비용은 JSON에 보존했다. 통계적 유의성·외부 우월성·새 독립 Test 증명이 아니다.']
        lines+=['','같은 GPU의 mean 대비 native 비용:', '', '|GPU|dataset|길이|arm|추론비|학습step비|peak메모리비|비용적격|','|---|---|---:|---|---:|---:|---:|---|']
        for x in cost:
            v=x['ratios_vs_mean'];lines.append(f"|{x['host']}|{x['dataset']}|{x['length']}|{x['arm']}|{v['inference_median_seconds']:.4f}|{v['training_step_median_seconds']:.4f}|{v['peak_allocated_bytes']:.4f}|{x['candidate_cost_eligible']}|")
    lines+=['','설계 한계: '+json.dumps(c['architecture'],ensure_ascii=False)]
    write(b/'analysis/selected_conditions.json',{'conditions':records,'seed_summary':summary,'decision':decision})
    (b/'analysis/VALIDATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    return decision

def finalize(stage):
    b,c=local_contract(stage);completion=b/'analysis/completion_receipt.json'
    if completion.exists():
        r=op.read(completion)
        assert r['contract_sha256']==op.sha_json(c)
        for p,v in r['local_files'].items():assert op.sha_file(b/p)==v,'Finalized local drift '+p
        return {'stage':stage,'status':'already_finalized_local_rawSHA_verified','remote_calls':0,'decision':r['decision']}
    cache={h:terminal_cache(b,c,h) for h in c['hosts']}
    if not all(cache.values()):return {'stage':stage,'status':'pending','remote_calls':0,'reason':'All scientific terminals and actual shutdown not yet verified'}
    claim=b/'retrieval/finalize.claim'
    if claim.exists():return {'stage':stage,'status':'claimed_or_failed_preserved','remote_calls':0,'user_action_required':True}
    write(claim,{'contract_sha256':op.sha_json(c),'created_unix':time.time(),'automatic_retry':False})
    try:
        for h in c['hosts']:retrieve_host(b,c,h)
        records=selected_records(b,c);decision=report(b,c,records)
        files=preserved_local_files(b)
        result={'stage':stage,'status':'complete','contract_sha256':op.sha_json(c),'completed_conditions':len(records),
            'actual_owner_shutdown_verified':True,'originals_rawSHA_verified':True,'final_Validation_report_verified':True,
            'local_files':files,'decision':decision,'completed_unix':time.time(),'CPU_binary_reinference_audit':False,'Test_evaluation':False,'new_independent_evaluation':False}
        write(completion,result);return {'stage':stage,'status':'complete','completed_conditions':len(records),'decision':decision}
    except BaseException as exc:
        write(b/'retrieval/failure.json',{'type':type(exc).__name__,'message':str(exc),'created_unix':time.time(),'automatic_retry':False})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=tuple(op.STAGES),required=True);a=p.parse_args();print(json.dumps(finalize(a.stage),ensure_ascii=False))
