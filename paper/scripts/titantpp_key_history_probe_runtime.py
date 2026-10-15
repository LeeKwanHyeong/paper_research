"""One owned native diagnostic pipeline: qualification3, onValidation3, onTrain3, offTrain3, offValidation3."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

SCHEMA='titantpp_key_history_probe_v1'
REMOTE='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_key_history_probe_20261016_v1_5090'
DEADLINE=1792679647.915156
ARM='titantpp_cnn_key_only_v1'
STAGES=('qualification','on_validation','on_train','off_train','off_validation')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def safe(root,name):
    root=Path(root).resolve();rel=Path(name);p=root/rel
    require(not rel.is_absolute() and '..' not in rel.parts and p.resolve().is_relative_to(root),'Unsafe diagnostic path')
    require(not any(x.is_symlink() for x in (p,*p.parents) if x.is_relative_to(root)),'Symlink forbidden')
    return p


def verify_contract(root,expected):
    root=Path(root).resolve();require(sha(root/'contract.json')==expected,'Contract raw SHA differs')
    c=read(root/'contract.json')
    require(c.get('schema')==SCHEMA and c.get('contract_sha256')==canonical({k:v for k,v in c.items() if k!='contract_sha256'}),'Contract canonical/schema differs')
    require(c.get('allowed_splits')==['train','validation'] and c.get('stages')==list(STAGES) and c.get('conditions')==['on','off']
            and c.get('validation_tolerance')=={'rtol':1e-5,'atol':1e-5}
            and c.get('max_parallel_per_host')==3 and c.get('new_training') is False and c.get('frozen_selection') is True
            and c.get('checkpoint_reselection') is False and c.get('held_out_test_evaluated') is False
            and c.get('historical_CNN_inference') is False and c.get('automatic_retry') is False,'Forbidden diagnostic scope')
    require(c.get('intervention')=='key_history_taps_only' and c.get('history_tap_multipliers')==[1,0]
            and c.get('tap_masks')=={'on':[1,1,1],'off':[1,0,0]}, 'History-tap intervention differs')
    require(c.get('host')=='5090' and c.get('remote_root')==REMOTE and c.get('deadline_unix')==DEADLINE,'Host/root/deadline differs')
    require(c.get('worker_timeout_seconds')==10800 and c.get('pipeline_timeout_seconds')==43200,'Bounded timeout differs')
    jobs=c['jobs'];require(len(jobs)==3 and {j['seed'] for j in jobs}=={42,52,62}
            and all(j['id']==f"raf_spare_parts__{j['seed']}__{ARM}" and j['arm']==ARM
                    and j['dataset']=='raf_spare_parts' and j['evaluation_host']=='5090' for j in jobs),'Current candidate3 scope differs')
    for name,digest in {**c['input_files'],**c['operation_files']}.items():
        require(sha(safe(root,name))==digest,'Input/operation SHA differs: '+name)
    require(c['operation_files'].get('operation/runtime.py')==sha(__file__),'Runtime operation changed')
    spec=importlib.util.spec_from_file_location('current_rows_worker_guard',root/'operation/evaluate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(module.verify_contract(root,expected)==c,'Worker contract guard differs')
    return c


def process_probe(pid,expected_start=None):
    """Never turn query errors, mid-probe disappearance or PID reuse into absence."""
    p=Path('/proc')/str(pid)
    try:stat1=(p/'stat').read_text()
    except FileNotFoundError:return {'pid':pid,'state':'absent','explicit_absence':True}
    except OSError as e:return {'pid':pid,'state':'query_error','error':str(e)}
    try:
        end=stat1.rfind(')');parts=stat1[end+2:].split();start=int(parts[19]);ppid=int(parts[1])
        argv=(p/'cmdline').read_bytes().split(b'\0');argv=[x.decode(errors='replace') for x in argv if x]
        stat2=(p/'stat').read_text();start2=int(stat2[stat2.rfind(')')+2:].split()[19])
    except FileNotFoundError:return {'pid':pid,'state':'disappeared_during_probe'}
    except (OSError,ValueError,IndexError) as e:return {'pid':pid,'state':'query_error','error':str(e)}
    if start!=start2:return {'pid':pid,'state':'reused_during_probe','start_ticks':start,'later_start_ticks':start2}
    if expected_start is not None and start!=expected_start:return {'pid':pid,'state':'pid_reused','start_ticks':start,'expected_start_ticks':expected_start}
    return {'pid':pid,'state':'present','start_ticks':start,'ppid':ppid,'argv':argv,'process_state':parts[0]}


def ancestors():
    ids=set();pid=os.getpid()
    while pid>0 and pid not in ids:
        ids.add(pid);p=process_probe(pid)
        if p['state']!='present':break
        pid=p['ppid']
    return ids


def root_owners(root,excluded):
    out=[];errors=[];needle=str(root)
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) in excluded:continue
        try:
            if path.stat().st_uid!=os.getuid():continue
        except FileNotFoundError:continue
        except OSError as e:errors.append({'pid':int(path.name),'error':str(e)});continue
        p=process_probe(int(path.name))
        if p['state']=='present' and any(needle in x for x in p['argv']):out.append(p)
        elif p['state']=='query_error':errors.append(p)
    return {'owners':out,'query_errors':errors,'excluded_current_ancestors':sorted(excluded)}


def gpu_probe(uuid):
    r=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,timeout=20)
    require(r.returncode==0,'GPU UUID query failed: '+r.stderr.decode(errors='replace'))
    require(uuid in r.stdout.decode().strip().splitlines(),'Fixed GPU absent')
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,timeout=20)
    require(r.returncode==0,'GPU process query failed: '+r.stderr.decode(errors='replace'))
    pids=[]
    for line in r.stdout.decode().splitlines():
        if not line.strip():continue
        fields=[x.strip() for x in line.split(',')];require(len(fields)==2 and fields[0].isdigit(),'Malformed GPU process query')
        if fields[1]==uuid:pids.append(int(fields[0]))
    return {'gpu_uuid':uuid,'compute_pids':sorted(set(pids)),'query_actual_exit':0,'stdout_sha256':hashlib.sha256(r.stdout).hexdigest(),
            'stderr_sha256':hashlib.sha256(r.stderr).hexdigest(),'observed_unix':time.time()}


def stage_split(stage):
    require(stage in STAGES,'Unapproved diagnostic stage')
    return 'qualification' if stage=='qualification' else 'train' if stage in ('on_train','off_train') else 'validation'


def result_path(job,stage):return f"results/{job}/{stage}_metrics.json"


def validate_predictive_state(receipt,job):
    expected={'state_sha256','all_named_buffers_sha256','context_cache_sha256'}
    before=receipt.get('predictive_state_before');after=receipt.get('predictive_state_after')
    require(isinstance(before,dict) and set(before)==expected and before==after,
            'Predictive parameter/buffer/context state changed or is missing')
    require(all(type(v) is str and len(v)==64 and set(v)<=set('0123456789abcdef') for v in before.values()),
            'Predictive state hash malformed')
    require(before['state_sha256']==job['checkpoint_state_sha256'],'Predictive state differs from selected tensor')
    return before


def validate_result(root,c,job,stage):
    path=safe(root,result_path(job['id'],stage));r=read(path);split=stage_split(stage)
    condition='synthetic_both' if stage=='qualification' else 'on' if stage in ('on_validation','on_train') else 'off'
    require(r.get('status')=='complete' and r.get('job_id')==job['id'] and r.get('stage')==stage
            and r.get('split')==split and r.get('condition')==condition and r.get('contract_sha256')==c['contract_sha256'], 'Worker result identity/status differs')
    require(r.get('parameters_unchanged') is True and r.get('no_training') is True and r.get('branch_function_restored') is True
            and r.get('held_out_test_evaluated') is False and r.get('native_runtime_checked') is True
            and r.get('state_including_buffers_unchanged') is True
            and r.get('state_sha256_before')==r.get('state_sha256_after')==job['checkpoint_state_sha256']
            and r.get('checkpoint_sha256')==job['checkpoint']['sha256'] and r.get('selected_epoch')==job['selected_epoch'], 'Worker state/runtime proof incomplete')
    validate_predictive_state(r,job)
    require(r.get('intervention')=='key_history_taps_only' and r.get('current_tap_preserved') is True
            and r.get('history_taps_disabled') is (condition=='off')
            and r.get('original_full_function_checks_preserved') is True, 'Current/history tap proof differs')
    if stage=='qualification':
        require(r.get('qualification_passed') is True and r.get('real_data_loaded') is False,'Synthetic native qualification failed')
    else:
        require(r.get('population')==c['datasets']['raf_spare_parts']['populations'][split],'Worker target population differs')
        if stage=='on_validation':
            require(r.get('validation_match')=='PASS' and r.get('validation_replay_check',{}).get('six_metrics') is True
                    and r['validation_replay_check'].get('relative_tolerance')==1e-5 and r['validation_replay_check'].get('absolute_tolerance')==1e-5,'On Validation six metrics failed')
            require(r.get('original_population_match') is True and r.get('stored_prediction_rows_available') is False
                    and r.get('rowwise_replay') is None,'On Validation population proof/row availability differs')
        mp=safe(root,r['manifest_path']);require(sha(mp)==r['manifest_sha256'],'Rows manifest SHA differs');m=read(mp)
        require(all(m.get(k)==v for k,v in r['prediction_manifest'].items()) and m.get('job_id')==job['id']
                and m.get('stage')==stage and m.get('split')==split and m.get('condition')==condition
                and m.get('contract_sha256')==c['contract_sha256'],'Rows manifest identity differs')
        require(m['prediction_rows']==r['population']['target_count']==sum(x['rows'] for x in m['parts']),'Row coverage differs')
        for part in m['parts']:
            p=safe(mp.parent,part['path']);require(sha(p)==part['sha256'] and p.stat().st_size==part['bytes'],'Rows shard changed')
    return {'path':str(path.relative_to(root)),'sha256':sha(path)}


def stage_permit(root,c,rawsha,stage):
    require(stage in STAGES,'Unapproved diagnostic stage')
    now=time.time();require(now<c['deadline_unix'],'Absolute deadline expired')
    p={'schema':'titantpp_key_history_probe_permit_v1','authorized':True,'stage':stage,
       'contract_sha256':c['contract_sha256'],'contract_raw_sha256':rawsha,'host':'5090',
       'job_ids':sorted(j['id'] for j in c['jobs']),'evaluator_sha256':c['operation_files']['operation/evaluate.py'],
       'not_before_unix':now,'deadline_unix':min(c['deadline_unix'],now+c['pipeline_timeout_seconds']),
       'new_training':False,'checkpoint_reselection':False,'automatic_retry':False}
    if stage=='on_validation':p['qualification_gate_sha256']=sha(root/'qualification_gate.json')
    if stage in ('on_train','off_train','off_validation'):p['on_validation_gate_sha256']=sha(root/'on_validation_gate.json')
    if stage in ('off_train','off_validation'):p['on_train_gate_sha256']=sha(root/'on_train_gate.json')
    rel=f'permits/{stage}.json';write(root/rel,p);return rel,sha(root/rel)


def stable_child_identity(process,argv,timeout=2.):
    until=time.monotonic()+timeout;first=None;last={}
    while time.monotonic()<until:
        last=process_probe(process.pid)
        if last.get('state')=='present':
            if first is None:first=last['start_ticks']
            require(first==last['start_ticks'],'Spawned PID reused')
            if last.get('argv')==argv:return last
        if process.poll() is not None:break
        time.sleep(.01)
    raise RuntimeError('Spawned PID/argv identity unavailable: '+str(last))


def stop_owned(children,grace=2):
    live=[x['process'] for x in children.values() if x['process'].poll() is None]
    groups=[]
    for p in live:
        try:
            require(os.getpgid(p.pid)==p.pid and os.getsid(p.pid)==p.pid,'Child private group changed')
            os.killpg(p.pid,signal.SIGTERM);groups.append(p.pid)
        except ProcessLookupError:pass
    until=time.monotonic()+grace
    while time.monotonic()<until and any(p.poll() is None for p in live):time.sleep(.02)
    for group in groups:
        try:os.killpg(group,signal.SIGKILL)
        except ProcessLookupError:pass
    for p in live:p.wait(timeout=5)


def dispatch(root,rawsha):
    root=Path(root).resolve();c=verify_contract(root,rawsha)
    require(str(root)==REMOTE,'Native root differs')
    require(time.time()<c['deadline_unix'],'Absolute deadline expired')
    write(root/'runtime/pipeline.claim',{'pid':os.getpid(),'identity':process_probe(os.getpid()),'contract_sha256':c['contract_sha256'],'started_unix':time.time()})
    for job in c['jobs']:
        for stage in STAGES:require(not safe(root,result_path(job['id'],stage)).exists(),'Existing result; retry prohibited')
    children={};outcomes=[];failure=None;handlers={}
    def interrupted(signum,frame):raise RuntimeError('Diagnostic dispatcher interrupted: '+str(signum))
    def finish(pid):
        nonlocal failure
        x=children.pop(pid);p=x['process'];code=p.wait()
        for stream in (x['stdout'],x['stderr']):stream.flush();os.fsync(stream.fileno());stream.close()
        record={'job_id':x['job']['id'],'split':stage_split(x['stage']),'stage':x['stage'],'contract_sha256':c['contract_sha256'],
                'actual_child_exit_observed':True,'exit_code':code,'pid':pid,'start_ticks':x['identity'].get('start_ticks'),
                'launch_identity':x['identity'],'argv':x['argv'],'started_unix':x['started'],'finished_unix':time.time(),
                'stdout_sha256':sha(x['stdout_path']),'stderr_sha256':sha(x['stderr_path'])}
        try:
            require(code==0,'Native worker failed with exit '+str(code))
            record['receipt']=validate_result(root,c,x['job'],x['stage']);record['status']='complete'
        except Exception as error:record['status']='failed';record['error']=str(error);failure=failure or str(error)
        write(root/f"runtime/worker_exits/{x['stage']}_{x['job']['id']}.json",record);outcomes.append(record)
    try:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):handlers[sig]=signal.signal(sig,interrupted)
        for stage in STAGES:
            permit,permit_sha=stage_permit(root,c,rawsha,stage)
            for job in c['jobs']:
                for oldpid,old in list(children.items()):
                    if old['process'].poll() is not None:finish(oldpid)
                if failure:raise RuntimeError(failure)
                # Exactly three fresh owned children; no admission after a detected failure.
                verify_contract(root,rawsha)
                outpath=root/f"logs/{stage}_{job['id']}.stdout";errpath=root/f"logs/{stage}_{job['id']}.stderr"
                outpath.parent.mkdir(exist_ok=True);out=outpath.open('xb');err=errpath.open('xb')
                argv=[c['runtimes']['5090']['python'],str(root/'operation/evaluate.py'),'--root',str(root),'--contract-sha',rawsha,
                      '--job-id',job['id'],'--stage',stage,'--permit-path',permit,'--permit-sha',permit_sha]
                started=time.time()
                try:p=subprocess.Popen(argv,cwd=root,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,
                                       env={**os.environ,**c['environment'],'PYTHONDONTWRITEBYTECODE':'1'})
                except BaseException:out.close();err.close();raise
                identity={}
                children[p.pid]={'process':p,'job':job,'stage':stage,'stdout':out,'stderr':err,'stdout_path':outpath,
                                 'stderr_path':errpath,'identity':identity,'argv':argv,'started':started}
                identity=stable_child_identity(p,argv);children[p.pid]['identity']=identity
                write(root/f"runtime/worker_launches/{stage}_{job['id']}.json",{'identity':identity,'argv':argv,'stage':stage,'job_id':job['id'],
                        'contract_sha256':c['contract_sha256'],'max_parallel':3,'active_owned_children':len(children)})
            while children:
                for pid,x in list(children.items()):
                    if x['process'].poll() is not None:finish(pid)
                    elif time.time()-x['started']>c['worker_timeout_seconds']:raise TimeoutError('Owned diagnostic worker timeout')
                if failure:raise RuntimeError(failure)
                if time.time()>=min(c['deadline_unix'],read(root/permit)['deadline_unix']):raise TimeoutError('Diagnostic stage deadline expired')
                if children:time.sleep(.05)
            stage_results=[x for x in outcomes if x['stage']==stage]
            require(len(stage_results)==3 and all(x['exit_code']==0 and x['status']=='complete' for x in stage_results),'Incomplete stage')
            if stage in ('qualification','on_validation','on_train'):
                gate={'schema':f'titantpp_key_history_probe_{stage}_gate_v1','contract_sha256':c['contract_sha256'],
                      'job_ids':sorted(j['id'] for j in c['jobs']),'all_native_exit0':True,'receipts':{},'worker_exits':{}}
                for x in stage_results:
                    job=next(j for j in c['jobs'] if j['id']==x['job_id'])
                    gate['receipts'][x['job_id']]=validate_result(root,c,job,stage)
                    rel=f"runtime/worker_exits/{stage}_{x['job_id']}.json";gate['worker_exits'][x['job_id']]={'path':rel,'sha256':sha(root/rel)}
                write(root/f'{stage}_gate.json',gate)
        verify_contract(root,rawsha)
    except BaseException as error:
        failure=failure or f'{type(error).__name__}: {error}'
        stop_owned(children)
        for pid in list(children):finish(pid)
        write(root/'runtime/failure.json',{'error':failure,'traceback':traceback.format_exc(),'automatic_retry':False,'contract_sha256':c['contract_sha256']})
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
    result={'schema':'titantpp_key_history_probe_pipeline_v1','status':'complete' if failure is None else 'failed',
            'contract_sha256':c['contract_sha256'],'workers':outcomes,'qualification_count':sum(x['stage']=='qualification' and x['status']=='complete' for x in outcomes),
            'data_worker_count':sum(x['stage']!='qualification' and x['status']=='complete' for x in outcomes),'error':failure,
            'held_out_test_evaluated':False,'no_training':True,'dispatcher_own_exit_directly_observed':False}
    write(root/'runtime/pipeline_complete.json',result)
    return result


def supervise(root,rawsha):
    root=Path(root).resolve();c=verify_contract(root,rawsha)
    require(str(root)==REMOTE,'Native root differs')
    excluded=ancestors();owners=root_owners(root,excluded);gpu=gpu_probe(c['runtimes']['5090']['gpu_uuid'])
    require(not owners['owners'] and not owners['query_errors'] and not gpu['compute_pids'],'Fresh owned/GPU availability not verified')
    write(root/'runtime/admission.json',{'contract_sha256':c['contract_sha256'],'observed_unix':time.time(),'owners':owners,'gpu':gpu,
                                       'current_supervisor':process_probe(os.getpid()),'fresh_availability_verified':True})
    write(root/'runtime/attempt.json',{'contract_sha256':c['contract_sha256'],'started_unix':time.time(),'automatic_retry':False})
    argv=[sys.executable,str(root/'operation/runtime.py'),'--root',str(root),'--contract-sha',rawsha,'--mode','dispatch']
    p=None;identity=None;code=None;error=None;out=root/'runtime/pipeline.stdout';err=root/'runtime/pipeline.stderr';handlers={}
    def interrupted(signum,frame):raise RuntimeError('Diagnostic supervisor interrupted: '+str(signum))
    try:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):handlers[sig]=signal.signal(sig,interrupted)
        with out.open('xb') as so,err.open('xb') as se:
            p=subprocess.Popen(argv,cwd=root,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
            identity=stable_child_identity(p,argv)
            write(root/'runtime/pipeline_launch.json',{'identity':identity,'argv':argv,'contract_sha256':c['contract_sha256']})
            timeout=min(c['pipeline_timeout_seconds'],c['deadline_unix']-time.time());require(timeout>0,'Deadline expired')
            code=p.wait(timeout=timeout)
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        if p is not None and p.poll() is None:stop_owned({p.pid:{'process':p}},grace=10)
        if p is not None:code=p.wait()
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
    if p is not None:
        write(root/'runtime/pipeline_actual_exit.json',{'actual_child_exit_observed':True,'exit_code':code,'pid':p.pid,
                'launch_identity':identity,'contract_sha256':c['contract_sha256'],'error':error,'completed_unix':time.time(),
                'stdout_sha256':sha(out),'stderr_sha256':sha(err)})
    require(error is None and code==0,'Diagnostic pipeline failed: '+str(error or code))
    result=read(root/'runtime/pipeline_complete.json')
    require(result['status']=='complete' and result['qualification_count']==3 and result['data_worker_count']==12,'Incomplete native diagnostic')
    probes=[process_probe(x['pid'],x.get('start_ticks')) for x in result['workers']]
    probes.append(process_probe(p.pid,identity.get('start_ticks')))
    owners=root_owners(root,ancestors());gpu=gpu_probe(c['runtimes']['5090']['gpu_uuid'])
    require(all(x['state']=='absent' and x['explicit_absence'] for x in probes) and not owners['owners']
            and not owners['query_errors'] and not gpu['compute_pids'],'Terminal owned process/GPU absence unverified')
    write(root/'runtime/terminal.json',{'status':'complete','contract_sha256':c['contract_sha256'],'observed_unix':time.time(),
             'owner_probes':probes,'root_owners':owners,'gpu':gpu,'all_owned_absent':True,
             'pipeline_actual_exit':0,'supervisor_own_os_exit_directly_observed':False})
    return 0


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--contract-sha',required=True)
    p.add_argument('--mode',choices=['dispatch','supervise'],required=True);a=p.parse_args(argv)
    if a.mode=='dispatch':return 0 if dispatch(a.root,a.contract_sha)['status']=='complete' else 1
    return supervise(a.root,a.contract_sha)


if __name__=='__main__':raise SystemExit(main())
