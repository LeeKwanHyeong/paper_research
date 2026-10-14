"""Launch/observe the explicitly approved three-slot operational extension."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import time
from types import FunctionType
import uuid

from paper.scripts import advance_titantpp_gru_cap_manual2 as prior
from paper.scripts import launch_titantpp_gru_cap_manual2 as launch
from paper.scripts import titantpp_components_status_transport_v2 as transport
from paper.scripts import titantpp_gru_cap_parallel3_v2_runtime as runtime

PROJECT, BUNDLE = prior.PROJECT, prior.BUNDLE
OUTPUT = prior.OUTPUT.parent / 'hourly_monitor_parallel3_v2'
REPORT = OUTPUT.parent
CONTROL = BUNDLE / 'parallel3_v2_control'
OLD_MONITOR = '5d798de08dd9c87c0ce3783230d88b86866b9492de74738808fdf240a54aeafa'
OLD_COLLECTION = '99daf73bb1e8d895292adb5a7c7b234fe27b6f1e60af8b6c9a02e084be8b8ecd'
need, read, sha, canonical, write = runtime.need, runtime.read, runtime.sha, runtime.canonical, runtime.write


def gate(digest):
    need(sha(CONTROL / 'integrity.json') == digest, 'Parallel seal changed; remote0')
    seal = read(CONTROL / 'integrity.json')
    need(canonical(seal['files']) == seal['files_sha256'], 'Parallel dependency closure changed')
    for name, expected in seal['files'].items():
        need(sha(PROJECT / name) == expected, 'Parallel dependency changed: ' + name)
    prior.verify_extra_seal(prior.OUTPUT / 'monitor_integrity.json', OLD_MONITOR)
    prior.verify_extra_seal(BUNDLE / 'collection_integrity.json', OLD_COLLECTION)
    old_observer, old_collector = prior.routed_helpers(launch)
    old_collector.verify_seal(BUNDLE, prior.OUTPUT, OLD_COLLECTION)
    c, _ = launch.check_bundle()
    need(sha(BUNDLE / 'execution_contract.json') == runtime.CONTRACT_RAW, 'Original contract changed')
    return c, seal, old_observer, old_collector


INSTALL = r'''
import base64,hashlib,json,os,pathlib,shlex,subprocess,sys,time
p=json.load(sys.stdin);root=pathlib.Path(p['root'])
assert p['host'] in ('5080','5090') and root.name=='titantpp_gru_residual_cap_validation_20261014_v1_'+p['host']+'_manual2'
assert root.is_absolute() and root.resolve()==root
assert hashlib.sha256((root/'execution_contract.json').read_bytes()).hexdigest()==p['contract_raw_sha256']
c=json.loads((root/'execution_contract.json').read_text());spec=c['hosts'][p['host']]
assert spec['root']==str(root) and spec['python']==p['python']
old=root/'parallel3'
assert {x.name for x in old.iterdir()}=={'runtime.py','approval.json','pipeline.log'}
assert "TypeError: 'set' object is not subscriptable" in (old/'pipeline.log').read_text()
assert hashlib.sha256((old/'runtime.py').read_bytes()).hexdigest()=='021352e23ec41002ba1d7a11e4df9ff47a93332bedfb40bacad27b5f253b3300'
assert not (root/'failure.json').exists() and not (root/'parallel3_v2').exists()
assert json.loads((root/'status.json').read_text())['status']=='running'
raw=base64.b64decode(p['runtime_base64'],validate=True)
assert hashlib.sha256(raw).hexdigest()==p['approval']['runtime_raw_sha256']
folder=root/'parallel3_v2';folder.mkdir()
with (folder/'runtime.py').open('xb') as f:f.write(raw)
with (folder/'approval.json').open('x') as f:json.dump(p['approval'],f,ensure_ascii=False,indent=2);f.write('\n')
cmd=[spec['python'],str(folder/'runtime.py'),'--contract',str(root/'execution_contract.json'),'--host',p['host'],'--mode','pipeline']
shell='cd '+shlex.quote(spec['source_root'])+' && exec env '+' '.join(shlex.quote(k+'='+v) for k,v in spec['environment'].items())+' '+' '.join(shlex.quote(x) for x in cmd)+' > '+shlex.quote(str(folder/'pipeline.log'))+' 2>&1'
subprocess.run([spec['tmux_binary'],'new-session','-d','-s',spec['tmux']+'_parallel3_v2',shell],check=True,capture_output=True,timeout=20)
print(json.dumps({'host':p['host'],'root':str(root),'status':'parallel_extension_started','argv':cmd,'started_unix':time.time(),'automatic_retry':False}))
'''


def install(host, digest):
    c, seal, _, _ = gate(digest)
    spec = c['hosts'][host]
    path = CONTROL / 'install' / host
    need(not path.exists(), 'Existing install intent; no retry')
    path.mkdir(parents=True)
    approval = {'user_instruction': runtime.APPROVAL, 'host': host, 'maximum_concurrency': 3,
        'runtime_raw_sha256': sha(runtime.__file__), 'contract_raw_sha256': runtime.CONTRACT_RAW,
        'preserve_existing_training': True, 'same_nine_fits': True, 'automatic_retry': False, 'Test': False}
    write(path / 'intent.json', approval, True)
    payload = dict(host=host, root=spec['root'], python=spec['python'], approval=approval,
        contract_raw_sha256=runtime.CONTRACT_RAW, runtime_base64=base64.b64encode(Path(runtime.__file__).read_bytes()).decode())
    cmd = ' '.join(shlex.quote(x) for x in ['env','PYTHONDONTWRITEBYTECODE=1',spec['python'],'-c',INSTALL])
    out = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,cmd],
        input=json.dumps(payload).encode(),capture_output=True,timeout=120)
    for name, value in [('stdout',out.stdout),('stderr',out.stderr)]:
        (path / (name + '.bin')).write_bytes(value)
    write(path / 'exit.json', {'actual_transport_exit':out.returncode,'stdout_sha256':hashlib.sha256(out.stdout).hexdigest(),
        'stderr_sha256':hashlib.sha256(out.stderr).hexdigest()}, True)
    need(out.returncode == 0, 'Parallel install failed; preserved output, no retry')
    return json.loads(out.stdout)


def replace_one(text, old, new):
    need(text.count(old) == 1, 'Frozen transport anchor changed')
    return text.replace(old,new,1)


EXTRA_PROBES = r'''
def parallel_process_observations(state,files,pids,c,host):
 observations,processes,owners=collect_process_observations(state,files,pids,c,host)
 extra={}
 def add(pid,source,expected=None):
  status_pid(pid,source);row=extra.setdefault(pid,{'sources':[],'expected':[]});row['sources'].append(source)
  if expected:row['expected'].append(expected)
 spec=c['hosts'][host];script=spec['root']+'/parallel3_v2/runtime.py'
 dispatch=[spec['python'],script,'--contract',spec['root']+'/execution_contract.json','--host',host,'--mode','dispatch']
 wrapper=dispatch[:-1]+['pipeline']
 for name in ('parallel3_v2/launch.json','parallel3_v2/exit.json','parallel3_v2/supervisor.claim'):
  if name not in files:continue
  r=status_record(files,name);status_need(r['argv']==dispatch,'Parallel dispatcher argv differs')
  add(r['pid'],name,{'argv':dispatch,'ppid':r['ppid'],'pgrp':r['pid']});add(r['ppid'],name,{'argv':wrapper})
 if 'parallel3_v2/status.json' in files:
  s=status_record(files,'parallel3_v2/status.json')
  for row in s.get('active_jobs',{}).values():
   add(row['worker_group_pid'],'parallel3_v2/status.json',{'argv':row['worker_guard_argv'],'ppid':s['supervisor_pid'],'pgrp':row['worker_group_pid']})
 for name in files:
  if name.startswith('parallel3_v2/worker_exits/'):
   r=status_record(files,name)
   add(r['worker_group_pid'],name,{'argv':r['argv'],'ppid':r['supervisor_pid'],'pgrp':r['worker_group_pid']})
   for key in ('pid','ppid','supervisor_pid'):add(r[key],name)
 for pid,row in extra.items():
  try:
   actual=status_proc_read(pid); matched=actual['identity_stable_during_query'] and all(all(actual.get(k)==v for k,v in e.items()) for e in row['expected'])
   label='live' if matched else 'reused_pid'
   observations[str(pid)]={'pid':pid,'state':label,'actual':actual,'evidence_sources':row['sources']}
   processes[str(pid)]={**actual,'observation_state':label}
  except (FileNotFoundError,ProcessLookupError):
   observations[str(pid)]={'pid':pid,'state':'absent','absence_observed':True,'evidence_sources':row['sources']}
  except (OSError,ValueError,UnicodeError) as error:
   observations[str(pid)]={'pid':pid,'state':'query_error','absence_observed':False,'error':str(error)}
   processes[str(pid)]={'pid':pid,'observation_state':'query_error'}
 return observations,processes,sorted(set(owners)|set(extra))
'''


def status_script(launcher):
    original = transport.patched_remote_script(launcher)
    anchor = "   for name in dict.fromkeys(names):"
    original = replace_one(original, anchor,
        "   names += ['parallel3_v2/'+n for n in ('runtime.py','approval.json','handoff.json','launch.json','exit.json','complete.json','failure.json','status.json','supervisor.claim')]\n"
        "   names += ['parallel3_v2/worker_exits/'+j['id']+'.json' for j in c['jobs'] if j['host']==host]\n" + anchor)
    original = replace_one(original, 'collect_process_observations(state,files,pids,c,host)\n',
        'parallel_process_observations(state,files,pids,c,host)\n')
    text = EXTRA_PROBES + '\n' + original
    compile(text,'parallel3_v2-status','exec')
    return text


def acquire(bundle, launcher, c, host, diagnostics):
    text = status_script(launcher)
    # read_status_once supplies the same bounded diagnostic transport and writes raw snapshots.
    namespace = dict(vars(transport))
    namespace['patched_remote_script'] = lambda _: text
    for key in ('read_status_once',):
        fn = getattr(transport,key); namespace[key] = FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    # Constructor reads patched_remote_script through its original module globals;
    # provide a dedicated subclass instead of mutating the module.
    class Remote(transport.StatusOnlyRemote):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.remote_script = text
    namespace['StatusOnlyRemote'] = Remote
    return namespace['read_status_once'](bundle,launcher,c,host,diagnostics)


def live(snapshot, pid, expected=None):
    row = snapshot['process_observations'].get(str(pid), {})
    need(row.get('state') == 'live' and row.get('actual',{}).get('identity_stable_during_query') is True,
        'Live process identity unresolved: ' + str(pid))
    if expected:
        need(all(row['actual'].get(k) == v for k,v in expected.items()), 'Live process identity mismatch')


def parent_done(files, snapshot, c, host, h, obs):
    if 'failure.json' not in files:
        return False
    f = json.loads(files['failure.json']);first=h['remaining_jobs'][0]
    reason = (f.get('type')=='ValueError' and f.get('message') in ('GPU occupied before next fit','Deadline reached or fresh-run rule failed')) or (
        f.get('type')=='FileExistsError' and c['hosts'][host]['root']+'/claims/'+first['id']+'.json' in f.get('message',''))
    need(reason and f.get('cleanup') is None and f.get('active_job')==first and set(f['completed'])==set(h['retained_job_ids'])
        and hashlib.sha256(files['claims/'+first['id']+'.json']).hexdigest()==h['first_claim_raw_sha256'], 'Unexpected original failure')
    if 'phases/train_exit.json' not in files:
        return False
    launch,exited=[json.loads(files[n]) for n in ('phases/train_launch.json','phases/train_exit.json')]
    need(exited['os_exit_code']==1 and all(exited[k]==launch[k] for k in ('pid','ppid','argv','contract_sha256')),
        'Administrative parent exit not verified')
    for pid in (exited['pid'],exited['ppid']):
        if snapshot['process_observations'].get(str(pid),{}).get('state')!='absent':return False
    return True


def completed(snapshot,c,host,job,files,launcher,obs,is_new):
    ns = obs.verifier_namespace(launcher)
    if is_new:
        mapped=dict(files)
        mapped['worker_exits/'+job['id']+'.json']=files['parallel3_v2/worker_exits/'+job['id']+'.json']
        mapped['status.json']=files['parallel3_v2/status.json']
        ns['decode_snapshot']=lambda *_:mapped
        ns['_dispatch_identity']=lambda *_:json.loads(files['parallel3_v2/launch.json'])
        src=inspect.getsource(obs.legacy.verify_completed_job)
        src=replace_one(src,'c["hosts"][host]["operation_root"] + "/campaign.py"',
            'c["hosts"][host]["root"] + "/parallel3_v2/runtime.py"')
        exec(compile(src,obs.legacy.__file__,'exec'),ns)
    row=ns['verify_completed_job'](snapshot,c,host,job)
    exitname=('parallel3_v2/' if is_new else '')+'worker_exits/'+job['id']+'.json'
    worker=json.loads(files[exitname])
    need(worker['pid'] != worker['worker_group_pid'] and worker['ppid']==worker['worker_group_pid'], 'Scientific worker/guard identity missing')
    obs.require_absent(snapshot,{worker['pid'],worker['worker_group_pid']},'Completed worker not explicitly absent')
    return row


def verify(snapshot,c,host,launcher,obs):
    files=launcher.decode_snapshot(snapshot,c,host)
    h=json.loads(files['parallel3_v2/handoff.json']); approval=json.loads(files['parallel3_v2/approval.json'])
    need(approval['user_instruction']==runtime.APPROVAL and approval['maximum_concurrency']==3
        and approval['runtime_raw_sha256']==sha(runtime.__file__) and h['contract_sha256']==canonical(c)
        and hashlib.sha256(files['parallel3_v2/runtime.py']).hexdigest()==approval['runtime_raw_sha256']
        and h['host']==host and h['maximum_concurrency']==3,'Parallel ownership binding changed')
    pl,pc=[json.loads(files[n]) for n in ('parallel3_v2/launch.json','parallel3_v2/supervisor.claim')]
    expected=[c['hosts'][host]['python'],c['hosts'][host]['root']+'/parallel3_v2/runtime.py','--contract',
        c['hosts'][host]['root']+'/execution_contract.json','--host',host,'--mode','dispatch']
    need(pl['argv']==expected and pl['contract_sha256']==canonical(c)
        and all(pc[k]==pl[k] for k in ('pid','ppid','argv','contract_sha256')),'Parallel supervisor identity changed')
    alljobs=[j for j in c['jobs'] if j['host']==host]
    retained=h['retained_job_ids'];new={j['id'] for j in h['remaining_jobs']}
    need([j['id'] for j in alljobs]==retained+[j['id'] for j in h['remaining_jobs']] and not set(retained)&new,'Job ownership partition changed')
    need('parallel3_v2/failure.json' not in files,'Parallel failure preserved; no retry')
    ps=json.loads(files['parallel3_v2/status.json']);old=json.loads(files['status.json'])
    terminal_parent=parent_done(files,snapshot,c,host,h,obs)
    active={**old.get('active_jobs',{}),**ps.get('active_jobs',{})}
    need(len(active)<=3 and set(ps.get('active_jobs',{}))<=new,'Parallel worker cap/owner mismatch')
    rows=[]
    for job in alljobs:
        name=job['id'];run=f"run/{name}/runs/{job['arm']}/count_only_log_regression/seed_{job['seed']}"
        if 'run/'+name+'/terminal_manifest.json' in files:
            rows.append(completed(snapshot,c,host,job,files,launcher,obs,name in new));continue
        row={'job':job,'status':'waiting','saved_epoch':None,'best_epoch':None,'selected_checkpoint_file_present':False}
        if name in active:
            a=active[name];parent=json.loads(files['parallel3_v2/launch.json' if name in new else 'phases/train_launch.json'])
            live(snapshot,parent['pid'],{'argv':parent['argv'],'ppid':parent['ppid']});live(snapshot,parent['ppid'])
            live(snapshot,a['worker_group_pid'],{'argv':a['worker_guard_argv'],'ppid':parent['pid'],'pgrp':a['worker_group_pid']})
            matches=[pid for pid in snapshot['gpu_pids'] if snapshot['processes'][str(pid)].get('ppid')==a['worker_group_pid']]
            need(len(matches)==1,'Scientific GPU worker not uniquely observed');live(snapshot,matches[0])
            lease=json.loads(files['leases/'+name+'.json']);claim=json.loads(files['claims/'+name+'.json'])
            need(lease['contract_sha256']==canonical(c) and lease['job']==name and lease['supervisor_pid']==parent['pid']
                and lease['worker_group_pid']==a['worker_group_pid'] and snapshot['observed_unix']<lease['expires_unix']<=claim['deadline_unix'],
                'Live lease invalid')
            row['status']='running_before_first_checkpoint'
            saved=snapshot.get('checkpoints',{}).get(name)
            if saved:
                if saved['snapshot_complete']:
                    history=json.loads(files[run+'/history.json'])['history'];need(len(history)==saved['saved_epoch'],'Saved history mismatch')
                    finite=[(r['val_qty_rmse'],r['epoch']) for r in history if obs.finite(r.get('val_qty_rmse'))]
                    need(finite,'No finite saved Validation selector')
                    row.update(status='running',saved_epoch=saved['saved_epoch'],best_epoch=min(finite)[1],
                        selected_checkpoint_file_present=saved['best_selected_file_present'])
                else:row['status']='checkpoint_snapshot_unconfirmed'
        rows.append(row)
    groups={r['worker_group_pid'] for r in active.values()}
    need(all(snapshot['processes'][str(pid)].get('pgrp') in groups for pid in snapshot['gpu_pids']), 'Foreign GPU process present')
    result={'status':'training_running','jobs':rows,'completed':sum(r['status']=='complete' for r in rows),
        'total':len(alljobs),'identity_verified':True,'live_process_probes_verified':True,
        'original_dispatcher_administrative_terminal_verified':terminal_parent,'maximum_concurrency':3}
    if result['completed']==len(alljobs):
        need(terminal_parent and ps['status']=='complete' and not active and snapshot['gpu_pids']==[],'Terminal ownership incomplete')
        a,b=[json.loads(files[n]) for n in ('parallel3_v2/launch.json','parallel3_v2/exit.json')]
        need(b['os_exit_code']==0 and all(a[k]==b[k] for k in ('pid','ppid','argv','contract_sha256'))
            and json.loads(files['parallel3_v2/complete.json'])['actual_dispatcher_exit']==0,'Parallel actual exit incomplete')
        pids=set(snapshot['observed_owner_pids']);obs.require_absent(snapshot,pids,'Terminal owner process remains')
        result.update(status=obs.TERMINAL,owner_process_absence_verified=True,owner_process_probe_count=len(pids))
    return result


def helpers(c,obs,collector):
    original_verify=obs.verify_monitor
    obs.verify_monitor=lambda output,digest,contract,bundle=BUNDLE:original_verify(output,digest,contract,bundle)
    obs.verify_snapshot=lambda snapshot,contract,host,launcher:verify(snapshot,contract,host,launcher,obs)
    # Existing cache save/load, bridge, ETA, counts and immutable collection are reused.
    obs.observe_host.__defaults__ = obs.observe_host.__defaults__
    return obs,collector


def advance(digest):
    c,seal,obs,collector=gate(digest);obs,collector=helpers(c,obs,collector)
    monitor_sha=seal['monitor_integrity_raw_sha256'];collection_sha=seal['collection_integrity_raw_sha256']
    # These closures only check the new monitor/collection seals, while preserving
    # all original scientific and raw-evidence checks.
    old_monitor=obs.verify_monitor
    def verify_monitor(output,digest,contract,bundle=BUNDLE):
        path=Path(output)/'monitor_integrity.json';need(Path(output).resolve()==OUTPUT and sha(path)==digest,'Parallel monitor path/SHA changed')
        s=read(path);need(s['contract_sha256']==canonical(contract) and canonical(s['files'])==s['files_sha256'],'Parallel monitor closure changed')
        for name,value in s['files'].items():need(sha(PROJECT/name)==value,'Monitor code changed')
    obs.verify_monitor=verify_monitor
    # routed collector's original closure restricts output to manual2; retain the
    # underlying pinned verify_seal implementation with only its path guard removed.
    core=__import__('paper.scripts.collect_titantpp_gru_cap_validation',fromlist=['verify_seal'])
    source=replace_one(inspect.getsource(core.verify_seal),"path = bundle / 'collection_integrity.json'",
        "path = bundle / 'parallel3_v2_control/collection_integrity.json'")
    exec(compile(source,core.__file__,'exec'),collector.__dict__)
    collector.verify_seal(BUNDLE,OUTPUT,collection_sha)
    result=obs.observe(BUNDLE,OUTPUT,monitor_integrity_sha=monitor_sha,acquire=acquire)
    for pathkey,shakey in [('round_path','round_raw_sha256'),('markdown_path','markdown_raw_sha256'),('latest_path','latest_raw_sha256')]:
        need(sha(result[pathkey])==result[shakey],'Observation artifact changed')
    collection=collector.collect(Path(result['round_path']),result['round_raw_sha256'],collection_sha,bundle=BUNDLE,monitor=OUTPUT,report=REPORT)
    return {'observation':result,'collection':collection,'automatic_retry':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('install','observe'),required=True)
    p.add_argument('--host',choices=('5080','5090'));p.add_argument('--integrity-sha',required=True)
    a=p.parse_args()
    print(json.dumps(install(a.host,a.integrity_sha) if a.mode=='install' else advance(a.integrity_sha),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
