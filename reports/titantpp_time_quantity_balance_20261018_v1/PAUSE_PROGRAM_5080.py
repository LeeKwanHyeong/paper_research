import json,os,pathlib,signal,time
from datetime import datetime,timezone
from paper.scripts import titantpp_external_cap27_runtime as r
root=pathlib.Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_external_cap27_20261018_v1_5080')
c=r.read(root/'execution_contract.json')
r.require(r.sha_json(c)=='0027111a776082e3fbb70f4eb41c91f99d233b0302e5f09fb99296d89ae0bfdb','Contract identity changed')
jobs=r.validate_contract(c,'5080')
expected={'pid': 351034, 'start_ticks': 68069770, 'pgid': 351023, 'argv': ['/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12', '-u', '-m', 'paper.scripts.titantpp_external_cap27_runtime', '--contract', '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_external_cap27_20261018_v1_5080/execution_contract.json', '--host', '5080', '--mode', 'dispatch'], 'cwd': '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_external_cap27_20261018_v1_5080/source'}
evidence=root.parent/('titantpp_external_cap27_pause_20261018_v1_'+'5080')
evidence.mkdir(exist_ok=False)
def state(pid):
    raw=pathlib.Path('/proc',str(pid),'stat').read_text()
    tail=raw[raw.rfind(')')+2:].split()
    return {'state':tail[0],'ppid':int(tail[1]),'pgid':int(tail[2])}
def identity(pid):
    ident=r.proc_identity(pid)
    if ident is not None:
        ident.update(state(pid))
    return ident
def same(a,b):
    return a is not None and all(a.get(k)==b.get(k) for k in ('pid','start_ticks','argv','cwd','pgid'))
def await_stopped(before):
    until=time.monotonic()+5
    while time.monotonic()<until:
        current=identity(before['pid'])
        r.require(same(current,before),'Process identity disappeared/changed during pause')
        if current['state'] in ('T','t'): return current
        time.sleep(.02)
    raise RuntimeError('SIGSTOP did not reach stopped state')
intent={'at':datetime.now(timezone.utc).isoformat(),'host':'5080','expected_dispatcher':expected,
        'authorized_action':'SIGSTOP_only','automatic_resume':False}
r.write(evidence/'intent.json',intent,exclusive=True)
dispatcher=identity(expected['pid'])
r.require(same(dispatcher,expected),'Dispatcher identity changed; signal refused')
os.kill(dispatcher['pid'],signal.SIGSTOP)
dispatcher_after=await_stopped(dispatcher)
allowed={tuple(r.worker_command(c,'5080',root/'execution_contract.json',j,stage)):(j['id'],stage)
         for j in jobs for stage in r.STAGES}
workers=[]
for entry in pathlib.Path('/proc').iterdir():
    if not entry.name.isdigit(): continue
    try: ident=identity(int(entry.name))
    except (FileNotFoundError,ProcessLookupError,PermissionError): continue
    if ident is None or tuple(ident['argv']) not in allowed: continue
    r.require(ident['cwd']==str(root/'source') and ident['ppid']==dispatcher['pid']
              and ident['pgid']==ident['pid'],'Worker ownership differs; dispatcher remains paused')
    job,stage=allowed[tuple(ident['argv'])]
    if ident['state']=='Z':
        workers.append({'job_id':job,'stage':stage,'already_exited_unreaped':True,'identity':ident})
        continue
    # A process group is exclusive to this worker (new session at launch).
    r.require(same(identity(ident['pid']),ident),'Worker identity changed before signal')
    os.killpg(ident['pgid'],signal.SIGSTOP)
    workers.append({'job_id':job,'stage':stage,'before':ident,'after':await_stopped(ident)})
status=r.read(root/'status.json')
gpu=r.gpu_snapshot(c['hosts']['5080']['gpu_uuid'])
result={'status':'paused','observed_at':datetime.now(timezone.utc).isoformat(),'host':'5080',
        'campaign_root':str(root),'contract_sha256':r.sha_file(root/'execution_contract.json'),
        'dispatcher':dispatcher_after,'workers':workers,'campaign_status_at_pause':status,
        'gpu':gpu,'automatic_resume':False,'cuda_contexts_and_vram_retained':True,
        'wall_clock_deadlines_continue':True,'checkpoint_or_source_modified':False}
r.write(evidence/'receipt.json',result,exclusive=True)
print(json.dumps(result,ensure_ascii=False))
