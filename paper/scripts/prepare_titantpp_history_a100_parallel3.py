"""Prepare an isolated, explicitly authorized three-fit A100 operational contract.
Science129 and inputs7 stay byte-identical; no remote call or GPU operation.
"""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'search_artifacts/titantpp_contribution_history_a100_20261008_v1'
SCHEMA='titantpp_contribution_history_a100_parallel3_20261008_v1'
NEW=ROOT/'search_artifacts'/SCHEMA
REMOTE='/workspace/'+SCHEMA+'_a100'
FINGERPRINT='587d5c6711df57529bfd354a7f71a046f458d47eee14abbf1073a63ede2f1f98'
FIELDS=('stage','arms','seeds','runner_modules','datasets','dataset_sha256','source','input_files','training','architecture','design_sha256','diagnostic','measurement_arms','cost_limits','evaluation_scope','held_out_test_evaluated','claim_contract_v2_rawSHA','policy')


def require(ok,message):
 if not ok:raise ValueError(message)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():require(p.read_bytes()==data,'Existing prepared file differs: '+str(p))
 else:
  with p.open('xb') as f:f.write(data)

def write_json(p,x):write(p,(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())

def replace_node(text,name,replacement):
 tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
 lines=text.splitlines(keepends=True);return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])

HELPERS=r'''
def job_lease_path(root,job_id):
    require(job_id and '/' not in job_id and '..' not in job_id,'Unsafe lease job')
    return root/'leases'/(job_id+'.json')

def await_job_lease(root,job,deadline):
    until=min(time.time()+10,deadline)
    path=job_lease_path(root,job['id'])
    while time.time()<until:
        try:
            lease=read(path)
            if lease.get('worker_group_pid')==os.getppid():return lease
        except FileNotFoundError:pass
        time.sleep(.05)
    raise ValueError('Matching per-job timeout lease not published')

def owned_root_bytes(root):
    total=0
    for p in root.rglob('*'):
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:continue
    return total

def verify_timeout_parent(tokens,deadline,lease,job,*,parent_pid,parent_ppid,group_pid,python,script,contract,host):
    require(parent_pid==lease['worker_group_pid']==group_pid and parent_ppid==lease['supervisor_pid'],
        'Native timeout PID/PPID/group ownership mismatch')
    require(lease['job']==job['id'],'Foreign worker lease')
    require(len(tokens)==16 and Path(tokens[0]).name=='timeout'
        and tokens[1:3]==['--signal=TERM','--kill-after=15s'],'Native timeout argv shape mismatch')
    duration=float(tokens[3]);parsed_deadline=float(tokens[-1])
    require(math.isfinite(duration) and duration>0 and math.isfinite(parsed_deadline)
        and parsed_deadline==float(deadline),'Native timeout deadline mismatch')
    expected=[python,script,'--contract',contract,'--host',host,'--mode','fit','--job',job['id'],'--deadline']
    require(tokens[4:-1]==expected,'Native timeout command ownership mismatch')
    return True

def batch_groups(c,host):
    jobs=[j for j in c['jobs'] if j['host']==host]
    require(len(jobs)==27,'Expected all27 single-host jobs')
    batches=[jobs[i:i+3] for i in range(0,len(jobs),3)]
    require(all(len(b)==3 and len({(j['dataset'],j['seed']) for j in b})==1
        and [j['arm'] for j in b]==c['arms'] for b in batches),'Parallel group scope drift')
    return batches

def verify_gpu_membership(gpu_pids,processes,active,*,root,used_bytes,total_bytes):
    require(total_bytes>0 and 0<=used_bytes<=.8*total_bytes,'Aggregate GPU allocation exceeds80 percent')
    for pid in gpu_pids:
        proc=processes.get(pid);require(proc is not None,'GPU PID absent from process snapshot')
        owners=[item for item in active.values() if proc['ppid']==item['child'].pid]
        require(len(owners)==1,'Foreign GPU process outside active worker groups')
        item=owners[0]
        require(proc['pgrp']==item['child'].pid and '--job' in proc['argv'],'Foreign GPU process group')
        argv=proc['argv'];index=argv.index('--job')
        require(index+1<len(argv) and argv[index+1]==item['job']['id']
            and str(root/'operation/campaign.py') in argv,'Foreign GPU worker command')
    return True

def check_active_gpu(c,host,root,active):
    gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=15)
    rows=[row.strip().split(',') for row in gpu.stdout.splitlines() if row.strip()]
    matches=[r for r in rows if r[0].strip()==c['hosts'][host]['gpu_uuid']]
    require(len(matches)==1,'GPU disappeared from allocation')
    total,used=(float(matches[0][i].strip())*1024**2 for i in (1,2))
    pids=common.gpu_pids(c['hosts'][host]);processes={}
    for pid in pids:
        try:
            tokens=Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\x00').split('\x00')
            stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            processes[pid]={'ppid':int(stat[1]),'pgrp':int(stat[2]),'argv':tokens}
        except FileNotFoundError:
            # A vanished GPU PID must disappear in a fresh inventory too.
            require(pid not in common.gpu_pids(c['hosts'][host]),'GPU process identity unavailable')
    return verify_gpu_membership(set(processes),processes,active,root=root,used_bytes=used,total_bytes=total)

def write_parallel_pulse(c,host,root,start,active,results):
    now=time.time();require(now<start['deadline_unix'],'Campaign absolute deadline reached')
    state={}
    for name,item in active.items():
        require(now<item['deadline'],'Condition absolute deadline reached')
        lease={'job':name,'contract_sha256':sha_json(c),'issued_unix':now,
            'expires_unix':min(now+90,item['deadline']),'supervisor_pid':os.getpid(),
            'worker_group_pid':item['child'].pid}
        common.write_json(job_lease_path(root,name),lease)
        state[name]={'job':item['job'],'worker_group_pid':item['child'].pid,
            'condition_deadline_unix':item['deadline'],'worker_guard_argv':item['command']}
    common.write_json(root/'status.json',{'status':'running','active_jobs':state,'supervisor_pid':os.getpid(),
        'completed':results,'updated_unix':now,'deadline_unix':start['deadline_unix'],'mac_required':False,
        'maximum_concurrency':3,'supervisor_argv':[sys.executable,*sys.argv]})

def validate_terminal(c,root,job):
    folder=root/'run'/job['id'];manifest=read(folder/'terminal_manifest.json')
    require(manifest['scientific_success'] is True and manifest['status']=='complete'
        and manifest['job']==job and manifest['contract_sha256']==sha_json(c)
        and manifest['held_out_test_evaluated'] is False,'Missing/foreign successful terminal')
    for name,digest in manifest['files'].items():require(sha_file(safe_relative(folder,name))==digest,'Terminal integrity mismatch')
    return {'terminal_manifest_sha256':sha_file(folder/'terminal_manifest.json')}

def audit_completed_batch(c,host,root,batch):
    # Recheck paired exposure after every arm is terminal; simultaneous finishes
    # cannot skip the full triplet comparison.
    for job in batch:
        path=root/'run'/job['id']/'runs'/job['arm']/VARIANT/f"seed_{job['seed']}"/'exposure.json'
        audit_host_batch_prefix(c,host,root,job,read(path))
'''

PULSE=r'''
class PulseBudget:
    def __init__(self,root,digest,job,deadline,lease=False):
        self.root,self.digest,self.job,self.deadline,self.lease=root,digest,job,deadline,lease
        self.last=0.;self.last_storage_check=0.;self.monotonic_deadline=time.monotonic()+max(0,deadline-time.time())
    def __call__(self):
        now=time.time()
        require(now<self.deadline and time.monotonic()<self.monotonic_deadline,'Absolute time budget exhausted')
        if now-self.last<2:return
        if self.lease:
            r=read(job_lease_path(self.root,self.job))
            require(r['contract_sha256']==self.digest and r['job']==self.job and now<r['expires_unix']<=self.deadline
                and r['worker_group_pid']==os.getpgrp() and r['supervisor_pid']!=os.getpid(),
                'Server-owned per-job supervisor lease expired or foreign')
        require(shutil.disk_usage(self.root).free>5*1024**3,'Server free disk below5GiB')
        common.write_json(self.root/'progress'/(self.job+'.json'),{'job':self.job,'pid':os.getpid(),'updated_unix':now,
            'deadline_unix':self.deadline,'source':'genuine_phase_or_batch_progress','mac_required':False})
        rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
        require(rss<16*1024**3,'Owned process CPU RSS exceeds16GiB')
        if now-self.last_storage_check>30:
            require(owned_root_bytes(self.root)<8*1024**3,'Owned-root bytes exceed8GiB')
            self.last_storage_check=now
        import torch
        if torch.cuda.is_initialized():
            fraction=.30 if self.lease else .8
            require(torch.cuda.max_memory_allocated()<=fraction*torch.cuda.get_device_properties(0).total_memory,
                'Owned process GPU allocation exceeds qualified operational fraction')
        self.last=now
'''

DISPATCH=r'''
def dispatch(c,host,root):
    start=authorization(c,root);permit=verify_training_permit(c,root)
    require(read(root/'qualification/receipt.json')==permit['qualification'],'Local qualification changed')
    require(not common.gpu_pids(c['hosts'][host]),'GPU occupied before supervisor launch')
    require(shutil.which('timeout') is not None,'Independent native timeout guard unavailable')
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        common.write_json(root/'supervisor.claim',{'pid':os.getpid(),'started_unix':time.time(),
            'contract_sha256':sha_json(c),'workers':3},exclusive=True)
        results={};active={};batch=None
        try:
            for batch in batch_groups(c,host):
                validate(c);check_active_gpu(c,host,root,active)
                require(not active,'Prior batch not terminal')
                for job in batch:
                    check_active_gpu(c,host,root,active)
                    require(len(active)<3,'Parallel worker ceiling exceeded')
                    require(time.time()<start['deadline_unix'],'Campaign budget expired')
                    require(not (root/'run'/job['id']).exists(),'Run exists; no implicit restart')
                    deadline=min(start['deadline_unix'],time.time()+c['limits']['per_condition_seconds'])
                    remaining=int(deadline-time.time());require(remaining>0,'No remaining fit budget')
                    common.write_json(root/'claims'/f"{job['id']}.json",{'job':job,'contract_sha256':sha_json(c),
                        'deadline_unix':deadline,'maximum_concurrency':3},exclusive=True)
                    command=['timeout','--signal=TERM','--kill-after=15s',str(remaining),sys.executable,
                        str(Path(__file__).resolve()),'--contract',str(root/'execution_contract.json'),
                        '--host',host,'--mode','fit','--job',job['id'],'--deadline',str(deadline)]
                    log=root/'logs'/f"{job['id']}.log";log.parent.mkdir(exist_ok=True)
                    with log.open('xb') as stream:
                        child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'PYTHONHASHSEED':str(job['seed'])},
                            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    active[job['id']]={'job':job,'child':child,'deadline':deadline,'command':command}
                    write_parallel_pulse(c,host,root,start,active,results)
                while active:
                    write_parallel_pulse(c,host,root,start,active,results)
                    check_active_gpu(c,host,root,active)
                    finished=[]
                    for name,item in active.items():
                        rc=item['child'].poll()
                        if rc is None:continue
                        common.write_json(root/'worker_exits'/(name+'.json'),{'job':item['job'],'returncode':rc,
                            'worker_group_pid':item['child'].pid,'observed_unix':time.time(),
                            'contract_sha256':sha_json(c),'actual_child_exit_observed':True},exclusive=True)
                        require(rc==0,'Worker exited '+str(rc)+'; '+name+'; no automatic retry')
                        results[name]=validate_terminal(c,root,item['job']);finished.append(name)
                    for name in finished:del active[name]
                    if active:time.sleep(5)
                check_active_gpu(c,host,root,active)
                audit_completed_batch(c,host,root,batch)
            common.write_json(root/'status.json',{'status':'complete','completed':results,'completed_conditions':len(results),
                'endpoint_roles':len(results)*2,'updated_unix':time.time(),'mac_required':False,'maximum_concurrency':3})
        except BaseException as exc:
            stopped=[]
            for name,item in active.items():
                try:
                    stop_owned(item['child']);stopped.append({'job':name,'worker_group_pid':item['child'].pid,'returncode':item['child'].returncode})
                except BaseException as stop_exc:stopped.append({'job':name,'stop_error':str(stop_exc)})
            failure={'status':'failed','type':type(exc).__name__,'message':str(exc),'active_jobs':list(active),
                'active_batch':batch,'traceback':traceback.format_exc(),'completed':results,'owned_children_stopped':stopped,
                'updated_unix':time.time(),'automatic_retry':False}
            common.write_json(root/'failure.json',failure,exclusive=True);common.write_json(root/'status.json',failure)
            raise
'''



TRAINING_PERMIT=r'''
def verify_training_permit(c,root):
    p=read(root/'training_permit.json');s=read(root/'start_permit.json');q=read(root/'qualification/receipt.json')
    require(p['contract_sha256']==sha_json(c) and p['start_permit_sha256']==sha_json(s)
        and p['qualification_sha256']==sha_file(root/'qualification/receipt.json') and p['qualification']==q,'Training binding drift')
    require(q['status']=='passed' and q['contract_sha256']==sha_json(c),'Native qualification missing')
    admission_root=root/'concurrency_admission'
    a=read(admission_root/'receipt.json');exit_receipt=read(root/'concurrency_exit.json')
    require(p['concurrency_admission']==a and p['concurrency_admission_sha256']==sha_file(admission_root/'receipt.json')
        and p['concurrency_exit_sha256']==sha_json(exit_receipt),'Concurrency admission raw/exit binding drift')
    require(a['status']=='passed' and a['contract_sha256']==sha_json(c) and a['arms']==c['arms'] and a['workers']==3
        and a['shape']==[128,256] and a['warmup_steps']==5 and a['measure_steps']==20
        and a['simultaneous_ready_verified'] is True and a['capacity_admission_only'] is True
        and a['scientific_optimizer_updates']==0 and a['timing_is_scientific_efficiency_evidence'] is False,
        'Native three-worker admission incomplete')
    require(exit_receipt['phase']=='concurrency' and exit_receipt['actual_child_exit_observed'] is True
        and exit_receipt['returncode']==0 and exit_receipt['supervisor_exception'] is None
        and exit_receipt['contract_sha256']==sha_json(c),'Concurrency supervisor exit not verified')
    require(set(a['actual_child_exits'])==set(c['arms']) and all(x['actual_exit_observed'] is True
        and x['returncode']==0 and type(x['timeout_pid']) is int and x['timeout_pid']>0 for x in a['actual_child_exits'].values()),
        'Concurrency child exits incomplete')
    require(len({x['timeout_pid'] for x in a['actual_child_exits'].values()})==3,'Duplicate concurrency timeout owner')
    require(not a['final_GPU_observation']['gpu_pids'] and len(set(a['GPU_witness']['gpu_pids']))==3,
        'Three simultaneous GPU owners and final GPU absence required')
    peak=a['aggregate_peak_fraction_observed']
    require(type(peak) in (int,float) and math.isfinite(peak) and 0<=peak<=.8,'Concurrency aggregate memory over80 percent')
    idle=p['concurrency_actual_idle_observation']
    require(not idle['owned_processes'] and not idle['gpu_pids'],'Concurrency actual ownership exit incomplete')
    require(bool(a['files']),'Concurrency raw evidence manifest missing')
    for name,digest in a['files'].items():
        require(sha_file(safe_relative(admission_root,name))==digest,'Concurrency raw evidence drift '+name)
    return p
'''


def build_runtime(original):
 text=original.replace('titantpp_contribution_history_a100_20261008_v1',SCHEMA)
 text=text.replace("c['limits']['workers_per_host']==1", "c['limits']['workers_per_host']==3")
 text=text.replace("    require(set(c['hosts'])=={'a100'} and set(c['assignments'])=={'a100'},'Foreign A100 host scope')",
  "    require(set(c['hosts'])=={'a100'} and set(c['assignments'])=={'a100'},'Foreign A100 host scope')\n    require(c['parallelism']=={'maximum_workers':3,'grouping':'same_dataset_seed_three_arms','per_process_cuda_fraction':.30,'aggregate_gpu_fraction':.8},'Parallel operational scope drift')")
 text=replace_node(text,'PulseBudget',PULSE)
 text=replace_node(text,'dispatch',DISPATCH)
 text=replace_node(text,'verify_training_permit',TRAINING_PERMIT)
 text=text.replace('\ndef run_fit(c,host,root,job,deadline):','\n'+HELPERS+'\ndef run_fit(c,host,root,job,deadline):')
 old="""    lease=read(root/'server_lease.json')
    require(os.getpgrp()==lease['worker_group_pid'] and os.getppid()!=lease['supervisor_pid'],'Fit requires native timeout process group')
    parent=subprocess.run(['ps','-p',str(os.getppid()),'-o','args='],capture_output=True,text=True,check=True).stdout
    require(parent.strip().startswith('timeout --signal=TERM --kill-after=15s ') and str(deadline) in parent,'Native timeout parent required')"""
 new="""    lease=await_job_lease(root,job,deadline)
    parent_pid=os.getppid()
    tokens=Path(f'/proc/{parent_pid}/cmdline').read_bytes().decode().rstrip('\\x00').split('\\x00')
    parent_stat=Path(f'/proc/{parent_pid}/stat').read_text().rsplit(')',1)[1].split()
    verify_timeout_parent(tokens,deadline,lease,job,parent_pid=parent_pid,parent_ppid=int(parent_stat[1]),
        group_pid=os.getpgrp(),python=sys.executable,script=str(Path(__file__).resolve()),
        contract=str(root/'execution_contract.json'),host=host)"""
 require(old in text,'Original fit guard not found');text=text.replace(old,new)
 text=text.replace("    from paper.scripts.count_aware_tpp_backbone import training\n    start=authorization(c,root);q=read(root/'qualification/receipt.json');verify_training_permit(c,root)",
 "    from paper.scripts.count_aware_tpp_backbone import training\n    import torch\n    torch.cuda.set_per_process_memory_fraction(.30,0)\n    start=authorization(c,root);q=read(root/'qualification/receipt.json');verify_training_permit(c,root)")
 old_nodes={n.name:ast.get_source_segment(original,n) for n in ast.parse(original).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 new_nodes={n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 for name in old_nodes.keys()-{'validate','validate_a100_scope','PulseBudget','run_fit','dispatch','verify_training_permit'}:
  require(old_nodes[name]==new_nodes[name],'Unexpected science/qualification function changed '+name)
 # Everything after the process guard in fit is byte-identical.
 marker="    budget();data=next(d for d in c['datasets'] if d['dataset_id']==job['dataset'])"
 require(old_nodes['run_fit'].split(marker,1)[1]==new_nodes['run_fit'].split(marker,1)[1],'Scientific fit body changed')
 return text


def prepare():
 old=read(OLD/'execution_contract.json')
 require(canonical({k:old[k] for k in FIELDS})==FINGERPRINT,'Scientific identity drift')
 require(old['source']['files_sha256']=='c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a','Source129 drift')
 for name,digest in {**old['source']['files'],**old['input_files']}.items():
  src=OLD/'source'/name;require(sha(src)==digest,'Frozen source/input raw drift '+name)
  dst=NEW/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():shutil.copyfile(src,dst)
  require(sha(dst)==digest,'Copied raw mismatch '+name)
 for name,digest in old['operation']['files'].items():require(sha(OLD/name)==digest,'Serial operation drift')
 text=build_runtime((OLD/'operation/campaign.py').read_text())
 write(NEW/'operation/campaign.py',text.encode())
 c=deepcopy(old);c['schema']=SCHEMA
 host=c['hosts']['a100']
 for k in ('root','source_root','operation_root','output_dir','tmux'):
  host[k]=host[k].replace(old['schema'],SCHEMA)
 host['environment']={k:v.replace(old['schema'],SCHEMA) for k,v in host['environment'].items()}
 c['limits']['workers_per_host']=3
 c['parallelism']={'maximum_workers':3,'grouping':'same_dataset_seed_three_arms','per_process_cuda_fraction':.30,'aggregate_gpu_fraction':.8}
 c['deployment_ready']=False
 c['operation']={'files':{'operation/campaign.py':sha(NEW/'operation/campaign.py')}}
 c['operation']['files_sha256']=canonical(c['operation']['files'])
 c['operational_replacement']={'parent_contract_sha256':canonical(old),'parent_root':old['hosts']['a100']['root'],
  'reason':'User explicitly requested three simultaneous fits on the same A100; serial first fit failed before training at native timeout parent representation check',
  'parent_failure_and_no_training_proof_required':True,'same_original_billing_window':True,
  'cost_cap_usd':50,'fresh_fits_only':True,'automatic_retry':False}
 require(c['budget']==old['budget'],'Original cost/deadline changed')
 require(c['jobs']==old['jobs'],'Canonical execution identities changed')
 require(canonical({k:c[k] for k in FIELDS})==FINGERPRINT,'Scientific contract changed')
 write_json(NEW/'execution_contract.draft.json',c)
 print(json.dumps({'bundle':str(NEW),'science_fingerprint':FINGERPRINT,'source_files':len(c['source']['files']),
  'input_files':len(c['input_files']),'operation_sha256':sha(NEW/'operation/campaign.py'),
  'operation_closure_sha256':c['operation']['files_sha256'],'jobs':len(c['jobs']),'GPU_calls':0,'remote_calls':0}))


if __name__=='__main__':prepare()
