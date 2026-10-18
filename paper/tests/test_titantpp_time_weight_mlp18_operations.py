"""Mock-only time-weight queue/packaging checks; no research inference or SSH."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2] if 'paper/tests' in str(__file__) else Path('/Users/igwanhyeong/PycharmProjects/paper_research')
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def load(name, filename):
    try:
        return __import__(name, fromlist=['_'])
    except ModuleNotFoundError as error:
        if error.name != name: raise
        module=ModuleType(name)
        module.__file__=str(ROOT/'paper/scripts'/filename)
        sys.modules[name]=module
        exec(compile((Path('/tmp')/filename).read_text(),module.__file__,'exec'),module.__dict__)
        return module


r=load('paper.scripts.titantpp_time_weight_mlp18_runtime','titantpp_time_weight_mlp18_runtime.py')
m=load('paper.scripts.manage_titantpp_time_weight_mlp18','manage_titantpp_time_weight_mlp18.py')


def stopped():
    return [{'pid':p,'pgid':p,'start_ticks':p*2,'argv':['old',str(p)],'cwd':'/old','state':'T'} for p in (10,11,12,13)]


@pytest.fixture
def contract(tmp_path):
    hosts={}
    for host in ('5080','5090'):
        root=tmp_path/host; source=root/'source';source.mkdir(parents=True)
        (source/'science.py').write_text('frozen\n')
        files={}
        for ds in r.DATASETS:
            for name in ('train_validation.parquet','split_manifest.json'):
                p=source/'inputs/data'/ds/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic input\n')
                files[str(p.relative_to(source))]=r.sha_file(p)
        hosts[host]={'root':str(root),'source_root':str(source),'python':'/python','gpu_uuid':'GPU-'+host,
            'environment':{},'runtime_expected':{},'absolute_host_deadline_unix':r.time.time()+1000000,'paused_processes':stopped()}
    inventory={'science.py':r.sha_file(Path(hosts['5080']['source_root'])/'science.py')}
    return {'schema':r.SCHEMA,'hosts':hosts,'jobs':m.build_jobs(),'datasets':[{'dataset_id':d} for d in r.DATASETS],
        'source':{'files':inventory,'files_sha256':r.sha_json(inventory)},'input_files':files,
        'training':{'maximum_epochs':300,'minimum_epochs':40,'patience':40,'batch_size':128},
        'limits':{'fit_wall_seconds':129600,'queue_wall_seconds':604800,'evaluation_wall_seconds':7200},
        'evaluation_scope':'validation_only','held_out_test_authorized':False,
        'baseline_validation_references':[{'dataset':d,'seed':s,'time_alpha':1.,'reevaluated':False,'retrained':False,'sha256':'bound'} for d in r.DATASETS for s in (42,52,62)]}


def test_exact18_matrix_and8_10_assignment(contract):
    assert len(r.validate_contract(contract,'5080'))==8
    assert len(r.validate_contract(contract,'5090'))==10
    assert r.STAGES==('train','validation')


@pytest.mark.parametrize('field,value',[('time_alpha',1.),('time_alpha',True),('arm','historical_cnn'),('seed',32),('host','5090'),('id','../bad')])
def test_job_corruption_rejected(contract,field,value):
    c=deepcopy(contract);c['jobs'][0][field]=value
    with pytest.raises((ValueError,KeyError)):r.validate_contract(c,'5080')


@pytest.mark.parametrize('mutation',['test_auth','test_input','budget','baseline','deadline','paused'])
def test_scope_and_lineage_guards(contract,mutation):
    c=deepcopy(contract)
    if mutation=='test_auth':c['held_out_test_authorized']=True
    if mutation=='test_input':c['input_files']['inputs/data/yellow_trip_hourly/full_capped.parquet']='not-admitted'
    if mutation=='budget':c['training']['maximum_epochs']=301
    if mutation=='baseline':c['baseline_validation_references'][0]['reevaluated']=True
    if mutation=='deadline':c['hosts']['5080']['absolute_host_deadline_unix']=None
    if mutation=='paused':c['hosts']['5080']['paused_processes'][0]['state']='S'
    with pytest.raises((ValueError,KeyError)):r.validate_contract(c,'5080')


def test_pid_wrapper_includes_state_and_rechecks_startticks(tmp_path,monkeypatch):
    pid=42;proc=tmp_path/"proc";folder=proc/str(pid);folder.mkdir(parents=True)
    fields=['T','1','42']+['0']*16+['123']
    (folder/'stat').write_text('42 (paused worker) '+' '.join(fields))
    (folder/'cmdline').write_bytes(b'old\0worker\0');(folder/'cwd').symlink_to(tmp_path,target_is_directory=True)
    identity=r.proc_identity(pid,proc_root=proc)
    assert identity['pid']==pid and identity['state']=='T' and identity['ppid']==1 and identity['start_ticks']==123
    monkeypatch.setattr(r,'_base_proc_identity',lambda p,**kw:{**identity,'start_ticks':-1})
    with pytest.raises(ValueError,match='PID changed'):r.proc_identity(pid,proc_root=proc)


def test_paused_tree_is_still_stopped_and_no_unknown_descendant(contract,monkeypatch):
    identities={p['pid']:p for p in stopped()}
    monkeypatch.setattr(r,'proc_identity',lambda p:identities.get(p))
    monkeypatch.setattr(r,'process_parents',lambda:{11:10,12:10,13:10})
    assert len(r.paused_processes(contract,'5080'))==4
    identities[11]={**identities[11],'state':'S'}
    with pytest.raises(ValueError,match='identity/state'):r.paused_processes(contract,'5080')
    identities[11]={**identities[11],'state':'T'}
    monkeypatch.setattr(r,'process_parents',lambda:{11:10,12:10,13:10,14:11})
    with pytest.raises(ValueError,match='Unknown descendant'):r.paused_processes(contract,'5080')


def test_paused_gpu_contexts_allowed_unknown_gpu_rejected(contract,monkeypatch):
    monkeypatch.setattr(r,'paused_processes',lambda *a:stopped())
    monkeypatch.setattr(r,'gpu_snapshot',lambda u:{'compute_pids':[11,12,13,100]})
    assert r.gpu_state(contract,'5080',[100])[0]['compute_pids']==[11,12,13,100]
    with pytest.raises(ValueError,match='Unidentified'):r.gpu_state(contract,'5080')
    assert r.available_slots([100],[11,12,13,100],[11,12,13])==2
    with pytest.raises(ValueError):r.available_slots([1,2,3,4],[],[])


def test_qualification_uses_free_memory_not_total(contract,monkeypatch):
    host='5080';target=contract['hosts'][host];monkeypatch.chdir(target['source_root'])
    monkeypatch.setattr(r,'gpu_state',lambda *a:({'compute_pids':[11,12,13]},stopped()))
    monkeypatch.setattr(r,'paused_processes',lambda *a:stopped())
    monkeypatch.setattr(r,'free_gpu_bytes',lambda *a:1000)
    monkeypatch.setattr(r,'runtime_check',lambda *a:{'gpu':{'total_memory_bytes':999999}})
    worker=ModuleType('paper.scripts.run_titantpp_time_weight_mlp18')
    worker.synthetic_check=lambda *a:{'status':'passed','real_data_loaded':False,'cuda_memory_peaks':{r.ARMS[0]:{'max_memory_reserved_bytes':100}}}
    monkeypatch.setitem(sys.modules,worker.__name__,worker)
    q=r.qualify(contract,host)
    assert q['parallel_memory_admission']['free_memory_before_qualification_bytes']==1000
    assert q['parallel_memory_admission']['three_worker_reserved_estimate_bytes']==300
    assert 'not a measured' in q['parallel_memory_admission']['limitation']
    with pytest.raises(ValueError,match='Existing'):r.qualify(contract,host)


def receipt(c,host,j,stage):
    root=Path(c['hosts'][host]['root']);folder=root/'run'/j['id'];folder.mkdir(parents=True,exist_ok=True)
    p=folder/(stage+'.txt');p.write_text('frozen selection\n')
    x={'status':'complete','job_id':j['id'],'stage':stage,'contract_sha256':r.sha_json(c),
        **{k:j[k] for k in ('dataset','arm','seed','time_alpha')},'held_out_test_evaluated':False,
        'files':{str(p.relative_to(root)):r.sha_file(p)}}
    if stage=='validation':x['validation_replay_passed']=True
    r.write(folder/(stage+'_receipt.json'),x,exclusive=True);return x


def test_receipt_binds_alpha_scope_and_files(contract):
    j=r.validate_contract(contract,'5080')[0];x=receipt(contract,'5080',j,'validation')
    path=Path(contract['hosts']['5080']['root'])/'run'/j['id']/'validation_receipt.json'
    assert r.verify_receipt(contract,'5080',j,'validation')['sha256']
    for update in ({'time_alpha':10.},{'held_out_test_evaluated':True},{'files':{}},{'validation_replay_passed':False}):
        r.write(path,{**x,**update})
        with pytest.raises(ValueError):r.verify_receipt(contract,'5080',j,'validation')
    with pytest.raises(ValueError):r.worker_command(contract,'5080',Path('/c'),j,'test')


def fake_queue(c,monkeypatch,failure_job=None):
    host='5080';root=Path(c['hosts'][host]['root']);(root/'logs').mkdir(exist_ok=True)
    jobs={j['id']:j for j in r.validate_contract(c,host)};instances=[];launched=[];maxcount=[0]
    class Child:
        def __init__(self,command,**kwargs):
            self.pid=1000+len(instances);self.command=command;self.cwd=str(kwargs['cwd']);self.returncode=None;self.ticks=0
            self.job=jobs[command[command.index('--job')+1]];self.stage=command[-1]
            assert kwargs['env']['PYTHONHASHSEED']==str(self.job['seed'])
            assert kwargs['start_new_session'] is True
            instances.append(self);launched.append((self.stage,self.job['id']))
            maxcount[0]=max(maxcount[0],sum(p.returncode is None for p in instances))
        def poll(self):
            if self.returncode is None:
                self.ticks+=1
                if self.ticks>=4:
                    self.returncode=1 if self.job['id']==failure_job else 0
                    if self.returncode==0:receipt(c,host,self.job,self.stage)
            return self.returncode
    monkeypatch.setattr(r.subprocess,'Popen',Child)
    monkeypatch.setattr(r,'gpu_state',lambda c,h,pids=():({'compute_pids':[11,12,13,*pids]},stopped()))
    monkeypatch.setattr(r,'proc_identity',lambda pid:next(({'pid':p.pid,'pgid':p.pid,'start_ticks':p.pid,'argv':p.command,'cwd':p.cwd,'state':'R'} for p in instances if p.pid==pid and p.returncode is None),None))
    monkeypatch.setattr(r.time,'sleep',lambda _:None)
    return launched,maxcount


def test_queue_has_at_most3_and_no_test(contract,monkeypatch):
    launched,maximum=fake_queue(contract,monkeypatch)
    result=r.run_stage(contract,'5080',Path('/c'),'train',r.time.monotonic())
    assert result['status']=='complete' and len(result['exits'])==8 and maximum[0]==3
    assert len(launched)==8
    with pytest.raises(ValueError):r.run_stage(contract,'5080',Path('/c'),'test',r.time.monotonic())


def test_failure_stops_new_admission_and_drains(contract,monkeypatch):
    j=r.validate_contract(contract,'5080')[0];launched,maximum=fake_queue(contract,monkeypatch,j['id'])
    result=r.run_stage(contract,'5080',Path('/c'),'train',r.time.monotonic())
    assert result['status']=='failed' and len(launched)==3 and len(result['exits'])==3
    assert len(result['pending_job_ids'])==5 and maximum[0]==3


def test_expired_absolute_deadline_starts_zero(contract,monkeypatch):
    launched,_=fake_queue(contract,monkeypatch);contract['hosts']['5080']['absolute_host_deadline_unix']=1.
    result=r.run_stage(contract,'5080',Path('/c'),'train',r.time.monotonic())
    assert result['status']=='failed' and not launched


def test_full_dispatch_only_train_then_validation(contract,monkeypatch):
    root=Path(contract['hosts']['5080']['root']);monkeypatch.chdir(contract['hosts']['5080']['source_root'])
    r.write(root/'qualification.json',{'status':'passed','host':'5080','contract_sha256':r.sha_json(contract)})
    launched,maximum=fake_queue(contract,monkeypatch);(root/'logs').rmdir()
    assert r.dispatch(contract,'5080',root/'execution_contract.json')==0
    assert [x[0] for x in launched]==['train']*8+['validation']*8
    final=r.read(root/'final_receipt.json')
    assert final['completed_fit_count']==final['validation_replay_count']==8
    assert final['test_count']==0 and final['held_out_test_evaluated'] is False
    assert final['prior_campaign_resumed'] is False and maximum[0]==3
    with pytest.raises(ValueError,match='Existing'):r.dispatch(contract,'5080',root/'execution_contract.json')


def test_collection_program_is_validation_only(contract):
    compile(m.collection_program(contract,'5080'),'<collection>','exec')
    assert "f['test_count']==0" in m.collection_program(contract,'5080')
    assert 'gpu,paused=r.gpu_state' in m.collection_program(contract,'5080')


def test_mock_prepare_packages_trainval6_only_and_preserves_inputs(contract,tmp_path,monkeypatch):
    import tarfile
    project=tmp_path/'project';project.mkdir();(project/'science.py').write_text('frozen\n')
    reference=tmp_path/'cap36';source=Path(contract['hosts']['5080']['source_root'])
    original={**contract['input_files']}
    for ds in r.DATASETS:
        original[f'inputs/data/{ds}/full_capped.parquet']='test-input-excluded'
    for name,digest in contract['input_files'].items():
        dst=reference/'input_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((source/name).read_bytes())
    previous={**contract,'input_files':original,'data_receipt':{'existing':'untouched'}}
    pause=tmp_path/'pause';pause.mkdir()
    for host in ('5080','5090'):
        m.write(pause/f'PAUSE_RECEIPT_{host}.json',{'status':'paused','dispatcher':stopped()[0],
            'workers':[{'after':p} for p in stopped()[1:]]})
        m.write(pause/f'PAUSE_COMPLETENESS_QA_{host}.json',{'all_live_direct_children_stopped':True,'stopped':stopped()})
    snapshot={'base_git_revision':'new-commit','files':{'science.py':m.sha(project/'science.py')},
        'files_sha256':m.canonical({'science.py':m.sha(project/'science.py')})}
    monkeypatch.setattr(m,'ROOT',project);monkeypatch.setattr(m,'BUNDLE',tmp_path/'bundle');monkeypatch.setattr(m,'REPORT',tmp_path/'report')
    monkeypatch.setattr(m,'REFERENCE_BUNDLE',reference);monkeypatch.setattr(m,'PAUSE_REPORT',pause)
    monkeypatch.setattr(m,'source_snapshot',lambda:snapshot)
    monkeypatch.setattr(m,'reference_contract',lambda:(previous,{'name':'existing'}))
    monkeypatch.setattr(m,'baseline_validation_references',lambda:contract['baseline_validation_references'])
    deadlines={h:contract['hosts'][h]['absolute_host_deadline_unix'] for h in ('5080','5090')}
    monkeypatch.setattr(m,'inherited_host_deadline',lambda h:(deadlines[h],{'extended':False}))
    before={name:m.sha(reference/'input_snapshot'/name) for name in contract['input_files']}
    result=m.prepare();c=m.read(m.BUNDLE/'execution_contract.json')
    assert result['planned_fit_count']==18 and result['host_fit_counts']=={'5080':8,'5090':10}
    assert len(c['input_files'])==6 and c['held_out_test_authorized'] is False
    assert c['hosts']['5080']['absolute_host_deadline_unix']==deadlines['5080']
    with tarfile.open(m.BUNDLE/'deployment.tar.gz') as archive:
        names=archive.getnames()
        assert 'source/sample_data' in names and not any('full_capped' in n for n in names)
        assert sum(n.endswith('train_validation.parquet') or n.endswith('split_manifest.json') for n in names)==6
    assert before=={name:m.sha(reference/'input_snapshot'/name) for name in contract['input_files']}
    with pytest.raises(ValueError,match='Existing'):m.prepare()


def test_source_snapshot_is941_frozen_plus3_new_only(tmp_path,monkeypatch):
    project=tmp_path/'project';project.mkdir();inventory={}
    for i in range(941):
        name=f'old/{i}.py';p=project/name;p.parent.mkdir(exist_ok=True);p.write_text(f'old{i}\n');inventory[name]=m.sha(p)
    for name in m.REQUIRED_NEW_SOURCE:
        p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('new bounded operation\n')
    spec=tmp_path/'spec.json';m.write(spec,{'cap36_lineage':{'source_inventory':inventory,'source_revision':'original','source_files_sha256':m.canonical(inventory)}})
    monkeypatch.setattr(m,'ROOT',project);monkeypatch.setattr(m,'PROBE_INPUTS',spec)
    def git(command,**kwargs):
        if command[1]=='ls-files':return '\n'.join(m.REQUIRED_NEW_SOURCE)+'\n'
        if command[1]=='diff':return ''
        if command[1]=='rev-parse':return 'new-commit\n'
        raise AssertionError(command)
    monkeypatch.setattr(m.subprocess,'check_output',git)
    result=m.source_snapshot()
    assert len(result['files'])==944 and result['model_science_source_revision']=='original'
    assert result['files']['old/0.py']==inventory['old/0.py']
    (project/'old/0.py').write_text('changed\n')
    with pytest.raises(ValueError,match='Original source changed'):m.source_snapshot()


def test_generated_supervisor_executes_exit0_child_and_json_reads_back(contract,tmp_path,monkeypatch):
    """Run the actual emitted supervisor, with a trivial local replacement child."""
    import json
    import shlex
    import subprocess
    c=deepcopy(contract);h=c['hosts']['5080'];h.update(python=sys.executable,tmux_binary='tmux',tmux='fake-local-only')
    source=Path(h['source_root'])
    package=source/'paper/scripts';package.mkdir(parents=True)
    (source/'paper/__init__.py').write_text('');(package/'__init__.py').write_text('')
    (package/'titantpp_time_weight_mlp18_runtime.py').write_text('raise SystemExit(0)\n')
    h['environment']['PYTHONPATH']=str(source)
    bundle=tmp_path/'supervisor_bundle';bundle.mkdir()
    (bundle/'deployment.tar.gz').write_bytes(b'not-transferred')
    (bundle/'execution_contract.json').write_text(json.dumps(c))
    monkeypatch.setattr(m,'BUNDLE',bundle);monkeypatch.setattr(m,'REPORT',tmp_path/'supervisor_report')
    monkeypatch.setattr(m,'verify_local',lambda **kwargs:c)
    launched=[]
    def local_transport(host,command,**kwargs):
        words=shlex.split(command)
        if words[0]=='tmux':
            shell=shlex.split(words[-1]);assert shell[:3]==[sys.executable,'-u','-c']
            launched.append(shell[3])
            completed=subprocess.run([sys.executable,'-u','-c',shell[3]],capture_output=True,timeout=20)
            assert completed.returncode==0,completed.stderr.decode()
            return completed
        return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
    monkeypatch.setattr(m,'ssh',local_transport)
    assert m.deploy('5080')['status']=='launched_not_yet_verified' and len(launched)==1
    raw=(Path(h['root'])/'dispatcher_exit.json').read_bytes()
    assert raw.endswith(b'\n') and not raw.endswith(b'\\n')
    value=json.loads(raw)
    assert value['actual_os_returncode']==0 and value['pid']>0
    assert value['supervisor_own_exit_directly_observed'] is False


def test_generated_collection_executes_and_tar_observation_json_reads_back(contract,tmp_path):
    """Run emitted collection code against a complete local fixture and stub runtime."""
    import io
    import json
    import subprocess
    import tarfile
    c=deepcopy(contract);host='5080';h=c['hosts'][host];root=Path(h['root'])
    jobs=r.validate_contract(c,host);ids=[j['id'] for j in jobs]
    final={'status':'complete','host':host,'contract_sha256':r.sha_json(c),
        'completed_fit_count':len(jobs),'validation_replay_count':len(jobs),'test_count':0,
        'stages':{stage:{'status':'complete','exits':{j:{'actual_os_returncode':0,'timeout':False,'pid':2000+i} for i,j in enumerate(ids)},
            'receipts':{j:{'sha256':'stub-record'} for j in ids}} for stage in r.STAGES}}
    (root/'execution_contract.json').write_text(json.dumps(c))
    (root/'final_receipt.json').write_text(json.dumps(final))
    (root/'dispatcher_exit.json').write_text(json.dumps({'pid':9999,'actual_os_returncode':0})+'\n')
    # Only host-observation helpers are stubbed. The actual generated Python,
    # byte serialization and streaming tar execute in a separate local process.
    bootstrap='''import sys,types,json,pathlib,hashlib
paper=types.ModuleType('paper');paper.__path__=[]
scripts=types.ModuleType('paper.scripts');scripts.__path__=[]
r=types.ModuleType('paper.scripts.titantpp_time_weight_mlp18_runtime')
r.read=lambda path:json.loads(pathlib.Path(path).read_text())
def require(ok,message):
 if not ok:raise ValueError(message)
r.require=require
r.sha_json=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
r.validate_contract=lambda c,h:[j for j in c['jobs'] if j['host']==h]
r.STAGES=('train','validation')
r.verify_receipt=lambda *args:None
r.gpu_state=lambda c,h:({'compute_pids':[11,12,13]},c['hosts'][h]['paused_processes'])
r.proc_identity=lambda pid:None
r.now=lambda:'fixture-observed-time'
sys.modules['paper']=paper;sys.modules['paper.scripts']=scripts
sys.modules[r.__name__]=r;paper.scripts=scripts;scripts.titantpp_time_weight_mlp18_runtime=r
'''+ '\nexec('+repr(m.collection_program(c,host))+')\n'
    completed=subprocess.run([sys.executable,'-c',bootstrap],capture_output=True,timeout=20)
    assert completed.returncode==0,completed.stderr.decode()
    with tarfile.open(fileobj=io.BytesIO(completed.stdout),mode='r:gz') as archive:
        raw=archive.extractfile('collection_observation.json').read()
        assert raw.endswith(b'\n') and not raw.endswith(b'\\n')
        observation=json.loads(raw)
        assert observation['host']==host and observation['owned_pids_absent'] is True
        assert observation['prior_campaign_still_stopped'] is True
        assert len(observation['paused_processes'])==4
