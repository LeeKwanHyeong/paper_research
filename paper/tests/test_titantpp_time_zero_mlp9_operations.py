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


r=load('paper.scripts.titantpp_time_zero_mlp9_runtime','titantpp_time_zero_mlp9_runtime.py')
m=load('paper.scripts.manage_titantpp_time_zero_mlp9','manage_titantpp_time_zero_mlp9.py')


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
            'environment':{},'runtime_expected':{},'absolute_host_deadline_unix':r.time.time()+1000000,'paused_processes':stopped(),
            'wait_for_parent':{'root':str(tmp_path/'parent'/host),'contract_raw_sha256':'parent-raw','contract_canonical_sha256':'parent-canonical','source_files_sha256':'parent-source','dispatcher_identity':{'pid':500,'start_ticks':123,'argv':['parent'],'cwd':'/parent','pgid':500}}}
    inventory={'science.py':r.sha_file(Path(hosts['5080']['source_root'])/'science.py')}
    return {'schema':r.SCHEMA,'hosts':hosts,'jobs':m.build_jobs(),'datasets':[{'dataset_id':d} for d in r.DATASETS],
        'source':{'files':inventory,'files_sha256':r.sha_json(inventory)},'input_files':files,
        'training':{'maximum_epochs':300,'minimum_epochs':40,'patience':40,'batch_size':128},
        'limits':{'fit_wall_seconds':129600,'queue_wall_seconds':604800,'evaluation_wall_seconds':7200},
        'evaluation_scope':'validation_only','held_out_test_authorized':False,
        'parent_time_weight18':{'raw_sha256':'parent-raw','canonical_sha256':'parent-canonical','source_files_sha256':'parent-source'},
        'baseline_validation_references':[{'dataset':d,'seed':s,'time_alpha':1.,'reevaluated':False,'retrained':False,'sha256':'bound'} for d in r.DATASETS for s in (42,52,62)]}


def test_exact9_matrix_and4_5_assignment(contract):
    assert len(r.validate_contract(contract,'5080'))==4
    assert len(r.validate_contract(contract,'5090'))==5
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
    monkeypatch.setattr(r,'predecessor_gate',lambda *a,**kw:{'status':'complete'})
    worker=ModuleType('paper.scripts.run_titantpp_time_zero_mlp9')
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
    monkeypatch.setattr(r,'predecessor_gate',lambda *a,**kw:{'status':'complete'})
    return launched,maxcount


def test_queue_has_at_most3_and_no_test(contract,monkeypatch):
    launched,maximum=fake_queue(contract,monkeypatch)
    result=r.run_stage(contract,'5080',Path('/c'),'train',r.time.monotonic())
    assert result['status']=='complete' and len(result['exits'])==4 and maximum[0]==3
    assert len(launched)==4
    with pytest.raises(ValueError):r.run_stage(contract,'5080',Path('/c'),'test',r.time.monotonic())


def test_failure_stops_new_admission_and_drains(contract,monkeypatch):
    j=r.validate_contract(contract,'5080')[0];launched,maximum=fake_queue(contract,monkeypatch,j['id'])
    result=r.run_stage(contract,'5080',Path('/c'),'train',r.time.monotonic())
    assert result['status']=='failed' and len(launched)==3 and len(result['exits'])==3
    assert len(result['pending_job_ids'])==1 and maximum[0]==3


def test_expired_absolute_deadline_starts_zero(contract,monkeypatch):
    launched,_=fake_queue(contract,monkeypatch);contract['hosts']['5080']['absolute_host_deadline_unix']=1.
    result=r.run_stage(contract,'5080',Path('/c'),'train',r.time.monotonic())
    assert result['status']=='failed' and not launched


def test_full_dispatch_only_train_then_validation(contract,monkeypatch):
    root=Path(contract['hosts']['5080']['root']);monkeypatch.chdir(contract['hosts']['5080']['source_root'])
    r.write(root/'qualification.json',{'status':'passed','host':'5080','contract_sha256':r.sha_json(contract)})
    launched,maximum=fake_queue(contract,monkeypatch);(root/'logs').rmdir()
    assert r.dispatch(contract,'5080',root/'execution_contract.json')==0
    assert [x[0] for x in launched]==['train']*4+['validation']*4
    final=r.read(root/'final_receipt.json')
    assert final['completed_fit_count']==final['validation_replay_count']==4
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
    parent=deepcopy(contract)
    parent_binding={'raw_sha256':'parent-raw','canonical_sha256':m.canonical(parent),'source_files_sha256':'parent-source'}
    parent_report=tmp_path/'parent_report';(parent_report/'observations').mkdir(parents=True)
    for host in ('5080','5090'):
        m.write(parent_report/'observations'/m.PARENT_OBSERVATIONS[host],{'dispatcher_current_identity':{'pid':500,'cwd':parent['hosts'][host]['source_root'],'argv':['parent'],'start_ticks':123,'pgid':500},'start.json':{'pid':500},'contract_sha256':m.canonical(parent)})
    monkeypatch.setattr(m,'PARENT_WEIGHT_REPORT',parent_report)
    monkeypatch.setattr(m,'parent_weight_contract',lambda:(parent,parent_binding))
    deadlines={h:contract['hosts'][h]['absolute_host_deadline_unix'] for h in ('5080','5090')}
    monkeypatch.setattr(m,'inherited_host_deadline',lambda h:(deadlines[h],{'extended':False}))
    before={name:m.sha(reference/'input_snapshot'/name) for name in contract['input_files']}
    result=m.prepare();c=m.read(m.BUNDLE/'execution_contract.json')
    assert result['planned_fit_count']==9 and result['host_fit_counts']=={'5080':4,'5090':5}
    assert len(c['input_files'])==6 and c['held_out_test_authorized'] is False
    assert c['hosts']['5080']['absolute_host_deadline_unix']==deadlines['5080']
    with tarfile.open(m.BUNDLE/'deployment.tar.gz') as archive:
        names=archive.getnames()
        assert 'source/sample_data' in names and not any('full_capped' in n for n in names)
        assert sum(n.endswith('train_validation.parquet') or n.endswith('split_manifest.json') for n in names)==6
    assert before=={name:m.sha(reference/'input_snapshot'/name) for name in contract['input_files']}
    with pytest.raises(ValueError,match='Existing'):m.prepare()


def test_source_snapshot_is944_frozen_plus3_new_only(tmp_path,monkeypatch):
    project=tmp_path/'project';project.mkdir();inventory={}
    for i in range(944):
        name=f'old/{i}.py';p=project/name;p.parent.mkdir(exist_ok=True);p.write_text(f'old{i}\n');inventory[name]=m.sha(p)
    for name in m.REQUIRED_NEW_SOURCE:
        p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('new bounded operation\n')
    spec=tmp_path/'spec.json';m.write(spec,{'cap36_lineage':{'source_inventory':inventory,'source_revision':'original','source_files_sha256':m.canonical(inventory)}})
    monkeypatch.setattr(m,'ROOT',project)
    monkeypatch.setattr(m,'parent_weight_contract',lambda:({'source':{'files':inventory,'files_sha256':m.canonical(inventory),'model_science_source_revision':'original','model_science_source_files_sha256':'science941'}},{}))
    def git(command,**kwargs):
        if command[1]=='ls-files':return '\n'.join(m.REQUIRED_NEW_SOURCE)+'\n'
        if command[1]=='diff':return ''
        if command[1]=='rev-parse':return 'new-commit\n'
        raise AssertionError(command)
    monkeypatch.setattr(m.subprocess,'check_output',git)
    result=m.source_snapshot()
    assert len(result['files'])==947 and result['model_science_source_revision']=='original'
    assert result['files']['old/0.py']==inventory['old/0.py']
    (project/'old/0.py').write_text('changed\n')
    with pytest.raises(ValueError,match='Sealed parent source changed'):m.source_snapshot()


def test_generated_supervisor_executes_exit0_child_and_json_reads_back(contract,tmp_path,monkeypatch):
    """Run the actual emitted supervisor, with a trivial local replacement child."""
    import json
    import shlex
    import subprocess
    c=deepcopy(contract);h=c['hosts']['5080'];h.update(python=sys.executable,tmux_binary='tmux',tmux='fake-local-only')
    source=Path(h['source_root'])
    package=source/'paper/scripts';package.mkdir(parents=True)
    (source/'paper/__init__.py').write_text('');(package/'__init__.py').write_text('')
    (package/'titantpp_time_zero_mlp9_runtime.py').write_text('raise SystemExit(0)\n')
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
    assert m.deploy('5080')['status']=='cpu_wait_queue_launched_not_yet_verified' and len(launched)==1
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
r=types.ModuleType('paper.scripts.titantpp_time_zero_mlp9_runtime')
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
sys.modules[r.__name__]=r;paper.scripts=scripts;scripts.titantpp_time_zero_mlp9_runtime=r
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


def make_parent_fixture(c,tmp_path,monkeypatch,*,complete=True):
    from paper.scripts import titantpp_time_weight_mlp18_runtime as parent
    from paper.scripts import manage_titantpp_time_weight_mlp18 as parent_manager
    host='5080';p=deepcopy(c);p['schema']=parent.SCHEMA;p['jobs']=parent_manager.build_jobs()
    parent_root=tmp_path/'parent_complete';parent_root.mkdir();p['hosts'][host].update(root=str(parent_root),source_root=str(parent_root/'source'))
    cp=parent_root/'execution_contract.json';parent.write(cp,p,exclusive=True)
    binding=c['hosts'][host]['wait_for_parent'];binding.update(root=str(parent_root),contract_raw_sha256=parent.sha_file(cp),
        contract_canonical_sha256=parent.sha_json(p),source_files_sha256=p['source']['files_sha256'])
    c['parent_time_weight18'].update(raw_sha256=binding['contract_raw_sha256'],canonical_sha256=binding['contract_canonical_sha256'],source_files_sha256=binding['source_files_sha256'])
    monkeypatch.setattr(r,'paused_processes',lambda *a:stopped())
    monkeypatch.setattr(r,'gpu_state',lambda *a:({'compute_pids':[11,12,13]},stopped()))
    if not complete:
        monkeypatch.setattr(r,'proc_identity',lambda pid:binding['dispatcher_identity'] if pid==500 else None)
        return parent_root,p
    jobs=parent.validate_contract(p,host);stages={}
    for stage in parent.STAGES:
        exits={};receipts={}
        for index,job in enumerate(jobs):
            folder=parent_root/'run'/job['id'];folder.mkdir(parents=True,exist_ok=True)
            binary=folder/'selected.pt'
            if not binary.exists():binary.write_bytes(b'fixed binary')
            value={'status':'complete','job_id':job['id'],'stage':stage,'contract_sha256':parent.sha_json(p),
                **{k:job[k] for k in ('dataset','arm','seed','time_alpha')},'held_out_test_evaluated':False,
                'selected_checkpoint':str(binary),'selected_epoch':3,'checkpoint_sha256':parent.sha_file(binary),
                'files':{str(binary.relative_to(parent_root)):parent.sha_file(binary)}}
            if stage=='validation':value['validation_replay_passed']=True
            path=folder/(stage+'_receipt.json');parent.write(path,value,exclusive=True)
            receipts[job['id']]={'path':str(path.relative_to(parent_root)),'sha256':parent.sha_file(path)}
            exits[job['id']]={'actual_os_returncode':0,'timeout':False,'pid':2000+index}
        stages[stage]={'status':'complete','exits':exits,'receipts':receipts}
    files={str(q.relative_to(parent_root)):parent.sha_file(q) for q in (parent_root/'run').rglob('*') if q.is_file()}
    manifest=parent_root/'artifact_manifest.json';parent.write(manifest,{'files':files},exclusive=True)
    final={'status':'complete','host':host,'contract_sha256':parent.sha_json(p),'completed_fit_count':len(jobs),
        'validation_replay_count':len(jobs),'test_count':0,'held_out_test_evaluated':False,'owned_worker_pids_absent':True,
        'stages':stages,'artifact_manifest':{'path':'artifact_manifest.json','sha256':parent.sha_file(manifest),'file_count':len(files)}}
    parent.write(parent_root/'final_receipt.json',final,exclusive=True)
    parent.write(parent_root/'dispatcher_exit.json',{'pid':500,'actual_os_returncode':0},exclusive=True)
    monkeypatch.setattr(r,'proc_identity',lambda pid:None)
    return parent_root,p


def test_parent_pending_is_cpu_wait_and_cannot_qualify(contract,tmp_path,monkeypatch):
    make_parent_fixture(contract,tmp_path,monkeypatch,complete=False)
    assert r.predecessor_gate(contract,'5080')['status']=='waiting'
    with pytest.raises(ValueError,match='no new GPU work'):r.predecessor_gate(contract,'5080',require_complete=True)
    h=contract['hosts']['5080'];monkeypatch.chdir(h['source_root'])
    monkeypatch.setattr(r,'runtime_check',lambda *a:pytest.fail('GPU qualification before predecessor complete'))
    with pytest.raises(ValueError,match='no new GPU work'):r.qualify(contract,'5080')


def test_parent_complete_gate_checks_every_stage_exit_and_binary(contract,tmp_path,monkeypatch):
    root,_=make_parent_fixture(contract,tmp_path,monkeypatch)
    gate=r.predecessor_gate(contract,'5080',require_complete=True)
    assert gate['status']=='complete' and gate['parent_owned_pids_absent'] is True
    binary=next((root/'run').glob('*/selected.pt'));binary.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA|binary|artifact'):r.predecessor_gate(contract,'5080',require_complete=True)


def test_parent_nonzero_dispatcher_exit_blocks(contract,tmp_path,monkeypatch):
    root,_=make_parent_fixture(contract,tmp_path,monkeypatch)
    r.write(root/'dispatcher_exit.json',{'pid':500,'actual_os_returncode':1})
    with pytest.raises(ValueError,match='dispatcher exit'):r.predecessor_gate(contract,'5080',require_complete=True)


def test_wait_queue_only_qualifies_after_parent_gate_and_retains_deadline(contract,monkeypatch):
    host='5080';h=contract['hosts'][host];monkeypatch.chdir(h['source_root'])
    calls=[];gates=iter(({'status':'waiting'},{'status':'complete'}))
    monkeypatch.setattr(r,'predecessor_gate',lambda *a,**kw:next(gates))
    monkeypatch.setattr(r.time,'sleep',lambda seconds:calls.append(('CPU_wait',seconds)))
    monkeypatch.setattr(r,'qualify_child',lambda *args:calls.append(('qualify',args[-1])))
    monkeypatch.setattr(r,'dispatch',lambda *args,**kw:calls.append(('dispatch',kw['queue_started'])) or 0)
    identity={'pid':os.getpid(),'start_ticks':123,'argv':['cpu','wait_dispatch'],'cwd':h['source_root'],'pgid':os.getpid(),'state':'S'}
    monkeypatch.setattr(r,'proc_identity',lambda pid:identity if pid==os.getpid() else None)
    before=h['absolute_host_deadline_unix']
    assert r.wait_dispatch(contract,host,Path('/contract'))==0
    assert [x[0] for x in calls]==['CPU_wait','qualify','dispatch']
    assert h['absolute_host_deadline_unix']==before
    assert r.read(Path(h['root'])/'wait_start.json')['wait_process_identity']==identity
    with pytest.raises(ValueError,match='Existing'):r.wait_dispatch(contract,host,Path('/contract'))


def test_waiting_status_accepts_busy_parent_gpu(contract,monkeypatch):
    h=contract['hosts']['5080'];identity={'pid':800,'start_ticks':456,'argv':['cpu','wait_dispatch'],'cwd':h['source_root'],'pgid':800,'state':'S'}
    r.write(Path(h['root'])/'wait_start.json',{'wait_process_identity':identity},exclusive=True)
    monkeypatch.setattr(r,'proc_identity',lambda pid:identity if pid==800 else None)
    monkeypatch.setattr(r,'predecessor_gate',lambda *a,**kw:{'status':'waiting','parent_dispatcher_identity':{'pid':500}})
    monkeypatch.setattr(r,'gpu_snapshot',lambda *a:{'compute_pids':[11,12,13,2001,2002,2003]})
    monkeypatch.setattr(r,'paused_processes',lambda *a:stopped())
    value=r.status(contract,'5080')
    assert value['status']=='waiting_for_parent' and len(value['gpu']['compute_pids'])==6


@pytest.mark.parametrize("condition,expected",[("missing","wait_process_not_observed"),("failure","failed"),("exit0","wait_queue_exited"),("exit1","failed")])
def test_wait_status_never_calls_dead_or_failed_queue_healthy(contract,monkeypatch,condition,expected):
    h=contract['hosts']['5080'];root=Path(h['root'])
    identity={'pid':800,'start_ticks':456,'argv':['cpu','wait_dispatch'],'cwd':h['source_root'],'pgid':800,'state':'S'}
    r.write(root/'wait_start.json',{'wait_process_identity':identity},exclusive=True)
    monkeypatch.setattr(r,'proc_identity',lambda pid:None if condition=='missing' else identity)
    monkeypatch.setattr(r,'gpu_snapshot',lambda *a:{'compute_pids':[11,12,13,2001,2002,2003]})
    monkeypatch.setattr(r,'paused_processes',lambda *a:stopped())
    monkeypatch.setattr(r,'predecessor_gate',lambda *a,**kw:pytest.fail('No healthy predecessor evaluation for failed/exited/unobserved queue'))
    if condition=='failure':r.write(root/'wait_failure.json',{'failure':'preserved'},exclusive=True)
    if condition.startswith('exit'):r.write(root/'dispatcher_exit.json',{'actual_os_returncode':int(condition[-1])},exclusive=True)
    result=r.status(contract,'5080')
    assert result['status']==expected and result['status']!='waiting_for_parent'
    assert result['wait_process_live_verified']==(condition!='missing')


def test_wait_status_rejects_pid_reuse_or_changed_command(contract,monkeypatch):
    h=contract['hosts']['5080'];root=Path(h['root'])
    identity={'pid':800,'start_ticks':456,'argv':['cpu','wait_dispatch'],'cwd':h['source_root'],'pgid':800,'state':'S'}
    r.write(root/'wait_start.json',{'wait_process_identity':identity},exclusive=True)
    monkeypatch.setattr(r,'proc_identity',lambda pid:{**identity,'start_ticks':457})
    monkeypatch.setattr(r,'gpu_snapshot',lambda *a:{'compute_pids':[11,12,13]})
    monkeypatch.setattr(r,'paused_processes',lambda *a:stopped())
    with pytest.raises(ValueError,match='CPU wait process identity changed'):r.status(contract,'5080')
