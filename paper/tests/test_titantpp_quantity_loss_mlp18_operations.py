"""Fabricated-only lifecycle and scope tests; no data inference or remote work."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import pytest

from paper.scripts import manage_titantpp_quantity_loss_mlp18 as m
from paper.scripts import titantpp_quantity_loss_mlp18_runtime as r


def stopped():
    return [{'pid':p,'pgid':p,'start_ticks':p*2,'argv':['old',str(p)],'cwd':'/old','state':'T'} for p in (10,11,12,13)]


def proof():
    return {'schema':'time_head_quantity_only_verification_v1',
        **{k:'a'*64 for k in ('initial_time_head_sha256','final_time_head_sha256','checkpoint_time_head_sha256')},
        **{k:True for k in ('all_time_head_gradients_none','optimizer_state_absent','time_head_parameters_unchanged','time_nll_is_diagnostic')},
        'optimizer_steps_checked':2}


@pytest.fixture
def contract(tmp_path):
    hosts={}
    for host in ('5080','5090'):
        root=tmp_path/host; source=root/'source';source.mkdir(parents=True)
        (source/'science.py').write_text('frozen\n');files={}
        for ds in r.DATASETS:
            for name in ('train_validation.parquet','split_manifest.json'):
                p=source/'inputs/data'/ds/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic input\n')
                files[str(p.relative_to(source))]=r.sha_file(p)
        hosts[host]={'root':str(root),'source_root':str(source),'python':sys.executable,'gpu_uuid':'GPU-'+host,
            'environment':{},'runtime_expected':{},'absolute_host_deadline_unix':r.time.time()+1000000,'paused_processes':stopped(),
            'completed_predecessor':{'root':str(tmp_path/'parent'/host),'contract_raw_sha256':'parent-raw','contract_canonical_sha256':'parent-canonical',
                'source_files_sha256':'parent-source','dispatcher_identity':{'pid':500},'final_receipt_sha256':'final-sha','dispatcher_exit_sha256':'exit-sha'}}
    inventory={'science.py':r.sha_file(Path(hosts['5080']['source_root'])/'science.py')}
    return {'schema':r.SCHEMA,'hosts':hosts,'jobs':m.build_jobs(),'datasets':[{'dataset_id':d} for d in r.DATASETS],
        'source':{'files':inventory,'files_sha256':r.sha_json(inventory)},'input_files':files,
        'architecture':{'quantity_loss_modes':list(r.LOSS_MODES),'quantity_loss_normalization':'none','time_alpha':0.,'model_architecture_changed':False,'checkpoint_warmstart':False},
        'training':{'maximum_epochs':300,'minimum_epochs':40,'patience':40,'batch_size':128},
        'limits':{'fit_wall_seconds':129600,'queue_wall_seconds':604800,'evaluation_wall_seconds':7200},
        'evaluation_scope':'validation_only','held_out_test_authorized':False,
        'parent_time_zero9':{'raw_sha256':'parent-raw','canonical_sha256':'parent-canonical','source_files_sha256':'parent-source'},
        'baseline_validation_references':[{'dataset':d,'seed':s,'time_alpha':0.,'loss_mode':'log_mse','time_head_trained':False,
            'reevaluated':False,'retrained':False,'sha256':'bound'} for d in r.DATASETS for s in (42,52,62)]}


def test_exact18_matrix_8_10_and_validation_only(contract):
    assert len(r.validate_contract(contract,'5080'))==8
    assert len(r.validate_contract(contract,'5090'))==10
    assert r.STAGES==('train','validation') and r.TIME_ALPHAS==(0.,)


@pytest.mark.parametrize('field,value',[('time_alpha',1.),('time_alpha',False),('loss_mode','normalized_mse'),('arm','cnn'),('seed',32),('host','5090'),('id','../bad')])
def test_job_mutation_rejected(contract,field,value):
    c=deepcopy(contract);c['jobs'][0][field]=value
    with pytest.raises((ValueError,KeyError)):r.validate_contract(c,'5080')


@pytest.mark.parametrize('mutation',['test_auth','test_input','budget','baseline','normalization','deadline','paused','parent'])
def test_scientific_scope(contract,mutation):
    c=deepcopy(contract)
    if mutation=='test_auth':c['held_out_test_authorized']=True
    if mutation=='test_input':c['input_files']['inputs/data/x/full_capped.parquet']='test'
    if mutation=='budget':c['training']['minimum_epochs']=1
    if mutation=='baseline':c['baseline_validation_references'][0]['time_alpha']=1.
    if mutation=='normalization':c['architecture']['quantity_loss_normalization']='cap'
    if mutation=='deadline':c['hosts']['5080']['absolute_host_deadline_unix']=None
    if mutation=='paused':c['hosts']['5080']['paused_processes'][0]['state']='S'
    if mutation=='parent':c['hosts']['5080']['completed_predecessor']['final_receipt_sha256']=''
    with pytest.raises((ValueError,KeyError)):r.validate_contract(c,'5080')


def test_paused_descendant_and_gpu_allowlist(contract,monkeypatch):
    identities={p['pid']:p for p in stopped()}
    monkeypatch.setattr(r,'proc_identity',lambda p:identities.get(p))
    monkeypatch.setattr(r,'process_parents',lambda:{11:10,12:10,13:10})
    monkeypatch.setattr(r,'gpu_snapshot',lambda u:{'compute_pids':[11,12,13,100]})
    assert len(r.paused_processes(contract,'5080'))==4
    assert r.gpu_state(contract,'5080',[100])[0]['compute_pids'][-1]==100
    with pytest.raises(ValueError):r.gpu_state(contract,'5080')
    identities[11]={**identities[11],'state':'S'}
    with pytest.raises(ValueError):r.paused_processes(contract,'5080')
    identities[11]={**identities[11],'state':'T'}
    monkeypatch.setattr(r,'process_parents',lambda:{11:10,12:10,13:10,14:11})
    with pytest.raises(ValueError):r.paused_processes(contract,'5080')


def make_receipt(c,host,j,stage):
    root=Path(c['hosts'][host]['root']);folder=root/'run'/j['id'];folder.mkdir(parents=True,exist_ok=True)
    binary=folder/'selected.pt'
    if not binary.exists():binary.write_bytes(b'fixed selection')
    value={'schema':'titantpp_quantity_loss_mlp18_stage_v1','status':'complete','job_id':j['id'],'stage':stage,'contract_sha256':r.sha_json(c),
        **{k:j[k] for k in ('dataset','arm','seed','time_alpha','loss_mode')},'held_out_test_evaluated':False,
        'quantity_only':True,'time_head_trained':False,'quantity_loss_normalization':'none','time_head_no_training_verification':proof(),
        'selected_checkpoint':str(binary),'checkpoint_sha256':r.sha_file(binary),'selected_epoch':3,'state_sha256':'s'*64,'target_identity':{'target_count':4},
        'files':{str(binary.relative_to(root)):r.sha_file(binary)}}
    if stage=='validation':value['validation_replay_passed']=True
    r.write(folder/(stage+'_receipt.json'),value,exclusive=True)
    return value


def test_receipt_selected_binary_and_quantity_proof(contract):
    j=r.validate_contract(contract,'5080')[0]
    make_receipt(contract,'5080',j,'train');v=make_receipt(contract,'5080',j,'validation')
    p=Path(contract['hosts']['5080']['root'])/'run'/j['id']/'validation_receipt.json'
    assert r.verify_receipt(contract,'5080',j,'validation')['sha256']
    for mutation in ({'loss_mode':'raw_mae'},{'quantity_loss_normalization':'cap'},{'time_head_trained':True},{'time_head_no_training_verification':{}},{'selected_epoch':4},{'files':{}},{'held_out_test_evaluated':True}):
        r.write(p,{**v,**mutation})
        with pytest.raises(ValueError):r.verify_receipt(contract,'5080',j,'validation')
    with pytest.raises(ValueError):r.worker_command(contract,'5080',Path('/c'),j,'test')


def fake_queue(c,monkeypatch,failure=False):
    host='5080';root=Path(c['hosts'][host]['root']);(root/'logs').mkdir(exist_ok=True)
    jobs={j['id']:j for j in r.validate_contract(c,host)};instances=[];launched=[];maxcount=[0]
    bad=next(iter(jobs)) if failure else None
    class Child:
        def __init__(self,command,**kwargs):
            self.pid=1000+len(instances);self.command=command;self.cwd=str(kwargs['cwd']);self.returncode=None;self.ticks=0
            self.job=jobs[command[command.index('--job')+1]];self.stage=command[-1]
            assert kwargs['env']['PYTHONHASHSEED']==str(self.job['seed']) and kwargs['start_new_session'] is True
            instances.append(self);launched.append((self.stage,self.job['id']))
            maxcount[0]=max(maxcount[0],sum(p.returncode is None for p in instances))
        def poll(self):
            if self.returncode is None:
                self.ticks+=1
                if self.ticks>=4:
                    self.returncode=1 if self.job['id']==bad else 0
                    if self.returncode==0:make_receipt(c,host,self.job,self.stage)
            return self.returncode
    monkeypatch.setattr(r.subprocess,'Popen',Child)
    monkeypatch.setattr(r,'gpu_state',lambda c,h,pids=():({'compute_pids':[11,12,13,*pids]},stopped()))
    monkeypatch.setattr(r,'proc_identity',lambda pid:next(({'pid':p.pid,'pgid':p.pid,'start_ticks':p.pid,'argv':p.command,'cwd':p.cwd,'state':'R'} for p in instances if p.pid==pid and p.returncode is None),None))
    monkeypatch.setattr(r.time,'sleep',lambda _:None)
    monkeypatch.setattr(r,'predecessor_gate',lambda *a,**kw:{'status':'complete'})
    return launched,maxcount


def test_full8_fit_validation_dispatch_concurrency_and_no_repeat(contract,monkeypatch):
    root=Path(contract['hosts']['5080']['root']);monkeypatch.chdir(contract['hosts']['5080']['source_root'])
    r.write(root/'qualification.json',{'status':'passed','host':'5080','contract_sha256':r.sha_json(contract)})
    launched,maximum=fake_queue(contract,monkeypatch);(root/'logs').rmdir()
    assert r.dispatch(contract,'5080',root/'execution_contract.json')==0
    assert [s for s,j in launched]==['train']*8+['validation']*8 and maximum[0]==3
    f=r.read(root/'final_receipt.json')
    assert f['completed_fit_count']==f['validation_replay_count']==8 and f['test_count']==0
    assert f['owned_worker_pids_absent'] is True and f['prior_campaign_resumed'] is False
    with pytest.raises(ValueError,match='Existing'):r.dispatch(contract,'5080',root/'execution_contract.json')


def test_failure_stops_admission_preserves_running_children(contract,monkeypatch):
    launched,maximum=fake_queue(contract,monkeypatch,True)
    result=r.run_stage(contract,'5080',Path('/c'),'train',r.time.monotonic())
    assert result['status']=='failed' and len(launched)==3 and len(result['exits'])==3
    assert len(result['pending_job_ids'])==5


def test_deadline_starts_zero_and_forbids_test(contract,monkeypatch):
    launched,_=fake_queue(contract,monkeypatch);contract['hosts']['5080']['absolute_host_deadline_unix']=1.
    result=r.run_stage(contract,'5080',Path('/c'),'train',r.time.monotonic())
    assert result['status']=='failed' and not launched
    with pytest.raises(ValueError):r.run_stage(contract,'5080',Path('/c'),'test',r.time.monotonic())


def test_qualify_checks_free_memory_both_losses_and_three_concurrent(contract,monkeypatch):
    monkeypatch.chdir(contract['hosts']['5080']['source_root'])
    monkeypatch.setattr(r,'gpu_state',lambda *a:({'compute_pids':[11,12,13]},stopped()))
    monkeypatch.setattr(r,'paused_processes',lambda *a:stopped())
    monkeypatch.setattr(r,'free_gpu_bytes',lambda *a:1000)
    monkeypatch.setattr(r,'runtime_check',lambda *a:{'runtime':'same'})
    monkeypatch.setattr(r,'predecessor_gate',lambda *a,**kw:{'status':'complete'})
    calls=[]
    monkeypatch.setattr(r,'qualify_three_concurrent',lambda *a:calls.append('both') or {'status':'passed'})
    worker=ModuleType(r.WORKER_MODULE)
    worker.synthetic_check=lambda *a:{'status':'passed','real_data_loaded':False,'cuda_memory_peaks':{r.ARMS[0]:{'max_memory_reserved_bytes':100}}}
    monkeypatch.setitem(sys.modules,worker.__name__,worker)
    q=r.qualify(contract,'5080')
    assert calls==['both'] and q['parallel_memory_admission']['three_worker_reserved_estimate_bytes']==300
    with pytest.raises(ValueError):r.qualify(contract,'5080')


def test_actual_emitted_three_process_synthetic_admission(contract,monkeypatch):
    """Three ordinary local children replace CUDA calculations, keeping subprocess I/O real."""
    root=Path(contract['hosts']['5080']['root']);source=Path(contract['hosts']['5080']['source_root'])
    (root/'execution_contract.json').write_text(json.dumps(contract))
    package=source/'paper/scripts';package.mkdir(parents=True)
    (source/'paper/__init__.py').write_text('');(package/'__init__.py').write_text('')
    (package/'run_titantpp_quantity_loss_mlp18.py').write_text(
        "import time\ndef synthetic_condition(c,h,m):\n time.sleep(.03)\n return {'status':'passed','loss_mode':m,'real_data_loaded':False}\n")
    contract['hosts']['5080']['environment']['PYTHONPATH']=str(source)
    monkeypatch.setattr(r,'proc_identity',lambda pid:None)
    monkeypatch.setattr(r,'gpu_state',lambda *a:({'compute_pids':[11,12,13]},stopped()))
    result=r.qualify_three_concurrent(contract,'5080')
    assert result['native_worker_exit0_count']==6 and set(result['loss_modes'])==set(r.LOSS_MODES)
    assert all(len(rows)==3 for rows in result['loss_modes'].values())
    assert len(list(root.glob('admission_*.json')))==6


def test_launch_single_lifecycle_no_wait_loop(contract,monkeypatch):
    h=contract['hosts']['5080'];monkeypatch.chdir(h['source_root'])
    calls=[]
    monkeypatch.setattr(r,'predecessor_gate',lambda *a,**kw:calls.append('complete_parent') or {'status':'complete'})
    monkeypatch.setattr(r,'qualify_child',lambda *a:calls.append('qualify'))
    monkeypatch.setattr(r,'dispatch',lambda *a,**kw:calls.append('dispatch') or 0)
    assert r.launch_dispatch(contract,'5080',Path('/c'))==0
    assert calls==['complete_parent','qualify','dispatch']
    with pytest.raises(ValueError):r.launch_dispatch(contract,'5080',Path('/c'))


def test_source_snapshot_preserves947_adds3(tmp_path,monkeypatch):
    project=tmp_path/'project';project.mkdir();inventory={}
    for i in range(947):
        n=f'old/{i}.py';p=project/n;p.parent.mkdir(exist_ok=True);p.write_text(str(i));inventory[n]=m.sha(p)
    for n in m.REQUIRED_NEW_SOURCE:
        p=project/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('new only')
    monkeypatch.setattr(m,'ROOT',project)
    monkeypatch.setattr(m,'parent_zero_contract',lambda:({'source':{'files':inventory,'files_sha256':m.canonical(inventory),'model_science_source_revision':'original','model_science_source_files_sha256':'science941'}},{}))
    def git(cmd,**kw):
        if cmd[1]=='ls-files':return '\n'.join(m.REQUIRED_NEW_SOURCE)
        if cmd[1]=='diff':return ''
        if cmd[1]=='rev-parse':return 'newcommit\n'
        raise AssertionError(cmd)
    monkeypatch.setattr(m.subprocess,'check_output',git)
    got=m.source_snapshot();assert len(got['files'])==950 and got['parent_source_file_count']==947
    (project/'old/1.py').write_text('changed')
    with pytest.raises(ValueError,match='parent source changed'):m.source_snapshot()


def test_collection_program_and_exclusive_archive(contract,tmp_path):
    compile(m.collection_program(contract,'5080'),'<collect>','exec')
    assert 'titantpp_quantity_loss_mlp18_runtime' in m.collection_program(contract,'5080')
    assert "f['test_count']==0" in m.collection_program(contract,'5080')
    import tarfile
    p=tmp_path/'bad.tar';payload=tmp_path/'x';payload.write_text('x')
    with tarfile.open(p,'w') as f:f.add(payload,arcname='../escape')
    with pytest.raises(ValueError):m.safe_extract(p,tmp_path/'out')


def test_mock_prepare_preserves_inputs_and_deadline(contract,tmp_path,monkeypatch):
    import tarfile
    project=tmp_path/'project';project.mkdir();(project/'science.py').write_text('frozen\n')
    reference=tmp_path/'cap36';source=Path(contract['hosts']['5080']['source_root'])
    inputs={**contract['input_files']}
    for name in inputs:
        p=reference/'input_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((source/name).read_bytes())
    previous={**contract,'input_files':{**inputs,'inputs/data/x/full_capped.parquet':'excluded'},'data_receipt':{'unchanged':True}}
    parent=deepcopy(contract);parent_binding={'raw_sha256':'parent-raw','canonical_sha256':'parent-canonical','source_files_sha256':'parent-source'}
    parent_bundle=tmp_path/'parent_bundle'
    for host in ('5080','5090'):
        originals=parent_bundle/'retrieval'/host/'originals';originals.mkdir(parents=True)
        m.write(originals/'final_receipt.json',{'status':'complete'})
        m.write(originals/'dispatcher_exit.json',{'actual_os_returncode':0,'pid':500})
    snapshot={'base_git_revision':'commit','files':{'science.py':m.sha(project/'science.py')},'files_sha256':m.canonical({'science.py':m.sha(project/'science.py')})}
    monkeypatch.setattr(m,'ROOT',project);monkeypatch.setattr(m,'BUNDLE',tmp_path/'bundle');monkeypatch.setattr(m,'REPORT',tmp_path/'report')
    m.write(m.REPORT/'SCIENCE_SPEC.json',{'approved':'literal'})
    monkeypatch.setattr(m,'REFERENCE_BUNDLE',reference);monkeypatch.setattr(m,'PARENT_ZERO_BUNDLE',parent_bundle)
    monkeypatch.setattr(m,'source_snapshot',lambda:snapshot)
    monkeypatch.setattr(m,'reference_contract',lambda:(previous,{'name':'existing'}))
    monkeypatch.setattr(m,'baseline_validation_references',lambda:contract['baseline_validation_references'])
    monkeypatch.setattr(m,'parent_zero_contract',lambda:(parent,parent_binding))
    deadlines={h:contract['hosts'][h]['absolute_host_deadline_unix'] for h in ('5080','5090')}
    monkeypatch.setattr(m,'inherited_host_deadline',lambda h:(deadlines[h],{'extended':False}))
    before={n:m.sha(reference/'input_snapshot'/n) for n in inputs}
    result=m.prepare();c=m.read(m.BUNDLE/'execution_contract.json')
    assert result['planned_fit_count']==18 and result['host_fit_counts']=={'5080':8,'5090':10}
    assert c['architecture']['quantity_loss_normalization']=='none'
    assert c['hosts']['5080']['absolute_host_deadline_unix']==deadlines['5080']
    assert 'wait_for_parent' not in c['hosts']['5080']
    with tarfile.open(m.BUNDLE/'deployment.tar.gz') as f:
        names=f.getnames();assert 'source/sample_data' in names
        assert not any('full_capped' in n for n in names)
        assert sum(n.endswith('train_validation.parquet') or n.endswith('split_manifest.json') for n in names)==6
    assert before=={n:m.sha(reference/'input_snapshot'/n) for n in inputs}
    assert m.verify_local(archive=True)['schema']==r.SCHEMA
    r.write(m.REPORT/'SCIENCE_SPEC.json',{'tampered':True})
    with pytest.raises(ValueError,match='specification'):m.verify_local()
    with pytest.raises(ValueError,match='Existing'):m.prepare()


def parent_fixture(c,tmp_path,monkeypatch):
    """Use a fabricated completed alpha0 parent, retaining the real gate algorithm."""
    from paper.scripts import titantpp_time_zero_mlp9_runtime as old
    host='5080';p=deepcopy(c);root=tmp_path/'completed_parent';root.mkdir()
    p['hosts'][host].update(root=str(root),source_root=str(root/'source'))
    jobs=[{**j,'id':j['id']+'__old'} for j in c['jobs'] if j['host']==host and j['loss_mode']=='raw_mse']
    p['jobs']=jobs;cp=root/'execution_contract.json';old.write(cp,p,exclusive=True)
    h=c['hosts'][host]['completed_predecessor'];h.update(root=str(root),contract_raw_sha256=old.sha_file(cp),contract_canonical_sha256=old.sha_json(p),source_files_sha256=p['source']['files_sha256'])
    c['parent_time_zero9'].update(raw_sha256=h['contract_raw_sha256'],canonical_sha256=h['contract_canonical_sha256'],source_files_sha256=h['source_files_sha256'])
    monkeypatch.setattr(old,'validate_contract',lambda c,h:jobs)
    monkeypatch.setattr(old,'verify_receipt',lambda *a:None)
    stages={}
    for stage in r.STAGES:
        receipts={};exits={}
        for i,j in enumerate(jobs):
            folder=root/'run'/j['id'];folder.mkdir(parents=True,exist_ok=True)
            binary=folder/'selected.pt'
            if not binary.exists():binary.write_bytes(b'original binary')
            path=folder/f'{stage}_receipt.json';old.write(path,{'selected_checkpoint':str(binary),'checkpoint_sha256':old.sha_file(binary)},exclusive=True)
            receipts[j['id']]={'path':str(path.relative_to(root)),'sha256':old.sha_file(path)}
            exits[j['id']]={'actual_os_returncode':0,'timeout':False,'pid':3000+i}
        stages[stage]={'status':'complete','receipts':receipts,'exits':exits}
    inventory={str(p.relative_to(root)):old.sha_file(p) for p in (root/'run').rglob('*') if p.is_file()}
    old.write(root/'artifact_manifest.json',{'files':inventory},exclusive=True)
    final={'status':'complete','host':host,'contract_sha256':old.sha_json(p),'completed_fit_count':len(jobs),'validation_replay_count':len(jobs),
        'test_count':0,'held_out_test_evaluated':False,'owned_worker_pids_absent':True,'stages':stages,
        'artifact_manifest':{'path':'artifact_manifest.json','sha256':old.sha_file(root/'artifact_manifest.json'),'file_count':len(inventory)}}
    old.write(root/'final_receipt.json',final,exclusive=True);old.write(root/'dispatcher_exit.json',{'actual_os_returncode':0,'pid':500},exclusive=True)
    h.update(final_receipt_sha256=old.sha_file(root/'final_receipt.json'),dispatcher_exit_sha256=old.sha_file(root/'dispatcher_exit.json'))
    monkeypatch.setattr(r,'paused_processes',lambda *a:stopped())
    monkeypatch.setattr(r,'gpu_state',lambda *a:({'compute_pids':[11,12,13]},stopped()))
    monkeypatch.setattr(r,'proc_identity',lambda *a:None)
    return root


def test_completed_predecessor_gate_verifies_binary_and_all_exits(contract,tmp_path,monkeypatch):
    root=parent_fixture(contract,tmp_path,monkeypatch)
    result=r.predecessor_gate(contract,'5080',require_complete=True)
    assert result['status']=='complete' and result['parent_owned_pids_absent'] is True
    next((root/'run').glob('*/selected.pt')).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='binary|SHA'):r.predecessor_gate(contract,'5080')


def test_missing_or_changed_predecessor_exit_blocks(contract,tmp_path,monkeypatch):
    root=parent_fixture(contract,tmp_path,monkeypatch)
    (root/'dispatcher_exit.json').unlink()
    with pytest.raises(ValueError,match='missing'):r.predecessor_gate(contract,'5080')
    r.write(root/'dispatcher_exit.json',{'pid':500,'actual_os_returncode':1})
    with pytest.raises(ValueError,match='SHA'):r.predecessor_gate(contract,'5080')


def test_status_during_qualification_reports_unattributed_gpu_without_counting_fits(contract,monkeypatch):
    host='5080';root=Path(contract['hosts'][host]['root'])
    r.write(root/'qualification_attempt.json',{'host':host})
    monkeypatch.setattr(r,'paused_processes',lambda *a:stopped())
    monkeypatch.setattr(r,'gpu_snapshot',lambda *a:{'compute_pids':[11,12,13,5001,5002,5003]})
    out=r.status(contract,host)
    assert out['status']=='qualifying' and not out['jobs'] and out['gpu_ownership_verified_for_training'] is False


def test_generated_supervisor_executes_exit0_child_and_json_reads_back(contract,tmp_path,monkeypatch):
    """Run the actual emitted supervisor, with a trivial local replacement child."""
    import json
    import shlex
    import subprocess
    c=deepcopy(contract);h=c['hosts']['5080'];h.update(python=sys.executable,tmux_binary='tmux',tmux='fake-local-only')
    source=Path(h['source_root'])
    package=source/'paper/scripts';package.mkdir(parents=True)
    (source/'paper/__init__.py').write_text('');(package/'__init__.py').write_text('')
    (package/'titantpp_quantity_loss_mlp18_runtime.py').write_text('raise SystemExit(0)\n')
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
    assert m.deploy('5080')['status']=='launch_requested_not_yet_verified' and len(launched)==1
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
r=types.ModuleType('paper.scripts.titantpp_quantity_loss_mlp18_runtime')
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
sys.modules[r.__name__]=r;paper.scripts=scripts;scripts.titantpp_quantity_loss_mlp18_runtime=r
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



def test_native_synthetic_failure_preserves_all_started_child_exits(contract,monkeypatch):
    root=Path(contract['hosts']['5080']['root']);source=Path(contract['hosts']['5080']['source_root'])
    (root/'execution_contract.json').write_text(json.dumps(contract))
    package=source/'paper/scripts';package.mkdir(parents=True)
    (source/'paper/__init__.py').write_text('');(package/'__init__.py').write_text('')
    (package/'run_titantpp_quantity_loss_mlp18.py').write_text(
        "def synthetic_condition(c,h,m):\n return {'status':'failed','loss_mode':m,'real_data_loaded':False}\n")
    contract['hosts']['5080']['environment']['PYTHONPATH']=str(source)
    with pytest.raises(ValueError,match='condition failed'):r.qualify_three_concurrent(contract,'5080')
    records=[r.read(p) for p in root.glob('admission_*.json')]
    assert len(records)==3 and all(type(x['actual_os_returncode']) is int for x in records)
    assert all(x['stdout_sha256'] and x['stderr_sha256'] for x in records)
    assert not list(root.glob('admission_raw_mae_*'))
