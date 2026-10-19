"""Fabricated CPU checks only: candidate identity, common conditions and real save/replay."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from paper.scripts import run_titantpp_late_path18 as w
from paper.scripts import titantpp_late_path18_runtime as r


def common_contract(c, *, time_input='log1p', mode='log_mse'):
    c['schema']=w.CONTRACT_SCHEMA
    c['common_condition']={'time_input':time_input,'loss_mode':mode,'time_alpha':0.}
    c.setdefault('architecture',{}).update(arms=list(w.ARMS),time_alpha=0.,quantity_loss_mode=mode,
        quantity_loss_modes=[mode],time_input_transform=w.TIME_INPUT_TRANSFORMS[time_input],
        quantity_input_transform=w.QUANTITY_INPUT_TRANSFORM,model_architecture_changed=True,
        late_path_parameter_count=1024,quantity_loss_normalization='none',checkpoint_warmstart=False)
    return c


@pytest.mark.parametrize('time_input',w.TIME_INPUTS)
@pytest.mark.parametrize('arm',w.ARMS)
def test_factory_actual_candidate_and_feature_metadata(time_input,arm):
    model,meta=w.build_model(w.fabricated_data(),arm,time_input=time_input)
    cls=w.RawTimeLateObservationPath if time_input=='raw' else w.CountAwareTitanLateObservationPath
    assert type(model) is cls and model.late_path_mode==w.MODE_BY_ARM[arm]
    assert meta['candidate_name']==arm and meta['late_path_parameter_count']==1024
    assert meta['time_input_transform']==w.TIME_INPUT_TRANSFORMS[time_input]
    dt=torch.tensor([[1.,32.]]);q=torch.tensor([[2.,10.]]);mask=torch.ones_like(dt,dtype=torch.bool)
    f=model.continuous_features(dt,q,mask)
    expected=dt if time_input=='raw' else torch.log1p(dt)
    assert torch.equal(f[...,0],expected) and torch.equal(f[...,1],torch.log1p(q))


def test_initial_common_tensors_rng_capacity_and_head_oracle():
    result=w.initialization_oracle()
    assert result['status']=='passed' and len(result['seeds'])==6
    assert {(x['seed'],x['arm']) for x in result['seeds']}=={(s,a) for s in w.SEEDS for a in w.ARMS}
    assert all(x['common_parameter_buffer_and_rng_exact'] and x['extra_parameter_count']==1024 for x in result['seeds'])


@pytest.mark.parametrize('mode',w.LOSS_MODES)
@pytest.mark.parametrize('time_input',w.TIME_INPUTS)
def test_exact_quantity_gradient_optimizer_time_head_guard(mode,time_input):
    for arm in w.ARMS:
        result=w.quantity_oracle(mode,'cpu',arm=arm,time_input=time_input)
        assert result['loss_formula_and_adamw_oracle_exact']
        assert result['time_input_transform']==w.TIME_INPUT_TRANSFORMS[time_input]
        assert w.verify_time_head_proof(result['time_head_no_training_verification'])['optimizer_steps_checked']==2


def test_hooks_are_candidate_scoped_and_shared_aliases_restore_after_failure():
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import core,training,constants,observed_time
    w.inherited.install_hooks()
    names=((factory,'build_count_aware_model'),(factory,'validate_checkpoint_route'),(training,'build_model'),
        (training,'validate_checkpoint_route'),(training,'BACKBONE_LABELS'),(core,'target_outputs'),
        (observed_time,'ROLES'),(observed_time,'validate_launch'),(constants,'BACKBONE_LABELS'),
        (constants,'validate_model_role_contract'),(w.inherited,'ARMS'),(w.inherited,'build_model'))
    before=[getattr(m,n) for m,n in names]
    sealed=w.sha_file(w._PARENT_PATH)
    with pytest.raises(RuntimeError):
        with w.runtime_hooks('raw_mae',time_input='raw',arm=w.ARMS[1]):
            assert factory.build_count_aware_model is not before[0]
            assert w._ACTIVE_ARM==w.ARMS[1]
            raise RuntimeError('fabricated only')
    assert all(getattr(m,n) is value for (m,n),value in zip(names,before))
    assert w._MODEL_SCOPE_DEPTH==0 and w._ACTIVE_ARM is None and w._ACTIVE_TIME_INPUT=='log1p'
    assert w._parent._ACTIVE_LOSS_MODE is None and sealed==w.sha_file(w._PARENT_PATH)
    with pytest.raises(ValueError):w.run_stage({},'5080','unused','test')


@pytest.mark.parametrize('time_input',w.TIME_INPUTS)
def test_synthetic_qualification_both_arms_only_common_loss(time_input):
    c={'common_condition':{'time_input':time_input,'loss_mode':'raw_mae','time_alpha':0.}}
    out=w.synthetic_check(c,'5090',device='cpu')
    assert out['status']=='passed' and out['loss_modes']==['raw_mae']
    assert set(out['per_arm'])==set(w.ARMS)==set(out['cuda_memory_peaks'])
    assert not out['real_data_loaded'] and not out['held_out_test_evaluated']
    assert all(v['loss_mode']=='raw_mae' and v['time_input']==time_input and v['candidate_name']==arm for arm,v in out['per_arm'].items())
    with pytest.raises(ValueError):w.synthetic_condition(c,'5090','raw_mse',arm=w.ARMS[0],device='cpu')


@pytest.mark.parametrize('time_input,mode,arm',[(t,m,a) for t in w.TIME_INPUTS for m in w.LOSS_MODES for a in w.ARMS])
def test_fabricated_fit_selected_binary_exact_candidate_and_native_style_replay(tmp_path,time_input,mode,arm):
    from paper.tests.test_titantpp_quantity_loss_worker import fabricated_contract
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    configure_runtime('cpu',4)
    c,data=fabricated_contract(tmp_path);common_contract(c,time_input=time_input,mode=mode)
    job={'id':'fabricated_'+mode+'_'+arm,'host':'5090','dataset':'raf_spare_parts','seed':42,
        'arm':arm,'time_alpha':0.,'loss_mode':mode,'time_input':time_input}
    c['jobs']=[job]
    original_args,original_eval=w.training_args,w.evaluate_selected
    def tiny(*args,**kwargs):
        value=original_args(*args,**kwargs)
        assert (value.epochs,value.min_epochs,value.early_stopping_patience)==(300,40,40)
        assert value.model_role==w.late_metadata(mode=w.MODE_BY_ARM[arm])['model_role']
        value.epochs=value.min_epochs=value.early_stopping_patience=2;value.device='cpu'
        return value
    def cpu_eval(*args,**kwargs):return original_eval(*args,**kwargs,device='cpu')
    with patch('paper.scripts.quantity_comparison_runtime.configure_runtime',side_effect=lambda *a:configure_runtime('cpu',4)), \
        patch.object(w,'training_args',side_effect=tiny),patch.object(w,'evaluate_selected',side_effect=cpu_eval):
        tr=w.run_stage(c,'5090',job['id'],'train')
        cp=torch_load_checkpoint(Path(tr['selected_checkpoint']),map_location='cpu')
        assert cp['backbone']==tr['arm']==tr['candidate_name']==arm
        assert tr['summary']['backbone']==arm and cp['encoder_config']['candidate_name']==arm
        assert tr['time_input']==time_input and tr['late_path_mode']==w.MODE_BY_ARM[arm]
        with w.runtime_hooks(mode,time_input=time_input,arm=arm):
            w.validate_weight_binding(cp,0.,mode);w.validate_candidate_route(cp,arm)
            wrong=deepcopy(cp);wrong['backbone']='titantpp_history_mlp_width16'
            with pytest.raises(ValueError):w.validate_candidate_route(wrong,arm)
            with pytest.raises(ValueError):w.validate_candidate_route(cp,w.ARMS[1-w.ARMS.index(arm)])
            wrong=deepcopy(cp);wrong['encoder_config']['late_path_mode']='wrong'
            with pytest.raises(ValueError):w.validate_candidate_route(wrong,arm)
        history=w.read(Path(tr['selected_checkpoint']).parent/'history.json')['history']
        assert cp['best_epoch']==min(history,key=lambda row:row['val_qty_rmse'])['epoch']
        val=w.run_stage(c,'5090',job['id'],'validation')
        assert val['validation_replay_passed'] and val['arm']==arm and val['candidate_name']==arm
        assert val['time_input']==time_input and val['loss_mode']==mode and not val['held_out_test_evaluated']
        assert val['target_identity']==tr['target_identity'] and val['state_sha256']==tr['state_sha256']
        with pytest.raises(ValueError):w.run_stage(c,'5090',job['id'],'train')


def runtime_contract(tmp_path,*,time_input='log1p',mode='log_mse'):
    from paper.tests.test_titantpp_quantity_loss_mlp18_operations import contract as old_fixture
    c=old_fixture.__wrapped__(tmp_path);common_contract(c,time_input=time_input,mode=mode)
    c['parent_raw_time27']={'contract_raw_sha256':'parent-raw','contract_canonical_sha256':'parent-canonical','source_files_sha256':'parent-source'}
    c['jobs']=[{'id':f'{d}__{a}__seed{s}__loss{mode}','dataset':d,'arm':a,'seed':s,'loss_mode':mode,
        'time_alpha':0.,'time_input':time_input,'host':'5080' if d=='yellow_trip_hourly' or (d=='intermittent_frozen_5000' and s==42) else '5090'}
        for d in r.DATASETS for a in r.ARMS for s in (42,52,62)]
    for ref in c['baseline_validation_references']:
        ref.update(loss_mode=mode,time_input=time_input,time_input_transform=w.TIME_INPUT_TRANSFORMS[time_input])
    return c


@pytest.mark.parametrize('time_input',w.TIME_INPUTS)
@pytest.mark.parametrize('mode',w.LOSS_MODES)
def test_runtime_exact18_shared_condition_host8_10_and_no_test(tmp_path,time_input,mode):
    c=runtime_contract(tmp_path,time_input=time_input,mode=mode)
    assert len(r.validate_contract(c,'5080'))==8 and len(r.validate_contract(c,'5090'))==10
    assert r.MODULE=='paper.scripts.titantpp_late_path18_runtime'
    assert r.WORKER_MODULE=='paper.scripts.run_titantpp_late_path18'
    assert r.STAGES==('train','validation')
    for mutation in ('job_loss','job_time','architecture','baseline','count','test'):
        changed=deepcopy(c)
        if mutation=='job_loss':changed['jobs'][0]['loss_mode']='raw_mae' if mode!='raw_mae' else 'log_mse'
        if mutation=='job_time':changed['jobs'][0]['time_input']='raw' if time_input=='log1p' else 'log1p'
        if mutation=='architecture':changed['architecture']['late_path_parameter_count']=2048
        if mutation=='baseline':changed['baseline_validation_references'][0]['time_input']='wrong'
        if mutation=='count':changed['jobs'].pop()
        if mutation=='test':changed['held_out_test_authorized']=True
        with pytest.raises((ValueError,KeyError)):r.validate_contract(changed,'5080')


def test_runtime_receipt_binds_candidate_mode_and_input_to_selected_binary(tmp_path):
    from paper.tests.test_titantpp_quantity_loss_mlp18_operations import proof
    c=runtime_contract(tmp_path,time_input='raw',mode='raw_mse');host='5080'
    job=r.validate_contract(c,host)[0]
    folder=Path(c['hosts'][host]['root'])/'run'/job['id'];folder.mkdir(parents=True)
    binary=folder/'selected.pt';binary.write_bytes(b'fabricated selected state')
    value={'schema':w.SCHEMA,'status':'complete','job_id':job['id'],'stage':'train','contract_sha256':r.sha_json(c),
        **{k:job[k] for k in ('dataset','arm','seed','time_alpha','loss_mode','time_input')},
        'held_out_test_evaluated':False,'quantity_only':True,'time_head_trained':False,
        'quantity_loss_normalization':'none','time_head_no_training_verification':proof(),
        'candidate_name':job['arm'],'late_path_mode':w.MODE_BY_ARM[job['arm']],'late_path_parameter_count':1024,
        'time_input_transform':w.TIME_INPUT_TRANSFORMS['raw'],'quantity_input_transform':w.QUANTITY_INPUT_TRANSFORM,
        'selected_checkpoint':str(binary),'checkpoint_sha256':r.sha_file(binary),'selected_epoch':2,
        'state_sha256':'s'*64,'target_identity':{'target_count':4},'files':{str(binary.relative_to(Path(c['hosts'][host]['root']))):r.sha_file(binary)}}
    path=folder/'train_receipt.json';r.write(path,value,exclusive=True)
    assert r.verify_receipt(c,host,job,'train')['sha256']==r.sha_file(path)
    for key,bad in (('candidate_name','titantpp_history_mlp_width16'),('late_path_mode','hidden'),
        ('time_input_transform','log1p_nonnegative_float32'),('late_path_parameter_count',2048)):
        changed=deepcopy(value);changed[key]=bad;r.write(path,changed)
        with pytest.raises(ValueError):r.verify_receipt(c,host,job,'train')
    r.write(path,value)
    binary.write_bytes(b'modified')
    with pytest.raises(ValueError):r.verify_receipt(c,host,job,'train')


def test_native_admission_has_three_per_arm_six_total_and_exact_condition(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from paper.tests.test_titantpp_quantity_loss_mlp18_operations import stopped
    c=runtime_contract(tmp_path,time_input='raw',mode='raw_mae')
    commands=[];pids=[]
    def popen(command,**kwargs):
        arm=command[-1]
        payload={'status':'passed','real_data_loaded':False,'loss_mode':'raw_mae',
            'candidate_name':arm,'time_input':'raw'}
        kwargs['stdout'].write(json.dumps(payload).encode())
        pid=1000+len(commands);commands.append(command);pids.append(pid)
        return SimpleNamespace(pid=pid,wait=lambda **kw:0,poll=lambda:0,returncode=0)
    monkeypatch.setattr(r._impl.subprocess,'Popen',popen)
    monkeypatch.setattr(r._impl,'proc_identity',lambda pid:None)
    monkeypatch.setattr(r._impl,'gpu_state',lambda *args,**kwargs:({'compute_pids':[]},stopped()))
    out=r._candidate_three_concurrent(c,'5080')
    assert out['native_worker_exit0_count']==6 and out['maximum_simultaneous_workers']==3
    assert set(out['per_arm'])==set(w.ARMS) and all(len(rows)==3 for rows in out['per_arm'].values())
    assert [cmd[-1] for cmd in commands]==[w.ARMS[0]]*3+[w.ARMS[1]]*3
    assert all(cmd[-2]=='raw_mae' for cmd in commands) and len(set(pids))==6


def test_qualification_own_cuda_context_allowance_cannot_escape_scope(tmp_path,monkeypatch):
    import os
    c=runtime_contract(tmp_path)
    h=c['hosts']['5080'];monkeypatch.chdir(h['source_root'])
    live={'pid':os.getpid(),'pgid':os.getpid(),'start_ticks':8,'cwd':h['source_root'],
        'argv':[h['python'],'-u','-m',r.MODULE,'--contract',str(Path(h['root'])/'execution_contract.json'),'--host','5080','--mode','qualify']}
    r.write(Path(h['root'])/'qualification_attempt.json',{'host':'5080','contract_sha256':r.sha_json(c)},exclusive=True)
    monkeypatch.setattr(r._impl,'proc_identity',lambda pid:deepcopy(live))
    calls=[]
    def gpu(c,host,active_pids=()):calls.append(tuple(active_pids));return {'compute_pids':list(active_pids)},[]
    monkeypatch.setattr(r._impl,'gpu_state',gpu)
    def run_checks(c,host):
        assert r._impl.gpu_state(c,host)[0]['compute_pids']==[os.getpid()]
        with pytest.raises(ValueError):r._impl.gpu_state(deepcopy(c),host)
        return {'status':'passed'}
    monkeypatch.setattr(r,'_original_three',run_checks)
    result=r.qualify_three_concurrent(c,'5080')
    assert result['qualification_self_context_allowed'] and calls==[(os.getpid(),)]
    assert r._impl.gpu_state is gpu


def test_manager_prepare_real_archive_runtime_matrix_with_fabricated_completed_parent(tmp_path,monkeypatch):
    """Exercise the generated contract/package across the manager/runtime boundary."""
    import tarfile
    from paper.scripts import manage_titantpp_late_path18 as manager
    project=tmp_path/'project';project.mkdir()
    parent=runtime_contract(tmp_path/'parent-fixture',time_input='raw',mode='raw_mse')
    parent.update(data_receipt={'scope':'fabricated TrainVal only'},stored_cap_mlp_reference={'preserved':True})
    parent_inputs=project/'search_artifacts'/manager.PARENT_NAME/'input_snapshot'
    parent_bundle=project/'search_artifacts'/manager.PARENT_NAME
    for name,digest in parent['input_files'].items():
        source=Path(parent['hosts']['5080']['source_root'])/name
        target=parent_inputs/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        assert r.sha_file(target)==digest
    for host in ('5080','5090'):
        originals=parent_bundle/'retrieval'/host/'originals';originals.mkdir(parents=True)
        r.write(originals/'final_receipt.json',{'status':'complete','unchanged':'parent'},exclusive=True)
        r.write(originals/'dispatcher_exit.json',{'pid':500 if host=='5080' else 600,'actual_os_returncode':0},exclusive=True)
    condition={'time_input':'raw','loss_mode':'raw_mse','time_alpha':0.}
    refs=[]
    for time_input in w.TIME_INPUTS:
        for mode in w.LOSS_MODES:
            for ds in r.DATASETS:
                for seed in (42,52,62):
                    host='5080' if ds=='yellow_trip_hourly' or (ds=='intermittent_frozen_5000' and seed==42) else '5090'
                    refs.append({'dataset':ds,'seed':seed,'host':host,'arm':'titantpp_history_mlp_width16',
                        'loss_mode':mode,'time_input':time_input,'time_alpha':0.,'time_head_trained':False,
                        'time_input_transform':w.TIME_INPUT_TRANSFORMS[time_input],
                        'reevaluated':False,'retrained':False,'sha256':'f'*64})
    binding={'contract_raw_sha256':'parent-raw','contract_canonical_sha256':'parent-canonical',
        'source_files_sha256':'parent-source','completion_sha256':'completed-parent'}
    selection={'status':'selected','selected_condition':condition}
    report=project/'reports/new';report.mkdir(parents=True)
    r.write(report/'SCIENCE_SPEC.json',{'scope':'fabricated manager boundary'},exclusive=True)
    source_path=project/'science.py';source_path.write_text('fabricated = True\n')
    files={'science.py':r.sha_file(source_path)}
    source={'base_git_revision':'fabricated-local','files':files,'files_sha256':r.sha_json(files)}
    bundle=project/'search_artifacts/new'
    parent_before={str(p):r.sha_file(p) for p in parent_bundle.rglob('*') if p.is_file()}
    monkeypatch.setattr(manager._impl,'ROOT',project)
    monkeypatch.setattr(manager._impl,'BUNDLE',bundle)
    monkeypatch.setattr(manager._impl,'REPORT',report)
    monkeypatch.setattr(manager,'verify_parent_completion',lambda:(deepcopy(parent),{},deepcopy(refs),deepcopy(selection),deepcopy(binding)))
    monkeypatch.setattr(manager,'source_snapshot',lambda:deepcopy(source))
    made=manager.prepare()
    actual=r.read(bundle/'execution_contract.json')
    assert made['planned_fit_count']==18 and made['host_fit_counts']=={'5080':8,'5090':10}
    assert made['baseline_reference_count']==9 and actual['common_condition']==condition
    assert len(r.validate_contract(actual,'5080'))==8 and len(r.validate_contract(actual,'5090'))==10
    assert all(x['loss_mode']=='raw_mse' and x['time_input']=='raw' for x in actual['baseline_validation_references'])
    assert all(actual['hosts'][h]['absolute_host_deadline_unix']==parent['hosts'][h]['absolute_host_deadline_unix'] for h in ('5080','5090'))
    assert all(actual['hosts'][h]['completed_predecessor']['root']==parent['hosts'][h]['root'] for h in ('5080','5090'))
    assert len(actual['input_files'])==6 and actual['packaging']['test_inputs_packaged'] is False
    with tarfile.open(bundle/'deployment.tar.gz','r:gz') as archive:
        names=set(archive.getnames())
        assert 'execution_contract.json' in names and 'source/science.py' in names
        assert all('test.parquet' not in n and 'full_capped.parquet' not in n for n in names)
        assert all('source/'+name in names for name in actual['input_files'])
        packaged=r.json.loads(archive.extractfile('execution_contract.json').read())
        assert packaged==actual
    assert parent_before=={str(p):r.sha_file(p) for p in parent_bundle.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='no overwrite'):manager.prepare()
    assert parent_before=={str(p):r.sha_file(p) for p in parent_bundle.rglob('*') if p.is_file()}
