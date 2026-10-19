"""Fabricated-only v2 predecessor/matrix checks; no SSH or research inference."""
from copy import deepcopy
from pathlib import Path

import pytest

from paper.scripts import titantpp_late_path18_v2_runtime as r
from paper.scripts import titantpp_late_path18_runtime as old
from paper.scripts import titantpp_quantity_loss_mlp18_runtime_v2 as parent
from paper.tests.test_titantpp_quantity_loss_mlp18_operations import contract as old_fixture, make_receipt, stopped


def candidate_contract(tmp_path,mode='log_mse'):
    c=old_fixture.__wrapped__(tmp_path)
    c['schema']=r.SCHEMA
    c['common_condition']={'time_input':'log1p','loss_mode':mode,'time_alpha':0.}
    c['architecture'].update(arms=list(r.ARMS),quantity_loss_mode=mode,quantity_loss_modes=[mode],
        time_input_transform='log1p_nonnegative_float32',quantity_input_transform='log1p_nonnegative_float32',
        model_architecture_changed=True,late_path_parameter_count=1024)
    c['parent_quantity_loss27']={'contract_raw_sha256':r._PARENT_RAW_SHA256,
        'contract_canonical_sha256':r._PARENT_CANONICAL_SHA256,'source_files_sha256':'parent-source'}
    for h in c['hosts'].values():
        h['completed_predecessor'].update(contract_raw_sha256=r._PARENT_RAW_SHA256,
            contract_canonical_sha256=r._PARENT_CANONICAL_SHA256)
    c['jobs']=[{'id':f'{d}__{a}__seed{s}__loss{mode}','dataset':d,'arm':a,'seed':s,'loss_mode':mode,
        'time_input':'log1p','time_alpha':0.,'host':'5080' if d=='yellow_trip_hourly' or (d=='intermittent_frozen_5000' and s==42) else '5090'}
        for d in r.DATASETS for s in (42,52,62) for a in r.ARMS]
    for ref in c['baseline_validation_references']:
        ref.update(time_input='log1p',time_input_transform='log1p_nonnegative_float32',loss_mode=mode)
    return c


@pytest.mark.parametrize('mode',r.LOSS_MODES)
def test_same18_science_logged_inputs_shared_loss_and_host8_10(tmp_path,mode):
    c=candidate_contract(tmp_path,mode)
    assert len(r.validate_contract(c,'5080'))==8
    assert len(r.validate_contract(c,'5090'))==10
    assert r.SCHEMA==old.SCHEMA=='titantpp_late_path18_v1'
    assert r.WORKER_MODULE==old.WORKER_MODULE=='paper.scripts.run_titantpp_late_path18'
    assert r.MODULE=='paper.scripts.titantpp_late_path18_v2_runtime'
    assert r.STAGES==('train','validation') and r.ARMS==old.ARMS
    assert 'parent_raw_time27' not in c


@pytest.mark.parametrize('mutation',('raw_time','raw_quantity','wrong_parent_sha','wrong_parent_canonical',
    'rawtime_lineage','missing_lineage','extra_loss','mixed_job_loss','mixed_job_time','test','checkpoint_warmstart','capacity','budget','deadline'))
def test_v2_scope_and_predecessor_mutation_rejected(tmp_path,mutation):
    c=candidate_contract(tmp_path)
    if mutation=='raw_time':c['common_condition']['time_input']='raw';c['architecture']['time_input_transform']='raw_nonnegative_float32'
    if mutation=='raw_quantity':c['architecture']['quantity_input_transform']='raw_nonnegative_float32'
    if mutation=='wrong_parent_sha':c['parent_quantity_loss27']['contract_raw_sha256']='wrong'
    if mutation=='wrong_parent_canonical':c['parent_quantity_loss27']['contract_canonical_sha256']='wrong'
    if mutation=='rawtime_lineage':c['parent_raw_time27']=deepcopy(c['parent_quantity_loss27'])
    if mutation=='missing_lineage':c.pop('parent_quantity_loss27')
    if mutation=='extra_loss':c['architecture']['quantity_loss_modes']=['log_mse','raw_mse']
    if mutation=='mixed_job_loss':c['jobs'][0]['loss_mode']='raw_mae'
    if mutation=='mixed_job_time':c['jobs'][0]['time_input']='raw'
    if mutation=='test':c['held_out_test_authorized']=True
    if mutation=='checkpoint_warmstart':c['architecture']['checkpoint_warmstart']=True
    if mutation=='capacity':c['architecture']['late_path_parameter_count']=2048
    if mutation=='budget':c['training']['maximum_epochs']=301
    if mutation=='deadline':c['hosts']['5080']['absolute_host_deadline_unix']=None
    with pytest.raises((ValueError,KeyError)):r.validate_contract(c,'5080')


def test_private_modules_and_preserved_old_runtime_are_unchanged():
    assert r._impl is not old._impl and r._adapter is not old
    assert old.MODULE=='paper.scripts.titantpp_late_path18_runtime'
    assert r._impl.MODULE==r.MODULE and r._impl.WORKER_MODULE==old.WORKER_MODULE
    assert r._impl.run_stage.__globals__ is r._impl.__dict__
    assert r._impl.predecessor_gate is r.predecessor_gate
    assert old._impl.predecessor_gate is old.predecessor_gate
    assert r.sha_file(r._V1_PATH)==r._V1_SHA256
    assert r._PARENT_RAW_SHA256=='bca53c976981c168bd0c7850c89a178e154e8e851498d1a2646756febd36441b'
    assert r._PARENT_CANONICAL_SHA256=='4b574670362e6469826fcbb4c1e355508594b76a00a125d4e418ca03a0ca71a2'


def completed_parent(tmp_path,monkeypatch,host='5080'):
    pc=old_fixture.__wrapped__(tmp_path/'parent')
    cp=Path(pc['hosts'][host]['root'])/'execution_contract.json'
    r.write(cp,pc,exclusive=True)
    raw,canonical=r.sha_file(cp),r.sha_json(pc)
    # Fabricated evidence gets its own golden bytes; the production constants
    # are separately checked above and never edited in source or on disk.
    monkeypatch.setattr(r,'_PARENT_RAW_SHA256',raw)
    monkeypatch.setattr(r,'_PARENT_CANONICAL_SHA256',canonical)
    c=candidate_contract(tmp_path/'new')
    c['parent_quantity_loss27'].update(source_files_sha256=pc['source']['files_sha256'])
    c['hosts'][host]['absolute_host_deadline_unix']=pc['hosts'][host]['absolute_host_deadline_unix']
    root=Path(pc['hosts'][host]['root']);jobs=parent.validate_contract(pc,host)
    stages={};pid=1000
    for stage in parent.STAGES:
        receipts={};exits={}
        for job in jobs:
            make_receipt(pc,host,job,stage)
            receipts[job['id']]=parent.verify_receipt(pc,host,job,stage)
            exits[job['id']]={'pid':pid,'actual_os_returncode':0,'timeout':False};pid+=1
        stages[stage]={'status':'complete','receipts':receipts,'exits':exits}
    inventory={str(p.relative_to(root)):r.sha_file(p) for p in root.rglob('*') if p.is_file()}
    mp=root/'artifact_manifest.json';r.write(mp,{'files':inventory},exclusive=True)
    final={'status':'complete','host':host,'contract_sha256':canonical,'completed_fit_count':len(jobs),
        'validation_replay_count':len(jobs),'test_count':0,'held_out_test_evaluated':False,
        'owned_worker_pids_absent':True,'stages':stages,
        'artifact_manifest':{'path':str(mp.relative_to(root)),'sha256':r.sha_file(mp),'file_count':len(inventory)}}
    fp=root/'final_receipt.json';r.write(fp,final,exclusive=True)
    xp=root/'dispatcher_exit.json';r.write(xp,{'pid':3000,'actual_os_returncode':0},exclusive=True)
    c['hosts'][host]['completed_predecessor'].update(root=str(root),contract_raw_sha256=raw,
        contract_canonical_sha256=canonical,source_files_sha256=pc['source']['files_sha256'],
        dispatcher_identity={'pid':3000},final_receipt_sha256=r.sha_file(fp),dispatcher_exit_sha256=r.sha_file(xp))
    monkeypatch.setattr(r._impl,'paused_processes',lambda *args:stopped())
    monkeypatch.setattr(r._impl,'proc_identity',lambda pid:None)
    monkeypatch.setattr(r._impl,'gpu_state',lambda *args,**kwargs:({'compute_pids':[11,12,13]},stopped()))
    return c,pc,root,final


@pytest.mark.parametrize('host',('5080','5090'))
def test_actual_quantity_parent_receipt_gate_worker_exit_manifest_pid_and_gpu(tmp_path,monkeypatch,host):
    c,pc,root,final=completed_parent(tmp_path,monkeypatch,host)
    before={str(p):r.sha_file(p) for p in root.rglob('*') if p.is_file()}
    result=r.predecessor_gate(c,host,require_complete=True)
    assert result['status']=='complete'
    assert result['predecessor_campaign']=='titantpp_quantity_loss_mlp18_20261019_v2'
    assert result['parent_owned_pids_absent'] and result['parent_gpu_pids_absent']
    assert before=={str(p):r.sha_file(p) for p in root.rglob('*') if p.is_file()}
    assert len(parent.validate_contract(pc,host))=={'5080':8,'5090':10}[host]


@pytest.mark.parametrize('mutation',('scientific_fail','timeout','missing_stage','dispatcher_fail','parent_failure',
    'empty_manifest','selected_binary_changed','receipt_changed','owner_alive','gpu_owner','deadline_changed'))
def test_quantity_parent_gate_rejects_failure_incomplete_or_changed_evidence(tmp_path,monkeypatch,mutation):
    c,pc,root,final=completed_parent(tmp_path,monkeypatch)
    h=c['hosts']['5080']['completed_predecessor'];job=parent.validate_contract(pc,'5080')[0]
    if mutation=='scientific_fail':final['stages']['train']['exits'][job['id']]['actual_os_returncode']=1
    if mutation=='timeout':final['stages']['validation']['exits'][job['id']]['timeout']=True
    if mutation=='missing_stage':final['stages'].pop('validation')
    if mutation=='dispatcher_fail':r.write(root/'dispatcher_exit.json',{'pid':3000,'actual_os_returncode':1});h['dispatcher_exit_sha256']=r.sha_file(root/'dispatcher_exit.json')
    if mutation=='parent_failure':r.write(root/'dispatch_failure.json',{'status':'failed'},exclusive=True)
    if mutation=='empty_manifest':
        mp=root/'artifact_manifest.json';r.write(mp,{'files':{}})
        final['artifact_manifest'].update(sha256=r.sha_file(mp),file_count=0)
    if mutation=='selected_binary_changed':(root/'run'/job['id']/'selected.pt').write_bytes(b'mutated')
    if mutation=='receipt_changed':r.write(root/'run'/job['id']/'validation_receipt.json',{'status':'complete'})
    if mutation=='owner_alive':monkeypatch.setattr(r._impl,'proc_identity',lambda pid:{'pid':pid})
    if mutation=='gpu_owner':monkeypatch.setattr(r._impl,'gpu_state',lambda *args,**kwargs:({'compute_pids':[3000]},stopped()))
    if mutation=='deadline_changed':c['hosts']['5080']['absolute_host_deadline_unix']+=1.
    r.write(root/'final_receipt.json',final);h['final_receipt_sha256']=r.sha_file(root/'final_receipt.json')
    with pytest.raises(ValueError):r.predecessor_gate(c,'5080',require_complete=True)


def test_v2_manager_prepare_real_runtime_archive_after_only_fabricated_loss27(tmp_path,monkeypatch):
    import json
    import tarfile
    from paper.scripts import manage_titantpp_late_path18_v2 as manager

    project=tmp_path/'project';project.mkdir()
    pc=old_fixture.__wrapped__(tmp_path/'parent-data')
    pc.update(data_receipt={'scope':'fabricated TrainVal only'},stored_cap_mlp_reference={'preserved':True})
    origin=project/'search_artifacts/titantpp_quantity_loss_mlp18_20261019_v2'
    for name,digest in pc['input_files'].items():
        p=origin/'input_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((Path(pc['hosts']['5080']['source_root'])/name).read_bytes())
        assert r.sha_file(p)==digest
    for host in ('5080','5090'):
        originals=origin/'retrieval'/host/'originals';originals.mkdir(parents=True)
        r.write(originals/'final_receipt.json',{'status':'complete','preserved':True},exclusive=True)
        r.write(originals/'dispatcher_exit.json',{'pid':500 if host=='5080' else 600,'actual_os_returncode':0},exclusive=True)
    condition={'time_input':'log1p','loss_mode':'raw_mse','time_alpha':0.}
    refs=[{**deepcopy(ref),'time_input':'log1p','time_input_transform':'log1p_nonnegative_float32','loss_mode':mode}
        for mode in ('log_mse','raw_mse','raw_mae') for ref in pc['baseline_validation_references']]
    selection={'status':'selected','selected_condition':condition,'condition_scores':[{}, {}, {}]}
    binding={'contract_raw_sha256':r._PARENT_RAW_SHA256,'contract_canonical_sha256':r._PARENT_CANONICAL_SHA256,
        'source_files_sha256':pc['source']['files_sha256'],'completion_sha256':'completed-loss27'}
    report=project/'reports/new-v2';report.mkdir(parents=True)
    r.write(report/'SCIENCE_SPEC.json',{'scope':'fabricated boundary v2'},exclusive=True)
    p=project/'science.py';p.write_text('fabricated = True\n');files={'science.py':r.sha_file(p)}
    source={'base_git_revision':'fabricated-local-v2','files':files,'files_sha256':r.sha_json(files)}
    bundle=project/'search_artifacts/new-v2'
    before={str(p):r.sha_file(p) for p in origin.rglob('*') if p.is_file()}
    monkeypatch.setattr(manager._impl,'ROOT',project)
    monkeypatch.setattr(manager._impl,'BUNDLE',bundle)
    monkeypatch.setattr(manager._impl,'REPORT',report)
    monkeypatch.setattr(manager._adapter,'verify_parent_completion',lambda:(deepcopy(pc),{},deepcopy(refs),deepcopy(selection),deepcopy(binding)))
    monkeypatch.setattr(manager._adapter,'source_snapshot',lambda:deepcopy(source))
    made=manager.prepare();actual=r.read(bundle/'execution_contract.json')
    assert made['planned_fit_count']==18 and made['host_fit_counts']=={'5080':8,'5090':10}
    assert made['baseline_reference_count']==9 and actual['common_condition']==condition
    assert len(r.validate_contract(actual,'5080'))==8 and len(r.validate_contract(actual,'5090'))==10
    assert 'parent_quantity_loss27' in actual and 'parent_raw_time27' not in actual
    assert all(ref['loss_mode']=='raw_mse' and ref['time_input']=='log1p' for ref in actual['baseline_validation_references'])
    assert all(actual['hosts'][h]['absolute_host_deadline_unix']==pc['hosts'][h]['absolute_host_deadline_unix'] for h in ('5080','5090'))
    assert all(actual['hosts'][h]['tmux']==f'late_path18_{h}_20261019_v2' for h in ('5080','5090'))
    assert all(actual['hosts'][h]['completed_predecessor']['root']==pc['hosts'][h]['root'] for h in ('5080','5090'))
    assert len(actual['input_files'])==6 and actual['packaging']['test_inputs_packaged'] is False
    with tarfile.open(bundle/'deployment.tar.gz','r:gz') as archive:
        assert json.loads(archive.extractfile('execution_contract.json').read())==actual
        names=set(archive.getnames())
        assert 'source/science.py' in names and all('source/'+n in names for n in actual['input_files'])
        assert all('test.parquet' not in n and 'full_capped.parquet' not in n for n in names)
    assert before=={str(p):r.sha_file(p) for p in origin.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='no overwrite'):manager.prepare()
    assert before=={str(p):r.sha_file(p) for p in origin.rglob('*') if p.is_file()}
