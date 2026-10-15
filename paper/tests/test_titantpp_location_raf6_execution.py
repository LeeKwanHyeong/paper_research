"""Synthetic protocol tests; no research rows, checkpoints or remote calls."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import pytest
from paper.scripts import titantpp_location_raf6_runtime as runtime
from paper.scripts import advance_titantpp_location_raf6 as followup
from paper.scripts import prepare_titantpp_location_raf6_execution as prep
from paper.scripts import deploy_titantpp_location_raf6 as deploy


def contract():
    remote='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'+runtime.SCHEMA+'_5090'
    return {'schema':runtime.SCHEMA,'jobs':runtime.expected_jobs(),'arms':list(runtime.ARMS),
        'training':deepcopy(runtime.TRAINING),'parallelism':deepcopy(runtime.PARALLELISM),
        'evaluation_scope':'validation_only','held_out_test_evaluated':False,'test_evaluation':False,
        'limits':{'total_wall_seconds':168*3600,'per_condition_seconds':36*3600,'automatic_retry':False},
        'hosts':{'5090':{'root':remote,'source_root':remote+'/source','operation_root':remote+'/operation',
            'gpu_uuid':'GPU-c9adc246-ef66-7906-bc07-6e152fa9952f',
            'python':'/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12'}},
        'source':{'files':{},'files_sha256':runtime.sha_json({})},
        'comparison_runtime_limitation':'new candidate arms on5090, historicalCNN5080',
        'primary_contrast':'Value−Query'}


def test_six_unique_candidate_jobs_same_runtime_two_waves():
    c=contract();runtime.validate(c,verify_files=False)
    assert len(c['jobs'])==len({j['id'] for j in c['jobs']})==6
    assert c['parallelism']['maximum_workers']==3
    for arm in runtime.ARMS:
        assert {j['seed'] for j in c['jobs'] if j['arm']==arm}=={42,52,62}
    assert {j['dataset'] for j in c['jobs']}=={'raf_spare_parts'}
    assert {j['host'] for j in c['jobs']}=={'5090'}


@pytest.mark.parametrize('change',['gpu','root','jobs','budget','test','concurrency','retry'])
def test_contract_scope_drift_blocks(change):
    c=contract()
    if change=='gpu':c['hosts']['5090']['gpu_uuid']='foreign'
    elif change=='root':c['hosts']['5090']['root']+='_old'
    elif change=='jobs':c['jobs'][0]['seed']=7
    elif change=='budget':c['training']['maximum_epochs']=301
    elif change=='test':c['test_evaluation']=True
    elif change=='concurrency':c['parallelism']['maximum_workers']=4
    elif change=='retry':c['limits']['automatic_retry']=True
    with pytest.raises(ValueError):runtime.validate(c,verify_files=False)



def test_foreign_gpu_worker_rejected():
    active={'job':{'child':SimpleNamespace(pid=20),'command':['timeout','--signal=TERM','--kill-after=15s','40','python','mine']}}
    with pytest.raises(ValueError,match='Foreign'):
        runtime.verify_gpu_membership({30},{30:{'ppid':99,'pgrp':99,'argv':['python','foreign']}},active,script='mine',used_bytes=1,total_bytes=10)


def test_exclusive_write_and_preserved_failure(tmp_path):
    prep.write(tmp_path/'failure.json',{'original':True})
    before=(tmp_path/'failure.json').read_bytes()
    with pytest.raises(FileExistsError):prep.write(tmp_path/'failure.json',{'original':False})
    with patch.object(runtime,'validate',return_value=None),pytest.raises(ValueError,match='no automatic retry'):
        runtime.authorization({},tmp_path)
    assert (tmp_path/'failure.json').read_bytes()==before


def test_remote_programs_compile_and_are_new_scope_only():
    compile(deploy.REMOTE_CODE,'deployment','exec')
    compile(followup.REMOTE_CODE,'observation','exec')
    assert 'scientific_worker_count\')==6' in followup.REMOTE_CODE
    assert 'supervisor_wrapper_own_os_exit_directly_observed' in followup.REMOTE_CODE
    assert 'titantpp_backbone_location_raf6_20261015_v1' in followup.REMOTE


def test_six_selected_models_reuse_historical_baseline(tmp_path):
    c=contract();endpoint={'count':6690,'qty_rmse':4.,'qty_mae':2.,'time_nll':1.,
        'tail':{'qty_rmse':8.,'qty_mae':5.,'time_nll':3.}}
    c['baseline_jobs']=[{'seed':seed,'original':{'selected_epoch':2,'validation_reference':endpoint}} for seed in runtime.SEEDS]
    c['stored_channel_reference_jobs']=[{'seed':seed,'reference_role':'stored_prior_loop_qkv_channel_not_fresh_primary_control',
        'original':{'selected_epoch':2,'validation_reference':endpoint}} for seed in runtime.SEEDS]
    for j in c['jobs']:
        folder=tmp_path/'originals/run'/j['id']
        prefix=folder/'runs'/j['arm']/'count_only_log_regression'/f"seed_{j['seed']}"
        followup.write(prefix/'endpoint_replays.json',{'best_epoch':2,'selected':endpoint})
        followup.write(prefix/'history.json',{'history':[{'epoch':1,'val_qty_rmse':5.},
            {'epoch':2,'val_qty_rmse':4.},{'epoch':3,'val_qty_rmse':4.}]})
        followup.write(folder/'terminal_manifest.json',{'job':j,'held_out_test_evaluated':False,
            'files':{str(p.relative_to(folder)):followup.sha(p) for p in prefix.iterdir()}})
    report=tmp_path/'report';report.mkdir()
    with patch.object(followup,'REPORT',report):result=followup.analyze(c,tmp_path/'originals')
    assert result['new_fit_count']==6 and len(result['rows'])==12 and len(result['seed_differences'])==18
    assert set(result['aggregate'])=={'CNN','StoredChannel','Query','Value'}
    assert all(r['best_epoch']==2 for r in result['rows'])
    assert all(d['RMSE']['absolute']==0 for d in result['seed_differences'])


def test_eta_uses_only_current_fit_own_finite_completed_timings():
    snapshot={'observed_unix':100,'all_training_terminal_verified':False,'failure':False,
        'jobs':[{'job':{'arm':'a','seed':42},'state':'running',
            'history':{'history':[{'epoch':1,'val_qty_rmse':8.},{'epoch':2,'val_qty_rmse':7.}]},
            'timing':{'epochs':[{'elapsed_seconds':float('nan')},{'elapsed_seconds':0},
                {'elapsed_seconds':2.},{'elapsed_seconds':4.}]}}]}
    row=followup.brief_round(snapshot)['rows'][0]
    assert row['timing_samples']==2 and row['conditional_stop_epoch']==42
    assert row['conditional_stop_seconds']==120 and row['epoch300_seconds']==894


def terminal_fixture(root):
    """Complete fabricated six-fit launch/permit/exit chain, no tensor/GPU data."""
    c=contract();c['hosts']['5090']['root']=str(root)
    digest=runtime.sha_json(c);counter=iter(range(900000001,900000100))
    base=[c['hosts']['5090']['python'],str(root/'operation/runtime.py'),'--contract',str(root/'execution_contract.json'),'--host','5090']
    def put(name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,sort_keys=True)+'\n');return value
    def launch(name,argv,observer,**kw):
        return put(name,{'pid':next(counter),'start_ticks':100,'observer_pid':observer['pid'],
            'observer_start_ticks':observer['start_ticks'],'argv':argv,'contract_sha256':digest,**kw})
    def child(name,l,argv,**kw):
        return put(name,{'pid':next(counter),'start_ticks':101,'ppid':l['pid'],'guard_start_ticks':l['start_ticks'],
            'argv':argv,'contract_sha256':digest,**kw})
    def actual(name,ln,l,**kw):
        return put(name,{'worker_group_pid':l['pid'],'supervisor_pid':l['observer_pid'],'argv':l['argv'],
            'contract_sha256':digest,'returncode':0,'actual_child_exit_observed':True,
            'launch_sha256':runtime.sha_file(root/ln),**kw})
    wrapper={'pid':next(counter),'start_ticks':99}
    pl=launch('pipeline_launch.json',base+['--mode','pipeline'],wrapper)
    pipeline=put('pipeline.claim',{'pid':pl['pid'],'start_ticks':pl['start_ticks'],'ppid':wrapper['pid'],
        'wrapper_start_ticks':wrapper['start_ticks'],'argv':pl['argv'],'contract_sha256':digest})
    approval=put('approval.json',{'approved':True,'hosts':['5090'],'fresh_fit_count':6,
        'maximum_concurrent_workers':3,'contract_sha256':digest})
    start=put('start_permit.json',{'host':'5090','started_at_unix':1.,'deadline_unix':1.+168*3600,
        'contract_sha256':digest,'approval_sha256':runtime.sha_json(approval)})
    guard=lambda tail:['timeout','--signal=TERM','--kill-after=15s','300']+base+tail
    ql=launch('qualification/launch.json',guard(['--mode','native']),pipeline,phase='native')
    q=child('qualification/receipt.json',ql,base+['--mode','native'],host='5090',status='passed',
        source_closure=c['source']['files_sha256'],held_out_test_evaluated=False,scientific_optimizer_updates=0,
        synthetic={a:{'arm':a,'status':'passed','contract_sha256':digest,'active_added_parameters':1024,
            'nonlinear_location':'q' if a==runtime.ARMS[0] else 'v'} for a in runtime.ARMS})
    for phase in ('ready','trained'):
        put('qualification/admission/'+phase+'_concurrency.json',{'workers':[{}, {}, {}],
            'contract_sha256':digest,'memory':{'gpu_uuid':c['hosts']['5090']['gpu_uuid'],
            'total_bytes':100,'used_bytes':70,'limit_fraction':.8}})
    for i,a,seed in runtime.ADMISSION_JOBS:
        prefix='qualification/admission/'+i
        l=launch(prefix+'.launch.json',base+['--mode','admission','--admission-id',i],q,admission_id=i,arm=a,seed=seed)
        put(prefix+'.ready.json',{'pid':l['pid'],'start_ticks':l['start_ticks'],'contract_sha256':digest})
        actual(prefix+'.actual_exit.json',prefix+'.launch.json',l,admission_id=i,arm=a,seed=seed)
    q['capacity_check']={'passed':True,'maximum_workers':3,'simultaneous_synthetic_workers_executed':True,
        'worker_actual_exit0_count':3,'evidence_files':{str(p.relative_to(root)):runtime.sha_file(p)
            for p in (root/'qualification/admission').iterdir()}}
    put('qualification/receipt.json',q);actual('qualification/actual_exit.json','qualification/launch.json',ql,phase='native')
    put('training_permit.json',{'contract_sha256':digest,'qualification':q,
        'qualification_sha256':runtime.sha_file(root/'qualification/receipt.json'),
        'start_permit_sha256':runtime.sha_json(start)})
    dl=launch('dispatcher_launch.json',guard(['--mode','dispatch']),pipeline,phase='dispatch')
    dispatcher=child('supervisor.claim',dl,base+['--mode','dispatch'])
    for j in c['jobs']:
        name=j['id'];deadline=1000.
        put('claims/'+name+'.json',{'job':j,'deadline_unix':deadline,'contract_sha256':digest})
        tail=['--mode','fit','--job',name,'--deadline',str(deadline)]
        l=launch('worker_launches/'+name+'.json',guard(tail),dispatcher,job=j)
        wi=child('worker_identities/'+name+'.json',l,base+tail,job=j,deadline_unix=deadline,
            pgrp=l['pid'],supervisor_pid=dispatcher['pid'],supervisor_start_ticks=dispatcher['start_ticks'],
            launch_sha256=runtime.sha_file(root/'worker_launches'/ (name+'.json')))
        actual('worker_exits/'+name+'.json','worker_launches/'+name+'.json',l,job=j,
            worker_identity_sha256=runtime.sha_file(root/'worker_identities'/(name+'.json')),
            worker_claim_sha256=runtime.sha_file(root/'claims'/(name+'.json')))
        prefix='runs/'+j['arm']+'/count_only_log_regression/seed_'+str(j['seed'])+'/'
        folder=root/'run'/name;folder.mkdir(parents=True)
        files={}
        for f in ('best_val_qty_rmse_model.pt','last_epoch_state.pt'):
            p=folder/(prefix+f);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fabricated-not-a-checkpoint');files[prefix+f]=runtime.sha_file(p)
        put('run/'+name+'/terminal_manifest.json',{'job':j,'status':'complete','scientific_success':True,
            'contract_sha256':digest,'held_out_test_evaluated':False,'evaluation_scope':'validation_only','files':files})
    put('status.json',{'status':'complete','contract_sha256':digest,'completed':{j['id']:{} for j in c['jobs']}})
    put('train_complete.json',{'status':'complete','contract_sha256':digest,'scientific_worker_count':6,'dispatcher_actual_exit':0})
    actual('dispatcher_exit.json','dispatcher_launch.json',dl,phase='dispatch')
    actual('pipeline_exit.json','pipeline_launch.json',pl)
    put('execution_contract.json',c)
    put('deployment_manifest.json',{'contract_sha256':digest,'files':{'execution_contract.json':runtime.sha_file(root/'execution_contract.json'),'approval.json':runtime.sha_file(root/'approval.json')}})
    return c,put


def run_observer(root,probe=None,gpu=''):
    import io
    output=SimpleNamespace(buffer=io.BytesIO());scope={'ROOT':str(root),'MANIFEST_SHA':runtime.sha_file(root/'deployment_manifest.json')}
    code=followup.REMOTE_CODE
    if probe:
        code=code.replace('pipeline_probe=probe(', 'probe=TEST_PROBE\npipeline_probe=probe(',1);scope['TEST_PROBE']=probe
    with patch('sys.stdout',output),patch('subprocess.check_output',return_value=gpu):exec(compile(code,'observer_fixture','exec'),scope)
    size,body=output.buffer.getvalue().split(b'\n',1)
    return json.loads(body[:int(size)])


def test_fully_bound_terminal_six_workers_and_permit_pass(tmp_path):
    c,_=terminal_fixture(tmp_path)
    assert runtime.verify_training_permit(c,tmp_path)
    r=run_observer(tmp_path)
    assert r['all_training_terminal_verified'] is True
    assert len(r['jobs'])==6 and all(x['state']=='complete' for x in r['jobs'])
    assert all(r['proof_bindings'].values())
    assert r['supervisor_wrapper_own_os_exit_directly_observed'] is False


@pytest.mark.parametrize('missing',['training_permit.json','qualification/receipt.json','start_permit.json','pipeline_launch.json','supervisor.claim'])
def test_missing_proof_never_terminal(tmp_path,missing):
    terminal_fixture(tmp_path);(tmp_path/missing).unlink()
    r=run_observer(tmp_path)
    assert not r['all_training_terminal_verified']
    assert all(j['state']!='complete' for j in r['jobs'])


def test_false_terminal_foreign_worker_exit_and_missing_identity(tmp_path):
    c,put=terminal_fixture(tmp_path);name=c['jobs'][0]['id']
    e=runtime.read(tmp_path/'worker_exits'/(name+'.json'));e['job']=c['jobs'][1];put('worker_exits/'+name+'.json',e)
    (tmp_path/'worker_identities'/(c['jobs'][1]['id']+'.json')).unlink()
    r=run_observer(tmp_path)
    assert not r['all_training_terminal_verified']
    assert r['jobs'][0]['state']==r['jobs'][1]['state']=='unconfirmed'


def test_changed_admission_memory_evidence_rejected(tmp_path):
    c,put=terminal_fixture(tmp_path)
    p='qualification/admission/trained_concurrency.json';v=runtime.read(tmp_path/p);v['memory']['used_bytes']=90;put(p,v)
    with pytest.raises(ValueError,match='Admission evidence drift'):runtime.verify_training_permit(c,tmp_path)
    with pytest.raises(AssertionError,match='admission evidence drift'):run_observer(tmp_path)


def test_admission_gpu_aggregate_limit_before_release():
    c=contract();gpu=c['hosts']['5090']['gpu_uuid']
    with patch.object(runtime.subprocess,'run',return_value=SimpleNamespace(stdout=gpu+', 100, 81\n')):
        with pytest.raises(ValueError,match='over80'):runtime.admission_memory(c,'5090')
    with patch.object(runtime.subprocess,'run',return_value=SimpleNamespace(stdout=gpu+', 100, 80\n')):
        assert runtime.admission_memory(c,'5090')['aggregate_fraction']==.8


def test_qualification_pending_is_waiting_not_false_failure(tmp_path):
    c,_=terminal_fixture(tmp_path)
    for folder in ('run','claims','worker_exits','worker_identities','worker_launches'):
        import shutil
        shutil.rmtree(tmp_path/folder)
    for f in ('training_permit.json','qualification/receipt.json','qualification/actual_exit.json','supervisor.claim',
              'dispatcher_launch.json','dispatcher_exit.json','pipeline_exit.json','train_complete.json','status.json'):
        (tmp_path/f).unlink()
    r=run_observer(tmp_path)
    assert not r['failure'] and not r['all_training_terminal_verified']
    assert all(j['state']=='waiting' for j in r['jobs'])


def test_reused_pid_cannot_claim_running(tmp_path):
    c,put=terminal_fixture(tmp_path);name=c['jobs'][0]['id'];wi=runtime.read(tmp_path/'worker_identities'/(name+'.json'))
    supervisor=runtime.read(tmp_path/'supervisor.claim');pipeline=runtime.read(tmp_path/'pipeline.claim');launch=runtime.read(tmp_path/'worker_launches'/(name+'.json'))
    (tmp_path/'run'/name/'terminal_manifest.json').unlink()
    put('status.json',{'status':'running','contract_sha256':runtime.sha_json(c),'active_jobs':{name:{}},
        'supervisor_pid':supervisor['pid'],'supervisor_start_ticks':supervisor['start_ticks'],'supervisor_argv':supervisor['argv']})
    put('leases/'+name+'.json',{'contract_sha256':runtime.sha_json(c),'job':name,
        'supervisor_pid':supervisor['pid'],'supervisor_start_ticks':supervisor['start_ticks'],
        'worker_group_pid':launch['pid'],'worker_group_start_ticks':launch['start_ticks'],'expires_unix':999.})
    states={v['pid']:{**v,'state':'live'} for v in (wi,supervisor,pipeline)}
    states[launch['pid']]={**launch,'state':'live','ppid':supervisor['pid']}
    def probe(pid):return states.get(pid,{'pid':pid,'state':'absent'})
    gpu=c['hosts']['5090']['gpu_uuid']+', '+str(wi['pid'])+'\n'
    with patch('time.time',return_value=100.):
        assert run_observer(tmp_path,probe,gpu)['jobs'][0]['state']=='running'
        states[wi['pid']]['start_ticks']+=1
        assert run_observer(tmp_path,probe,gpu)['jobs'][0]['state']=='unconfirmed'


def test_candidate_only_admission_and_initial_state_contract():
    assert len(runtime.ADMISSION_JOBS)==3 and len({x[0] for x in runtime.ADMISSION_JOBS})==3
    assert all(a in runtime.ARMS and a!='titantpp_cnn' for _,a,_ in runtime.ADMISSION_JOBS)
    import inspect
    assert 'runner.initial_states(data,seed,arms=ARMS)' in inspect.getsource(runtime.bootstrap)
    assert 'runner.initial_states(data,s,arms=ARMS)' in inspect.getsource(runtime.native_qualification)


def test_scope_has_only_location_candidates_not_prior_qkv_or_cnn():
    assert runtime.ARMS==('titantpp_cnn_query_nonlinear_v1','titantpp_cnn_value_nonlinear_v1')
    assert prep.ARMS==runtime.ARMS and runtime.SCHEMA==prep.SCHEMA
    assert runtime.ADMISSION_JOBS==(('query42',runtime.ARMS[0],42),('query52',runtime.ARMS[0],52),('value62',runtime.ARMS[1],62))
    assert 'titantpp_cnn_channel_nonlinear_v1' not in followup.REMOTE_CODE
    assert 'titantpp_cnn_joint_nonlinear_v1' not in followup.REMOTE_CODE


def stored_reference_fixture(root, *, held_out=False, split='validation'):
    parent=root/'parent';report=root/'report';parent.mkdir();report.mkdir()
    jobs=[];rows=[];files={}
    for seed in runtime.SEEDS:
        arm='titantpp_cnn_channel_nonlinear_v1';jid=f'raf_spare_parts__{seed}__{arm}'
        jobs.append({'id':jid,'arm':arm,'seed':seed})
        endpoint=parent/'retrieval/originals/run'/jid/'runs'/arm/'count_only_log_regression'/f'seed_{seed}'/'endpoint_replays.json'
        prep.write(endpoint,{'best_epoch':3,'selected':{'count':6690,'qty_rmse':4.,'qty_mae':2.}})
        files[str(endpoint.relative_to(root))]=prep.sha(endpoint)
        rows.append({'arm':'Channel','seed':seed,'best_epoch':3,'RMSE':4.,'MAE':2.})
    validation=report/'THREE_SEED_VALIDATION.json'
    prep.write(validation,{'split':split,'dataset':'raf_spare_parts','held_out_test_evaluated':held_out,'rows':rows})
    files[str(validation.relative_to(root))]=prep.sha(validation)
    receipt=parent/'retrieval/final_receipt.json'
    prep.write(receipt,{'status':'complete','new_fit_count':6,'held_out_test_evaluated':False,
        'all_training_terminal_verified':True,'files':files})
    return parent,report,receipt,{'jobs':jobs}


def test_stored_channel_reads_only_pinned_validation_and_is_not_fresh_job(tmp_path):
    parent,report,receipt,c=stored_reference_fixture(tmp_path)
    with patch.object(prep,'ROOT',tmp_path),patch.object(prep,'PARENT',parent),patch.object(prep,'PARENT_REPORT',report),patch.object(prep,'PARENT_RECEIPT_SHA',prep.sha(receipt)):
        jobs,pins=prep.stored_channel_references(c)
    assert len(jobs)==3 and {j['seed'] for j in jobs}=={42,52,62}
    assert all(j['reference_role']=='stored_prior_loop_qkv_channel_not_fresh_primary_control' for j in jobs)
    assert not any(j['arm'] in runtime.ARMS for j in jobs)
    assert len(pins)==5


@pytest.mark.parametrize('change',['receipt','endpoint','held_out','split'])
def test_stored_reference_tamper_or_mixed_result_rejected(tmp_path,change):
    parent,report,receipt,c=stored_reference_fixture(tmp_path,held_out=change=='held_out',split='test' if change=='split' else 'validation')
    receipt_sha=prep.sha(receipt)
    if change=='receipt':receipt.write_text(receipt.read_text()+' ')
    if change=='endpoint':
        endpoint=next((parent/'retrieval/originals').rglob('endpoint_replays.json'))
        endpoint.write_text(endpoint.read_text()+' ')
    with patch.object(prep,'ROOT',tmp_path),patch.object(prep,'PARENT',parent),patch.object(prep,'PARENT_REPORT',report),patch.object(prep,'PARENT_RECEIPT_SHA',receipt_sha),pytest.raises(ValueError):
        prep.stored_channel_references(c)


def test_native_qualification_never_runs_existing_cnn_inference():
    import inspect
    code=inspect.getsource(runtime.native_qualification)
    assert 'baseline_reference_initial_state_sha' in code
    assert 'build_model(data,runner.CNN)' not in code
    assert 'selected.pt' not in code and 'target_outputs' not in code
    assert "result['active_added_parameters']==1024" in inspect.getsource(runtime.synthetic_arm)


@pytest.mark.parametrize('change',['parameters','location'])
def test_wrong_native_location_or_parameter_budget_cannot_claim_terminal(tmp_path,change):
    c,put=terminal_fixture(tmp_path)
    q=runtime.read(tmp_path/'qualification/receipt.json')
    result=q['synthetic'][runtime.ARMS[0]]
    if change=='parameters':result['active_added_parameters']=3072
    else:result['nonlinear_location']='v'
    put('qualification/receipt.json',q)
    p=runtime.read(tmp_path/'training_permit.json');p['qualification']=q
    p['qualification_sha256']=runtime.sha_file(tmp_path/'qualification/receipt.json')
    put('training_permit.json',p)
    with pytest.raises(ValueError,match='Candidate synthetic qualification missing'):
        runtime.verify_training_permit(c,tmp_path)
    assert run_observer(tmp_path)['all_training_terminal_verified'] is False
