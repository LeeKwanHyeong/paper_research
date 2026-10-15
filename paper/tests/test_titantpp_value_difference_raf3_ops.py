"""Synthetic protocol tests; no research rows, checkpoints or remote calls."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import pytest
from paper.scripts import titantpp_value_difference_raf3_runtime as runtime
from paper.scripts import advance_titantpp_value_difference_raf3 as followup
from paper.scripts import prepare_titantpp_value_difference_raf3 as prep
from paper.scripts import deploy_titantpp_value_difference_raf3 as deploy


def contract():
    remote='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'+runtime.SCHEMA+'_5090'
    return {'schema':runtime.SCHEMA,'jobs':runtime.expected_jobs(),'arms':list(runtime.ARMS),
        'training':deepcopy(runtime.TRAINING),'parallelism':deepcopy(runtime.PARALLELISM),
        'evaluation_scope':'validation_only','held_out_test_evaluated':False,'test_evaluation':False,
        'limits':{'total_wall_seconds':168*3600,'per_condition_seconds':36*3600,'automatic_retry':False,
            'inherited_absolute_host_deadline_unix':runtime.INHERITED_DEADLINE,
            'inherited_start_permit_sha256':runtime.INHERITED_START_SHA},
        'hosts':{'5090':{'root':remote,'source_root':remote+'/source','operation_root':remote+'/operation',
            'gpu_uuid':'GPU-c9adc246-ef66-7906-bc07-6e152fa9952f',
            'python':'/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12'}},
        'source':{'files':{},'files_sha256':runtime.sha_json({})},
        'comparison_runtime_limitation':'new candidate arms on5090, historicalCNN5080',
        'primary_contrast':'ValueDifference−CNN'}


def test_three_unique_candidate_jobs_same_runtime_one_wave():
    c=contract();runtime.validate(c,verify_files=False)
    assert len(c['jobs'])==len({j['id'] for j in c['jobs']})==3
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
    assert 'scientific_worker_count\')==3' in followup.REMOTE_CODE
    assert 'supervisor_wrapper_own_os_exit_directly_observed' in followup.REMOTE_CODE
    assert 'titantpp_backbone_value_difference_raf3_20261016_v1' in followup.REMOTE


def analysis_fixture(tmp_path):
    c=contract();endpoint={'count':6690,'qty_rmse':4.,'qty_mae':2.,'time_nll':1.,
        'tail':{'qty_rmse':8.,'qty_mae':5.,'time_nll':3.}}
    c['baseline_jobs']=[{'seed':seed,'original':{'selected_epoch':2,'validation_reference':endpoint}} for seed in runtime.SEEDS]
    for j in c['jobs']:
        folder=tmp_path/'originals/run'/j['id']
        prefix=folder/'runs'/j['arm']/'count_only_log_regression'/f"seed_{j['seed']}"
        followup.write(prefix/'endpoint_replays.json',{'best_epoch':2,'selected':endpoint})
        followup.write(prefix/'history.json',{'history':[{'epoch':1,'val_qty_rmse':5.},
            {'epoch':2,'val_qty_rmse':4.},{'epoch':3,'val_qty_rmse':4.}]})
        followup.write(folder/'terminal_manifest.json',{'job':j,'held_out_test_evaluated':False,
            'status':'complete','scientific_success':True,'evaluation_scope':'validation_only',
            'files':{str(p.relative_to(folder)):followup.sha(p) for p in prefix.iterdir()}})
    report=tmp_path/'report';report.mkdir()
    return c,report,tmp_path/'originals'


def test_three_selected_models_reuse_historical_baseline(tmp_path):
    c,report,originals=analysis_fixture(tmp_path)
    with patch.object(followup,'REPORT',report):result=followup.analyze(c,originals)
    assert result['new_fit_count']==3 and len(result['rows'])==6 and len(result['seed_differences'])==3
    assert set(result['aggregate'])=={'CNN','ValueDifference'}
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
    """Complete fabricated three-fit launch/permit/exit chain, no tensor/GPU data."""
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
    approval=put('approval.json',{'approved':True,'hosts':['5090'],'fresh_fit_count':3,
        'maximum_concurrent_workers':3,'contract_sha256':digest})
    start=put('start_permit.json',{'host':'5090','started_at_unix':runtime.INHERITED_DEADLINE-50000.,'deadline_unix':runtime.INHERITED_DEADLINE,
        'contract_sha256':digest,'approval_sha256':runtime.sha_json(approval)})
    guard=lambda tail:['timeout','--signal=TERM','--kill-after=15s','300']+base+tail
    ql=launch('qualification/launch.json',guard(['--mode','native']),pipeline,phase='native')
    q=child('qualification/receipt.json',ql,base+['--mode','native'],host='5090',status='passed',
        source_closure=c['source']['files_sha256'],held_out_test_evaluated=False,scientific_optimizer_updates=0,
        synthetic={a:{'arm':a,'status':'passed','contract_sha256':digest,'active_trainable_parameter_count':79363,'registered_parameter_count':104095,'active_local_cnn_parameters':512,
            'active_value_difference_parameters':128,'original_value_kernel_absent':True,'value_operation':'adjacent_observed_differences'} for a in runtime.ARMS})
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
        name=j['id'];deadline=runtime.INHERITED_DEADLINE-1000.
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
    put('train_complete.json',{'status':'complete','contract_sha256':digest,'scientific_worker_count':3,'dispatcher_actual_exit':0})
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


def test_fully_bound_terminal_three_workers_and_permit_pass(tmp_path):
    c,_=terminal_fixture(tmp_path)
    assert runtime.verify_training_permit(c,tmp_path)
    r=run_observer(tmp_path)
    assert r['all_training_terminal_verified'] is True
    assert len(r['jobs'])==3 and all(x['state']=='complete' for x in r['jobs'])
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
        'worker_group_pid':launch['pid'],'worker_group_start_ticks':launch['start_ticks'],'expires_unix':runtime.INHERITED_DEADLINE-2000.})
    states={v['pid']:{**v,'state':'live'} for v in (wi,supervisor,pipeline)}
    states[launch['pid']]={**launch,'state':'live','ppid':supervisor['pid']}
    def probe(pid):return states.get(pid,{'pid':pid,'state':'absent'})
    gpu=c['hosts']['5090']['gpu_uuid']+', '+str(wi['pid'])+'\n'
    with patch('time.time',return_value=runtime.INHERITED_DEADLINE-40000.):
        assert run_observer(tmp_path,probe,gpu)['jobs'][0]['state']=='running'
        states[wi['pid']]['start_ticks']+=1
        assert run_observer(tmp_path,probe,gpu)['jobs'][0]['state']=='unconfirmed'


def test_candidate_only_admission_and_initial_state_contract():
    assert len(runtime.ADMISSION_JOBS)==3 and len({x[0] for x in runtime.ADMISSION_JOBS})==3
    assert all(a in runtime.ARMS and a!='titantpp_cnn' for _,a,_ in runtime.ADMISSION_JOBS)
    import inspect
    assert 'runner.initial_states(data,seed,arms=ARMS)' in inspect.getsource(runtime.bootstrap)
    assert 'runner.initial_states(data,s,arms=ARMS)' in inspect.getsource(runtime.native_qualification)


def test_scope_has_only_difference_candidate_not_prior_arms_or_cnn():
    assert runtime.ARMS==('titantpp_cnn_value_difference_v1',)
    assert prep.ARMS==runtime.ARMS and runtime.SCHEMA==prep.SCHEMA
    assert runtime.ADMISSION_JOBS==tuple(('difference'+str(s),runtime.ARMS[0],s) for s in (42,52,62))
    assert 'titantpp_cnn_query_nonlinear_v1' not in followup.REMOTE_CODE
    assert 'titantpp_cnn_value_nonlinear_v1' not in followup.REMOTE_CODE


def parent_fixture(root, *, held_out=False):
    parent=root/'parent';parent.mkdir()
    c={'pinned':'synthetic'};prep.write(parent/'execution_contract.json',c)
    start=parent/'retrieval/originals/start_permit.json'
    prep.write(start,{'host':'5090','contract_sha256':prep.canonical(c),'started_at_unix':1.,'deadline_unix':1.+168*3600})
    original=parent/'retrieval/originals/synthetic.txt';original.write_text('not research data')
    receipt=parent/'retrieval/final_receipt.json'
    prep.write(receipt,{'status':'complete','new_fit_count':6,'held_out_test_evaluated':held_out,
        'all_training_terminal_verified':True,'files':{str(p.relative_to(root)):prep.sha(p) for p in (start,original)}})
    return parent,receipt,start,original


def test_parent_training_deadline_is_bound_to_unchanged_original_receipt(tmp_path):
    parent,receipt,start,_=parent_fixture(tmp_path)
    with patch.object(prep,'ROOT',tmp_path),patch.object(prep,'PARENT',parent),patch.object(prep,'PARENT_RECEIPT_SHA',prep.sha(receipt)),patch.object(prep,'PARENT_START_SHA',prep.sha(start)):
        pins,value=prep.parent_training_pins()
    assert len(pins)==3 and value['deadline_unix']==1.+168*3600


@pytest.mark.parametrize('change',['receipt','original','held_out','unbound_deadline'])
def test_parent_tamper_or_mixed_result_rejected(tmp_path,change):
    parent,receipt,start,original=parent_fixture(tmp_path,held_out=change=='held_out')
    receipt_sha,start_sha=prep.sha(receipt),prep.sha(start)
    if change=='receipt':receipt.write_text(receipt.read_text()+' ')
    elif change=='original':original.write_text('tampered')
    elif change=='unbound_deadline':start_sha='0'*64
    with patch.object(prep,'ROOT',tmp_path),patch.object(prep,'PARENT',parent),patch.object(prep,'PARENT_RECEIPT_SHA',receipt_sha),patch.object(prep,'PARENT_START_SHA',start_sha),pytest.raises(ValueError):
        prep.parent_training_pins()


def test_native_qualification_never_runs_existing_cnn_inference():
    import inspect
    code=inspect.getsource(runtime.native_qualification)
    assert 'baseline_reference_initial_state_sha' in code
    assert 'build_model(data,runner.CNN)' not in code
    assert 'selected.pt' not in code and 'target_outputs' not in code
    assert "result['active_local_cnn_parameters']==512" in inspect.getsource(runtime.synthetic_arm)


@pytest.mark.parametrize('change',['parameters','difference_parameters','old_v','operation','active_count','registered_count'])
def test_wrong_native_difference_contract_cannot_claim_terminal(tmp_path,change):
    c,put=terminal_fixture(tmp_path)
    q=runtime.read(tmp_path/'qualification/receipt.json');result=q['synthetic'][runtime.ARMS[0]]
    if change=='parameters':result['active_local_cnn_parameters']=576
    elif change=='difference_parameters':result['active_value_difference_parameters']=192
    elif change=='old_v':result['original_value_kernel_absent']=False
    elif change=='active_count':result['active_trainable_parameter_count']+=64
    elif change=='registered_count':result['registered_parameter_count']+=64
    else:result['value_operation']='free_three_tap'
    put('qualification/receipt.json',q)
    permit=runtime.read(tmp_path/'training_permit.json');permit['qualification']=q
    permit['qualification_sha256']=runtime.sha_file(tmp_path/'qualification/receipt.json');put('training_permit.json',permit)
    with pytest.raises(ValueError,match='Candidate synthetic qualification missing'):
        runtime.verify_training_permit(c,tmp_path)
    assert run_observer(tmp_path)['all_training_terminal_verified'] is False


def test_new_start_uses_remaining_prior_lease_without_extension():
    c=contract();now=runtime.INHERITED_DEADLINE-40000.
    start=runtime.new_start_permit(c,'5090',{'approved':True},now)
    assert start['deadline_unix']==runtime.INHERITED_DEADLINE
    assert start['deadline_unix']-now==40000. < 168*3600
    assert runtime.valid_start_deadline(c,start)
    start['deadline_unix']=now+168*3600
    assert not runtime.valid_start_deadline(c,start)
    with pytest.raises(ValueError,match='Expired'):
        runtime.new_start_permit(c,'5090',{},runtime.INHERITED_DEADLINE)


@pytest.mark.parametrize('delta',[-1.,1.,168*3600])
def test_altered_start_deadline_cannot_claim_terminal(tmp_path,delta):
    c,put=terminal_fixture(tmp_path);start=runtime.read(tmp_path/'start_permit.json')
    start['deadline_unix']+=delta;put('start_permit.json',start)
    permit=runtime.read(tmp_path/'training_permit.json');permit['start_permit_sha256']=runtime.sha_json(start);put('training_permit.json',permit)
    with pytest.raises(ValueError,match='Start approval'):
        runtime.verify_training_permit(c,tmp_path)
    assert not run_observer(tmp_path)['all_training_terminal_verified']


def test_nonempty_pinned_gpu_cannot_trigger_collection(tmp_path):
    c,_=terminal_fixture(tmp_path)
    result=run_observer(tmp_path,gpu=c['hosts']['5090']['gpu_uuid']+', 123456789\n')
    assert not result['all_training_terminal_verified'] and 'archive_sha256' not in result


@pytest.mark.parametrize('change',['tamper','missing_selected','missing_exit','live_owner'])
def test_terminal_collection_requires_all_binary_exit_and_absence_proofs(tmp_path,change):
    c,put=terminal_fixture(tmp_path);name=c['jobs'][0]['id']
    if change=='tamper':
        target=next((tmp_path/'run'/name).rglob('last_epoch_state.pt'));target.write_bytes(b'changed')
        with pytest.raises(AssertionError,match='Terminal SHA'):run_observer(tmp_path)
        return
    if change=='missing_selected':
        path='run/'+name+'/terminal_manifest.json';m=runtime.read(tmp_path/path)
        m['files']={n:d for n,d in m['files'].items() if not n.endswith('best_val_qty_rmse_model.pt')};put(path,m)
    elif change=='missing_exit':(tmp_path/'worker_exits'/(name+'.json')).unlink()
    if change=='live_owner':result=run_observer(tmp_path,lambda pid:{'pid':pid,'state':'unconfirmed','reason':'query_error'})
    else:result=run_observer(tmp_path)
    assert not result['all_training_terminal_verified'] and 'archive_sha256' not in result


def test_preparation_spec_tamper_stops_before_remote_or_bundle_write(tmp_path):
    report=tmp_path/'report';report.mkdir();(report/'SCIENCE_SPEC.json').write_text('{}')
    with patch.object(prep,'REPORT',report),patch.object(prep.subprocess,'check_output') as call,pytest.raises(ValueError,match='Scientific specification'):
        prep.prepare()
    call.assert_not_called()


@pytest.mark.parametrize('change',['duplicate_seed','count','nonfinite','candidate_scope','nonfirst_tie','terminal_scope'])
def test_bad_validation_comparison_rejected_without_outputs(tmp_path,change):
    c,report,originals=analysis_fixture(tmp_path)
    if change=='duplicate_seed':c['baseline_jobs'][1]['seed']=42
    elif change=='count':c['baseline_jobs'][0]['original']['validation_reference']['count']=1
    elif change=='nonfinite':c['baseline_jobs'][0]['original']['validation_reference']['qty_rmse']=float('nan')
    elif change=='candidate_scope':c['jobs'][0]['arm']='titantpp_cnn'
    else:
        folder=originals/'run'/c['jobs'][0]['id'];mp=folder/'terminal_manifest.json';manifest=followup.read(mp)
        if change=='terminal_scope':manifest['evaluation_scope']='mixed'
        else:
            endpoint=next(folder.rglob('endpoint_replays.json'));v=followup.read(endpoint);v['best_epoch']=3
            endpoint.write_text(json.dumps(v));manifest['files'][str(endpoint.relative_to(folder))]=followup.sha(endpoint)
        mp.write_text(json.dumps(manifest))
    with patch.object(followup,'REPORT',report),pytest.raises(ValueError):followup.analyze(c,originals)
    assert not list(report.iterdir())


def test_unfavorable_seed_is_retained_and_zero_reference_ratio_is_na(tmp_path):
    c,report,originals=analysis_fixture(tmp_path)
    # Each seed's baseline is copied separately; no score selection or exclusion.
    for j in c['baseline_jobs']:j['original']['validation_reference']=deepcopy(j['original']['validation_reference'])
    c['baseline_jobs'][0]['original']['validation_reference']['qty_mae']=0.
    c['baseline_jobs'][1]['original']['validation_reference']['qty_mae']=1.
    c['baseline_jobs'][2]['original']['validation_reference']['qty_mae']=4.
    with patch.object(followup,'REPORT',report):result=followup.analyze(c,originals)
    diff={r['seed']:r for r in result['seed_differences']}
    assert diff[42]['MAE']=={'absolute':2.,'relative_percent':None}
    assert diff[52]['MAE']['absolute']==1. and diff[62]['MAE']['absolute']==-2.
    assert result['aggregate']['CNN']['MAE']['sample_sd']==pytest.approx(2.0816659994661326)
    assert 'N/A' in (report/'THREE_SEED_VALIDATION.md').read_text()
