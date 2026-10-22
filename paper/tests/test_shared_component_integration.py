"""Independent CPU integration through the unchanged generic scientific engine."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from models.TPPs.CountAwareComponentAblation import ARMS, build_model
from paper.scripts import run_shared_component_ablation as worker
from paper.scripts import run_titantpp_shared_history_cap as fixture
from paper.tests.test_shared_component_runner import contract as contract_fixture


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('dataset,max_len', [('raf_spare_parts',84),('yellow_trip_hourly',256),('intermittent_frozen_5000',256)])
@pytest.mark.parametrize('seed',[42,52,62])
def test_all_four_arms_common_initialization_real_factories(dataset,max_len,seed):
    data=fixture.fabricated_data(); data['dataset_id']=dataset; data['loader']['max_seq_len']=max_len
    states={}; rng={}; counts={}
    for arm in ARMS:
        torch.manual_seed(seed); model,_=build_model(data,arm)
        states[arm]={k:v.clone() for k,v in model.state_dict().items() if 'identity' not in k}
        rng[arm]=torch.get_rng_state().clone(); counts[arm]=sum(p.numel() for p in model.parameters())
        assert len(model.encoder.layers)==2
        persistent=[tuple(p.shape) for name,p in model.named_parameters() if name.endswith('attn.persistent_mem')]
        assert persistent==[(1,16,64),(1,16,64)]
    common=set.intersection(*(set(s) for s in states.values()))
    assert common and all(torch.equal(states['full'][k],s[k]) for s in states.values() for k in common)
    assert all(torch.equal(rng['full'],r) for r in rng.values())
    assert counts['full']-counts['no_prototype']==4096
    assert counts['full']-counts['no_history']==9600
    assert counts['full']-counts['encoder_only']==13696


@pytest.mark.parametrize('arm',ARMS)
@pytest.mark.parametrize('epochs',[1,2])
def test_actual_engine_selected_not_final_replay_and_proof(tmp_path,arm,epochs):
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256,torch_load_checkpoint
    args,frame,data,interface,quantity=fixture._engine_fixture(tmp_path,'titantpp_history_mlp_shared15_v1')
    args.epochs=args.min_epochs=args.early_stopping_patience=epochs
    args.batch_size=2; args.model_role='observed_time_joint_v1'; args.time_alpha=1.; args.max_val_batches=None
    interface.update(time_alpha=1.,time_head_trained=True,time_loss_graph_excluded=False)
    data.update(dataset_id='raf_spare_parts',quantity_boundaries_all_train_rows=quantity['boundaries'])
    data['loader'].update(batch_size=2,lookback_weeks=84)
    data['model'].update(lambda_log_qty=1.,lambda_tail=0.,time_head_lr_multiplier=1.)
    loader=make_loader(frame,target_split='validation',shuffle=False,generator=None,**data['loader'])
    _,population=worker.population_identity(loader.dataset,'validation')
    assert population['target_count']>=6
    data['inherited_data_identity']={'populations':{'train':{},'validation':population}}
    c=contract_fixture(tmp_path)
    c['datasets'][0].update(deepcopy(data)); c['datasets'][0]['loader'].update(batch_size=128,num_workers=0,drop_last=False,train_shuffle=True,validation_shuffle=False)
    job=next(j for j in c['jobs'] if j['dataset']==data['dataset_id'] and j['arm']==arm and j['seed']==42)
    interface['source_files_sha256']=c['source']['files_sha256']
    original=training.evaluate; n=0
    def score(**kwargs):
        nonlocal n
        result=original(**kwargs); n+=1
        if epochs==2 and n==2: result={**result,'qty_rmse':result['qty_rmse']+1000.}
        return result
    with worker._engine_hooks(data,arm) as tracked,mock.patch.object(training,'evaluate',side_effect=score):
        summary,_,_=training.train_one(args=args,frame=frame,quantity_contract=quantity,interface_meta=interface,backbone=arm,quantity_variant=worker.VARIANT,seed=42)
    path=Path(summary['checkpoint_path']); payload=torch_load_checkpoint(path,map_location='cpu')
    assert summary['completed_epochs']==epochs and payload['best_epoch']==1
    history=json.loads((path.parent/'history.json').read_text())['history']
    assert fixture.select_best_epoch(history)==1
    proof=tracked['tracker'].proof(payload)
    worker.raw9.verify_time_head_proof(proof,payload['model_state_dict'])
    if epochs==2:
        last=torch_load_checkpoint(path.parent/'last_epoch_state.pt',map_location='cpu')
        assert last['epoch']==2 and last['model_state_sha256']!=payload['model_state_sha256']
    receipt={'host':'5080','selected_checkpoint':str(path.relative_to(tmp_path)),'checkpoint_sha256':worker.sha_file(path),'state_sha256':canonical_state_dict_sha256(payload['model_state_dict']),'selected_epoch':1,'training_origin':'fresh_fit','time_head_training_verification':tracked['tracker'].proof()}
    folder=tmp_path/'run'/job['id']; folder.mkdir(parents=True)
    (folder/'train_receipt.json').write_text(json.dumps(receipt))
    a=worker.evaluate_selected(c,data,job,receipt,tmp_path,frame,'validation',tmp_path/'ref.gz',device='cpu')
    b=worker.evaluate_selected(c,data,job,receipt,tmp_path,frame,'validation',tmp_path/'replay.gz',device='cpu')
    assert gzip.open(tmp_path/'ref.gz','rb').read()==gzip.open(tmp_path/'replay.gz','rb').read()
    assert a['metrics']==b['metrics'] and b['prediction_rows']==population['target_count'] and b['parameters_unchanged']
    for field in ['qty_rmse','qty_mae','time_nll']:
        assert b['metrics']['overall'][field]==pytest.approx(summary['best_val_'+field],abs=1e-5)
    with torch.no_grad():
        next(p for name,p in tracked['tracker'].model.named_parameters() if 'time' in name).add_(.125)
    after=tracked['tracker'].proof(payload); worker.raw9.verify_time_head_proof(after,payload['model_state_dict'])
    assert after['checkpoint_time_head_sha256']==proof['checkpoint_time_head_sha256']
    assert after['final_time_head_sha256']!=proof['final_time_head_sha256']
    assert worker.sha_file(path)==receipt['checkpoint_sha256']
    missing=deepcopy(payload); missing.pop('time_head_training_verification')
    invalid_path=path.parent/'missing_proof.pt'; torch.save(missing,invalid_path)
    invalid_receipt={**receipt,'selected_checkpoint':str(invalid_path.relative_to(tmp_path)),'checkpoint_sha256':worker.sha_file(invalid_path)}
    with pytest.raises(ValueError,match='[Pp]roof|verification'):
        worker._load_model(c,data,job,invalid_receipt,tmp_path,'cpu')
    corrupt=deepcopy(receipt); corrupt['checkpoint_sha256']='0'*64
    (folder/'train_receipt.json').write_text(json.dumps(corrupt))
    with pytest.raises(ValueError,match='checkpoint bytes changed'):
        worker.load_selected(c,'5080',job['id'])


@pytest.mark.parametrize('arm',ARMS)
def test_first_training_stage_can_create_receipt_without_preexisting_receipt(tmp_path,arm):
    from paper.scripts import manage_shared_component_ablation as manager
    from paper.scripts import run_multilag_detail_execution as base
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    args,frame,data,interface,quantity=fixture._engine_fixture(tmp_path,'titantpp_history_mlp_shared15_v1')
    args.epochs=args.min_epochs=args.early_stopping_patience=1
    args.batch_size=2; args.model_role='observed_time_joint_v1'; args.time_alpha=1.; args.max_val_batches=None
    interface.update(time_alpha=1.,time_head_trained=True,time_loss_graph_excluded=False)
    data.update(dataset_id='raf_spare_parts',quantity_boundaries_all_train_rows=quantity['boundaries'])
    data['loader'].update(batch_size=128,lookback_weeks=84,num_workers=0,drop_last=False,train_shuffle=True,validation_shuffle=False)
    data['model'].update(lambda_log_qty=1.,lambda_tail=0.,time_head_lr_multiplier=1.)
    data['optimizer']={'name':'AdamW','lr':.001,'weight_decay':.01,'grad_clip':1.}
    loader=make_loader(frame,target_split='validation',shuffle=False,generator=None,**{k:data['loader'][k] for k in ['batch_size','lookback_weeks','max_seq_len']})
    _,population=worker.population_identity(loader.dataset,'validation')
    data['inherited_data_identity']={'populations':{'train':{},'validation':population}}
    c=contract_fixture(tmp_path); c['datasets'][0]=data
    job=next(j for j in c['jobs'] if j['dataset']==data['dataset_id'] and j['arm']==arm and j['seed']==42)
    interface['source_files_sha256']=c['source']['files_sha256']
    folder=tmp_path/'run'/job['id']; folder.mkdir(parents=True)
    args.output_dir=folder
    assert not (folder/'train_receipt.json').exists()
    with mock.patch.object(manager,'verify_native_qualification',return_value={}),mock.patch.object(base,'prepare_admitted_data',return_value=(frame,{'synthetic':True})),mock.patch.object(worker,'_training_args',return_value=args),mock.patch.object(worker,'_training_interface',return_value=interface):
        receipt=worker._stage_train(c,'5080',job,data,tmp_path,folder,'cpu')
    assert receipt['status']=='complete' and receipt['new_training_epochs']==1
    assert receipt['validation_reference']['prediction_rows']==population['target_count']
    assert not (folder/'train_receipt.json').exists(), 'Only run_stage publishes final immutable receipt'
    (folder/'train_receipt.json').write_text(json.dumps(receipt))
    from paper.scripts import profile_shared_component_ablation as profiler
    model,selected,checkpoint,profile_data=profiler._load_selected(c,'5080',job['id'])
    assert profile_data==data and selected['selected_epoch']==receipt['selected_epoch']
    assert checkpoint['binary_sha256']==receipt['checkpoint_sha256']
    import polars as pl
    full=pl.concat([frame,frame.filter(pl.col('chronological_split')=='validation').with_columns((pl.col('seq')+2).alias('seq'),pl.lit('test').alias('chronological_split'))]).sort(['oper_part_no','seq'])
    full_path=tmp_path/'fabricated_full.parquet'; full.write_parquet(full_path)
    test_loader=make_loader(full,target_split='test',shuffle=False,generator=None,**{k:data['loader'][k] for k in ['batch_size','lookback_weeks','max_seq_len']})
    _,test_population=worker.population_identity(test_loader.dataset,'test')
    data['full_data_identity']={'data':{'path':str(full_path),'sha256':worker.sha_file(full_path)},'populations':{'test':test_population}}
    with mock.patch.object(worker,'_check_source'),mock.patch.object(manager,'verify_global_gate',side_effect=RuntimeError('incomplete global Validation')),mock.patch.object(worker,'_evaluation_frame') as load_frame:
        with pytest.raises(RuntimeError,match='incomplete global Validation'):
            worker.run_stage(c,'5080',job['id'],'test',device='cpu')
        load_frame.assert_not_called()
    with mock.patch.object(worker,'_check_source'),mock.patch.object(manager,'verify_global_gate',return_value={}),mock.patch.object(base,'prepare_admitted_data',return_value=(frame,{'synthetic':True})):
        result=worker.run_stage(c,'5080',job['id'],'test',device='cpu')
    assert result['prediction_rows']==test_population['target_count']==8
    assert result['held_out_test_evaluated'] is True and result['state_sha256']==receipt['state_sha256']
    rows=[json.loads(row) for row in gzip.open(folder/'test_predictions.jsonl.gz','rt')]
    assert {row['split'] for row in rows}=={'test'} and {row['target_seq'] for row in rows}=={11,12}
    assert worker.sha_file(full_path)==data['full_data_identity']['data']['sha256']
    from paper.tests.test_shared_component_profile import Hooks,batch
    c['campaign']='shared_component_ablation_20261022_v1'
    c['cost']={'batch_size':128,'preparation_seed':42,'warmup_batches':5,'measured_batches':32,'repetitions':3,'modes':['train','eval'],'fresh_process_per_rep':True,'same_fit_host':True,'maximum_parallel_workers':1,'automatic_retry':False}
    batches=[batch() for _ in range(37)]; manifest_rows=[]; targets=[]
    for index,value in enumerate(batches):
        ids=[['fabricated',index*128+i] for i in range(128)]; targets.extend(ids)
        manifest_rows.append({'batch':index,'phase':'warmup' if index<5 else 'measured','target_split':'train','targets':ids,'tensor_sha256':profiler.tensor_sha(value)})
    cached=tmp_path/'cost'/'batches'/f"{data['dataset_id']}.pt"; cached.parent.mkdir(parents=True); torch.save(batches,cached)
    manifest={'dataset':data['dataset_id'],'split':'train','synthetic':False,'preparation_seed':42,'file_sha256':profiler.sha_file(cached),'batches':manifest_rows,'target_ids_sha256':profiler.canonical_sha(targets),'contract_sha256':profiler.canonical_sha(c),'contract_file_sha256':'fake-test-contract'}
    cached.with_suffix('.json').write_text(json.dumps(manifest))
    # Only CUDA transport/clock/identity are replaced; actual selected loading,
    # model forward, protocol checks and JSON serialization are exercised.
    with mock.patch.object(profiler,'_runtime',return_value={'device':'cpu_fake_cuda_boundary'}),mock.patch.object(profiler,'_guard'),mock.patch.object(profiler,'CudaHooks',return_value=Hooks()),mock.patch.object(torch.nn.Module,'to',lambda self,*a,**kw:self):
        measured=profiler.profile(c,'5080',job['id'],'eval',1,contract_file_sha256='fake-test-contract')
    saved=json.loads((tmp_path/'cost'/'results'/job['id']/'eval_rep1.json').read_text())
    assert saved['status']=='complete' and saved['checkpoint_binary_unchanged']
    assert saved['selected']['checkpoint_sha256']==receipt['checkpoint_sha256']
    assert len(saved['rows'])==37 and saved['optimizer_steps']==0
    assert worker.sha_file(tmp_path/receipt['selected_checkpoint'])==receipt['checkpoint_sha256']


def test_profile_rejects_nonfinite_parameter_created_by_last_optimizer_step():
    from paper.scripts import profile_shared_component_ablation as p
    from paper.tests.test_shared_component_profile import Tiny,Hooks,batch,output
    model=Tiny(); optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    original=optimizer.step; steps=0
    def corrupt_last(*args,**kwargs):
        nonlocal steps
        result=original(*args,**kwargs); steps+=1
        if steps==2:
            with torch.no_grad(): model.weight.fill_(float('nan'))
        return result
    optimizer.step=corrupt_last
    with pytest.raises(ValueError,match='[Nn]onfinite|finite'):
        p.measure_batches(model,[batch(2)]*2,'train',optimizer,Hooks(),output,grad_clip=1.,warmup=1,measured=1)


def _qualification_evidence(c,host,root):
    from paper.scripts import manage_shared_component_ablation as m
    jobs=[j for j in c['jobs'] if j['host']==host]
    unique={(j['dataset'],j['arm']):j for j in jobs}
    exits=[]; capacities={}
    pairs=[(j,'synthetic') for j in unique.values()]+[(next(j for j in jobs if j['arm']==a),'capacity') for a in ARMS]
    for job,stage in pairs:
        entry={'job_id':job['id'],'stage':stage,'host':host,'contract_sha256':m.canonical(c),'actual_os_returncode':0,'timeout':False,'supervisor_error':None,'resource_incident':None}
        m.write(root/'control'/f"{job['id']}.{stage}.exit.json",entry); exits.append(entry)
    for arm in ARMS:
        receipt={'status':'qualified','host':host,'arm':arm,'contract_sha256':m.canonical(c),'source_files_sha256':c['source']['files_sha256'],'synthetic':True,'real_data_loaded':False,'cost_complete':False,'checkpoint_saved':False,
                 'rows':[{'mode':mode,'batch_size':128,'padded_length':256,'optimizer_steps':int(mode=='train'),'elapsed_seconds':.1,'memory_baseline_bytes':{'allocated':1,'reserved':2},'memory_peak_bytes':{'allocated':2,'reserved':3}} for mode in ['eval','train']]}
        path=f'cost/qualification/{arm}.json'; m.write(root/path,receipt); capacities[arm]={'path':path,'sha256':m.sha(root/path)}
    value={'status':'complete','host':host,'contract_sha256':m.canonical(c),'actual_child_exits':exits,'capacity_receipts':capacities}
    m.write(root/'qualification.complete.json',value)
    return value


@pytest.mark.parametrize('fault',['none','empty_rows','small_batch','missing_train_mode','nonfinite_time','invalid_memory','original_exit_mutated','foreign_host'])
def test_native_gate_requires_real_full_shape_measurement_evidence(tmp_path,fault):
    from paper.scripts import manage_shared_component_ablation as m
    c=contract_fixture(tmp_path); value=_qualification_evidence(c,'5080',tmp_path)
    if fault=='original_exit_mutated':
        entry=value['actual_child_exits'][0]; entry['actual_os_returncode']=1
    elif fault=='foreign_host':
        value['actual_child_exits'][0]['host']='5090'
    elif fault!='none':
        item=value['capacity_receipts']['full']; receipt=m.read(tmp_path/item['path'])
        if fault=='empty_rows': receipt['rows']=[]
        if fault=='small_batch': receipt['rows'][0]['batch_size']=2
        if fault=='missing_train_mode': receipt['rows'][1]['mode']='eval'
        if fault=='nonfinite_time': receipt['rows'][0]['elapsed_seconds']=float('inf')
        if fault=='invalid_memory': receipt['rows'][0]['memory_peak_bytes']['allocated']=0
        (tmp_path/item['path']).write_text(json.dumps(receipt)); item['sha256']=m.sha(tmp_path/item['path'])
    (tmp_path/'qualification.complete.json').write_text(json.dumps(value))
    if fault=='none': assert m.verify_native_qualification(c,'5080',tmp_path)['status']=='complete'
    else:
        with pytest.raises((ValueError,RuntimeError)): m.verify_native_qualification(c,'5080',tmp_path)


def test_global_gate_rejects_actual_exit_from_wrong_gpu_host(tmp_path):
    from paper.scripts import manage_shared_component_ablation as m
    from paper.tests.test_manage_shared_component_ablation import make_gate
    c=contract_fixture(tmp_path); gate=make_gate(c,tmp_path)
    evidence=next(iter(gate['jobs'].values())); item=evidence['train_exit']
    value=m.read(tmp_path/item['path']); value['host']='5090' if evidence['host']=='5080' else '5080'
    (tmp_path/item['path']).write_text(json.dumps(value)); item['sha256']=m.sha(tmp_path/item['path'])
    (tmp_path/'global_validation_gate.json').write_text(json.dumps(gate))
    with pytest.raises(RuntimeError): m.verify_global_gate(c,tmp_path)


def test_actual_time_interface_uses_component_contract_and_neutral_design_envelope(tmp_path):
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from paper.scripts.run_count_aware_tpp_backbone_control import derive_train_time_contract
    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import target_dt_sha256
    args,frame,data,interface,quantity=fixture._engine_fixture(tmp_path,'titantpp_history_mlp_shared15_v1')
    data['dataset_id']='raf_spare_parts'; data['loader'].update(batch_size=128,lookback_weeks=84)
    temporal=derive_train_time_contract(frame,lookback_weeks=84,max_seq_len=84)
    stats={'train_time_scale':temporal['time_scale'],'train_log_scaled_mean':temporal['target_log_scaled_mean'],'train_log_scaled_std':temporal['target_log_scaled_std']}
    for split in ['train','validation']:
        loader=make_loader(frame,target_split=split,shuffle=False,generator=None,**data['loader'])
        ds=loader.dataset; targets=torch.tensor([max(1.,float(ds.dt_lists[p][i+1])) for p,i in ds.index],dtype=torch.float64)
        stats['expected_'+split+'_targets']=len(targets)
        stats['expected_'+split+'_target_dt_sha256']=target_dt_sha256(targets)
    data['time_statistics']=stats
    c=contract_fixture(tmp_path); c['science_spec_sha256']='a'*64; c['science']={'retained':{'hidden_dim':64}}
    c['architecture']={'schema':'shared_component_architecture_v1','arms':{a:{} for a in ARMS},'retained':{'hidden_dim':64}}
    c['design_sha256']=worker._canonical(c['architecture'])
    actual=worker._training_interface(c,data,frame,'full')
    assert actual['time_alpha']==1. and actual['time_head_trained']
    assert actual['source_files_sha256']==c['source']['files_sha256']
    assert actual['backbone_design']['arm']=='full'
    assert actual['backbone_design']['schema']==worker.SCHEMA
    assert 'mode_by_arm' not in actual['backbone_design']


def test_reuse_reads_source_legacy_and_preserves_checkpoint_bytes(tmp_path):
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts import manage_shared_component_ablation as manager
    from paper.scripts import run_multilag_detail_execution as base
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import torch_load_checkpoint,canonical_state_dict_sha256
    import shutil
    original=tmp_path/'source'/'legacy'; original.mkdir(parents=True)
    args,frame,data,interface,quantity=fixture._engine_fixture(original,'titantpp_history_mlp_shared15_v1')
    args.epochs=args.min_epochs=args.early_stopping_patience=1; args.batch_size=2; args.time_alpha=1.
    interface.update(worker.raw9.input_identity())
    with worker.raw9._engine_hooks('titantpp_history_mlp_shared15_v1'):
        summary,_,_=training.train_one(args=args,frame=frame,quantity_contract=quantity,interface_meta=interface,backbone='titantpp_history_mlp_shared15_v1',quantity_variant=worker.VARIANT,seed=52)
    checkpoint=Path(summary['checkpoint_path']); payload=torch_load_checkpoint(checkpoint,map_location='cpu')
    worker.raw9.validate_checkpoint(payload,'titantpp_history_mlp_shared15_v1')
    data.update(dataset_id='raf_spare_parts',quantity_boundaries_all_train_rows=quantity['boundaries'])
    data['loader'].update(batch_size=128,lookback_weeks=84,num_workers=0,drop_last=False,train_shuffle=True,validation_shuffle=False)
    data['model'].update(lambda_log_qty=1.,lambda_tail=0.,time_head_lr_multiplier=1.)
    data['optimizer']={'name':'AdamW','lr':.001,'weight_decay':.01,'grad_clip':1.}
    loader=make_loader(frame,target_split='validation',shuffle=False,generator=None,**{k:data['loader'][k] for k in ['batch_size','lookback_weeks','max_seq_len']})
    _,population=worker.population_identity(loader.dataset,'validation')
    data['inherited_data_identity']={'populations':{'train':{},'validation':population}}
    c=contract_fixture(tmp_path); c['datasets'][0]=data; c['hosts']['5090']['source_root']=str(tmp_path/'source')
    job=next(j for j in c['jobs'] if j['dataset']==data['dataset_id'] and j['arm']=='full' and j['seed']==52)
    files={'checkpoint':checkpoint,'history':checkpoint.parent/'history.json'}
    for key in ['train_receipt','validation_receipt','old_train_exit','old_validation_exit']:
        path=original/(key+'.json'); path.write_text(json.dumps({'status':'complete','actual_os_returncode':0})); files[key]=path
    job['reuse']={k:{'path':str(v.relative_to(tmp_path/'source')),'sha256':worker.sha_file(v)} for k,v in files.items()}
    job['reuse'].update(selected_epoch=payload['best_epoch'],canonical_tensor_sha256=canonical_state_dict_sha256(payload['model_state_dict']))
    before={str(p):worker.sha_file(p) for p in files.values()}
    folder=tmp_path/'run'/job['id']; folder.mkdir(parents=True)
    with mock.patch.object(manager,'verify_native_qualification',return_value={}),mock.patch.object(base,'prepare_admitted_data',return_value=(frame,{'synthetic':True})):
        receipt=worker._stage_train(c,'5090',job,data,tmp_path,folder,'cpu')
    assert receipt['new_training_epochs']==0 and receipt['training_origin']=='reused_verified_fit'
    copied=tmp_path/receipt['selected_checkpoint']
    assert copied.read_bytes()==checkpoint.read_bytes()
    assert {str(p):worker.sha_file(p) for p in files.values()}==before
    (folder/'train_receipt.json').write_text(json.dumps(receipt))
    from paper.scripts import profile_shared_component_ablation as profiler
    _,_,proof,_=profiler._load_selected(c,'5090',job['id'])
    assert proof['binary_sha256']==before[str(checkpoint)]


def test_native_runtime_configures_before_identity_and_rejects_foreign_gpu(tmp_path):
    from paper.scripts import quantity_comparison_runtime as runtime
    from paper.scripts import manage_shared_component_ablation as manager
    import sys
    c=contract_fixture(tmp_path); spec=c['hosts']['5080']; spec.update(python=sys.executable,environment={},runtime_expected={'threads':4,'interop_threads':1},gpu_uuid='expected-gpu')
    events=[]
    def configure(*args,**kwargs): events.append('configure')
    def identity(*args):
        assert events==['configure']; return {'threads':4,'interop_threads':1,'gpu':{'uuid':'expected-gpu'}}
    with mock.patch.object(manager,'guard'),mock.patch.object(runtime,'configure_runtime',side_effect=configure),mock.patch.object(runtime,'runtime_identity',side_effect=identity):
        assert worker._native_runtime(c,'5080','cuda:0')['gpu']['uuid']=='expected-gpu'
    with mock.patch.object(manager,'guard'),mock.patch.object(runtime,'configure_runtime'),mock.patch.object(runtime,'runtime_identity',return_value={'threads':4,'interop_threads':1,'gpu':{'uuid':'foreign'}}):
        with pytest.raises(ValueError,match='GPU'): worker._native_runtime(c,'5080','cuda:0')


@pytest.mark.parametrize('fault',['none','resumed_predecessor','reused_pid','foreign_gpu'])
def test_guard_preserves_paused_predecessor_and_refuses_other_gpu_owner(tmp_path,fault):
    from paper.scripts import manage_shared_component_ablation as m
    import time
    c=contract_fixture(tmp_path); expected={'pid':7,'start_ticks':1,'pgid':7,'argv':['old','worker'],'cwd':'/old'}
    c['hosts']['5080'].update(absolute_host_deadline_unix=time.time()+60,boot_id='boot',paused_processes=[expected])
    live={**expected,'state':'T'}; pids='7\n'
    if fault=='resumed_predecessor': live['state']='R'
    if fault=='reused_pid': live['start_ticks']=2
    if fault=='foreign_gpu': pids+='99\n'
    with mock.patch.object(Path,'read_text',return_value='boot\n'),mock.patch.object(m,'proc_identity',return_value=live),mock.patch.object(m.subprocess,'run',return_value=mock.Mock(stdout=pids)),mock.patch.object(m.os,'killpg') as kill:
        if fault=='none': assert m.guard(c,'5080')['paused_preserved']==1
        else:
            with pytest.raises(RuntimeError): m.guard(c,'5080')
        kill.assert_not_called()


@pytest.mark.parametrize('fault',['sha','parent'])
def test_full_input_rejects_changed_bytes_or_changed_admitted_parent(tmp_path,fault):
    from paper.scripts import run_multilag_detail_execution as base
    import polars as pl
    _,frame,data,_,_=fixture._engine_fixture(tmp_path,'titantpp_history_mlp_shared15_v1')
    full=pl.concat([frame,frame.filter(pl.col('chronological_split')=='validation').with_columns((pl.col('seq')+2).alias('seq'),pl.lit('test').alias('chronological_split'))]).sort(['oper_part_no','seq'])
    path=tmp_path/'full.parquet'; full.write_parquet(path)
    data['full_data_identity']={'data':{'path':str(path),'sha256':worker.sha_file(path)}}
    if fault=='sha': data['full_data_identity']['data']['sha256']='0'*64
    else:
        full.with_columns((pl.col('demand_qty')+1).alias('demand_qty')).write_parquet(path)
        data['full_data_identity']['data']['sha256']=worker.sha_file(path)
    with mock.patch.object(base,'prepare_admitted_data',return_value=(frame,{})):
        with pytest.raises(ValueError,match='changed'): worker._evaluation_frame(data,'test')
