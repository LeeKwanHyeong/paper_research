"""Fabricated-only raw-time input treatment; no research rows, GPUs, SSH or Test."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
import torch.nn.functional as F
from paper.scripts import run_titantpp_raw_time_mlp27 as w


def test_raw_time_feature_only_no_scale_inverse_transform_or_quantity_change():
    dts=torch.tensor([[0.,1.,2.,64.,-1.]],dtype=torch.float64)
    q=torch.tensor([[0.,1.,4.,65.,3.]],dtype=torch.float64)
    mask=torch.tensor([[False,True,True,True,False]])
    before=(dts.clone(),q.clone(),mask.clone(),torch.get_rng_state().clone())
    result=w.RawTimeHistoryMLP16.continuous_features(dts,q,mask)
    reference=w.CountAwareTitanHistoryWidth.continuous_features(dts,q,mask)
    assert result.dtype==torch.float32
    assert torch.equal(result[...,0],dts.float().clamp_min(0)*mask)
    assert torch.equal(result[...,1],reference[...,1])
    assert result[0,3,0].item()==64. and result[0,3,1].item()==torch.log1p(torch.tensor(65.)).item()
    assert torch.equal(dts,before[0]) and torch.equal(q,before[1]) and torch.equal(mask,before[2])
    assert torch.equal(torch.get_rng_state(),before[3])


@pytest.mark.parametrize('seed',(42,52,62))
def test_constructor_rng_and_every_parameter_buffer_identical_to_parent(seed):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    w.inherited.install_hooks()
    torch.manual_seed(seed);reference,old_meta=w.sealed_parent.build_model(w.fabricated_data(),w.ARMS[0]);old_rng=torch.get_rng_state().clone()
    torch.manual_seed(seed);model,metadata=w.build_model(w.fabricated_data(),w.ARMS[0]);new_rng=torch.get_rng_state().clone()
    assert type(model) is w.RawTimeHistoryMLP16 and type(reference) is w.CountAwareTitanHistoryWidth
    assert canonical_state_dict_sha256(model.state_dict())==canonical_state_dict_sha256(reference.state_dict())
    assert torch.equal(old_rng,new_rng)
    assert [(n,tuple(p.shape),p.requires_grad) for n,p in model.named_parameters()]==[(n,tuple(p.shape),p.requires_grad) for n,p in reference.named_parameters()]
    assert list(model.named_buffers())[0][0]==list(reference.named_buffers())[0][0]
    assert {k:v for k,v in metadata.items() if k not in w.feature_labels()}==old_meta
    assert model.time_head_contract()==reference.time_head_contract()


@pytest.mark.parametrize('mode',w.LOSS_MODES)
def test_exact_three_losses_time_graph_excluded(mode):
    z=torch.nn.Parameter(torch.tensor([-.5,.2,1.1],dtype=torch.float64));t=torch.nn.Parameter(torch.tensor([.1,.2,.3],dtype=torch.float64))
    prediction=torch.expm1(F.softplus(z));truth=torch.tensor([1.,5.,3.],dtype=torch.float64)
    log_loss=(F.softplus(z)-torch.log1p(truth)).square()
    value={'joint_loss':t.square()+log_loss,'time_loss':t.square(),'quantity_train_loss':log_loss,'log_qty_loss':log_loss,'pred_qty':prediction,'true_qty':truth}
    with w.runtime_hooks(mode):out=w.make_weighted_target_outputs(lambda *a,**k:value,0.)(None,lambda_log_qty=1.)
    expected=log_loss if mode=='log_mse' else ((prediction-truth).square() if mode=='raw_mse' else (prediction-truth).abs())
    assert out['joint_loss'] is out['quantity_train_loss'] and torch.equal(out['joint_loss'],expected)
    if mode=='log_mse':assert out['joint_loss'] is log_loss
    assert torch.autograd.grad(out['joint_loss'].mean(),t,allow_unused=True,retain_graph=True)[0] is None
    actual=torch.autograd.grad(out['joint_loss'].mean(),z,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected.mean(),z)[0]
    assert torch.equal(actual,expected_grad)
    assert out['time_loss'] is value['time_loss'] and out['true_qty'] is truth


@pytest.mark.parametrize('mode',w.LOSS_MODES)
def test_real_raw_time_mlp_two_step_gradient_adamw_and_frozen_time_head(mode):
    result=w.quantity_oracle(mode,'cpu')
    assert result['loss_formula_and_adamw_oracle_exact'] and result['shared_nonzero_gradient_observed']
    proof=w.verify_time_head_proof(result['time_head_no_training_verification'])
    assert proof['optimizer_steps_checked']==2 and proof['all_time_head_gradients_none'] and proof['optimizer_state_absent']


def test_target_gap_quantity_blocking_and_duration_head_target_unchanged():
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    torch.manual_seed(42);model,_=w.build_model(w.fabricated_data(),w.ARMS[0]);model.eval()
    dts=torch.tensor([[0.,2.,64.,3.],[0.,4.,8.,5.]]);q=torch.tensor([[1.,2.,6.,10.],[1.,3.,5.,2.]])
    mask=torch.ones_like(dts,dtype=torch.bool);saved=(dts.clone(),q.clone(),mask.clone());seen=[]
    before=canonical_state_dict_sha256(model.state_dict());original=model.log_observation_dt
    def spy(hidden,target):seen.append(target.detach().clone());return original(hidden,target)
    model.log_observation_dt=spy
    with torch.no_grad():
        result=target_outputs(model,dts,mask,q,lambda_log_qty=1.)
        changed=dts.clone();changed[:,-1]=torch.tensor([19.,29.]);changed_q=q.clone();changed_q[:,-1]+=100
        alt=target_outputs(model,changed,mask,changed_q,lambda_log_qty=1.)
        single=target_outputs(model,dts[:1],mask[:1],q[:1],lambda_log_qty=1.)
    assert torch.equal(seen[0],dts[:,-1]) and torch.equal(seen[1],changed[:,-1])
    assert torch.equal(result['pred_qty'],alt['pred_qty'])
    torch.testing.assert_close(result['pred_qty'][:1],single['pred_qty'],rtol=1e-5,atol=1e-5)
    assert torch.equal(result['true_qty'],q[:,-1]) and not torch.equal(result['time_loss'],alt['time_loss'])
    assert all(torch.equal(a,b) for a,b in zip(saved,(dts,q,mask)))
    assert canonical_state_dict_sha256(model.state_dict())==before


def bound_payload(mode):
    identity=w.weight_identity(0.,mode);objective=w.loss_identity(mode);labels=w.feature_labels()
    interface={'time_weight_identity':identity,'quantity_loss_identity':objective,'loss_mode':mode,**labels}
    return {'time_alpha':0.,'quantity_only':True,'time_head_trained':False,'time_weight_identity':identity,
        'quantity_loss_identity':objective,'loss_mode':mode,'quantity_loss_normalization':'none',**labels,
        'encoder_config':deepcopy(labels),'interface_meta':interface,
        'resume_identity':{'interface_meta':deepcopy(interface),'time_weight_identity':identity,'quantity_loss_identity':objective,
            **labels,'arguments':{'time_alpha':0.,'loss_mode':mode,**labels}}}


@pytest.mark.parametrize('mutation',('top','encoder','interface','resume','arguments','old_log_time'))
def test_feature_binding_tamper_or_old_log_time_checkpoint_rejected(mutation):
    p=bound_payload('log_mse');w.validate_weight_binding(p,0.)
    if mutation=='top':p['time_input_transform']='log1p'
    if mutation=='encoder':p['encoder_config'].pop('input_feature_identity')
    if mutation=='interface':p['interface_meta']['quantity_input_transform']='raw'
    if mutation=='resume':p['resume_identity'].pop('input_feature_identity')
    if mutation=='arguments':p['resume_identity']['arguments']['time_input_transform']='log1p'
    if mutation=='old_log_time':
        for k in w.feature_labels():p.pop(k)
    with pytest.raises(ValueError):w.validate_weight_binding(p,0.)


def test_scoped_factory_hooks_restore_every_shared_alias_on_failure():
    from paper.scripts.count_aware_tpp_backbone import core,training
    from paper.scripts import run_multilag_detail_execution as base,run_titantpp_history_width as width
    w.inherited.install_hooks()
    original=(width.CountAwareTitanHistoryWidth,core.target_outputs,training.target_outputs,training.build_model,
        training.build_optimizer,training._resume_identity,base.time_interface,w.sealed_parent.SCHEMA,w.sealed_parent.LOSS_MODES,w.sealed_parent.runtime_hooks)
    source=w.sha_file(w._PARENT_PATH)
    with pytest.raises(RuntimeError):
        with w.runtime_hooks('raw_mae'):
            assert width.CountAwareTitanHistoryWidth is w.RawTimeHistoryMLP16
            raise RuntimeError('fabricated failure')
    assert original==(width.CountAwareTitanHistoryWidth,core.target_outputs,training.target_outputs,training.build_model,
        training.build_optimizer,training._resume_identity,base.time_interface,w.sealed_parent.SCHEMA,w.sealed_parent.LOSS_MODES,w.sealed_parent.runtime_hooks)
    assert w._FEATURE_SCOPE_DEPTH==0 and w._parent._ACTIVE_LOSS_MODE is None and w.sha_file(w._PARENT_PATH)==source
    with pytest.raises(ValueError):w.run_stage({},'5080','unused','test')


def test_synthetic_three_loss_admission_and_target_independence():
    out=w.synthetic_check(device='cpu')
    assert out['loss_modes']==list(w.LOSS_MODES) and not out['real_data_loaded'] and not out['held_out_test_evaluated']
    assert set(out['cuda_memory_peaks'])==set(w.ARMS)
    init=out['initialization_oracle'];assert init['status']=='passed' and not init['real_data_loaded']
    assert [x['seed'] for x in init['seeds']]==list(w.SEEDS)
    assert all(x['parent_initial_state_sha256']==x['raw_time_initial_state_sha256'] for x in init['seeds'])
    for mode in w.LOSS_MODES:
        assert out['per_loss'][mode]['quantity_oracle']['loss_formula_and_adamw_oracle_exact']
        assert out['per_loss'][mode]['time_input_transform']==w.TIME_INPUT_TRANSFORM


def test_fabricated_three_loss_train_checkpoint_selection_and_validation(tmp_path):
    from paper.tests.test_titantpp_quantity_loss_worker import fabricated_contract
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    configure_runtime('cpu',4);c,data=fabricated_contract(tmp_path);c['schema']=w.CONTRACT_SCHEMA
    c['architecture'].update(time_input_transform=w.TIME_INPUT_TRANSFORM,quantity_input_transform=w.QUANTITY_INPUT_TRANSFORM)
    c['jobs']=[{'id':'fabricated_'+mode,'host':'5090','dataset':'raf_spare_parts','seed':42,'arm':w.ARMS[0],'time_alpha':0.,'loss_mode':mode} for mode in w.LOSS_MODES]
    original_args=w.training_args;original_eval=w.evaluate_selected
    def tiny(*args,**kwargs):
        result=original_args(*args,**kwargs)
        assert (result.epochs,result.min_epochs,result.early_stopping_patience)==(300,40,40)
        result.epochs=result.min_epochs=result.early_stopping_patience=2;result.device='cpu';return result
    def cpu_eval(*args,**kwargs):return original_eval(*args,**kwargs,device='cpu')
    with patch('paper.scripts.quantity_comparison_runtime.configure_runtime',side_effect=lambda *a:configure_runtime('cpu',4)), \
        patch.object(w,'training_args',side_effect=tiny),patch.object(w,'evaluate_selected',side_effect=cpu_eval):
        for job in c['jobs']:
            train=w.run_stage(c,'5090',job['id'],'train')
            assert train['summary']['completed_epochs']==2
            w.validate_weight_binding(train['summary'],0.,job['loss_mode'])
            cp=torch_load_checkpoint(Path(train['selected_checkpoint']),map_location='cpu')
            w.validate_weight_binding(cp,0.,job['loss_mode'])
            assert cp['encoder_config']['time_input_transform']==w.TIME_INPUT_TRANSFORM
            history=w.read(next(Path(train['selected_checkpoint']).parent.glob('history.json')))['history']
            assert cp['best_epoch']==min(history,key=lambda row:row['val_qty_rmse'])['epoch']
            validation=w.run_stage(c,'5090',job['id'],'validation')
            assert validation['validation_replay_passed'] and validation['loss_mode']==job['loss_mode']
            assert validation['time_input_transform']==w.TIME_INPUT_TRANSFORM and not validation['held_out_test_evaluated']
            assert validation['target_identity']==train['target_identity'] and validation['state_sha256']==train['state_sha256']
            assert validation['time_head_no_training_verification']['optimizer_state_absent']
            with pytest.raises(ValueError):w.run_stage(c,'5090',job['id'],'train')
