"""Literal raw quantity-loss controls on fabricated tensors and temporary data."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from paper.scripts import run_titantpp_quantity_loss_mlp18 as w


@pytest.mark.parametrize('mode',w.LOSS_MODES)
def test_literal_raw_loss_and_inverse_chain_gradient_without_time_graph(mode):
    z=torch.nn.Parameter(torch.tensor([-1.,.5,1.2],dtype=torch.float64))
    t=torch.nn.Parameter(torch.tensor([.1,.2,.3],dtype=torch.float64))
    prediction=torch.expm1(F.softplus(z))
    truth=torch.tensor([10.,0.,2.],dtype=torch.float64)
    truth[1]=prediction.detach()[1]  # MAE's exact-equality subgradient is zero.
    log_loss=(F.softplus(z)-torch.log1p(truth)).square()
    value={'joint_loss':t.square()+log_loss,'time_loss':t.square(),'quantity_train_loss':log_loss,
        'log_qty_loss':log_loss,'pred_qty':prediction,'true_qty':truth,'sentinel':object()}
    with w.runtime_hooks(mode):
        out=w.make_weighted_target_outputs(lambda *a,**k:value,0.)(None,lambda_log_qty=1.)
    expected=(prediction-truth).square() if mode=='raw_mse' else (prediction-truth).abs()
    assert torch.equal(out['joint_loss'],expected)
    assert out['joint_loss'] is out['quantity_train_loss'] and out['pred_qty'] is prediction
    assert out['log_qty_loss'] is log_loss and out['sentinel'] is value['sentinel']
    gt,gz=torch.autograd.grad(out['joint_loss'].mean(),(t,z),allow_unused=True)
    assert gt is None
    slope=2*(prediction.detach()-truth) if mode=='raw_mse' else (prediction.detach()-truth).sign()
    torch.testing.assert_close(gz,slope*torch.exp(z.detach())/3,rtol=1e-12,atol=1e-12)
    assert w.loss_identity(mode)['quantity_loss_normalization']=='none'
    assert w.loss_identity(mode)['prediction_clipped_to_cap'] is False


def test_detached_raw_prediction_and_foreign_modes_rejected():
    z=torch.tensor([.2],requires_grad=True);log_loss=z.square()
    value={'joint_loss':log_loss,'time_loss':log_loss,'quantity_train_loss':log_loss,'log_qty_loss':log_loss,
        'pred_qty':z.detach(),'true_qty':torch.ones(1)}
    with w.runtime_hooks('raw_mse'),pytest.raises(ValueError,match='detached'):
        w.make_weighted_target_outputs(lambda *a,**k:value,0.)(None,lambda_log_qty=1.)
    for mode in ('log_mse','raw_mse_cap_normalized',None):
        with pytest.raises(ValueError):w.loss_identity(mode)
    with pytest.raises(ValueError):w.run_stage({},'5080','unused','test')


@pytest.mark.parametrize('mode',w.LOSS_MODES)
def test_real_mlp16_two_update_gradient_and_adamw_oracle(mode):
    result=w.quantity_oracle(mode,'cpu')
    assert result['literal_raw_gradient_and_adamw_oracle_exact'] is True
    assert result['shared_nonzero_gradient_observed'] is True
    proof=w.verify_time_head_proof(result['time_head_no_training_verification'])
    assert proof['optimizer_steps_checked']==2 and proof['optimizer_state_absent'] is True


def test_hooks_restore_sealed_workers_and_training_aliases_on_error():
    from paper.scripts.count_aware_tpp_backbone import core,training
    from paper.scripts import run_multilag_detail_execution as base
    w.inherited.install_hooks()
    before=(core.target_outputs,training.target_outputs,training.build_model,training.build_optimizer,
        training._resume_identity,training.train_epoch_with_telemetry,base.time_interface,
        w.sealed_zero.SCHEMA,w.sealed_zero.make_weighted_target_outputs,w.sealed_zero.weight_identity)
    source_sha=w.sha_file(w._SEALED_PATH)
    with pytest.raises(RuntimeError):
        with w.runtime_hooks('raw_mae'):
            with pytest.raises(ValueError):
                with w.runtime_hooks('raw_mse'):pass
            raise RuntimeError('fabricated failure')
    assert before==(core.target_outputs,training.target_outputs,training.build_model,training.build_optimizer,
        training._resume_identity,training.train_epoch_with_telemetry,base.time_interface,
        w.sealed_zero.SCHEMA,w.sealed_zero.make_weighted_target_outputs,w.sealed_zero.weight_identity)
    assert w._ACTIVE_LOSS_MODE is None and w.sha_file(w._SEALED_PATH)==source_sha


def test_fresh_initial_state_rng_and_head_are_unchanged_across_losses():
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    states=[];rngs=[];metadata=[]
    for mode in w.LOSS_MODES:
        with w.runtime_hooks(mode):
            torch.manual_seed(52);model,meta=w.build_model(w.fabricated_data(),w.ARMS[0])
            states.append(canonical_state_dict_sha256(model.state_dict()));rngs.append(torch.get_rng_state().clone());metadata.append(meta)
    torch.manual_seed(52);reference,meta=w.sealed_zero.build_model(w.fabricated_data(),w.ARMS[0])
    assert states==[canonical_state_dict_sha256(reference.state_dict())]*2
    assert torch.equal(rngs[0],rngs[1]) and torch.equal(rngs[0],torch.get_rng_state())
    assert metadata==[meta,meta]


def test_both_loss_synthetic_admission_and_target_blocking():
    result=w.synthetic_check(device='cpu')
    assert result['loss_modes']==list(w.LOSS_MODES) and set(result['cuda_memory_peaks'])==set(w.ARMS)
    assert result['real_data_loaded'] is False and result['held_out_test_evaluated'] is False
    for mode in w.LOSS_MODES:
        assert result['per_loss'][mode]['loss_mode']==mode
        assert result['per_loss'][mode]['quantity_oracle']['literal_raw_gradient_and_adamw_oracle_exact'] is True


def bound_payload(mode):
    identity=w.weight_identity(0.,mode);loss=w.loss_identity(mode)
    interface={'time_weight_identity':identity,'quantity_loss_identity':loss,'loss_mode':mode}
    return {'time_alpha':0.,'quantity_only':True,'time_head_trained':False,'time_weight_identity':identity,
        'quantity_loss_identity':loss,'loss_mode':mode,'quantity_loss_normalization':'none','interface_meta':interface,
        'resume_identity':{'interface_meta':copy.deepcopy(interface),'time_weight_identity':identity,
            'quantity_loss_identity':loss,'arguments':{'time_alpha':0.,'loss_mode':mode}}}


def test_loss_identity_cannot_be_relabelled_or_normalized():
    value=bound_payload('raw_mse');w.validate_weight_binding(value,0.)
    with pytest.raises(ValueError):w.validate_weight_binding({**value,'loss_mode':'raw_mae'},0.,'raw_mse')
    for key,new in (('loss_mode','raw_mae'),('quantity_loss_normalization','cap'),('time_alpha',.1),('time_head_trained',True)):
        with pytest.raises(ValueError):w.validate_weight_binding({**value,key:new},0.)


def fabricated_contract(root):
    import numpy as np
    import polars as pl
    from paper.tests.test_titantpp_time_weight_mlp18 import contract_at
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from paper.scripts.run_count_aware_tpp_backbone_control import derive_train_time_contract
    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import target_dt_sha256
    c=contract_at(root);c['schema']=w.CONTRACT_SCHEMA
    c['source']={'base_git_revision':'fabricated','files_sha256':'fixture'}
    c['design_sha256']='fixture';c['architecture']={'models':list(w.ARMS)}
    c['jobs']=[{'id':'fabricated_'+mode,'host':'5090','dataset':'raf_spare_parts','seed':42,
        'arm':w.ARMS[0],'time_alpha':0.,'loss_mode':mode} for mode in w.LOSS_MODES]
    raw=pl.DataFrame([{'oper_part_no':entity,'seq':i+1,'delta_t':[0,1,2,1,3,1][i],
        'demand_qty':float([1,3,8,10,10,2][i]),'chronological_split':'train' if i<4 else 'validation'}
        for entity in ('fabricated_a','fabricated_b') for i in range(6)])
    path=root/'train_validation.parquet';manifest=root/'manifest.json';raw.write_parquet(path);manifest.write_text('{"fabricated":true}')
    data=c['datasets'][0];data['loader']={'batch_size':128,'lookback_weeks':84,'max_seq_len':84,
        'num_workers':0,'drop_last':False,'train_shuffle':True,'validation_shuffle':False}
    frame=prepare_count_frame(raw);populations={}
    def dataset(split):return make_loader(frame,target_split=split,batch_size=128,lookback_weeks=84,max_seq_len=84,shuffle=False,generator=None).dataset
    for split in ('train','validation'):_,populations[split]=w.population_identity(dataset(split),split)
    data['inherited_data_identity']={'data':{'path':str(path),'sha256':w.sha_file(path)},
        'split_manifest':{'path':str(manifest),'sha256':w.sha_file(manifest)},'populations':populations}
    train_q=raw.filter(pl.col('chronological_split')=='train')['demand_qty'].to_numpy().astype('<f8')
    target_q,_=w.population_identity(dataset('train'),'train')
    data['statistics']={'train_log_mean':float(np.log1p(train_q).mean()),'train_log_std':float(np.log1p(train_q).std()),
        'raw_scale':float(np.sqrt(np.square(target_q).mean())),
        'all_train_quantity_sha256':hashlib.sha256(b'quantity_comparison_all_train_row_quantity_v1\0'+train_q.tobytes()).hexdigest()}
    temporal=derive_train_time_contract(frame,lookback_weeks=84,max_seq_len=84)
    data['model'].update(time_scale=temporal['time_scale'],time_initial_location=temporal['target_log_scaled_mean'],
        time_initial_scale=temporal['target_log_scaled_std'],time_sigma_floor=.001,
        time_observation_contract={'mode':'positive_integer_round_clamp_v1','unit':'month','top_code':None})
    data['time_statistics']={'train_time_scale':temporal['time_scale'],'train_log_scaled_mean':temporal['target_log_scaled_mean'],
        'train_log_scaled_std':temporal['target_log_scaled_std']}
    for split in ('train','validation'):
        ds=dataset(split);values=torch.tensor([max(1.,float(ds.dt_lists[p][e+1])) for p,e in ds.index],dtype=torch.float64)
        data['time_statistics'][f'expected_{split}_targets']=len(ds)
        data['time_statistics'][f'expected_{split}_target_dt_sha256']=target_dt_sha256(values)
    return c,data


def test_two_epoch_training_metadata_selection_and_validation_replay(tmp_path):
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.count_aware_tpp_backbone import training
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    configure_runtime('cpu',4);c,data=fabricated_contract(tmp_path)
    original_args,original_eval=w.training_args,w.evaluate_selected
    def tiny(*args,**kwargs):
        a=original_args(*args,**kwargs)
        assert (a.epochs,a.min_epochs,a.early_stopping_patience)==(300,40,40)
        a.epochs=a.min_epochs=a.early_stopping_patience=2;a.device='cpu';return a
    def cpu_eval(*args,**kwargs):return original_eval(*args,**kwargs,device='cpu')
    with patch('paper.scripts.quantity_comparison_runtime.configure_runtime',side_effect=lambda *a:configure_runtime('cpu',4)), \
        patch.object(w,'training_args',side_effect=tiny),patch.object(w,'evaluate_selected',side_effect=cpu_eval):
        for job in c['jobs']:
            train=w.run_stage(c,'5090',job['id'],'train')
            assert train['summary']['completed_epochs']==2
            w.validate_weight_binding(train['summary'],0.,job['loss_mode'])
            checkpoint=torch_load_checkpoint(Path(train['selected_checkpoint']),map_location='cpu')
            w.validate_weight_binding(checkpoint,0.,job['loss_mode'])
            assert checkpoint['loss_mode']==job['loss_mode']
            validation=w.run_stage(c,'5090',job['id'],'validation')
            assert validation['validation_replay_passed'] is True and validation['loss_mode']==job['loss_mode']
            assert validation['quantity_loss_normalization']=='none' and validation['held_out_test_evaluated'] is False
            proof=w.verify_time_head_proof(validation['time_head_no_training_verification'])
            assert proof['optimizer_steps_checked']==2 and proof['optimizer_state_absent'] is True
            assert train['target_identity']==validation['target_identity']
            wrong={**job,'loss_mode':'raw_mae' if job['loss_mode']=='raw_mse' else 'raw_mse'}
            with pytest.raises(ValueError):w._load_selected(data,wrong,train,tmp_path,'cpu')
            with pytest.raises(ValueError):w.run_stage(c,'5090',job['id'],'train')
