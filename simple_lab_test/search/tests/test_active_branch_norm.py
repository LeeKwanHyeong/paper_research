from copy import deepcopy
import io
import json
from pathlib import Path
import pytest
import torch
from models.TPPs.CountAwareTitanActiveBranchNorm import ARM,ROLE
from models.TPPs.CountAwareFactory import validate_checkpoint_route
from paper.scripts import run_active_branch_norm as r
from paper.scripts.verify_active_branch_norm import check
from simple_lab_test.search.tests.test_multilag_detail_execution import admitted_contract,cpu_training


def test_formula(cpu_training):
    assert all(v is True for k,v in check().items() if k!='device')


def test_real_shape_initialization(cpu_training):
    c=json.loads(Path('search_artifacts/titantpp_core_ablation_20260928_v1/frozen_execution/execution_contract.json').read_text())
    for d in c['datasets']:
        for seed in (42,52,62):
            r.initial_states(d,seed)
            torch.manual_seed(seed);base,_=r.build_model(d,'titantpp_history_mlp')
            torch.manual_seed(seed);new,_=r.build_model(d,ARM)
            assert sum(p.numel() for p in base.parameters())==sum(p.numel() for p in new.parameters())
            dt=torch.ones(2,8);q=torch.ones(2,8);mask=torch.ones(2,8,dtype=torch.bool)
            from paper.scripts.count_aware_tpp_backbone.core import target_outputs
            rng=torch.get_rng_state()
            a=target_outputs(base,dt,mask,q,lambda_log_qty=1.)
            torch.set_rng_state(rng)
            b=target_outputs(new,dt,mask,q,lambda_log_qty=1.)
            for k in a:
                if isinstance(a[k],torch.Tensor):assert torch.equal(a[k],b[k]),k
            with pytest.raises(ValueError):new.load_state_dict(base.state_dict())
            stream=io.BytesIO();torch.save(new.state_dict(),stream);stream.seek(0)
            new.load_state_dict(torch.load(stream,weights_only=True))


@pytest.mark.parametrize('dataset_id',['yellow_trip_hourly','intermittent_frozen_5000','insta_market_basket'])
def test_train_and_endpoint_contract(cpu_training,admitted_contract,tmp_path,dataset_id):
    c=deepcopy(admitted_contract);c.update(model_role=ROLE,design_sha256='synthetic',architecture={})
    d=next(x for x in c['datasets'] if x['dataset_id']==dataset_id)
    d['steps_per_epoch']=(d['inherited_data_identity']['populations']['train']['target_count']+127)//128
    frame,_=r.base.prepare_admitted_data(d);interface=r.time_interface(d,frame,c)
    initial=r.initial_states(d,42)
    args=r.base.training_args(c,d,tmp_path/'train');args.device='cpu';args.epochs=args.min_epochs=args.early_stopping_patience=3
    args.execution_role='synthetic_norm_contract_test'
    quantity={'boundaries':d['quantity_boundaries_all_train_rows'],'strata':[{'label':f'bin_{i}'} for i in range(5)]}
    from paper.scripts.count_aware_tpp_backbone import training
    with r.shared.audited_training(training,d,lambda:None,lambda *_:None) as exposure:
        summary,_,_=training.train_one(args=args,frame=frame,quantity_contract=quantity,interface_meta=interface,backbone=ARM,quantity_variant=r.VARIANT,seed=42)
    assert summary['initial_state_sha256']==initial[ARM]
    run=args.output_dir/'runs'/ARM/r.VARIANT/'seed_42'
    history=r.read(run/'history.json')['history']
    r.prior.audit_arm(history,summary,exposure,d,{'minimum_epochs':3,'maximum_epochs':3,'patience':3})
    ident=training._resume_identity(args=args,backbone=ARM,quantity_variant=r.VARIANT,seed=42,monitor=args.checkpoint_monitor,quantity_contract=quantity,interface_meta=interface)
    for name,epoch in [('best_val_qty_rmse_model.pt',summary['best_epoch']),('last_epoch_state.pt',3)]:
        result=r.replay_checkpoint(run/name,d,frame,lambda:None,device='cpu',expected_arm=ARM,expected_identity=ident,expected_initial=initial[ARM],expected_epoch=epoch)
        for k,h in [('qty_rmse','val_qty_rmse'),('qty_mae','val_qty_mae'),('time_nll','val_time_nll')]:
            assert result[k]==pytest.approx(history[epoch-1][h],abs=1e-8)
        payload=torch.load(run/name,weights_only=False);validate_checkpoint_route(payload,ARM)
        with pytest.raises(ValueError):validate_checkpoint_route(payload,'titantpp_history_mlp')
        payload['model_state_dict']['active_branch_norm_identity'][0]^=1
        with pytest.raises(ValueError):validate_checkpoint_route(payload,ARM)
