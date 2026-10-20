"""MAC9: exact optimized construction, joint fits, causal state and all-nine Test gate.

Only fabricated CPU rows are used by these tests; no project Test data is read.
"""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from paper.scripts import run_titans_mac_accuracy9 as worker


def contract_fixture(root):
    from tests.test_titantpp_shared_external_extra42_worker import contract_fixture as old_fixture
    result = old_fixture(root)
    result.update(schema=worker.CONTRACT_SCHEMA)
    result['hosts'] = {'5090': {'root': str(root)}}
    result['jobs'] = [{"id": f"{dataset}__{arm}__seed{seed}", "host": "5090",
        "dataset": dataset, "arm": arm, "seed": seed, "time_input": "log1p",
        "loss_mode": "log_mse", "time_alpha": 1.0}
        for dataset in worker.DATASETS for arm in worker.ARMS for seed in worker.SEEDS]
    return result


@pytest.fixture
def contract(tmp_path):
    return contract_fixture(tmp_path / "mac9")


def test_exact_approved_nine_jobs_and_common_settings(contract, tmp_path):
    assert worker.ARMS == ('titantpp_titans_mac',)
    assert worker.SEEDS == (42, 52, 62) and len(contract['jobs']) == 9
    for job in contract['jobs']:
        selected, data, root, folder = worker.select_job(contract, '5090', job['id'])
        assert selected == job and folder == root / 'run' / job['id']
        with worker.candidate_factory(job['arm']):
            args = worker.training_args(contract, data, tmp_path, job['arm'])
        assert (args.epochs,args.min_epochs,args.early_stopping_patience)==(300,40,40)
        assert (args.batch_size,args.lr,args.weight_decay,args.grad_clip)==(128,.001,.01,1.)
        assert args.time_alpha == 1. and args.loss_mode == 'log_mse'
        assert args.model_role == 'observed_time_joint_v1'
        assert args.titans_memory_gradient_clip == 1.


@pytest.mark.parametrize('kind', ['seed','duplicate','host','old_model','alpha','cap','window','budget','warm','permission'])
def test_changed_contract_rejected_without_reading_input(contract, kind):
    if kind=='seed': contract['jobs'][0]['seed']=72
    elif kind=='duplicate': contract['jobs'][1]=deepcopy(contract['jobs'][0])
    elif kind=='host': contract['jobs'][0]['host']='5080'
    elif kind=='old_model': contract['jobs'][0]['arm']='titantpp'
    elif kind=='alpha': contract['common_condition']['time_alpha']=0
    elif kind=='cap': contract['datasets'][0]['cap_value']=7
    elif kind=='window': next(d for d in contract['datasets'] if d['dataset_id']==contract['jobs'][0]['dataset'])['loader']['max_seq_len']=13
    elif kind=='budget': contract['training']['maximum_epochs']=301
    elif kind=='warm': contract['training']['warm_start']=True
    else: contract['execution_authorization']['test_evaluation']=False
    with patch.object(Path,'read_bytes',side_effect=AssertionError('Unexpected data access')):
        with pytest.raises(ValueError): worker.select_job(contract,'5090',contract['jobs'][0]['id'])


def test_scoped_factory_exact_builder_and_exception_restoration():
    from paper.scripts.count_aware_tpp_backbone import training, observed_time
    old_build, old_backbones = training.build_model, observed_time.BACKBONES
    with pytest.raises(RuntimeError,match='fabricated'):
        with worker.candidate_factory(worker.ARM):
            assert training.build_model is worker.mac_runtime.factory
            assert worker.ARM in observed_time.BACKBONES
            with worker.candidate_factory(worker.ARM):
                assert training.build_model is worker.mac_runtime.factory
            raise RuntimeError('fabricated')
    assert training.build_model is old_build and observed_time.BACKBONES is old_backbones
    with pytest.raises(ValueError):
        with worker.candidate_factory('titantpp'): pass


@pytest.mark.parametrize('seed', worker.SEEDS)
def test_common_heads_match_frozen_shared_and_exact_benchmark_factory(seed):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    data=worker.prepared.fabricated_data()
    torch.manual_seed(seed);a,ma=worker.build_model(data,worker.ARM)
    after=torch.random.get_rng_state()
    torch.manual_seed(seed);b,mb=worker.mac_runtime.build_model(data)
    assert ma==mb and canonical_state_dict_sha256(a.state_dict())==canonical_state_dict_sha256(b.state_dict())
    assert torch.equal(after,torch.random.get_rng_state())
    assert sum(p.numel() for p in a.parameters())==122310
    assert a.titans_mac_encoder.neural_memory.gradient_max_norm==1.
    assert worker.verify_common_head_initialization(data,seed)['shared15_common_head_exact']


@pytest.fixture(scope='module')
def synthetic(): return worker.synthetic_check(device='cpu')


def test_cpu_joint_optimizer_causality_state_and_roundtrip(synthetic):
    assert synthetic['status']=='passed' and all(synthetic['checks'].values())
    assert synthetic['parameter_count']=={worker.ARM:122310}
    assert synthetic['real_data_loaded'] is False and synthetic['held_out_test_evaluated'] is False
    proof=worker.verify_time_head_proof(synthetic['time_head_training_verification'][worker.ARM])
    assert proof['optimizer_steps_checked']==2
    assert proof['all_trainable_parameters_in_optimizer']
    memory=synthetic['online_memory_verification'][worker.ARM]
    assert memory['write_calls']>0 and memory['finite_state_updates']>0 and memory['sequence_local_reset']


def test_online_updates_remain_active_but_no_cross_batch_state_persists():
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    configure_runtime('cpu',4);torch.manual_seed(42)
    model,_=worker.build_model(worker.prepared.fabricated_data(),worker.ARM)
    model.eval().requires_grad_(False)
    dt,qty,mask=worker.prepared.fabricated_batch()
    dt,qty,mask=[x.repeat(1,4) for x in (dt,qty,mask)]
    mask[1,18:]=False
    before=worker.context_identity(model)
    proof=worker.verify_sequence_local_online_memory(model,dt,mask,qty)
    assert proof['finite_state_updates']>0 and worker.context_identity(model)==before
    original=target_outputs(model,dt,mask,qty,lambda_log_qty=1.)
    q,t=qty.clone(),dt.clone();target=mask.sum(1)-1;idx=torch.arange(len(target))
    q[idx,target]+=1234;t[idx,target]+=123;q[~mask]=999;t[~mask]=999
    changed=target_outputs(model,t,mask,q,lambda_log_qty=1.)
    torch.testing.assert_close(original['pred_qty'],changed['pred_qty'],rtol=1e-5,atol=1e-5)
    assert 'write_sequence' not in model.titans_mac_encoder.neural_memory.__dict__


def test_missing_gate_blocks_test_before_native_or_full_input(contract):
    with patch.object(worker,'prepare_evaluation_frame',side_effect=AssertionError('Test accessed')), \
         patch.object(worker,'check_native_runtime',side_effect=AssertionError('CUDA initialized')):
        with pytest.raises(FileNotFoundError):
            worker.run_stage(contract,'5090',contract['jobs'][0]['id'],'test')


def test_all9_gate_requires_actual_integer_exit0(contract):
    root=Path(contract['hosts']['5090']['root']);root.mkdir()
    gate={'status':'passed','host':'5090','contract_sha256':worker.sha_json(contract),'jobs':{}}
    worker.write_exclusive(root/'validation_gate.json',gate)
    with pytest.raises(ValueError,match='All 9'):worker.verify_validation_gate(contract,'5090',root)
    gate['jobs']={job['id']:{} for job in contract['jobs']}
    job=contract['jobs'][0];folder=root/'run'/job['id'];folder.mkdir(parents=True)
    p,e=folder/'validation_receipt.json',folder/'validation_exit.json'
    worker.write_exclusive(p,{})
    worker.write_exclusive(e,{'job_id':job['id'],'stage':'validation','actual_os_returncode':False,
        'timeout':False,'pid':123,'start_ticks':123,'observed_exit_at':'fabricated'})
    gate['jobs'][job['id']]={'validation_receipt_sha256':worker.sha_file(p),'validation_exit_sha256':worker.sha_file(e)}
    (root/'validation_gate.json').write_text(json.dumps(gate))
    with pytest.raises(ValueError,match='actual child exit0'):worker.verify_validation_gate(contract,'5090',root)


@pytest.mark.parametrize('stage',['resume','deploy','other',None])
def test_invalid_stage_rejected_before_any_dependencies(contract,stage):
    with patch.object(worker,'verify_frozen_dependencies',side_effect=AssertionError('Unexpected read')):
        with pytest.raises(ValueError):worker.run_stage(contract,'5090',contract['jobs'][0]['id'],stage)


def test_cli_rejects5080_and_stdout_json(contract,capsys,tmp_path):
    path=tmp_path/'contract.json';path.write_text(json.dumps(contract))
    with patch.object(worker,'read',side_effect=AssertionError('Unexpected read')):
        with pytest.raises(SystemExit):worker.main(('--contract',str(path),'--host','5080','--stage','train'))
    capsys.readouterr()
    def synthetic(*args,**kw):
        print('fabricated progress');return {'status':'passed'}
    with patch.object(worker,'synthetic_check',side_effect=synthetic):
        worker.main(('--contract',str(path),'--host','5090','--stage','synthetic'))
    result=capsys.readouterr();assert json.loads(result.out)=={'status':'passed'}
    assert result.err.strip()=='fabricated progress'


@pytest.mark.parametrize("arm", worker.ARMS)
def test_fabricated_actual_engine_checkpoint_selection_and_native_replay(tmp_path, arm):
    import polars as pl
    from paper.scripts import run_multilag_detail_execution as base
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    root = tmp_path / "execution"
    fixture = root / "fabricated"
    fixture.mkdir(parents=True)
    args, _, fixture_data, interface, _ = worker.prepared._engine_fixture(fixture, worker.prepared.ARMS[0])
    frame = prepare_count_frame(pl.read_parquet(args.data))
    args.model_role = worker.role_for(arm)
    args.time_alpha = 1.0
    args.titans_memory_gradient_clip = 1.0
    for key, value in worker.input_identity().items():
        setattr(args, key, value)
    interface.update(worker.input_identity())
    contract = contract_fixture(root)
    contract["source"]["base_git_revision"] = args.source_revision
    data = next(value for value in contract["datasets"] if value["dataset_id"] == "raf_spare_parts")
    data.update({**fixture_data, "dataset_id": "raf_spare_parts",
                 "quantity_boundaries_all_train_rows": [2., 5., 8., 10.]})
    data["loader"].update(batch_size=128, lookback_weeks=84, max_seq_len=84, num_workers=0,
                          drop_last=False, train_shuffle=True, validation_shuffle=False)
    data["model"].update(lambda_log_qty=1., lambda_tail=0., time_head_lr_multiplier=1.)
    data["inherited_data_identity"] = {
        "data": {"path": str(args.data), "sha256": args.data_sha256},
        "split_manifest": {"path": str(args.split_manifest), "sha256": args.split_manifest_sha256},
        "populations": {}}
    for split in ("train", "validation"):
        loader = make_loader(frame, target_split=split, batch_size=128, lookback_weeks=84,
                             max_seq_len=84, shuffle=False, generator=None)
        data["inherited_data_identity"]["populations"][split] = worker.population_identity(loader.dataset, split)[1]
    # Fabricated held-out rows exercise the frozen Test path without opening
    # project held-out data or weakening the production all-9 gate.
    raw = pl.read_parquet(args.data)
    extra = pl.DataFrame([{"oper_part_no": entity, "seq": seq, "delta_t": 2,
        "demand_qty": quantity, "chronological_split": "test"}
        for entity in raw["oper_part_no"].unique().to_list()
        for seq, quantity in ((11, 13.), (12, 7.))])
    full_frame = prepare_count_frame(pl.concat([raw, extra]).sort(["oper_part_no", "seq"]))
    full_path = fixture / "fabricated_full.parquet"
    full_frame.write_parquet(full_path)
    full_loader = make_loader(full_frame, target_split="test", batch_size=128,
                             lookback_weeks=84, max_seq_len=84, shuffle=False, generator=None)
    data["full_data_identity"] = {"data": {"path": str(full_path), "sha256": worker.sha_file(full_path)},
        "populations": {"test": worker.population_identity(full_loader.dataset, "test")[1]}}
    model, encoder = worker.build_model(data, arm)
    contract["frozen_conditions"] = {"models": {arm: {
        "expected_encoder_config_by_dataset": {"raf_spare_parts": encoder},
        "expected_parameter_count_by_dataset": {"raf_spare_parts": sum(p.numel() for p in model.parameters())}}}}
    job = next(value for value in contract["jobs"] if value["dataset"] == "raf_spare_parts"
               and value["arm"] == arm and value["seed"] == 42)
    args.output_dir = root / "run" / job["id"]
    interface.update(source_files_sha256=contract["source"]["files_sha256"],
                     execution_contract_sha256=worker.sha_json(contract))
    with patch("paper.scripts.quantity_comparison_runtime.configure_runtime",
               side_effect=lambda device, threads: configure_runtime("cpu", threads)), \
         patch.object(worker, "check_native_runtime", return_value={"fabricated_only": True}), \
         patch.object(worker, "training_args", return_value=args), \
         patch.object(worker, "training_interface", return_value=interface), \
         patch.object(base, "prepare_admitted_data", return_value=(frame, {"fabricated_only": True})):
        evaluate = worker.evaluate_selected
        with patch.object(worker, "evaluate_selected", side_effect=lambda *values, **kw:
                          evaluate(*values, **{**kw, "device": "cpu"})):
            trained = worker.run_stage(contract, "5090", job["id"], "train")
            replay = worker.run_stage(contract, "5090", job["id"], "validation")
            with patch.object(worker, "verify_validation_gate", return_value={"fabricated_only": True}) as gate:
                tested = worker.run_stage(contract, "5090", job["id"], "test")
                assert gate.call_count == 2
    assert trained["summary"]["completed_epochs"] == 2
    assert trained["selected_epoch"] == replay["selected_epoch"]
    assert trained["validation_reference"]["metrics"] == replay["metrics"]
    assert replay["prediction_rows"] == 8 and replay["metrics"]["tail"]["count"] == 4
    assert replay["parameters_unchanged"] and replay["context_unchanged"] and replay["native_runtime_checked"]
    assert replay["validation_replay_passed"] and replay["individual_predictions_match"]
    assert not trained["held_out_test_evaluated"] and not replay["held_out_test_evaluated"]
    assert trained["checkpoint_sha256"] == replay["checkpoint_sha256"]
    assert tested["held_out_test_evaluated"] and tested["stage"] == "test"
    assert tested["prediction_rows"] == 8 and tested["metrics"]["tail"]["count"] == 4
    assert tested["selected_epoch"] == trained["selected_epoch"]
    assert tested["checkpoint_sha256"] == trained["checkpoint_sha256"]
    assert tested["state_sha256"] == trained["state_sha256"]
    assert not tested["validation_replay_passed"] and tested["parameters_unchanged"]
    proof = worker.verify_time_head_proof(trained["time_head_training_verification"])
    assert proof["all_trainable_parameters_in_optimizer"]
    worker.verify_train_receipt(contract, "5090", job, root, trained)
    worker.verify_inventory(root, replay["files"])
    for stage in ("train", "validation"):
        with pytest.raises(ValueError):
            worker.run_stage(contract, "5090", job["id"], stage)


def test_frozen_raw9_dependency_bytes_remain_unchanged():
    worker.verify_frozen_dependencies()
