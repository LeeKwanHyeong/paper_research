"""CPU and metadata checks for the bounded last Instacart arm transfer."""
from copy import deepcopy
from pathlib import Path
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from paper.scripts import run_mixed_quantity_last_arm_transfer as transfer
from paper.scripts import run_mixed_quantity_partition as old
from paper.scripts import quantity_comparison_engine as engine
from paper.scripts import mixed_quantity_objective as primitives
from paper.scripts.quantity_objective_comparison import QuantityCase
from paper.scripts.time_quantity_diagnostic import _group
from simple_lab_test.search.tests.test_mixed_quantity_engine import make_model, statistics, tensors, objectives

ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / 'search_artifacts/mixed_quantity_last_arm_transfer_preparation_v1/inputs'


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def contract():
    previous = old.read_json(ROOT / 'paper/contracts/mixed_quantity_partition_5080_5090_v1.json')
    calibration = old.read_json(INPUTS / 'calibration.json')
    hosts, qualifications = {}, {}
    for role, (old_role, _) in transfer.ROLES.items():
        h = deepcopy(previous['hosts'][old_role])
        base = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/last_arm_test_' + h['server_alias']
        h.update(old_role=old_role, old_wrapper_path=h['wrapper_path'], wrapper_path=base+'/wrapper.py',
                 output_dir=base+'/run', probe_output_dir=base+'/probe', tmux_session='last_'+h['server_alias'],
                 probe_tmux_session='last_probe_'+h['server_alias'])
        hosts[role] = h
        qualifications[role] = old.read_json(INPUTS/'receipt.json') if role.endswith('5080') else previous['parent']['qualification_receipt']
    return {'schema': transfer.SCHEMA, 'status': 'frozen_pending_explicit_approval',
            'original_partition': {'contract':previous, 'sha256':old.sha_json(previous)},
            'wrapper': {'sha256':transfer.sha_file(transfer.__file__)}, 'hosts':hosts,
            'qualifications':qualifications, 'deadline_unix':transfer.DEADLINE,
            'limits':deepcopy(old.LIMITS), 'policy':deepcopy(transfer.POLICY), 'probe_policy':deepcopy(transfer.PROBE_POLICY),
            'calibration':{'receipt':calibration,'file_sha256':old.sha_file(INPUTS/'calibration.json'),
                           'paths':{role:str(INPUTS/'calibration.json') for role in hosts}},
            'anchor':{'file_sha256':old.sha_file(INPUTS/'last_epoch_state.pt'),
                      'state_sha256':'10ec1dec593e5e0244d36afb9065105caf97528aed64ec28c104cbb2beb817a3',
                      'paths':{role:str(INPUTS/'last_epoch_state.pt') for role in hosts}}}


def handoff(contract):
    source = Path(transfer.partition(contract)['hosts']['instacart_5080']['output_dir'])/transfer.DATASET/'mixed_original'
    return {'schema':'mixed_quantity_last_arm_handoff_v1','transfer_contract_sha256':transfer.sha_json(contract),
            'observed_old_process_dead':True,'matched_never_started':True,'original_process_ids':[3855,3880],
            'committed_epoch':80,'source_dir':str(source),
            'persisted_file_hashes':{str(source/name):'a'*64 for name in ('contract.json','last_epoch_state.pt')}}


def probe(contract, role, record=None):
    row = {'outputs':{key:[1.,2.]*256 for key in ('pred_qty','time_loss','log_qty_loss','raw_qty_loss','objective_loss')},
           'gradients':{key:{'parameters':[{'name':key,'shape':[2]}],'values':[1.,2.]} for key in ('encoder','time_head','quantity_head')}}
    return {'schema':'mixed_quantity_cross_runtime_probe_v1','status':'passed','role':role,
            'transfer_contract_sha256':transfer.sha_json(contract),'handoff_sha256':transfer.sha_json(record or handoff(contract)),
            'policy':deepcopy(transfer.PROBE_POLICY),'runtime':contract['hosts'][role]['runtime_expected'],
            'optimizer_steps':0,'calibration_file_sha256':contract['calibration']['file_sha256'],
            'anchor_file_sha256':contract['anchor']['file_sha256'], 'anchor_state_sha256':contract['anchor']['state_sha256'],
            'input_tensors_sha256':'d'*64,'initial_state_sha256':contract['calibration']['receipt']['initial_model_state_sha256'],'indices_sha256':'f'*64,
            'anchors':{anchor:{case:deepcopy(row) for case in old.CASES} for anchor in transfer.PROBE_POLICY['anchors']}}


def test_real_frozen_contract_and_exact_native_identity(contract):
    result=transfer.validate_contract(contract,old,wrapper_path=transfer.__file__,now=transfer.DEADLINE-60)
    assert result['new_training_arms']==result['resumed_arms']==1
    assert transfer.native_identity(contract,'mixed_continuation_5080',old)['execution_contract_sha256']==transfer.OLD_5080_SHA
    assert transfer.training_identity(contract,'mixed_continuation_5080',old)==transfer.native_identity(contract,'mixed_continuation_5080',old)
    identity=transfer.training_identity(contract,'matched_fresh_5090',old)
    assert identity['runtime']==contract['hosts']['matched_fresh_5090']['runtime_expected']
    assert identity['calibration_provenance_identity']==contract['calibration']['receipt']['identity']
    assert identity['cross_runtime_comparison'] is True
    assert identity['transfer_contract_sha256']==transfer.sha_json(contract)


@pytest.mark.parametrize('change',['deadline','budget','policy','tolerance','old_parent','old_runtime','old_qualification','source','oldoutput','calibidentity'])
def test_pinned_research_scope_and_old_identity_cannot_be_weakened(contract,change):
    if change=='deadline': contract['deadline_unix']+=1
    elif change=='budget': contract['limits']['output_bytes_per_host']*=2
    elif change=='policy': contract['policy']['automatic_retry']=True
    elif change=='tolerance': contract['probe_policy']['rtol']=.1
    elif change=='old_parent': contract['original_partition']['contract']['arms'][-1]['host']='continuation_5090'
    elif change=='old_runtime': contract['hosts']['mixed_continuation_5080']['runtime_expected']['numpy']='forged'
    elif change=='old_qualification': contract['qualifications']['mixed_continuation_5080']['execution_contract_sha256']='a'*64
    elif change=='source': contract['hosts']['mixed_continuation_5080']['source_root']='/tmp/changed_source'
    elif change=='oldoutput': contract['hosts']['mixed_continuation_5080']['output_dir']=transfer.partition(contract)['hosts']['instacart_5080']['output_dir']
    else: contract['calibration']['receipt']['identity']['execution_contract_sha256']='a'*64
    with pytest.raises(ValueError): transfer.validate_contract(contract,old)


def test_handoff_rejects_live_old_owner_and_started_last_arm_and_other_source(contract):
    original=handoff(contract)
    transfer.validate_handoff(contract,original)
    for key in ('observed_old_process_dead','matched_never_started'):
        record=deepcopy(original);record[key]=False
        with pytest.raises(ValueError,match='ownership'):transfer.validate_handoff(contract,record)
    record=deepcopy(original);record['source_dir']=record['source_dir'].replace('mixed_original','B_log_original')
    with pytest.raises(ValueError,match='Only original mixed'):transfer.validate_handoff(contract,record)
    record=deepcopy(original);record['persisted_file_hashes']['/tmp/unassigned']='b'*64
    with pytest.raises(ValueError,match='Unassigned'):transfer.validate_handoff(contract,record)
    record=deepcopy(original);record['committed_epoch']=120
    with pytest.raises(ValueError,match='Incomplete'):transfer.validate_handoff(contract,record)


def test_approval_is_transfer_specific_not_old_partition_approval(contract):
    approval={'status':'approved_by_user','scope':transfer.APPROVAL_SCOPE,'user_instruction':'Synthetic user authorization',
              'transfer_contract_sha256':transfer.sha_json(contract)}
    transfer.validate_approval(contract,approval)
    approval['scope']=old.APPROVAL_SCOPE
    with pytest.raises(ValueError,match='approval'):transfer.validate_approval(contract,approval)


def test_small_cross_runtime_differences_pass_large_or_nonfinite_fail(contract):
    a,b=probe(contract,'mixed_continuation_5080'),probe(contract,'matched_fresh_5090')
    b['anchors']['B_final_epoch120']['mixed_original']['outputs']['pred_qty'][0]+=1e-6
    result=transfer.compare_probes(contract,a,b)
    assert result['passed'] and len(result['checks'])==48
    b['anchors']['B_final_epoch120']['mixed_original']['gradients']['encoder']['values'][0]+=.1
    assert transfer.compare_probes(contract,a,b)['passed'] is False
    b['anchors']['initial']['mixed_original']['outputs']['time_loss'][0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):transfer.compare_probes(contract,a,b)


@pytest.mark.parametrize('field',['input_tensors_sha256','initial_state_sha256','indices_sha256','anchor_state_sha256','handoff_sha256'])
def test_cross_runtime_exact_identity_checks(contract,field):
    a,b=probe(contract,'mixed_continuation_5080'),probe(contract,'matched_fresh_5090')
    b[field]='0'*64
    with pytest.raises(ValueError,match='identity mismatch'):transfer.compare_probes(contract,a,b)


def test_5080_resume_survives_cross_host_gate_failure_but_matched_cannot_launch(contract):
    h=handoff(contract)
    a,b=probe(contract,'mixed_continuation_5080',h),probe(contract,'matched_fresh_5090',h)
    compatibility=transfer.compare_probes(contract,a,b)
    transfer.validate_training_gates(contract,'matched_fresh_5090',h,b,compatibility)
    compatibility['passed']=False
    transfer.validate_training_gates(contract,'mixed_continuation_5080',h,a,compatibility)
    with pytest.raises(ValueError,match='cross-host'):transfer.validate_training_gates(contract,'matched_fresh_5090',h,b,compatibility)
    with pytest.raises(ValueError):transfer.validate_training_gates(contract,'mixed_continuation_5080',h,None)
    compatibility=transfer.compare_probes(contract,a,b);compatibility['checks'].pop()
    with pytest.raises(ValueError):transfer.validate_training_gates(contract,'matched_fresh_5090',h,b,compatibility)


def test_legacy_time_head_gradients_are_not_misclassified_as_encoder():
    model=make_model()
    groups={key:[n for n,p in model.named_parameters() if _group(n)==key] for key in ('encoder','time_head','quantity_head')}
    assert all(groups.values())
    assert {'b_t','w_raw','v_t.weight'}<=set(groups['time_head'])
    assert sum(map(len,groups.values()))==len(list(model.parameters()))


def test_calibration_bytes_identity_and_coefficients_preserved(contract,monkeypatch):
    before=json.dumps(contract['calibration'],sort_keys=True)
    def forbidden(*args,**kwargs):pytest.fail('Recalibration is prohibited')
    monkeypatch.setattr(primitives,'calibrate_mixed_objective',forbidden)
    for role in transfer.ROLES:
        calibration=transfer._evidence(contract,role)
        frozen=primitives.objectives_from_calibration(calibration)
        assert frozen[1].alpha==.160021665648455 and frozen[2].quantity_scale==.8340005759765311
    assert json.dumps(contract['calibration'],sort_keys=True)==before


def synthetic_identity():
    return {'source':{'revision':'synthetic'},'data':{'kind':'synthetic'},'runtime':{'device':'cpu'},'execution_contract_sha256':'a'*64}


def run(path,stop,resume=False,identity=None):
    dataset=TensorDataset(*tensors())
    train=DataLoader(dataset,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(41))
    validation=DataLoader(dataset,batch_size=2,generator=torch.Generator().manual_seed(42))
    return engine.run_case(model=make_model(),train_loader=train,validation_loader=validation,case=QuantityCase.B_LOG_ORIGINAL,
                           mixed_objective=objectives()[1],statistics=statistics(),output_dir=path,epochs=120,seed=42,
                           identity=identity or synthetic_identity(),stop_after_epochs=stop,resume=resume)


def test_same_host_copy_repair_resume_preserves_optimizer_rng_and_selectors(tmp_path):
    source,target,full=(tmp_path/name for name in ('source','target','full'))
    run(source,2)
    old.write_json(source/'summary.json',{'stale':True})
    hashes={str(p):old.sha_file(p) for p in source.rglob('*') if p.is_file()}
    old.copy_and_repair_arm({'source_dir':str(source),'mode':'resume'},target,{'persisted_file_hashes':hashes},
                            identity=synthetic_identity(),objective=objectives()[1],engine=engine)
    a,b=run(target,3,True),run(full,3)
    assert a==b
    states=[torch.load(p/'last_epoch_state.pt',weights_only=False) for p in (target,full)]
    for key in ('model_state_sha256','optimizer_state_sha256','rng_state_sha256'):
        assert states[0][key]==states[1][key]
    assert a['selectors']==b['selectors']
    assert {str(p):old.sha_file(p) for p in source.rglob('*') if p.is_file()}==hashes
    with pytest.raises(ValueError,match='identity'):
        run(target,4,True,{**synthetic_identity(),'execution_contract_sha256':'b'*64})


def test_selector_is_strict_earliest_at_exact_tie():
    selectors=engine._selector_template()
    state=make_model().state_dict()
    metrics={'raw_quantity_rmse':1.5,'legacy_time_loss':2.}
    engine.update_selectors(selectors,metrics,epoch=1,global_step=2,state=state)
    first=deepcopy(engine._public_selectors(selectors))
    changed={name:value+1 for name,value in state.items()}
    engine.update_selectors(selectors,metrics,epoch=2,global_step=4,state=changed)
    assert engine._public_selectors(selectors)==first


def test_actual_probe_graph_has_512_outputs_three_gradient_groups_and_zero_optimizer_steps(monkeypatch):
    model=make_model()
    initial={name:value.detach().clone() for name,value in model.state_dict().items()}
    trained={name:value.detach().clone() for name,value in initial.items()}
    trained['quantity_head.weight'].fill_(.01)
    batches=[tuple(t.repeat(32,1) for t in tensors()) for _ in range(4)]
    def forbidden(*args,**kwargs):pytest.fail('Probe instantiated an optimizer')
    monkeypatch.setattr(torch.optim,'AdamW',forbidden)
    result=transfer.probe_anchors(model,batches,statistics(),objectives(),
                                  {'initial':initial,'B_final_epoch120':trained},device='cpu',check=lambda:None)
    for anchor in result.values():
        for row in anchor.values():
            assert all(len(values)==512 for values in row['outputs'].values())
            assert all(group['values'] for group in row['gradients'].values())
    assert all(torch.equal(value,trained[name]) for name,value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
