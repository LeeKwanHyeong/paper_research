"""CPU-only guards and native-kernel tests; no real samples or checkpoints."""
from copy import deepcopy
from datetime import datetime
import math
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[2] if Path(__file__).parent.name == 'tests' else Path.cwd()
SOURCE = PROJECT / 'search_artifacts/titantpp_cnn_gru54_3seed_dual_20261005_v1/source'
RUNTIME = Path(__file__).with_name('titantpp_time_head_refit_runtime.py')
if not RUNTIME.exists():
    RUNTIME = Path(__file__).parents[1] / 'scripts/titantpp_time_head_refit_runtime.py'
spec = importlib.util.spec_from_file_location('time_refit_runtime_under_test', RUNTIME)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


@pytest.fixture(scope='module')
def scientific():
    engine, core, loader, runner = r.native(SOURCE)
    from models.TPPs.CountAwareTPP import SharedTimeCountModel
    from models.TPPs.positive_integer_time import positive_integer_log_mass
    from paper.scripts.titantpp_gru_control_checks import synthetic_data
    torch.set_num_threads(1)
    return SimpleNamespace(engine=engine, core=core, digest=runner.canonical_state_dict_sha256,
        shared=SharedTimeCountModel, kernel=positive_integer_log_mass, synthetic_data=synthetic_data)


@pytest.fixture
def toy(scientific):
    class NativeHead(scientific.shared):
        def __init__(self):
            super().__init__(hidden_dim=64, train_log_mean=1.2, train_log_std=.8,
                quantity_variant='count_only_log_regression', time_head_mode='heteroscedastic_lognormal_duration',
                time_scale=1., time_initial_location=0., time_initial_scale=.8,
                time_observation_contract={'mode':'positive_integer_round_clamp_v1','unit':'hour','top_code':None})
            self.register_buffer('frozen_marker', torch.tensor([11.]))
            with torch.no_grad():
                self.v_t.weight.zero_()
        def encode_task_states(self, dts, quantities, mask, memory_write_mask=None):
            observed = mask if memory_write_mask is None else memory_write_mask
            values = (torch.log1p(dts) + torch.log1p(quantities)) * observed
            states = values.unsqueeze(-1).expand(-1,-1,64).contiguous()
            return states, states
    m = NativeHead()
    r.freeze(m)
    return m


def batch(n=6):
    dt = torch.tensor([[0.,0.,1.,2.,3.],[0.,1.,1.,2.,2.]] * (n//2))
    qty = torch.tensor([[0.,0.,4.,7.,12.],[0.,3.,5.,8.,14.]] * (n//2))
    mask = torch.tensor([[False,False,True,True,True],[False,True,True,True,True]] * (n//2))
    return (torch.zeros_like(dt),dt,mask,torch.zeros_like(dt),qty)


def cache(m, scientific, n=6):
    frozen = r.state_sha(m, scientific.digest, frozen_only=True)
    values, first = r.extract_cache(m, scientific.core, [batch(n)], 'cpu', scientific.digest, frozen)
    return values, first, frozen


def test_exact_native_interval_kernel_and_target_hiding(toy, scientific):
    values, first, frozen = cache(toy, scientific)
    original = scientific.core.target_outputs(toy, first[1], first[2], first[4], lambda_log_qty=1.)
    assert torch.equal(values['native_nll'], original['time_loss'])
    assert torch.equal(values['prediction'], original['pred_qty'])
    assert not values['hidden'].requires_grad and not values['hidden'].is_inference()
    with torch.no_grad():
        loc, scale, _ = toy._lognormal_time_terms(values['hidden'], values['dt'])
        expected = -scientific.kernel(values['dt'],loc,scale,time_scale=1.,top_code=None)
    assert torch.equal(values['native_nll'], expected)
    changed = list(first)
    changed[1], changed[4] = changed[1].clone(), changed[4].clone()
    changed[1][:,-1] = 29.
    changed[4][:,-1] = 990.
    hidden = r.extract_batch(toy, scientific.core, changed, 'cpu')
    assert torch.equal(hidden['hidden'], values['hidden'])
    assert torch.equal(hidden['prediction'], values['prediction'])
    assert not torch.equal(hidden['native_nll'], values['native_nll'])
    r.guard_frozen(toy, scientific.digest, frozen)


def test_actual_cnn_gru_no_grad_repeated_forward_and_two_scalar_step(scientific):
    torch.manual_seed(52)
    m, _ = scientific.engine.build_model(scientific.synthetic_data(8), r.ARM)
    r.freeze(m)
    dt = torch.randint(1,8,(4,8)).float()
    qty = torch.randint(1,20,(4,8)).float()
    mask = torch.ones_like(dt,dtype=torch.bool)
    inputs = (torch.zeros_like(dt),dt,mask,torch.zeros_like(dt),qty)
    values = r.extract_batch(m, scientific.core, inputs, 'cpu')
    frozen = r.state_sha(m, scientific.digest, frozen_only=True)
    whole = r.state_sha(m, scientific.digest)
    saved = r.scalars(m)
    opt = torch.optim.Adam([m.b_t,m.w_raw],lr=.001)
    r.cache_nll(m,values,slice(None),'cpu').mean().backward()
    assert all(p.grad is None for name,p in m.named_parameters() if name not in r.PARAMETERS)
    assert all(torch.isfinite(getattr(m,name).grad).all() for name in r.PARAMETERS)
    opt.step()
    assert r.state_sha(m,scientific.digest) != whole
    r.spot_check(m,scientific.core,inputs,'cpu',values,scientific.digest,frozen)
    r.restore_scalars(m,saved)
    assert r.state_sha(m,scientific.digest) == whole
    assert torch.equal(r.extract_batch(m,scientific.core,inputs,'cpu')['native_nll'],values['native_nll'])


@pytest.mark.parametrize('mutation',['train','extra_grad','buffer','memory','weight'])
def test_freeze_rejects_real_mutations(toy, scientific, mutation):
    expected = r.state_sha(toy, scientific.digest, frozen_only=True)
    if mutation == 'train': toy.train()
    elif mutation == 'extra_grad': toy.v_t.weight.requires_grad_(True)
    elif mutation == 'buffer': toy.frozen_marker.add_(1)
    elif mutation == 'memory': toy._ctx_mem = torch.ones(1)
    else:
        with torch.no_grad(): toy.v_t.weight.add_(.1)
    with pytest.raises(ValueError): r.guard_frozen(toy,scientific.digest,expected)


def test_real_cached_train_updates_preserve_every_quantity_and_no_val_gradient(toy,scientific,monkeypatch):
    train, first, frozen = cache(toy,scientific)
    validation = {key:value.clone() for key,value in train.items()}
    baseline = r.cache_metrics(toy,validation,'cpu',threshold=13.)
    original_nll = r.cache_nll
    calls = []
    def checked(model, data, indices, device):
        calls.append((data is train,torch.is_grad_enabled()))
        if torch.is_grad_enabled(): assert data is train
        return original_nll(model,data,indices,device)
    monkeypatch.setattr(r,'cache_nll',checked)
    result = r.fit_cached(toy,train,validation,core=scientific.core,first_batch=first,device='cpu',
        digest=scientific.digest,frozen_sha=frozen,seed=52,threshold=13.,fitting=r.FITTING)
    assert result['selected']['time_nll'] < baseline['time_nll']
    assert 1 <= result['best_epoch'] <= result['completed_epochs'] <= 40
    assert result['optimizer_steps'] > 0
    assert 'epoch_seconds' not in result['history'][0] and 'completed_utc' not in result['history'][0]
    for row in result['history'][1:]:
        assert math.isfinite(row['epoch_seconds']) and row['epoch_seconds'] > 0
        assert datetime.fromisoformat(row['completed_utc']).utcoffset().total_seconds() == 0
    assert any(is_train and grad for is_train,grad in calls)
    assert any(not is_train and not grad for is_train,grad in calls)
    for key in ('qty_rmse','qty_mae','qty_sse'):
        assert result['E0'][key] == result['selected'][key] == result['last'][key]
        assert result['E0']['tail'][key] == result['selected']['tail'][key] == result['last']['tail'][key]
    r.guard_frozen(toy,scientific.digest,frozen)
    assert r.choose_best(result['history'])['epoch'] == result['best_epoch']


def test_worsening_validation_retains_identity_and_stops_after_patience(toy,scientific):
    train, first, frozen = cache(toy,scientific)
    validation = {key:value.clone() for key,value in train.items()}
    validation['dt'].fill_(1.)
    identity = r.scalars(toy)
    result = r.fit_cached(toy,train,validation,core=scientific.core,first_batch=first,device='cpu',
        digest=scientific.digest,frozen_sha=frozen,seed=62,threshold=13.,fitting=r.FITTING)
    assert result['best_epoch'] == 0
    assert result['completed_epochs'] == 10
    assert result['selected'] == result['E0']
    assert all(torch.equal(getattr(toy,k).detach().cpu(),identity[k]) for k in r.PARAMETERS)


def test_strict_earliest_finite_time_selector():
    rows = [{'epoch':i,'validation':{'time_nll':v}} for i,v in enumerate([3.,2.,2.,float('nan'),4.])]
    assert r.choose_best(rows)['epoch'] == 1
    with pytest.raises(ValueError): r.choose_best(rows[3:4])


@pytest.fixture
def contract(tmp_path):
    c = r.read(PROJECT/'search_artifacts/titantpp_time_head_bias_refit_5080_20261006_v1/draft_contract.json')
    c['source']['root'] = str(SOURCE)
    c['data_root']=str(SOURCE)
    c['runtime']['source_root']=str(SOURCE)
    c['root']=str(tmp_path)
    c['limits']['max_owned_root_bytes']=8*1024**3
    c['jobs'] = [deepcopy(c['jobs'][0])]
    j=c['jobs'][0]
    j['output_dir']=str(tmp_path/'run'/'fit')
    checkpoint=tmp_path/'original.pt'
    checkpoint.write_bytes(b'not deserialized during dry-run')
    j['checkpoint']['path']=str(checkpoint)
    j['checkpoint']['sha256']=r.sha(checkpoint)
    job={'id':j['canonical_original_id'],'dataset':j['dataset'],'seed':j['seed'],'arm':r.ARM}
    selected={**j['baseline_metrics'],'state_sha256':j['checkpoint']['state_sha256']}
    endpoint=tmp_path/'endpoint_replays.json'
    r.write(endpoint,{'evaluation_scope':'validation_only','held_out_test_evaluated':False,
        'job':job,'best_epoch':j['checkpoint']['epoch'],'selected':selected})
    j['source_endpoint']={'path':str(endpoint),'sha256':r.sha(endpoint)}
    terminal=tmp_path/'source_terminal.json'
    keys=f'runs/{r.ARM}/count_only_log_regression/seed_{j["seed"]}/'
    r.write(terminal,{'scientific_success':True,'status':'complete','held_out_test_evaluated':False,'job':job,
        'files':{keys+'best_val_qty_rmse_model.pt':j['checkpoint']['sha256'],keys+'endpoint_replays.json':r.sha(endpoint)}})
    j['source_terminal']={'path':str(terminal),'sha256':r.sha(terminal)}
    return c


@pytest.mark.parametrize('change',[ 'parameters','test','fit_validation','device','seed','arm','population','epochs','retry','sourceclosure'])
def test_contract_rejects_scope_and_science_drift(contract,change):
    c=deepcopy(contract)
    if change=='parameters': c['fitting']['parameters'].append('v_t.weight')
    elif change=='test': c['held_out_test_evaluated']=True
    elif change=='fit_validation': c['fit_split']='validation'
    elif change=='device': c['device']='cpu'
    elif change=='seed': c['jobs'][0]['seed']=53
    elif change=='arm': c['jobs'][0]['arm']='titantpp'
    elif change=='population': c['jobs'][0]['baseline_metrics']['count']-=1
    elif change=='epochs': c['fitting']['maximum_epochs']=130
    elif change=='retry': c['limits']['automatic_retry']=True
    else: c['source']['files_sha256']='0'*64
    with pytest.raises(ValueError): r.validate_contract(c,verify_files=False)


def test_original_metadata_job_binding_rejects_wrong_endpoint(contract):
    j=contract['jobs'][0]
    endpoint=r.read(j['source_endpoint']['path'])
    endpoint['job']['seed']=62
    Path(j['source_endpoint']['path']).unlink()
    r.write(j['source_endpoint']['path'],endpoint)
    j['source_endpoint']['sha256']=r.sha(j['source_endpoint']['path'])
    with pytest.raises(ValueError,match='job identity'): r.validate_contract(contract)


def test_safe_source_paths_and_symlink_escape(tmp_path):
    with pytest.raises(ValueError): r.confined(tmp_path,'../outside')
    with pytest.raises(ValueError): r.confined(tmp_path,'/absolute')
    (tmp_path/'link').symlink_to('/etc')
    with pytest.raises(ValueError): r.confined(tmp_path,'link/passwd')


def test_cli_dry_run_checks_sha_without_checkpoint_torch_or_dataset_load(contract,tmp_path):
    r.validate_contract(contract)
    path=tmp_path/'contract.json'
    r.write(path,contract)
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
    proc=subprocess.run([sys.executable,str(RUNTIME),'--contract',str(path),'--job',contract['jobs'][0]['id'],'--dry-run'],
        capture_output=True,text=True,env=env,timeout=20)
    assert proc.returncode==0,proc.stderr
    result=json.loads(proc.stdout)
    assert result['data_loaded'] is False and result['checkpoint_loaded'] is False and result['device_initialized'] is False
    assert result['optimizer_updates']==0
    Path(contract['jobs'][0]['checkpoint']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint SHA'): r.validate_contract(contract)


def permit_fixture(c,tmp_path):
    first=c['jobs'][0]
    q=tmp_path/'receipt.json'
    receipt={'status':'passed','contract_sha256':r.canonical(c),'host':'5080','real_optimizer_updates':0,
        'job':first['id'],'representative_job':first['id'],'checkpoint_sha256':first['checkpoint']['sha256'],
        'source_files_sha256':r.SOURCE_CLOSURE,'device':'cuda:0','evaluation_scope':'validation_only',
        'held_out_test_evaluated':False,'quantity_output_unchanged':True,'trainable_tensors':list(r.PARAMETERS),
        'trainable_scalar_count':2,'runtime':{**c['runtime']['runtime_expected'],'gpu':{'uuid':c['runtime']['gpu_uuid']}}}
    r.write(q,receipt)
    now=r.time.time()
    p={'schema':'time_head_refit_training_permit_v1','contract_sha256':r.canonical(c),'job':first['id'],
        'owner_pid':os.getppid(),'issued_at_unix':now,'deadline_unix':now+30,'qualification':{'path':str(q),'sha256':r.sha(q)}}
    path=tmp_path/'permit.json'
    r.write(path,p)
    return path,p,q,receipt


@pytest.mark.parametrize('change',['valid','owner','job','expired','budget','qualification','representative','uuid','kernel_scope'])
def test_training_permit_rejects_foreign_owner_receipt_and_deadline(contract,tmp_path,change):
    path,p,q,receipt=permit_fixture(contract,tmp_path)
    if change=='owner': p['owner_pid']+=1
    elif change=='job': p['job']='another'
    elif change=='expired': p['issued_at_unix']-=100;p['deadline_unix']-=100
    elif change=='budget': p['deadline_unix']=p['issued_at_unix']+3601
    elif change=='qualification': p['qualification']['sha256']='0'*64
    elif change in ('representative','uuid','kernel_scope'):
        if change=='representative': receipt['job']='wrong'
        elif change=='uuid': receipt['runtime']['gpu']['uuid']='wrong'
        else: receipt['held_out_test_evaluated']=True
        q.unlink();r.write(q,receipt);p['qualification']['sha256']=r.sha(q)
    path.unlink();r.write(path,p)
    if change=='valid': r.Permit(contract,contract['jobs'][0],path,p['owner_pid'])()
    else:
        with pytest.raises(ValueError): r.Permit(contract,contract['jobs'][0],path,os.getppid())


def test_failed_qualification_and_fit_preserve_failure_and_forbid_retry(contract,tmp_path,monkeypatch):
    def fail(*args,**kwargs): raise ValueError('synthetic load failure')
    monkeypatch.setattr(r,'load_job',fail)
    output=tmp_path/'qualification'
    with pytest.raises(ValueError,match='synthetic load'): r.qualify(contract,contract['jobs'][0],{},output,'cpu')
    assert r.read(output/'failure.json')['automatic_retry_allowed'] is False
    with pytest.raises(ValueError,match='Fresh qualification'): r.qualify(contract,contract['jobs'][0],{},output,'cpu')
    with pytest.raises(ValueError,match='synthetic load'): r.execute(contract,contract['jobs'][0],{},lambda:None,'cpu')
    assert r.read(Path(contract['jobs'][0]['output_dir'])/'failure.json')['automatic_retry_allowed'] is False
    with pytest.raises(ValueError,match='Fresh output'): r.execute(contract,contract['jobs'][0],{},lambda:None,'cpu')


def test_owned_supervisor_precreated_directory_is_claimed_once(tmp_path):
    output=tmp_path/'job'
    output.mkdir()
    r.write(output/'training_permit.json',{'owner_pid':os.getppid()})
    r.claim_output(output,{'training_permit.json','worker_process.json'},job='synthetic')
    assert r.read(output/'runtime_claim.json')['job']=='synthetic'
    with pytest.raises(ValueError,match='Fresh output'):
        r.claim_output(output,{'training_permit.json','worker_process.json'},job='synthetic')


def test_storage_limit_is_rechecked_independently_of_frequent_pulses(contract,tmp_path,monkeypatch):
    contract['limits']['max_owned_root_bytes']=100
    contract['root']=str(tmp_path/'owned')
    owned=Path(contract['root']);owned.mkdir()
    clock=[0.]
    monkeypatch.setattr(r.time,'monotonic',lambda:clock[0])
    resources=r.Resources(contract)
    resources()
    (owned/'growing').write_bytes(b'x'*101)
    for i in range(1,30):
        clock[0]=float(i);resources()
    clock[0]=30.
    with pytest.raises(ValueError,match='storage budget'): resources()


def test_operational_chain_rejects_missing_approval_and_changed_operation(contract,tmp_path,monkeypatch):
    own=tmp_path/'operation';own.mkdir()
    worker=own/'titantpp_time_head_refit_runtime.py';worker.write_bytes(RUNTIME.read_bytes())
    contract['operation']={'files':{'operation/titantpp_time_head_refit_runtime.py':r.sha(worker)}}
    contract['operation']['files_sha256']=r.canonical(contract['operation']['files'])
    path=tmp_path/'execution_contract.json';r.write(path,contract)
    with pytest.raises(FileNotFoundError): r.verify_operational_authorization(contract,path)
    worker.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Operation file SHA'): r.verify_operational_authorization(contract,path)


def test_complete_operational_chain_matches_parent_science_subset(contract,tmp_path,monkeypatch):
    own=tmp_path/'operation';own.mkdir()
    worker=own/'titantpp_time_head_refit_runtime.py';worker.write_bytes(RUNTIME.read_bytes())
    contract['operation']={'files':{'operation/titantpp_time_head_refit_runtime.py':r.sha(worker)}}
    contract['operation']['files_sha256']=r.canonical(contract['operation']['files'])
    parent=r.read(PROJECT/'search_artifacts/titantpp_cnn_gru54_3seed_dual_20261005_v1/execution_contract.json')
    parent['hosts']['5080']['source_root']=str(SOURCE)
    contract['runtime']['environment']={**parent['hosts']['5080']['environment'],'MPLCONFIGDIR':str(tmp_path/'cache/matplotlib'),'XDG_CACHE_HOME':str(tmp_path/'cache')}
    for key,value in contract['runtime']['environment'].items(): monkeypatch.setenv(key,value)
    parentroot=tmp_path/'parent';parentroot.mkdir()
    r.write(parentroot/'execution_contract.json',parent)
    contract['parent_root']=str(parentroot)
    contract['parent_contract_sha256']=r.canonical(parent)
    path=tmp_path/'execution_contract.json';r.write(path,contract)
    approval={'approved':True,'contract_sha256':r.canonical(contract),'hosts':['5080'],'user_instruction':'Synthetic CPU fixture'}
    r.write(tmp_path/'approval.json',approval)
    now=r.time.time()
    start={'contract_sha256':r.canonical(contract),'approval_sha256':r.canonical(approval),
        'issued_at_unix':now,'deadline_unix':now+23400}
    r.write(tmp_path/'start_permit.json',start)
    assert r.verify_operational_authorization(contract,path)==start
    parent['datasets'][-1]['statistics']['train_log_mean']+=.01
    (parentroot/'execution_contract.json').unlink();r.write(parentroot/'execution_contract.json',parent)
    with pytest.raises(ValueError,match='Parent scientific binding'): r.verify_operational_authorization(contract,path)


@pytest.mark.parametrize('precreated',[(),('claim.json',),('worker_process.json',),('claim.json','worker_process.json')])
def test_qualification_accepts_only_supervisor_evidence_subset(contract,tmp_path,monkeypatch,precreated):
    output=tmp_path/'qualification';output.mkdir()
    for name in precreated:
        r.write(output/name,{'owner_pid':os.getppid(),'pid':os.getpid()})
    def fail(*args,**kwargs): raise ValueError('synthetic qualification entered')
    monkeypatch.setattr(r,'load_job',fail)
    with pytest.raises(ValueError,match='synthetic qualification entered'):
        r.qualify(contract,contract['jobs'][0],{},output,'cpu')
    assert r.read(output/'runtime_claim.json')['owner_pid']==os.getppid()
    assert r.read(output/'failure.json')['automatic_retry_allowed'] is False
    with pytest.raises(ValueError,match='Fresh qualification'):
        r.qualify(contract,contract['jobs'][0],{},output,'cpu')
