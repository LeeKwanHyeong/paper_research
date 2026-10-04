"""Independent synthetic and immutable-manifest review; no real model inference."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

e = module('deep_independent_evaluate', HERE/'evaluate.py')
p = module('deep_independent_pipeline', HERE/'pipeline.py')

# Reuse the behavioral regression suite while replacing its worker/controller.
# Every fixture writes into pytest tmp_path; no real campaign process is launched.
for name in ('test_evaluate', 'test_pipeline'):
    shared = module('deep_reused_' + name, HERE.parent/'width16'/f'{name}.py')
    if name == 'test_evaluate': shared.e = e
    else: shared.p = p
    for key, value in vars(shared).items():
        if key.startswith('test_') and callable(value): globals()['reused_' + key] = value
        # pytest needs the test_ prefix for collected function aliases.
        if key.startswith('test_') and callable(value): globals()['test_reused_' + key[5:]] = value


def test_only_native_two_setting_override_differs_from_original_evaluator():
    derivation = json.loads((HERE/'evaluator_derivation.json').read_text())
    parent = ROOT/derivation['parent_path']
    assert e.sha256_file(parent) == derivation['parent_sha256']
    native = ast.parse((HERE/'evaluate.py').read_text())
    worker = next(n for n in native.body if isinstance(n, ast.FunctionDef) and n.name == 'run_worker')
    removed = []
    class NativeOverride(ast.NodeTransformer):
        def visit_If(self, node):
            if (isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Attribute)
                and node.test.left.attr == 'model' and len(node.test.comparators)==1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == 'deep_renewal_event_native_nb'):
                call = node.body[0].value
                assert isinstance(call.func,ast.Attribute) and call.func.attr=='update'
                assert {k.arg: ast.literal_eval(k.value) for k in call.keywords} == {
                    'quantity_variant':'shifted_nb_quantity_nll', 'time_head_mode':'shifted_nb_duration'}
                removed.append(node)
                return None
            return self.generic_visit(node)
    NativeOverride().visit(worker)
    assert len(removed)==1
    assert ast.dump(native,include_attributes=False)==ast.dump(ast.parse(parent.read_text()),include_attributes=False)


def test_real_seal_twelve_routes_three_frozen_sources_and_preflight_bindings():
    contract=e.read_json(HERE/'execution_contract.json')
    registry=e.read_json(HERE/'evaluation_registry.json')
    manifest=e.read_json(HERE/'dataset_manifest.json')
    seal=e.read_json(HERE/'code_seal.json')
    assert len(registry['rows'])==12 and len(registry['bundles'])==3
    assert contract['reuse_conditions']==[] and contract['reuse_split_count']==0
    assert contract['validation_gate']==dict(new_conditions=12,rtol=1e-5,atol=1e-5,all_required_before_test=True)
    assert contract['approval']['models']==['deep_renewal_event_native_nb']
    assert contract['approval']['retraining_authorized'] is False
    assert contract['resources']['concurrent_workers']==1
    assert contract['resources']['shared_runtime_changes'] is False
    assert contract['manuscript_comparison_included'] is False
    for path,expected in seal['files'].items(): assert e.sha256_file(ROOT/path)==expected
    assert e.sha256_file(HERE/'evaluation_registry.json')==contract['registry_sha256']
    assert e.sha256_file(HERE/'dataset_manifest.json')==contract['dataset_manifest_sha256']
    assert e.sha256_file(HERE/'evaluate.py')==contract['evaluator_sha256']
    assert e.sha256_file(ROOT/contract['common_scope_path'])==contract['common_scope_sha256']
    seen=set()
    for row in registry['rows']:
        for split in ('validation','test'):
            args=SimpleNamespace(root=ROOT,registry=str(HERE/'evaluation_registry.json'),
                dataset_manifest=str(HERE/'dataset_manifest.json'),contract=str(HERE/'execution_contract.json'),
                dataset=row['dataset'],model=row['model'],seed=row['seed'],split=split)
            root,bound,bundle,spec,data,identities=e.validate_inputs(args)
            assert bound==row
            assert spec['loader']==next(x for x in manifest['datasets'] if x['dataset']==row['dataset'])['loader']
            assert data['sha256']==contract['dataset_sha256'][row['dataset']]
            pop=contract['population_references'][row['dataset']][split]
            assert pop['data_file_sha256']==data['sha256'] and pop['loader']==spec['loader']
            assert pop['expected_target_count']==data['populations'][split]['target_count']
        if row['evaluator_source_bundle'] not in seen:
            e.verify_source(ROOT,bundle)
            canonical=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
            original=e.read_json(ROOT/bundle['contract_path'])
            assert original['source']['files']==bundle['source_files']
            assert canonical(bundle['source_files'])==bundle['source_closure_sha256']
            assert canonical(original)==bundle['canonical_sha256']
            seen.add(row['evaluator_source_bundle'])
        assert e.sha256_file(ROOT/row['checkpoint_path'])==row['checkpoint_file_sha256']
        assert e.sha256_file(ROOT/row['validation_replay_path'])==row['validation_replay_sha256']
        ref=contract['validation_references'][p.identity(row)]
        assert ref['state_sha256']==row['state_tensor_sha256']
        assert ref['count']==row['validation_count']
    preflight=e.read_json(HERE/'local_preflight.json')
    assert preflight['status']=='passed' and preflight['conditions']==12 and preflight['inference_started'] is False
    assert preflight['registry_sha256']==contract['registry_sha256']
    indexed={(r['dataset'],r['seed']):r for r in registry['rows']}
    for value in preflight['rows']:
        row=indexed[(value['dataset'],value['seed'])]
        assert value['strict_native_reconstruction'] is True and value['test_read'] is False
        assert value['state_tensor_sha256']==row['state_tensor_sha256']


def fake_prediction(tmp_path,monkeypatch,split='test'):
    monkeypatch.setattr(p,'HERE',tmp_path)
    (tmp_path/'execution_contract.json').write_text('{}')
    row=dict(dataset='synthetic',model='deep_renewal_event_native_nb',seed=42,selected_epoch=5,
             checkpoint_file_sha256='cp',state_tensor_sha256='state')
    population=dict(target_identity_sha256='target',truth_sha256='truth',loader={'history':64},
                    data_file_sha256='data',expected_target_count=2)
    result=dict(count=2,qty_rmse=1.0,qty_mae=1.0,time_nll=0.5)
    receipt={**row,**population,'split':split,'contract_sha256':p.sha(tmp_path/'execution_contract.json'),
             'runner_sha256':'runner','registry_sha256':'registry','dataset_manifest_sha256':'manifest',
             'runtime':{'cpu_peak_rss_bytes':1024,'cuda_peak_reserved_bytes':1024}}
    (tmp_path/'receipt.json').write_text('{}')
    contract=dict(evaluator_sha256='runner',registry_sha256='registry',dataset_manifest_sha256='manifest',
                  population_references={'synthetic':{split:population}},
                  validation_references={p.identity(row):result},resources={'rss_limit_gib':1,'gpu_memory_limit_gib':1})
    monkeypatch.setattr(p,'load_predictions',lambda folder:(receipt,None))
    monkeypatch.setattr(p,'metrics',lambda frame:result)
    return row,receipt,contract,result


@pytest.mark.parametrize('field',['target_identity_sha256','truth_sha256','loader','data_file_sha256','expected_target_count'])
def test_population_and_history_mismatch_cannot_be_accepted(tmp_path,monkeypatch,field):
    row,receipt,contract,_=fake_prediction(tmp_path,monkeypatch)
    p.check_prediction(tmp_path,row,'test',contract)
    receipt[field]='substituted'
    with pytest.raises(AssertionError):p.check_prediction(tmp_path,row,'test',contract)


@pytest.mark.parametrize('field',['qty_rmse','qty_mae','time_nll'])
def test_validation_metric_mismatch_fails_gate(tmp_path,monkeypatch,field):
    row,receipt,contract,result=fake_prediction(tmp_path,monkeypatch,'validation')
    contract['validation_references']=copy.deepcopy(contract['validation_references'])
    result[field]+=0.1
    with pytest.raises(AssertionError):p.check_prediction(tmp_path,row,'validation',contract)


def test_existing_condition_directory_stops_before_gpu_probe_or_process(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'HERE',tmp_path)
    row=dict(dataset='synthetic',model='deep_renewal_event_native_nb',seed=42)
    directory=tmp_path/'runs'/'test'/p.identity(row);directory.mkdir(parents=True)
    sentinel=directory/'prior_evidence.txt';sentinel.write_text('preserve')
    monkeypatch.setattr(p,'output',lambda command:pytest.fail('No GPU query expected'))
    monkeypatch.setattr(p.subprocess,'Popen',lambda *a,**kw:pytest.fail('No process expected'))
    with pytest.raises(AssertionError,match='Existing output is preserved'):
        p.run_condition(row,'test',{'resources':{}},0)
    assert sentinel.read_text()=='preserve'
