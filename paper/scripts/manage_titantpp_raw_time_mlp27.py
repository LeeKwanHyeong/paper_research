"""Prepare an isolated raw-time27 experiment only after verified local parent completion."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import math
from pathlib import Path

from paper.scripts import titantpp_raw_time_mlp27_runtime as _runtime

_BASE_PATH = Path(__file__).with_name('manage_titantpp_quantity_loss_mlp18.py')
_BASE_SHA = '331aa17a98532217bb4cdda8865cf09530cc54b5c31f4968456655e93be27e8d'
_runtime.require(_runtime.sha_file(_BASE_PATH) == _BASE_SHA, 'Sealed predecessor manager changed')
_spec = importlib.util.spec_from_file_location('paper.scripts._raw_time27_manager_private', _BASE_PATH)
_impl = importlib.util.module_from_spec(_spec)
_source = _BASE_PATH.read_text().replace('titantpp_quantity_loss_mlp18', 'titantpp_raw_time_mlp27')
_source = _source.replace('quantity_loss18_', 'raw_time27_')
exec(compile(_source, str(_BASE_PATH), 'exec'), _impl.__dict__)
_impl.runtime = _runtime
_impl.NAME = 'titantpp_raw_time_mlp27_20261019_v1'
_impl.BUNDLE = _impl.ROOT / 'search_artifacts' / _impl.NAME
_impl.REPORT = _impl.ROOT / 'reports' / _impl.NAME
_impl.MODULE = _runtime.MODULE
_impl.REQUIRED_NEW_SOURCE = ('paper/scripts/run_titantpp_raw_time_mlp27.py', 'paper/scripts/titantpp_raw_time_mlp27_runtime.py', 'paper/scripts/manage_titantpp_raw_time_mlp27.py')
PARENT_NAME = 'titantpp_quantity_loss_mlp18_20261019_v2'
PARENT_CONTRACT_RAW_SHA = 'bca53c976981c168bd0c7850c89a178e154e8e851498d1a2646756febd36441b'
PARENT_CONTRACT_CANONICAL_SHA = '4b574670362e6469826fcbb4c1e355508594b76a00a125d4e418ca03a0ca71a2'


def parent_contract():
    path = _impl.ROOT / 'search_artifacts' / PARENT_NAME / 'execution_contract.json'
    _impl.require(_impl.sha(path) == PARENT_CONTRACT_RAW_SHA, 'Fixed v2 parent contract changed')
    c = _impl.read(path)
    _impl.require(_impl.canonical(c) == PARENT_CONTRACT_CANONICAL_SHA, 'Fixed v2 canonical contract changed')
    return c, path


def verify_parent_completion():
    """Read and hash local completed artifacts; do not run the old finisher/comparator."""
    from paper.scripts import manage_titantpp_quantity_loss_mlp18_v2 as parent
    c, cp = parent_contract()
    checked = parent.verify_local(archive=True)
    _impl.require(checked == c, 'Parent manager resolved a different experiment')
    finish_path = parent.REPORT / 'completion_v1/complete.json'
    completion = parent.read(finish_path)
    fp = parent.BUNDLE / 'retrieval/final_receipt.json'; final = parent.read(fp)
    _impl.require(completion['status'] == 'complete' and completion['final_receipt_path'] == str(fp) and completion['final_receipt_sha256'] == parent.sha(fp), 'Parent finite completion not verified')
    expected = {'new_fit_count': 18, 'validation_replay_count': 18, 'baseline_reference_count': 9, 'comparison_model_count': 27, 'seed_count': 3, 'time_alpha': 0., 'time_head_no_training_verified_count': 18, 'actual_scientific_child_exit0_count': 36, 'actual_dispatcher_exit0_count': 2}
    _impl.require(final['status'] == 'complete' and final['held_out_test_evaluated'] is False and final['baseline_reevaluated'] is False and all(final.get(k) == v for k, v in expected.items()), 'Parent final counts/scope differ')
    _impl.require(final['source_bindings'] and final['archive_bindings'] and final['report_files'], 'Empty parent completion inventory')
    for group in ('source_bindings', 'archive_bindings', 'report_files'):
        for name, digest in final[group].items():
            _impl.require(parent.sha(name) == digest, 'Parent completed binding changed: ' + name)
    for host in ('5080', '5090'):
        folder = parent.BUNDLE / 'retrieval' / host; originals = folder / 'originals'
        receipt = parent.read(folder / 'receipt.json')
        _impl.require(receipt['status'] == 'complete' and receipt['host'] == host and receipt['held_out_test_evaluated'] is False, 'Parent collection incomplete')
        _impl.require(parent.sha(folder / 'original.tar.gz') == receipt['archive_sha256'] and parent.sha(folder / 'manifest.json') == receipt['manifest_sha256'], 'Parent archive/manifest changed')
        inventory = parent.read(folder / 'manifest.json')['files']
        _impl.require(inventory and len(inventory) == receipt['file_count'], 'Parent inventory empty/incomplete')
        for name, digest in inventory.items():
            _impl.require(parent.sha(parent.runtime.local_path(originals, name)) == digest, 'Parent original changed: ' + name)
        parent.verify_collected(c, host, originals)
    comparison_path = parent.REPORT / 'comparison_v1/THREE_LOSS_VALIDATION.json'
    comparison = parent.read(comparison_path)
    _impl.require(final['report_files'].get(str(comparison_path)) == parent.sha(comparison_path) and comparison['status'] == 'complete' and comparison['scope'] == 'validation_only' and comparison['comparison_model_count'] == 27 and comparison['held_out_test_evaluated'] is False, 'Parent comparison lineage incomplete')
    records = comparison['records']
    expected_keys = {(d, s, m) for d in _runtime.DATASETS for s in (42, 52, 62) for m in _runtime.LOSS_MODES}
    _impl.require(len(records) == 27 and {(r['dataset'], r['seed'], r['loss_mode']) for r in records} == expected_keys, 'Parent comparison endpoint matrix differs')
    return c, final, comparison, {'name': PARENT_NAME, 'contract_path': str(cp), 'contract_raw_sha256': parent.sha(cp), 'contract_canonical_sha256': parent.canonical(c), 'source_files_sha256': c['source']['files_sha256'], 'completion_path': str(finish_path), 'completion_sha256': parent.sha(finish_path), 'final_receipt_path': str(fp), 'final_receipt_sha256': parent.sha(fp), 'comparison_path': str(comparison_path), 'comparison_sha256': parent.sha(comparison_path), 'retrained': False, 'reevaluated': False}


def baseline_validation_references(*, verified=None):
    from paper.scripts import manage_titantpp_quantity_loss_mlp18_v2 as parent
    c, final, comparison, binding = verified if verified is not None else verify_parent_completion()
    refs = parent.baseline_validation_references()
    _impl.require(refs == c['baseline_validation_references'], 'Parent stored log0 references changed')
    for ref in refs: ref['time_input_transform'] = 'log1p_nonnegative_float32'
    for job in c['jobs']:
        host = job['host']; originals = parent.BUNDLE / 'retrieval' / host / 'originals'
        path = originals / 'run' / job['id'] / 'validation_receipt.json'; receipt = parent.read(path)
        _impl.require(final['source_bindings'].get(str(path)) == parent.sha(path), 'Parent endpoint SHA missing')
        parent.runtime.verify_time_head_proof(receipt['time_head_no_training_verification'])
        selected = Path(receipt['selected_checkpoint']); remote = Path(c['hosts'][host]['root'])
        _impl.require(selected.is_absolute() and selected.is_relative_to(remote), 'Parent checkpoint outside host root')
        local_cp = parent.runtime.local_path(originals, str(selected.relative_to(remote)))
        _impl.require(parent.sha(local_cp) == receipt['checkpoint_sha256'], 'Parent selected binary changed')
        histories = list((path.parent / 'runs').glob('*/*/seed_*/history.json'))
        _impl.require(len(histories) == 1, 'Missing parent selected history')
        finite = [(r['val_qty_rmse'], r['epoch']) for r in parent.read(histories[0])['history'] if type(r.get('val_qty_rmse')) in (int, float) and math.isfinite(r['val_qty_rmse'])]
        _impl.require(finite and min(finite, key=lambda x: x[0])[1] == receipt['selected_epoch'], 'Parent first finite RMSE selection differs')
        refs.append({'dataset': job['dataset'], 'host': host, 'seed': job['seed'], 'arm': job['arm'], 'loss_mode': job['loss_mode'], 'time_alpha': 0., 'time_input_transform': 'log1p_nonnegative_float32', 'quantity_only': True, 'time_head_trained': False, 'selected_epoch': receipt['selected_epoch'], 'selected_checkpoint': str(local_cp), 'checkpoint_sha256': receipt['checkpoint_sha256'], 'state_sha256': receipt['state_sha256'], 'target_identity': receipt['target_identity'], 'metrics': receipt['metrics'], 'path': str(path), 'sha256': parent.sha(path), 'completion_receipt_path': binding['final_receipt_path'], 'completion_receipt_sha256': binding['final_receipt_sha256'], 'time_head_no_training_verification': receipt['time_head_no_training_verification'], 'retrained': False, 'reevaluated': False, 'held_out_test_evaluated': False})
    keys = {(r['dataset'], r['seed'], r['loss_mode']): r for r in comparison['records']}
    _impl.require(len(refs) == 27, 'Expected27 stored log-time endpoints')
    for ref in refs:
        old = keys[ref['dataset'], ref['seed'], ref['loss_mode']]
        _impl.require(all(ref[k] == old[k] for k in ('host', 'time_alpha', 'selected_epoch', 'target_identity', 'metrics')), 'Parent comparison/endpoint differs')
    return sorted(refs, key=lambda r: (r['dataset'], r['seed'], r['loss_mode']))


def source_snapshot():
    c, cp = parent_contract(); original = c['source']['files']
    _impl.require(len(original) == 952 and _impl.canonical(original) == c['source']['files_sha256'], 'Expected byte-identical parent952')
    for name, digest in original.items(): _impl.require(_impl.sha(_impl.ROOT / name) == digest, 'Parent source changed: ' + name)
    required = set(_impl.REQUIRED_NEW_SOURCE)
    tracked = set(_impl.subprocess.check_output(['git', 'ls-files', '--', *required], cwd=_impl.ROOT, text=True).splitlines())
    _impl.require(tracked == required and not _impl.subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', *required], cwd=_impl.ROOT, text=True).strip(), 'Commit only the new raw-time operators before prepare')
    files = {**original, **{name: _impl.sha(_impl.ROOT / name) for name in required}}
    revision = _impl.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_impl.ROOT, text=True).strip()
    return {'base_git_revision': revision, 'files': files, 'files_sha256': _impl.canonical(files), 'model_science_source_revision': c['source']['model_science_source_revision'], 'model_science_source_files_sha256': c['source']['model_science_source_files_sha256'], 'parent_source_files_sha256': c['source']['files_sha256'], 'parent_source_file_count': 952, 'new_input_adapter': 'raw observed gap; quantity log1p unchanged'}


def prepare():
    _impl.require(not _impl.BUNDLE.exists(), 'Existing raw-time bundle; no overwrite/retry')
    verified = verify_parent_completion(); parent, _, _, binding = verified
    baseline = baseline_validation_references(verified=verified); source = source_snapshot()
    inputs = deepcopy(parent['input_files'])
    _impl.require(len(inputs) == 6, 'Only TrainVal/manifest6')
    hosts = deepcopy(parent['hosts'])
    for host, h in hosts.items():
        root = f'/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/{_impl.NAME}_{host}'
        old = _impl.ROOT / 'search_artifacts' / PARENT_NAME / 'retrieval' / host / 'originals'
        fp, xp = old / 'final_receipt.json', old / 'dispatcher_exit.json'
        exit_record = _impl.read(xp)
        h.update(root=root, source_root=root + '/source', tmux=f'raw_time27_{host}_20261019_v1', completed_predecessor={'root': parent['hosts'][host]['root'], **{k: binding[k] for k in ('contract_raw_sha256', 'contract_canonical_sha256', 'source_files_sha256')}, 'dispatcher_identity': {'pid': exit_record['pid']}, 'final_receipt_sha256': _impl.sha(fp), 'dispatcher_exit_sha256': _impl.sha(xp)})
        h['environment'].update(SOURCE_REVISION=source['base_git_revision'], PYTHONPATH=h['source_root'], MPLCONFIGDIR=root + '/cache/matplotlib', XDG_CACHE_HOME=root + '/cache')
    architecture = {'arms': list(_runtime.ARMS), 'time_alpha': 0., 'quantity_loss_modes': list(_runtime.LOSS_MODES), 'quantity_loss_normalization': 'none', 'model_architecture_changed': False, 'checkpoint_warmstart': False, 'time_input_transform': 'raw_nonnegative_float32', 'quantity_input_transform': 'log1p_nonnegative_float32', 'objective': 'log-MSE or literal raw-MSE/raw-MAE; time graph excluded', 'input_change_only': 'observed time gap log1p -> raw; quantity transform/head/optimizer/budget/selection unchanged'}
    contract = {'schema': _runtime.SCHEMA, 'created_at': _impl.now(), 'science_spec_path': str(_impl.REPORT / 'SCIENCE_SPEC.json'), 'science_spec_sha256': _impl.sha(_impl.REPORT / 'SCIENCE_SPEC.json'), 'source': source, 'hosts': hosts, 'jobs': _impl.build_jobs(), 'datasets': deepcopy(parent['datasets']), 'input_files': inputs, 'data_receipt': deepcopy(parent['data_receipt']), 'architecture': architecture, 'design_sha256': _impl.canonical(architecture), 'evaluation_scope': 'validation_only', 'authorization': 'User approved raw-time MLP16 three quantity losses x3 datasets x3 seeds=27; stored log-time27 reused; no Test', 'held_out_test_authorized': False, 'historical_cnn_retrained': False, 'historical_cnn_reevaluated': False, 'training': deepcopy(parent['training']), 'limits': deepcopy(parent['limits']), 'execution': {'maximum_concurrent_per_host': 3, 'completed_predecessor_required_before_gpu_work': True, 'validation_gate_scope': 'all assigned jobs on each host', 'existing_campaign_remains_stopped': True, 'recurring_automation': False}, 'comparison': {'scope': 'raw-time27 versus matched stored log-time27', 'model_count': 54, 'seeds': [42, 52, 62], 'sample_sd_ddof': 1}, 'baseline_validation_references': baseline, 'parent_quantity_loss27': binding, 'stored_cap_mlp_reference': deepcopy(parent['stored_cap_mlp_reference']), 'packaging': {'required_empty_source_directories': ['sample_data'], 'test_inputs_packaged': False}}
    for host in hosts: _runtime.validate_contract(contract, host)
    # All original inputs are verified before creation of the exclusive new bundle.
    origin_root = _impl.ROOT / 'search_artifacts' / PARENT_NAME / 'input_snapshot'
    for name, digest in inputs.items(): _impl.require(_impl.sha(_runtime.local_path(origin_root, name)) == digest, 'Parent TrainVal input changed')
    _impl.BUNDLE.mkdir(parents=True, exist_ok=False)
    for name in inputs:
        origin = _runtime.local_path(origin_root, name); destination = _runtime.local_path(_impl.BUNDLE / 'input_snapshot', name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with origin.open('rb') as src, destination.open('xb') as dst: _impl.shutil.copyfileobj(src, dst)
    _impl.write(_impl.BUNDLE / 'execution_contract.json', contract); archive = _impl.package_archive(contract, source)
    receipt = {'status': 'prepared', 'created_at': _impl.now(), 'source_revision': source['base_git_revision'], 'source_file_count': len(source['files']), 'contract_sha256': _impl.sha(_impl.BUNDLE / 'execution_contract.json'), 'canonical_contract_sha256': _impl.canonical(contract), 'archive_sha256': _impl.sha(archive), 'planned_fit_count': 27, 'host_fit_counts': {h: sum(j['host'] == h for j in contract['jobs']) for h in hosts}, 'baseline_reference_count': 27, 'held_out_test_evaluated': False, 'historical_cnn_retrained': False, 'parent_completion_sha256': binding['completion_sha256']}
    _impl.write(_impl.REPORT / 'PREPARED.json', receipt)
    return receipt


_original_verify_collected = _impl.verify_collected

def verify_collected(c, host, originals):
    result = _original_verify_collected(c, host, originals)
    for job in _runtime.validate_contract(c, host):
        for stage in _runtime.STAGES:
            receipt = _impl.read(Path(originals) / 'run' / job['id'] / f'{stage}_receipt.json')
            _impl.require(receipt.get('time_input_transform') == 'raw_nonnegative_float32' and receipt.get('quantity_input_transform') == 'log1p_nonnegative_float32', 'Collected input identity differs')
    return result


_impl.source_snapshot = source_snapshot
_impl.baseline_validation_references = baseline_validation_references
_impl.prepare = prepare
_impl.verify_collected = verify_collected
for _name, _value in tuple(_impl.__dict__.items()):
    if not _name.startswith('_'): globals()[_name] = _value

if __name__ == '__main__':
    raise SystemExit(_impl.main())
