"""Local-only, deterministic preparation. Never deploy, launch, or import model code."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = Path('/Users/igwanhyeong/PycharmProjects/paper_research')
OLD = Path('/private/tmp/titantpp_gru_controls_20261005_stage/titantpp_gru_controls_3seed_dual_20261005_v1')
NAME = 'titantpp_cnn_gru54_3seed_dual_20261005_v1'
ARM = 'titantpp_cnn_gru54'
HOST_DATA = {'5080': ['yellow_trip_hourly', 'raf_spare_parts'], '5090': ['intermittent_frozen_5000']}
SOURCE_SHA = '4e94229fae002dc678025ecf056862f832c64459969f403b3cb5ac8cea8ba7b0'
PARENT_SHA = '40184c7ff418f35b191ebfa309e462b9e88889443f36e4722cc5d06988d182cd'
WORKER_SHA = '7d961867f612f86e1b78346630764d7624472532f596aba133a3bf3d24da2f26'

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def require(condition, message):
    if not condition:
        raise ValueError(message)

def save(path, value):
    """Idempotent identical writes; a changed freeze requires a different output root."""
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        require(path.read_bytes() == data, 'Refuse replacing different frozen artifact: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)

def metrics(endpoint):
    return {key: endpoint[key] for key in ('qty_rmse', 'qty_mae', 'time_nll')}

def item_sha(item):
    value = item.get('file_sha256', item.get('sha256'))
    require(isinstance(value, str) and len(value) == 64, 'Missing original file SHA')
    return value

def verified_source(path, expected):
    require(sha(path) == expected, 'Input SHA mismatch: ' + str(path))
    return str(Path(path).resolve())

def stage_inputs(out, deployment_inputs):
    """Independent copies only: never hardlink writable staging to research originals."""
    staged = {}
    for relative, item in deployment_inputs.items():
        if 'local_source' not in item:
            continue
        target = out / 'payload' / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(item['local_source'], target)
        require(sha(target) == item['sha256'], 'Staged input differs: ' + relative)
        staged[relative] = {'path': str(target), 'sha256': item['sha256']}
    return staged

def expected_jobs():
    return [{'id': f'{dataset}__{seed}__{ARM}', 'host': host, 'dataset': dataset, 'seed': seed, 'arm': ARM}
            for host, datasets in HOST_DATA.items() for dataset in datasets for seed in (52, 62)]

def validate_contract(c, old, parent):
    require(c['schema'] == NAME and c['arms'] == [ARM] and c['jobs'] == expected_jobs(), 'Six-fit scope drift')
    require(c['source'] == old['source'] and len(c['source']['files']) == 123
            and digest(c['source']['files']) == SOURCE_SHA, 'Exact frozen123 source drift')
    require(len(parent['source']['files']) == 117 and digest(parent['source']['files']) == PARENT_SHA,
            'Parent117 source drift')
    require(all(c['source']['files'].get(p) == s for p, s in parent['source']['files'].items()), 'Parent117 scientific mismatch')
    require(c['training'] == old['training'] and c['datasets'] == old['datasets']
            and c['dataset_sha256'] == old['dataset_sha256'], 'Training/dataset drift')
    require(c['limits'] == old['limits'], 'Budget/resource policy drift')
    require(c['operation']['files_sha256'] == digest(c['operation']['files']), 'Operation closure mismatch')
    require(c['canonical_conditions'] == 9 and c['new_fits'] == 6 and c['already_terminal_reused'] == 3, 'Counts drift')
    require(len(c['reuse']) == 6 and all(x['seed'] in (52, 62) for x in c['reuse']), 'Exposure references drift')
    require(set(c['anchors42']) == {d for ds in HOST_DATA.values() for d in ds}, 'Anchor coverage mismatch')
    for host in HOST_DATA:
        for key in ('python', 'runtime_expected', 'gpu_uuid', 'tmux_binary'):
            require(c['hosts'][host][key] == old['hosts'][host][key], 'Runtime mutation: ' + key)
        for key, value in old['hosts'][host]['environment'].items():
            if key not in ('PYTHONPATH', 'MPLCONFIGDIR', 'XDG_CACHE_HOME'):
                require(c['hosts'][host]['environment'][key] == value, 'Runtime environment mutation: ' + key)
    for d, epochs in {'yellow_trip_hourly': (126, 166), 'raf_spare_parts': (9, 49), 'intermittent_frozen_5000': (12, 52)}.items():
        a = c['anchors42'][d]
        require((a['selected_epoch'], a['last_epoch']) == epochs, 'Anchor epochs drift')
        require(a['checkpoint_sha256'] == c['input_files'][a['checkpoint']]
                and a['last_checkpoint_sha256'] == c['input_files'][a['last_checkpoint']], 'Anchor binary binding mismatch')
    require(c['evaluation_scope'] == 'validation_only' and c['held_out_test_evaluated'] is False, 'Held-out boundary drift')
    return {'status': 'passed', 'checks': ['six_jobs_and_host_seed_scope', 'exact123_and_parent117',
            'scientific_training_and_dataset_equality', 'unchanged_native_runtime', 'anchor_epochs_and_SHA_bindings',
            'six_width4_exposure_references', 'operation_closure', 'budgets', 'held_out_lock']}

def prepare(args):
    out = args.output.resolve()
    require(out == HERE / 'contract', 'Output ownership is restricted to contract/')
    require(math.isfinite(args.issued_at) and args.issued_at > 0, 'Issued timestamp must be finite and positive')
    require(args.issued_at == int(args.issued_at), 'Issued timestamp must use whole Unix seconds')
    args.issued_at = int(args.issued_at)
    old = read(args.parent_bundle / 'execution_contract.json')
    parent_path = PROJECT / 'reports/titantpp_cnn_gru_a100_seed42_20261004_v1/execution_contract.json'
    parent = read(parent_path)
    registry = read(args.anchor_registry)
    require(registry['schema'] == 'titantpp_cnn_gru54_seed42_reuse_registry_v1', 'Wrong reuse registry schema')
    require(registry['status'] == 'PASS' and len(registry['rows']) == 3
            and all(row['model'] == ARM and row['seed'] == 42 and row['scientific_success'] is True
                    and row['status'] == 'complete' for row in registry['rows']), 'Incomplete anchor audit')
    source_files = old['source']['files']
    require(digest(source_files) == SOURCE_SHA and digest(parent['worker']['files']) == WORKER_SHA, 'Parent closures drift')
    operation_files = {f'operation/{name}': sha(HERE / 'operation' / name)
                       for name in ('campaign.py', 'diagnostic_adapter.py', 'monitor.py')}
    require(all((HERE / p).is_file() for p in operation_files), 'Operation files missing')
    c = copy.deepcopy(old)
    for key in ('reused_gru42',):
        c.pop(key, None)
    c.update(schema=NAME, arms=[ARM], seeds=[42, 52, 62], jobs=expected_jobs(), canonical_conditions=9,
             new_fits=6, already_terminal_reused=3, reference_baseline='titantpp_history_mlp_width16')
    c['architecture'] = {'arms': [ARM], 'inherited_model_contract': 'titantpp_cnn_gru_width16_v1',
        'model_changed': False, 'cnn': 'encoder1 event-only causal depthwise QKV kernel3',
        'correction': 'prefix GRU54', 'correction_parameters': 24732, 'cnn_parameters': 576,
        'fixed_alpha': 1, 'eligibility': 'at_least_two_observed_events_since_withheld_reset; padding_skipped',
        'loss_heads_and_selector': 'unchanged parent117 scientific source',
        'purpose': 'three-seed candidate performance against frozen external and strong internal benchmarks',
        'limits': 'Not a complete three-seed 2x2 interaction or recurrence-only causal experiment'}
    c['design_sha256'] = digest(c['architecture'])
    c['operation'] = {'files': operation_files, 'files_sha256': digest(operation_files),
        'entrypoint': 'operation/campaign.py', 'source_import_root': 'source',
        'qualified_scientific_import_closure': SOURCE_SHA, 'runtime_package_mutation': False}
    deployment_inputs = {}
    # Preserve only required prepared data and width4 exposure/reference inputs.
    c['input_files'] = {p: s for p, s in old['input_files'].items() if '/gru42/' not in p}
    for p, s in c['input_files'].items():
        deployment_inputs[p] = {'sha256': s, 'source_kind': 'existing_frozen_source_input', 'old_relative_path': p}
    c['reuse'] = [r for r in old['reuse'] if r['seed'] in (52, 62)]
    c['baseline_replays'] = {d: {seed: row for seed, row in refs.items() if seed in ('52', '62')}
                            for d, refs in old['baseline_replays'].items()}
    c['qualification_baseline_role'] = 'width4 original exposure and scientific compatibility only; primary baseline remains width16'
    c['anchors42'] = {}
    for row in registry['rows']:
        d = row['dataset']
        prefix = f'inputs/references/cnngru42/{d}'
        resolved = {}
        for key, filename in [('selected', 'best_val_qty_rmse_model.pt'), ('last', 'last_epoch_state.pt'),
                              ('terminal_manifest', 'terminal_manifest.json'), ('history', 'history.json'),
                              ('endpoint_replays', 'endpoint_replays.json'), ('input_receipt', 'input_receipt.json'),
                              ('exposure', 'exposure.json')]:
            item = row[key]
            checksum = item_sha(item)
            local = verified_source(item['path'], checksum)
            target = prefix + '/' + filename
            c['input_files'][target] = checksum
            deployment_inputs[target] = {'sha256': checksum, 'local_source': local, 'source_kind': 'audited_existing_original'}
            resolved[key] = target
        ep = read(row['endpoint_replays']['path'])
        require(ep['evaluation_scope'] == 'validation_only' and ep['held_out_test_evaluated'] is False,
                'Training endpoint is not validation-only')
        s, last = row['selected'], row['last']
        require(ep['best_epoch'] == s['epoch'] and ep['completed_epochs'] == last['epoch']
                and s['initial_state_sha256'] == last['initial_state_sha256'], 'Anchor endpoint/initial identity drift')
        history = read(row['history']['path'])['history']
        require([x['epoch'] for x in history] == list(range(1, last['epoch'] + 1)), 'Nonsequential anchor history')
        best = min((x for x in history if math.isfinite(x['val_qty_rmse'])), key=lambda x: x['val_qty_rmse'])
        require(best['epoch'] == s['epoch'], 'Anchor first-minimum selection drift')
        def history_metrics(epoch):
            return {m: history[epoch - 1][k] for m, k in
                    [('qty_rmse', 'val_qty_rmse'), ('qty_mae', 'val_qty_mae'), ('time_nll', 'val_time_nll')]}
        c['anchors42'][d] = {'job_id': f'{d}__42__{ARM}', 'checkpoint': resolved['selected'],
            'checkpoint_sha256': s['file_sha256'], 'last_checkpoint': resolved['last'],
            'last_checkpoint_sha256': last['file_sha256'], 'initial_state_sha256': s['initial_state_sha256'],
            'selected_epoch': s['epoch'], 'last_epoch': last['epoch'], 'metrics': metrics(ep['selected']),
            'last_metrics': metrics(ep['last']), 'source_revision': registry['source_revision'],
            'history_metrics': history_metrics(s['epoch']), 'last_history_metrics': history_metrics(last['epoch']),
            'endpoint_history_difference_policy': 'Preserve both originals; existing heterogeneous replay tolerance abs/rel1e-5, no history rewriting',
            'selected_state_sha256': s['state_tensor_sha256'], 'last_state_sha256': last['state_tensor_sha256'],
            'training_binding': {k: {'path': resolved[k], 'sha256': item_sha(row[k])}
                                 for k in ('terminal_manifest', 'history', 'endpoint_replays', 'input_receipt')},
            'reuse_registry_row_sha256': digest(row), 'heterogeneous_original_gpu': True,
            'test_exposure': 'Previously evaluated and used during candidate development; no new Test access in this campaign',
            'new_training': False}
        for key in ('history', 'endpoint_replays', 'terminal_manifest', 'input_receipt', 'exposure'):
            c['anchors42'][d][key] = resolved[key]
            c['anchors42'][d][key + '_sha256'] = item_sha(row[key])
        c['anchors42'][d]['endpoint'] = resolved['endpoint_replays']
        c['anchors42'][d]['endpoint_sha256'] = item_sha(row['endpoint_replays'])
    for host, cfg in c['hosts'].items():
        old_root = cfg['root']
        root = str(Path(old_root).parent / f'{NAME}_{host}')
        cfg.update(root=root, source_root=root + '/source', operation_root=root + '/operation',
                   tmux=f'{NAME}_{host}', output_dir=root + '/run', assigned_datasets=HOST_DATA[host])
        for key in ('PYTHONPATH', 'MPLCONFIGDIR', 'XDG_CACHE_HOME'):
            cfg['environment'][key] = cfg['environment'][key].replace(old_root, root)
    c['runtime_reuse_evidence'] = {}
    for host in HOST_DATA:
        evidence_path = args.parent_bundle / 'qualification' / f'receipt_{host}.json'
        evidence = read(evidence_path)
        require(evidence['status'] == 'passed' and evidence['host'] == host
                and evidence['contract_sha256'] == digest(old)
                and evidence['source_files_sha256'] == SOURCE_SHA, 'Old native evidence identity mismatch')
        c['runtime_reuse_evidence'][host] = {'receipt_path': str(evidence_path), 'receipt_sha256': sha(evidence_path),
            'recorded_runtime': evidence['runtime'], 'recorded_runtime_sha256': digest(evidence['runtime']),
            'role': 'prior native qualification evidence; new CNNGRU qualification still mandatory',
            'inherited_environment_must_remain_available': True}
    width_path = args.parent_bundle / 'width16_validation_refs.json'
    width = read(width_path)
    require(len(width['rows']) == 9 and len({(r['dataset'], r['seed']) for r in width['rows']}) == 9,
            'Nine unique frozen width16 references required')
    for row in width['rows']:
        for filename, path_key, sha_key in [('best_val_qty_rmse_model.pt', 'checkpoint_path', 'checkpoint_sha256'),
            ('endpoint_replays.json', 'endpoint_path', 'endpoint_sha256'), ('history.json', 'history_path', 'history_sha256'),
            ('terminal_manifest.json', 'terminal_manifest_path', 'terminal_manifest_sha256')]:
            local = verified_source(row[path_key], row[sha_key])
            target = f"inputs/references/width16/{row['dataset']}/{row['seed']}/{filename}"
            c['input_files'][target] = row[sha_key]
            deployment_inputs[target] = {'sha256': row[sha_key], 'local_source': local, 'source_kind': 'frozen_width16_original'}
    c['references'] = {'width16_registry_sha256': sha(width_path), 'seed42_registry_sha256': sha(args.anchor_registry),
        'width16_selected_checkpoints': 9, 'external_benchmark_requirement': {
            'family_count': 6, 'named_required_anchor': 'S2P2', 'other_five_aliases': 'bind from existing external registry before analysis'},
        'external_comparison_status': 'existing frozen validation bindings to be assembled separately; no training-time external evaluation'}
    c['reporting'].update(primary='unchanged full Validation raw RMSE selector; overall and tail RMSE/MAE/TimeNLL same selected epoch',
        candidate='CNN+GRU54 only; not proof of CNN/GRU interaction or independent recurrence benefit',
        benchmarks='same-split/seed/population frozen S2P2 three-seed, external five, MLP16 and strong internal references; missing evidence stays missing',
        independent_evaluation='Existing Test is development-exposed; no new Test or independent split creation in this campaign')
    c['transition'] = {'parent_campaign': old['schema'], 'parent_contract_sha256': digest(old),
        'original_33_condition_inventory': 'root supplies user-stopped inventory separately before deployment',
        'reason': 'user-directed prioritization of CNN+GRU54, not numerical failure',
        'preserve_completed_partial_and_unstarted': True, 'new_operation_mutates_old_campaign_artifacts': False}
    checks = validate_contract(c, old, parent)
    staged = stage_inputs(out, deployment_inputs)
    source_registry = {'schema': NAME + '_source_registry_v1', 'exact_source123': copy.deepcopy(c['source']),
        'inherited_parent117': copy.deepcopy(parent['source']), 'parent117_equal': True,
        'legacy_worker4': {**parent['worker'], 'role': 'legacy_reference_only', 'executed_by_new_operation': False},
        'operation': c['operation'], 'old_source_bundle': str(args.parent_bundle),
        'source_provenance': 'frozen123 copied exactly; parent117 unchanged; separate operations with own digest'}
    approval = {'approved': True, 'contract_sha256': digest(c), 'hosts': ['5080', '5090'],
        'user_instruction': '사용자 2026-10-05 직접 승인: 현재 GRU 대조 중지 후 CNN+GRU54로 새 3seed 검증',
        'authorization_evidence': 'root relay of direct current user authorization',
        'interpreted_concrete_scope': '6 fresh Train/Validation fits plus3 unchanged seed42 anchors; existing native runtimes; no new Test',
        'recorded_unix': args.issued_at}
    permit = {'contract_sha256': digest(c), 'approval_sha256': digest(approval),
              'started_at_unix': args.issued_at, 'deadline_unix': args.issued_at + 168 * 3600}
    deployment = {'schema': NAME + '_deployment_plan_v1', 'status': 'prepared_not_deployed',
        'contract_sha256': digest(c), 'source_sha256': SOURCE_SHA, 'source123_mode': 'copy_from_old_verified_snapshot_without_modification',
        'hosts': {h: {'old_root': old['hosts'][h]['root'], 'new_root': c['hosts'][h]['root'],
                      'source_root': c['hosts'][h]['source_root'], 'operation_root': c['hosts'][h]['operation_root'],
                      'runtime_reused': c['hosts'][h]['python']} for h in HOST_DATA},
        'source_root_input_files': deployment_inputs, 'root_operation_files': operation_files,
        'staged_additional_inputs': staged,
        'predeploy_requirements': ['old_user_stopped_inventory_and_exact_owner_exit', 'operation_source_freeze',
            'isolated_new_root', 'native_qualification', 'both_host_qualification_bound_training_permit'],
        'training_permit_created': False, 'remote_actions_performed': False}
    analysis = {'schema': NAME + '_analysis_v1', 'scientific_contract_sha256': digest(c),
        'evaluation_scope': 'full_validation_only', 'held_out_test_evaluated': False,
        'metrics': ['qty_rmse', 'qty_mae', 'time_nll', 'tail_qty_rmse', 'tail_qty_mae', 'tail_time_nll'],
        'seeds': [42, 52, 62], 'summary': 'per-dataset mean/sample SD and same-seed paired differences; no cross-unit raw RMSE aggregation',
        'benchmarks': 'reuse frozen external six including S2P2 plus strong internal MLP models; bind split/population/selector/likelihood before combining',
        'limitation': 'n3 descriptive, reused heterogeneous seed42; not a full factorial contribution experiment or independent held-out claim',
        'selector_changed': False, 'width16_registry_sha256': sha(width_path)}
    outputs = {'execution_contract.json': c, 'source_registry.json': source_registry, 'approval.json': approval,
        'start_permit.json': permit, 'deployment_metadata.json': deployment, 'analysis_contract.json': analysis,
        'current.json': {'campaign': NAME, 'canonical_sha256': digest(c), 'contract': 'execution_contract.json',
                         'status': 'prepared_not_deployed', 'training_started': False,
                         'source_closure_sha256': c['source']['files_sha256'],
                         'operation_closure_sha256': c['operation']['files_sha256'],
                         'hosts': {h: spec['root'] for h, spec in c['hosts'].items()}},
        'width16_validation_refs.json': width, 'contract_checks.json': {**checks, 'contract_sha256': digest(c),
                    'canonical_reencoding_stable': digest(json.loads(canonical(c))) == digest(c)}}
    for name, value in outputs.items():
        save(out / name, value)
    text = ('# CNN+GRU54 3seed 전환 계약\n\n상태: 로컬 준비 완료, 미배포·미학습.\n\n'
        '신규6조건: 5080 Taxi·RAF seed52/62, 5090 Intermittent seed52/62. '
        '기존 seed42 선택/마지막 checkpoint3쌍을 SHA와 원래 선택 epoch로 재사용합니다.\n\n'
        '기존123 source와 부모117 과학 코드는 그대로 유지합니다. A100 worker4는 legacy reference only이며 '
        '새 operation은 별도 SHA로 고정합니다. 공용 Runtime 패키지를 변경하지 않습니다.\n\n'
        '기존33조건은 사용자 방향 전환에 따른 중단으로 별도 보존합니다. 임의 재시작·Test·선택 변경은 없습니다. '
        '새 결과는 동일 Validation의 S2P2와 다른 외부/강한 내부 비교군에 연결하며, '
        '기존 Test 노출·seed42 이종GPU·n3 제한을 유지합니다.\n\n'
        '**배포·qualification·시작 — 다음 작업:** 부모가 기존 정확 소유 프로세스 종료 증거와 원본을 봉인하고, '
        '새 isolated root에 배포 후 양쪽 qualification과 training permit을 확인합니다. '
        '이 스크립트는 배포·원격 호출·학습·Test를 수행하지 않습니다.\n')
    readme = out / 'README.md'
    if readme.exists(): require(readme.read_text() == text, 'README freeze differs')
    else: readme.write_text(text)
    return {'status': 'prepared_not_deployed', 'contract_sha256': digest(c), 'contract': str(out / 'execution_contract.json'),
            'new_fits': 6, 'canonical_conditions': 9}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'contract')
    parser.add_argument('--parent-bundle', type=Path, default=OLD)
    parser.add_argument('--anchor-registry', type=Path, default=HERE / 'contract/reuse_seed42_registry.json')
    parser.add_argument('--issued-at', type=int, required=True, help='Explicit fixed Unix timestamp for preparation/start budget; reuse unchanged on rerun')
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False))

if __name__ == '__main__':
    main()
