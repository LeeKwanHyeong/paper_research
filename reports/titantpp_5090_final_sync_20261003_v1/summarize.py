"""Link all nine 5090 terminal fits to original files and passed CPU audits.

Rehashes existing artifacts only; performs no model forward/evaluation or network
operation. Older CPU audits are reused, not represented as newly executed.
"""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
BASE = ROOT / 'search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2'
SCOPE = json.loads((REPORT / 'scope.json').read_text())
EXPECTED = SCOPE['hosts']['5090']['contract_sha256']
CLOSURE = SCOPE['hosts']['5090']['source_closure_sha256']
AUDITS = [
    BASE / 'retrieved/completed13_20261002_v1/5090/terminal_audit.json',
    ROOT / 'search_artifacts/titantpp_completed22_audit_20261002_v1/retrieved/5090/terminal_audit.json',
    ROOT / 'search_artifacts/titantpp_structural3_final_audit_20261003_v1/retrieved/5090/terminal_audit.json',
    ROOT / 'search_artifacts/titantpp_5090_final_sync_20261003_v1/retrieved/5090/terminal_audit.json',
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    contract = read(BASE / 'execution_contract.json')
    assert canonical(contract) == EXPECTED
    assert canonical(contract['source']['files']) == CLOSURE
    snapshot_path = Path(SCOPE['hosts']['5090']['snapshot'])
    assert sha(snapshot_path) == SCOPE['hosts']['5090']['snapshot_sha256']
    snapshot = read(snapshot_path)
    server = snapshot['files']['status.json']
    assert server['status'] == 'complete' and server['completed_conditions'] == 9
    rows, reused_evidence = [], []
    for audit_path in AUDITS:
        audit = read(audit_path)
        assert audit['status'] == 'passed' and audit['host'] == '5090'
        assert audit['evaluation_scope'] == 'validation_only'
        assert audit['contract_sha256'] == EXPECTED and audit['source_closure_sha256'] == CLOSURE
        assert audit['new_model_forward_calls'] == audit['new_training_updates'] == audit['new_replay_calls'] == 0
        original = audit_path.parent / 'original'
        collection = read(original / 'collection_manifest.json')
        retrieval = read(audit_path.parent / 'retrieval_receipt.json')
        assert sha(audit_path.parent / 'original.tar') == audit['archive_sha256'] == retrieval['archive_sha256']
        for relative, evidence in collection['files'].items():
            file = original / relative
            assert file.is_file() and not file.is_symlink()
            assert sha(file) == evidence['sha256'] and file.stat().st_size == evidence['bytes']
        assert canonical(read(original / 'execution_contract.json')) == EXPECTED
        for relative, digest in contract['source']['files'].items():
            assert sha(original / 'source' / relative) == digest
        is_new = audit_path == AUDITS[-1]
        reused_evidence.append({
            'audit_path': str(audit_path), 'audit_sha256': sha(audit_path),
            'original_root': str(original), 'archive_sha256': audit['archive_sha256'],
            'collection_manifest_sha256': sha(original / 'collection_manifest.json'),
            'retrieval_receipt_sha256': sha(audit_path.parent / 'retrieval_receipt.json'),
            'original_files_rehashed': len(collection['files']),
            'cpu_audit_executed_this_request': is_new,
            'condition_count': len(audit['conditions']),
        })
        for condition in audit['conditions']:
            job = condition['job']
            assert job in contract['jobs'] and job['host'] == '5090' and condition['status'] == 'passed'
            job_root = original / 'run' / job['id']
            manifest_path = job_root / 'terminal_manifest.json'
            manifest = read(manifest_path)
            assert manifest['scientific_success'] and manifest['status'] == 'complete'
            assert sha(manifest_path) == server['completed'][job['id']]['terminal_manifest_sha256']
            checkpoint_paths = {}
            for relative, digest in manifest['files'].items():
                file = job_root / relative
                assert sha(file) == digest
                if file.suffix == '.pt':
                    assert sha(file) == condition['checkpoint_files'][file.name]
                    checkpoint_paths[file.name] = str(file)
            assert len(checkpoint_paths) == 2
            rows.append({
                'job': job, 'training_terminal': 'complete', 'scientific_success': True,
                'original_sync': 'verified', 'cpu_binary_audit': 'passed',
                'cpu_audit_mode': 'new' if is_new else 'reused_with_current_file_rehash',
                'completed_epochs': condition['completed_epochs'],
                'selected_epoch': condition['selected_epoch'],
                'selected_validation': condition['selected_validation'],
                'audit_path': str(audit_path), 'audit_sha256': sha(audit_path),
                'original_root': str(job_root), 'terminal_manifest_sha256': sha(manifest_path),
                'checkpoint_paths': checkpoint_paths, 'checkpoint_sha256': condition['checkpoint_files'],
                'model_tensor_sha256': condition['model_tensor_sha256'],
                'validation_replay_records_verified': condition['replay_records_audited_without_rerun'],
            })
    expected = {job['id'] for job in contract['jobs'] if job['host'] == '5090'}
    assert len(rows) == 9 and {row['job']['id'] for row in rows} == expected
    rows.sort(key=lambda row: (row['job']['seed'], contract['arms'].index(row['job']['arm'])))
    now = dt.datetime.now(dt.timezone.utc)
    result = {
        'status': 'passed', 'created_utc': now.isoformat(), 'evaluation_scope': 'validation_only',
        'contract_sha256': EXPECTED, 'source_closure_sha256': CLOSURE,
        'server_final_snapshot': str(snapshot_path), 'server_final_snapshot_sha256': sha(snapshot_path),
        'counts': {'completed': 9, 'failed': 0, 'running': 0, 'unstarted': 0,
                   'original_sync_verified': 9, 'cpu_binary_audit_passed': 9,
                   'checkpoints_verified': 18, 'replay_roles_verified': 18,
                   'newly_collected_conditions': 1, 'reused_conditions': 8},
        'new_retrieval': read(AUDITS[-1].parent / 'retrieval_receipt.json'),
        'evidence_bundles': reused_evidence, 'conditions': rows,
        'new_training_updates': 0, 'new_model_forward_calls': 0, 'new_replay_calls': 0,
        'remote_writes': False,
        'limitations': [
            'Saved validation records and CPU checkpoint state audit; no new validation or held-out/test prediction.',
            'The first eight CPU audits are reused after current original-file, checkpoint and archive SHA rechecks.',
            'The final condition uses the frozen 109-file source on the local CPU runtime, not the remote CUDA runtime.',
            'This report completes 5090 only. Remaining old RunPod CPU audits are outside this task.',
            'Deep Renewal remains a retained research record and is not added back to manuscript comparisons.',
        ],
    }
    with (REPORT / 'registry.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    with (REPORT / 'condition_registry.csv').open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'arm', 'seed', 'completed_epochs', 'selected_epoch', 'training_terminal',
                         'original_sync', 'cpu_binary_audit', 'cpu_audit_mode', 'val_qty_mae', 'val_qty_rmse',
                         'val_time_nll', 'audit_path', 'original_root'])
        for row in rows:
            writer.writerow([row['job']['dataset'], row['job']['arm'], row['job']['seed'],
                             row['completed_epochs'], row['selected_epoch'], row['training_terminal'],
                             row['original_sync'], row['cpu_binary_audit'], row['cpu_audit_mode'],
                             *[row['selected_validation'][key] for key in ('qty_mae', 'qty_rmse', 'time_nll')],
                             row['audit_path'], row['original_root']])
    names = {'titantpp_current_only_param_matched': 'Current-only',
             'titantpp_all_available_history_mlp': 'All-available',
             'deep_renewal_event_native_nb': 'Deep Renewal'}
    lines = [
        '# 5090 후속 실험 원본 동기화·CPU 감사 완료', '',
        f"작성: {now.astimezone(dt.timezone(dt.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')}", '',
        'Instacart 9/9조건의 학습 완료, 원본 동기화, CPU checkpoint 감사를 모두 확인했다. '
        '이번에는 마지막 Deep Renewal seed62의 원본 130파일(동결 source109 포함)·checkpoint 2개를 새로 회수하고 감사했다. '
        '기존 8조건은 이전 감사 결과를 재사용하고 현재 보존 원본·archive·checkpoint SHA를 재확인했다. 총 18개 checkpoint와 18개 selected/last validation replay 기록이 연결된다.', '',
        '| 모델 | Seed | 종료 / 선택 epoch | 학습 | 원본 | CPU 감사 |',
        '|---|---:|---:|---|---|---|',
    ]
    for row in rows:
        lines.append(f"| {names[row['job']['arm']]} | {row['job']['seed']} | {row['completed_epochs']} / {row['selected_epoch']} | 완료 | SHA 검증 | 통과 ({'신규' if row['cpu_audit_mode']=='new' else '기존 감사 재사용'}) |")
    last = next(row for row in rows if row['cpu_audit_mode'] == 'new')
    values = last['selected_validation']
    lines.extend(['',
        f"마지막 조건의 선택 epoch46 validation은 MAE {values['qty_mae']:.9f}, RMSE {values['qty_rmse']:.9f}, 시간 NLL {values['time_nll']:.9f}다. 동일 선택 epoch의 세 지표이며 test 결과가 아니다.", '',
        'CPU 감사는 파일·tensor SHA, strict 모델 로드, optimizer/RNG 복원, source·초기화·선택 epoch·전체 train/validation exposure와 저장된 재평가 기록을 대조했다. '
        '모델 forward, 새 학습, 새 replay, held-out/test 열람은 수행하지 않았다. 원격 읽기 전용 수집만 수행했으며 서버 Runtime과 프로세스는 변경하지 않았다.', '',
        'Deep Renewal은 원고 비교에서 제외된 상태를 유지한다. 이번 작업은 연구 원본 보존과 감사 완료이며 논문 주장을 바꾸지 않는다. '
        '이전 RunPod의 별도 CPU 감사 미완료 항목이나 신규 A100 routing/placement 실험에는 적용하지 않는다.', '',
        f'- 원계약 canonical SHA: `{EXPECTED}`', f'- 동결 source109 closure SHA: `{CLOSURE}`',
        f'- [신규 회수 영수증]({AUDITS[-1].parent / "retrieval_receipt.json"})',
        f'- [신규 CPU 감사]({AUDITS[-1]})',
        f'- [9조건 원본·감사 연결 목록]({REPORT / "registry.json"})',
        f'- [조건별 CSV]({REPORT / "condition_registry.csv"})', '',
        '**남은 작업**: 5090 동기화 범위에는 미완료 항목이 없다. 기존 RunPod의 별도 미완료 CPU 감사는 별도 작업으로 남는다.',
    ])
    with (REPORT / 'FINAL_REPORT.md').open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'passed', 'counts': result['counts'], 'report': str(REPORT / 'FINAL_REPORT.md')}))


if __name__ == '__main__':
    main()
