"""Bind 84 unchanged original selections plus 24 structural controls to local files.

All original checkpoints and prior audits stay in place. No network, forward pass,
data loader, held-out outcome, or mixed metric file is used by this script.
"""
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT / 'reports/titantpp_independent_evaluation_preparation_20261001_v1'
DESIGN = ROOT / 'reports/titantpp_independent_final_evaluation_design_20261003_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def key(row):
    return row['dataset'], row['model'], row['seed']


def main():
    scope = read(OUT / 'scope.json')
    for p, digest in scope['sources'].items():
        assert sha(ROOT / p) == digest, p
    old = read(PRIOR / 'evaluation_registry.json')
    original = read(DESIGN / 'checkpoint_manifest.json')
    structural = read(DESIGN / 'structural_checkpoint_manifest.json')
    fresh = read(OUT / 'checkpoint_audit.json')
    assert fresh['status'] == 'passed' and fresh['selected_checkpoints'] == 36
    missing_keys = {key(r) for r in original['rows'] if not r['current_local_path_present']}
    assert missing_keys == {key(r) for r in fresh['rows']}
    assert len(missing_keys) == 36
    previous = {key(r): r for r in old['rows']}
    original_by_key = {key(r): r for r in original['rows']}
    audits = {key(r): r for r in fresh['rows']}
    # Verify both prior local retrieval archives and every byte in their manifests.
    archive_checks = []
    for host in ('5080', '5090'):
        base = ROOT / 'search_artifacts/titantpp_independent_evaluation_preparation_20261001_v1' / host / 'attempt1'
        receipt = read(base / 'retrieval_receipt.json')
        assert sha(base / 'original.tar') == receipt['archive_sha256']
        manifest = read(base / 'original/collection_manifest.json')
        for name, rec in manifest['files'].items():
            assert sha(base / 'original' / name) == rec['sha256'], name
        archive_checks.append({'host': host, 'archive': rel(base / 'original.tar'),
                               'archive_sha256': receipt['archive_sha256'],
                               'files_verified': len(manifest['files']), 'bytes': receipt['bytes'],
                               'new_remote_calls': 0})
    bundles = {}

    def add_bundle(name, contract_path, source_root, builder):
        c = read(contract_path)
        assert canonical(c['source']['files']) == c['source']['files_sha256']
        workspace = OUT / 'evaluator_sources' / name
        for path, digest in c['source']['files'].items():
            src = source_root / path
            assert sha(src) == digest
            dst = workspace / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(src, dst)
            assert sha(dst) == digest
        (workspace / 'sample_data').mkdir(exist_ok=True)
        bundles[name] = {'contract_path': rel(contract_path), 'canonical_sha256': canonical(c),
                         'source_root': rel(workspace), 'source_files': c['source']['files'],
                         'source_closure_sha256': c['source']['files_sha256'], 'datasets': c['datasets'],
                         'selected_weight_audit_builder_module': builder,
                         'factory_module': 'models.TPPs.CountAwareFactory',
                         'factory_function': 'build_count_aware_model',
                         'factory_config': 'data.model excluding backbone/lambda_log_qty/lambda_tail/time_head_lr_multiplier, plus train_log_mean/train_log_std/max_seq_len',
                         'original_source_root': rel(source_root)}
        return bundles[name]

    for name, b in old['bundles'].items():
        add_bundle(name, ROOT / b['contract_path'], ROOT / b['source_root'],
                   {'core': 'paper.scripts.run_titantpp_core_ablation',
                    'additional': 'paper.scripts.run_additional_tpp',
                    'raf': None}.get(name, 'paper.scripts.run_local_detail_benchmark'))
    rows = []
    aliases = []
    for k, before in original_by_key.items():
        row = dict(previous[k])
        for identity in ('selected_epoch', 'state_tensor_sha256', 'contract_canonical_sha256', 'selection'):
            assert row[identity] == before[identity], (k, identity)
        cp = ROOT / row['checkpoint_path']
        assert cp.is_file() and sha(cp) == row['checkpoint_file_sha256']
        bundle = bundles[row['evaluator_source_bundle']]
        assert bundle['canonical_sha256'] == row['contract_canonical_sha256']
        row['source_closure_sha256'] = bundle['source_closure_sha256']
        row['current_local_path_present'] = True
        row['current_matches_recorded_sha256'] = True
        row['current_file_sha256'] = row['checkpoint_file_sha256']
        row['current_byte_check_utc'] = datetime.now(timezone.utc).isoformat()
        row['campaign'] = 'original84'
        row['frozen_source_directory'] = bundle['source_root']
        if k in audits:
            audit = audits[k]
            for identity in ('checkpoint_path', 'checkpoint_file_sha256', 'state_tensor_sha256', 'selected_epoch'):
                assert row[identity] == audit[identity]
            row['binary_status'] = 'fresh_CPU_strict_load_of_previously_retrieved_original'
            row['current_audit_path'] = rel(OUT / 'checkpoint_audit.json')
            row['current_audit_sha256'] = sha(OUT / 'checkpoint_audit.json')
            aliases.append({'dataset': k[0], 'model': k[1], 'seed': k[2],
                            'old_missing_local_path': before['checkpoint_path'],
                            'verified_existing_local_path': row['checkpoint_path'],
                            'checkpoint_file_sha256': row['checkpoint_file_sha256'],
                            'state_tensor_sha256': row['state_tensor_sha256'],
                            'resolution': 'already_retrieved_and_audited_on_20261001; stale local alias replaced'})
        else:
            prior_audit = ROOT / row['audit_path']
            assert prior_audit.is_file()
            row['current_audit_path'] = rel(prior_audit)
            row['current_audit_sha256'] = sha(prior_audit)
        rows.append(row)
    for before in structural['rows']:
        row = dict(before)
        name = {'5080': 'extension5080', '5090': 'extension5090', 'pro4500': 'extensionPRO4500'}[row['host']]
        source = ROOT / row['frozen_source_directory']
        if name not in bundles:
            add_bundle(name, source.parent / 'execution_contract.json', source,
                       'paper.scripts.run_pakdd_extension')
        b = bundles[name]
        assert b['canonical_sha256'] == row['contract_canonical_sha256']
        assert b['source_closure_sha256'] == row['source_closure_sha256']
        assert sha(ROOT / row['checkpoint_path']) == row['checkpoint_file_sha256']
        audit_path = ROOT / row['prior_cpu_audit_reused']
        assert sha(audit_path) == row['audit_sha256']
        a = read(audit_path)
        matched = next(x for x in a['conditions'] if (x['job']['dataset'], x['job']['arm'], x['job']['seed']) == key(row))
        assert matched['status'] == 'passed' and matched['strict_cpu_loading']
        assert matched['selected_epoch'] == row['selected_epoch']
        assert matched['model_tensor_sha256']['selected'] == row['state_tensor_sha256']
        assert matched['checkpoint_files']['best_val_qty_rmse_model.pt'] == row['checkpoint_file_sha256']
        row.update({'evaluator_source_bundle': name, 'campaign': 'structural24',
                    'frozen_source_directory': b['source_root'], 'local_binary_present': True,
                    'current_local_path_present': True, 'current_matches_recorded_sha256': True,
                    'current_file_sha256': row['checkpoint_file_sha256'],
                    'binary_status': 'prior_CPU_audit_reused_and_current_bytes_verified',
                    'current_audit_path': rel(audit_path), 'current_audit_sha256': sha(audit_path),
                    'validation_replay_path': row['selected_validation_replay_path']})
        rows.append(row)
    assert len(rows) == len({key(r) for r in rows}) == 108
    result = {'status': 'all_108_selected_binary_identities_verified', 'rows': rows, 'bundles': bundles,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'scope_sha256': sha(OUT / 'scope.json'),
              'execution_scope': 'validation_preparation_only', 'heldout_execution_authorized': False,
              'summary': {'original_conditions': 84, 'structural_conditions': 24, 'selected_conditions': 108,
                          'fresh_CPU_selected_loads': 36, 'prior_CPU_audits_reused': 72,
                          'stale_local_aliases_resolved': 36, 'still_missing': 0,
                          'frozen_source_bundles': len(bundles), 'new_remote_calls': 0,
                          'new_forward_calls': 0, 'heldout_read': False},
              'source_identity_note': 'Do not import current project model code. Use each frozen bundle in an isolated process.',
              'audit_scope': '36 legacy selected weights only; no last/optimizer binary audit was added. The other72 prior audits are reused.',
              'hardware_caveat': 'Intermittent structural seed62 uses PRO4500; retain provenance and do not mix with efficiency measurements.'}
    save('evaluation_registry.json', result)
    save('checkpoint_manifest.json', result)
    save('path_resolutions.json', {'rows': aliases, 'reason': 'design manifest retained pre-retrieval paths'})
    save('verification.json', {'status': 'passed', 'archive_checks': archive_checks,
                              'all108_checkpoint_file_SHA_verified': True, 'selected_identities_unchanged': True,
                              'all_frozen_source_file_SHA_verified': True, 'sources': scope['sources'],
                              'evaluation_registry_sha256': sha(OUT / 'evaluation_registry.json'), **result['summary']})
    cols = ['campaign', 'dataset', 'model', 'seed', 'selected_epoch', 'checkpoint_path',
            'checkpoint_file_sha256', 'state_tensor_sha256', 'evaluator_source_bundle',
            'contract_canonical_sha256', 'source_closure_sha256', 'current_audit_path', 'binary_status']
    with (OUT / 'condition_registry.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'report.md').write_text('''# 최종 평가 checkpoint 원본 연결

원래84조건과 구조대조24조건의 selected checkpoint 총108개를 모두 로컬 원본·파일SHA·tensorSHA·선택epoch·동결소스·계약에 연결했다.

설계 manifest의 누락36개는 실제 유실이 아니었다. 2026-10-01 준비 작업에서 이미 회수·감사된 원본이 다른 로컬 경로에 있었으며, 이번에는 두 보존 archive와755개파일을 SHA 재확인하고36개 selected weight를 원래6개 동결소스로 CPU strict-load 재검증했다. 네 모델(RMTPP·THP·NHP·SAHP) × Taxi·Intermittent·Instacart ×3seed에 해당한다. 원격 호출·추가 다운로드는0회다.

기존48개 원래모델과24개 구조대조는 이전 CPU 감사 및 현재 원본 파일SHA를 재사용했다.36개 재감사는 selected weight/tensor/route/epoch/validation 선택과 기록된exposure 범위이며, 없는 optimizer/last-state를 검증했다고 주장하지 않는다. 신규forward·학습·재평가·held-out 열람은0회다.

실행기는 `evaluation_registry.json`의 각 row `evaluator_source_bundle`을 `bundles`에 연결해 `contract_path`, `source_root`, `source_files`, `source_closure_sha256`, `datasets`, `factory_module`, `factory_function`을 사용한다. 공통 CountAwareFactory에 계약의 model 설정과 train통계 및 max_seq_len을 적용한다. 모델별 원래 selected epoch와 tensor는 변경되지 않았다. 서로 다른 frozen source는 별도 프로세스로 import한다. 모든 path는 프로젝트 root 기준이다.

`path_resolutions.json`은 낡은36개 alias와 보존원본 경로의 대응표이고, `checkpoint_audit.json`은 이번36개 CPU감사, `verification.json`은 archive·SHA 연결 검증이다. `condition_registry.csv`는108개 행을 제공한다.

이 연결 완료는 평가자료의 독립성 확립이나held-out 실행승인을 뜻하지 않는다. 원본데이터 적격성 검토와 공통평가실행기 검증은 별도 작업이다.
''')
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
