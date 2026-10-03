"""Aggregate 24 audited structural controls and 12 reused MLP validation fits."""
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from collect import ROOT, REPORT, DEST, read, sha

METRICS = ('qty_mae', 'qty_rmse', 'time_nll')
STRUCTURAL = ('titantpp_current_only_param_matched', 'titantpp_all_available_history_mlp')
DATASETS = ('yellow_trip_hourly', 'intermittent_frozen_5000', 'raf_spare_parts', 'insta_market_basket')


def save(name, value):
    (REPORT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    scope = read(REPORT / 'scope.json')
    prior_path = Path(scope['prior_comparison'])
    assert sha(prior_path) == scope['prior_comparison_sha256']
    prior = read(prior_path)
    rows = [dict(x) for x in prior['completed'] if x['paper_structural_scope']]
    assert len(rows) == 21
    prior_source_hashes = {x['path']: x['sha256'] for x in prior['sources']}
    sources = []
    for path in sorted({x['audit'] for x in rows}):
        p = Path(path)
        assert sha(p) == prior_source_hashes[path]
        audit = read(p)
        assert audit['status'] == 'passed' and audit['originals_unchanged']
        receipt = read(p.parent / 'retrieval_receipt.json')
        assert sha(p.parent / 'original.tar') == receipt['archive_sha256'] == audit['archive_sha256']
        sources.append({'path': path, 'sha256': sha(p), 'reused': True})
    # Reuse verified native selected validation records, without new prediction.
    base = [dict(x) for x in prior['reused_MLP']]
    assert len(base) == 12
    for row in base:
        src = ROOT / row['source']
        assert sha(src) == row['source_sha256']
        endpoint = read(src)
        assert endpoint['selected']['evaluation_scope'] == 'validation_only'
        assert endpoint['selected']['held_out_test_evaluated'] is False
        assert endpoint['best_epoch'] == row['selected_epoch']
        for key in METRICS:
            assert endpoint['selected'][key] == row[key]
        row['host'] = '5090' if row['dataset'] == 'insta_market_basket' else '5080'
        row['binary_CPU_audit'] = 'reused_prior_baseline_audit'
    retrievals = []
    for host in ('5090', 'pro4500'):
        directory = DEST / host
        audit = read(directory / 'terminal_audit.json')
        receipt = read(directory / 'retrieval_receipt.json')
        assert audit['status'] == 'passed' and audit['originals_unchanged']
        assert sha(directory / 'original.tar') == receipt['archive_sha256'] == audit['archive_sha256']
        manifest = read(directory / 'original/collection_manifest.json')
        for rel, record in manifest['files'].items():
            assert sha(directory / 'original' / rel) == record['sha256']
        sources.append({'path': str(directory / 'terminal_audit.json'),
                        'sha256': sha(directory / 'terminal_audit.json'), 'reused': False})
        retrievals.append({'host': host, 'files': receipt['files'], 'bytes': receipt['bytes'],
                           'source_files': receipt['source_files'], 'conditions': len(audit['conditions']),
                           'checkpoints': audit['verified_terminal_checkpoints']})
        for condition in audit['conditions']:
            job = condition['job']
            rows.append({'campaign': 'extension36', 'dataset': job['dataset'],
                         'model': job['arm'], 'seed': job['seed'], 'host': host,
                         'job_id': job['id'], 'training_state': 'complete',
                         'binary_CPU_audit': 'passed', 'selected_epoch': condition['selected_epoch'],
                         'completed_epochs': condition['completed_epochs'], **condition['selected_validation'],
                         'audit': str(directory / 'terminal_audit.json'),
                         'original_job_path': str(directory / 'original/run' / job['id']),
                         'contract_sha256': audit['contract_sha256'],
                         'source_closure_sha256': audit['source_closure_sha256'],
                         'reused_prior13_audit': False, 'paper_structural_scope': True})
    assert len(rows) == len({r['job_id'] for r in rows}) == 24
    groups = defaultdict(list)
    for row in base + rows:
        groups[row['dataset'], row['model']].append(row)
    aggregates = []
    for dataset in DATASETS:
        for model in ('titantpp_history_mlp', *STRUCTURAL):
            group = sorted(groups[dataset, model], key=lambda x: x['seed'])
            assert [x['seed'] for x in group] == [42, 52, 62]
            aggregate = {'dataset': dataset, 'model': model, 'seeds': [42, 52, 62],
                         'three_seed_complete': True,
                         'mean': {k: statistics.mean(x[k] for x in group) for k in METRICS},
                         'sample_sd': {k: statistics.stdev(x[k] for x in group) for k in METRICS},
                         'hosts_by_seed': {str(x['seed']): x['host'] for x in group}}
            if model != 'titantpp_history_mlp':
                baselines = sorted(groups[dataset, 'titantpp_history_mlp'], key=lambda x: x['seed'])
                aggregate['candidate_lower_paired_count'] = {k: sum(x[k] < b[k] for x, b in zip(group, baselines)) for k in METRICS}
                aggregate['candidate_vs_MLP_percent'] = {k: (aggregate['mean'][k] / statistics.mean(b[k] for b in baselines) - 1) * 100 for k in METRICS}
                aggregate['MLP_vs_candidate_percent'] = {k: (statistics.mean(b[k] for b in baselines) / aggregate['mean'][k] - 1) * 100 for k in METRICS}
                aggregate['paired'] = [{'seed': x['seed'], 'candidate_host': x['host'], 'MLP_host': b['host'],
                                        'candidate': {k: x[k] for k in METRICS}, 'MLP': {k: b[k] for k in METRICS},
                                        'candidate_vs_MLP_percent': {k: (x[k] / b[k] - 1) * 100 for k in METRICS}}
                                       for x, b in zip(group, baselines)]
            aggregates.append(aggregate)
    result = {'evaluation_scope': 'validation_only', 'created_unix': time.time(),
              'scope_sha256': sha(REPORT / 'scope.json'), 'completed': rows, 'reused_MLP': base,
              'groups': aggregates, 'sources': sources,
              'counts': {'newly_audited_structural_conditions': 3, 'new_checkpoints': 6,
                         'reused_structural_conditions': 21, 'structural_controls_audited': 24,
                         'reused_MLP_conditions': 12, 'three_seed_groups': 12},
              'aggregation': 'arithmetic mean and sample SD (ddof=1) across seeds 42,52,62; no pooling of endpoints',
              'percent_definition': '(candidate_mean / MLP_mean - 1) * 100; negative means lower error',
              'metric_selection': 'MAE, RMSE and time NLL at the same first minimum raw validation quantity RMSE epoch',
              'hardware_caveat': 'Intermittent structural-control seed62 used PRO4500; seeds42/52 and original MLP used 5080. No efficiency comparison or causal GPU attribution.',
              'not_claimed': ['full extension36 CPU audit completion', 'A100 CPU audit completion', 'independent held-out generalization', 'statistical significance from n=3']}
    save('comparison.json', result)
    fields = ['dataset', 'model', 'seed', 'host', 'selected_epoch', 'completed_epochs', *METRICS,
              'binary_CPU_audit', 'audit', 'original_job_path', 'source', 'source_sha256']
    with (REPORT / 'condition_registry.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(base + rows, key=lambda x: (DATASETS.index(x['dataset']), x['model'], x['seed'])))
    save('verification.json', {'status': 'passed', 'verified_unix': time.time(), 'retrievals': retrievals,
                              'new_conditions': 3, 'new_checkpoints': 6, 'reused_structural_conditions': 21,
                              'reused_baseline_conditions': 12, 'new_training_or_forward_calls': 0,
                              'new_replay_calls': 0, 'heldout_read': False, 'sources': sources})
    names = {'yellow_trip_hourly': 'Taxi', 'intermittent_frozen_5000': 'Intermittent',
             'raf_spare_parts': 'RAF', 'insta_market_basket': 'Instacart',
             'titantpp_history_mlp': 'Original MLP', 'titantpp_current_only_param_matched': 'Current-only',
             'titantpp_all_available_history_mlp': 'All-available'}
    lines = ['# 최종 구조 대조 원본 감사와 3seed 집계', '',
             '평가 범위는 validation-only다. 새 구조 대조 3조건/6 checkpoint가 감사에 통과했고 기존 구조 대조 21조건과 원래 MLP 12조건을 재사용했다. 4개 데이터 × 3개 모델 × 3seed의 36개 행이다. 이는 원래 후속 36조건 캠페인 전체 감사 완료를 뜻하지 않는다.', '',
             '| 데이터 | 모델 | MAE 평균 ± 표본 SD | RMSE 평균 ± 표본 SD | 시간 NLL 평균 ± 표본 SD |',
             '|---|---|---:|---:|---:|']
    for a in aggregates:
        cells = [f"{a['mean'][k]:.6f} ± {a['sample_sd'][k]:.6f}" for k in METRICS]
        lines.append('| ' + ' | '.join([names[a['dataset']], names[a['model']], *cells]) + ' |')
    lines += ['', '## 해석 범위', '',
              '- 각 seed의 최초 최소 raw 수량 RMSE epoch에서 MAE·RMSE·시간 NLL을 함께 읽었다. 평균은 seed별 지표의 산술평균이며 표준편차는 ddof=1이다.',
              '- Intermittent 구조 대조 seed62는 PRO4500, seed42/52 및 원래 MLP는 5080이다. 이종 GPU 출처를 유지하고 효율 측정에 혼합하지 않는다.',
              '- 학습·forward·validation replay를 새로 실행하지 않았다. strict CPU 모델 로드, optimizer/RNG/shuffle 복원, frozen source109 및 원본 SHA, 선택 epoch, 저장 지표, selected/last replay와 exposure를 검증했다.',
              '- Deep Renewal·A100의 추가 CPU 감사와 독립적인 최종 일반화 평가는 이 작업의 완료 주장에 포함하지 않는다.', '',
              '## 재현', '', '프로젝트 root에서 새 scope에 해당하는 최초 회수는 `collect.py 5090` 및 `collect_local_pro4500.py`, 감사는 `audit.py 5090`과 `audit.py pro4500`, 집계는 `summarize.py`를 사용했다. 수집·감사 스크립트는 기존 완료 결과를 덮어쓰지 않는다. 삭제된 PRO4500 Pod에 접속하지 않았다.', '',
              '원본과 감사 경로, contract/source SHA, seed별 짝 비교와 방향별 변화율은 `comparison.json`, 36개 조건별 지표는 `condition_registry.csv`, 감사 증거는 `verification.json`에 보존한다.']
    (REPORT / 'report.md').write_text('\n'.join(lines) + '\n')
    for a in aggregates:
        print(a['dataset'], a['model'], a['mean'], a['sample_sd'], a.get('candidate_vs_MLP_percent'), a.get('candidate_lower_paired_count'))


if __name__ == '__main__':
    main()
