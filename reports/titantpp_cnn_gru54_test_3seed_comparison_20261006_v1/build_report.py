"""Compare sealed nine-condition Test evidence with frozen external/internal references."""
from pathlib import Path
import csv
import importlib.util
import json
import math
import statistics
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUNDLE = ROOT / 'search_artifacts/titantpp_cnn_gru54_test_3seed_20261006_v1'
BASE = ROOT / 'reports/titantpp_three_dataset_final_comparison_20261005_v1'
CANDIDATE = 'titantpp_cnn_gru54'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows):
    require(bool(rows), 'Empty output CSV')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def run():
    # No baseline Test or mixed CSV is read before this sealed replay gate exists.
    e = module(BUNDLE / 'operation/evaluate_titantpp_cnn_gru_frozen_test.py', 'test3seed_evidence_audit')
    c = e.read(BUNDLE / 'execution_contract.json')
    reg = e.read(BUNDLE / 'evaluation_registry.json')
    retrieval = e.read(BUNDLE / 'retrieved/retrieval_receipt.json')
    require(retrieval['contract_sha256'] == e.sha(BUNDLE / 'execution_contract.json'), 'Retrieval contract SHA changed')
    require(e.sha(BUNDLE / 'retrieved/evidence.tar.gz') == retrieval['archive_sha256'], 'Retrieval archive SHA changed')
    for relative, digest in retrieval['files_SHA'].items():
        require(e.sha(e.confined(BUNDLE / 'retrieved', relative)) == digest, 'Retrieved file SHA changed: ' + relative)
    terminal = e.read(HERE / 'native_terminal_observation.json')
    exit_receipt = e.read(BUNDLE / 'retrieved/supervisor_process_exit.json')
    require(terminal['contract_sha256'] == retrieval['contract_sha256'] == exit_receipt['contract_sha256'], 'Terminal contract changed')
    require(terminal['root'] == c['resources']['root'] and terminal['gpu_uuid'] == c['resources']['gpu_uuid'], 'Terminal root/GPU changed')
    require(terminal['pipeline_status'] == 'complete' and terminal['completed_population_evaluations'] == 18, 'Native pipeline is incomplete')
    require(terminal['owned_processes'] == [] and terminal['gpu_processes'].strip() == '', 'Owned native processes remain')
    require(terminal['supervisor_exit'] == exit_receipt and exit_receipt['returncode'] == 0, 'Actual supervisor did not exit successfully')
    require(c['approval']['test_inference_authorized'] is True and c['approval']['retraining_authorized'] is False, 'Test authorization changed')
    require(e.sha(e.confined(BUNDLE, c['approval']['source_approval_path'])) == c['approval']['source_approval_sha256'], 'Test approval evidence SHA changed')
    gate = e.read(BUNDLE / 'retrieved/validation_gate.json')
    require(gate['passed'] is True and gate['conditions'] == 9, 'Fresh Validation9 gate did not pass')
    require(gate['contract_sha256'] == retrieval['contract_sha256'], 'Validation gate contract changed')
    require(len(gate['rows']) == 9, 'Validation gate must contain nine conditions')
    val_entries = {r['condition']: r for r in gate['rows']}
    require(set(val_entries) == {e.identity(r) for r in reg['rows']}, 'Validation gate identities changed')
    for row in reg['rows']:
        relative = f'runs/validation/{e.identity(row)}/receipt.json'
        path = BUNDLE / 'retrieved' / relative
        entry = val_entries[e.identity(row)]
        checked = e.check_receipt(path, row, 'validation', c, BUNDLE)
        require(entry['split'] == 'validation' and entry['receipt_path'] == relative, 'Validation receipt path/split changed')
        require(entry['receipt_sha256'] == e.sha(path) and entry['metrics'] == checked['metrics'], 'Validation receipt SHA/metrics changed')
    # All local Validation evidence is verified before opening Test-containing completion.
    completion = e.read(BUNDLE / 'retrieved/inference_completion.json')
    require(completion['contract_sha256'] == retrieval['contract_sha256'], 'Completion contract changed')
    require(completion['status'] == 'complete' and completion['conditions'] == 9, 'Test evaluation did not complete nine conditions')
    require(completion['full_population_splits'] == 18, 'Expected nine Validation and nine Test populations')
    require(all(completion[k] is False for k in ['new_training', 'selection_changed', 'raw_predictions_written', 'automatic_retry']), 'Frozen evaluation scope changed')
    require(len(gate['rows']) == len(completion['records']) == len(reg['rows']) == 9, 'Completion condition coverage changed')
    native_entries = {(r['condition'], r['split']): r for r in completion['completed']}
    require(len(native_entries) == 18, 'Duplicate/missing native split receipts')
    candidate_rows, cell_rows = [], []
    for row in reg['rows']:
        for split in ['validation', 'test']:
            condition = e.identity(row)
            relative = f'runs/{split}/{condition}/receipt.json'
            path = BUNDLE / 'retrieved' / relative
            checked = e.check_receipt(path, row, split, c, BUNDLE)
            entry = native_entries[(condition, split)]
            require(entry['receipt_path'] == relative and entry['receipt_sha256'] == e.sha(path), 'Native receipt path/SHA changed')
            require(checked['metrics'] == entry['metrics'], 'Native receipt metrics differ')
            if split == 'validation':
                require(entry in gate['rows'], 'Validation record differs from passed gate')
            receipt = e.read(path)
            for cell in receipt['metrics']['quantity_cells']:
                bounds = receipt['metrics']['quantity_boundaries']
                bin_id = cell['bin']
                cell_rows.append({'dataset': row['dataset'], 'model': CANDIDATE,
                    'seed': row['seed'], 'split': split, 'bin': bin_id,
                    'lower_exclusive': bounds[bin_id - 1] if bin_id else None,
                    'upper_inclusive': bounds[bin_id] if bin_id < len(bounds) else None,
                    'count': cell['count'], 'time_nll': cell['time_nll'],
                    'time_nll_sum': cell['time_nll_sum'],
                    'overall_mean_contribution': cell['time_nll_sum'] / receipt['metrics']['count'],
                    'NLL_sum_share_percent': 100 * cell['time_nll_sum'] / receipt['metrics']['time_nll_sum'],
                    'receipt_path': str(path.relative_to(ROOT)), 'receipt_sha256': e.sha(path)})
            for view in ['overall', 'tail']:
                m = receipt['metrics'] if view == 'overall' else receipt['metrics']['tail']
                candidate_rows.append({'dataset': row['dataset'], 'model': CANDIDATE,
                    'seed': row['seed'], 'split': split, 'view': view, 'count': m['count'],
                    'threshold': e.TAILS[row['dataset']] if view == 'tail' else None,
                    **{k: m[k] for k in e.METRICS}, 'selected_epoch': row['selected_epoch'],
                    'provenance': 'fresh_full_population_same_checkpoint_replay_SHA_verified',
                    'source': str(path.relative_to(ROOT)),
                    'source_sha256': e.sha(path),
                    'receipt_path': str(path.relative_to(ROOT)), 'receipt_sha256': e.sha(path),
                    'comparison_family': 'cnn_gru54_quantity_selected_frozen_3seed',
                    'time_nll_definition': 'recorded_positive_integer_lognormal_round_clamp_mass_nll',
                    'threshold_rule': 'raw_quantity > threshold' if view == 'tail' else None})
    # Existing reference collector validates all SHA, selected epoch and same population bindings.
    base = module(BASE / 'build_report.py', 'reference_comparison_for_test3seed')
    reference_rows, sources, evidence, selected, populations = base.collect_and_verify()
    for row in reg['rows']:
        for split in ['validation', 'test']:
            a = c['population_references'][row['dataset']][split]
            b = populations[row['dataset']][split]
            for key in ['target_identity_sha256', 'truth_sha256', 'loader', 'data_file_sha256', 'expected_target_count']:
                require(a[key] == b[key], 'Comparison population changed: ' + str((row['dataset'], split, key)))
    base.ORDER.append(CANDIDATE)
    base.LABELS[CANDIDATE] = 'TitanTPP CNN+GRU54'
    rows = reference_rows + candidate_rows
    base.check_coverage(rows)
    aggregate = base.summarize(rows)
    by_group = {(r['dataset'], r['model'], r['split'], r['view']): r for r in aggregate}
    write_csv(HERE / 'metrics_per_seed.csv', rows)
    write_csv(HERE / 'quantity_cell_time_nll.csv', cell_rows)
    time_readout = ['# 고정 checkpoint의 시간 일반화 위치 확인', '',
        '재평가의 집계 영수증만 분해했다. raw 예측·행·시간 분포를 새로 열거나 학습하지 않았다. 아래 bin은 원래 Train 경계로 고정한 실제 target 수량 구간이며 모델 선택·입력에 사용하지 않는 사후 진단이다.', '',
        '전체 TimeNLL은 각 사건 NLL 합/count이고, 각 수량 구간의 NLL 합을 더하면 전체 합이 된다. 각 열은 동일 세 seed 통계의 산술 평균이며, 합 비중은 seed별 비율을 평균했다.', '',
        '| 데이터 | 작은 수량 경계 | Validation 해당 구간 NLL | Test 해당 구간 NLL | Validation 전체 NLL 합 비중 | Test 전체 NLL 합 비중 |',
        '|---|---:|---:|---:|---:|---:|']
    for dataset, label in base.DATASETS.items():
        val_cells = [r for r in cell_rows if r['dataset'] == dataset and r['split'] == 'validation' and r['bin'] == 0]
        test_cells = [r for r in cell_rows if r['dataset'] == dataset and r['split'] == 'test' and r['bin'] == 0]
        require(len(val_cells) == len(test_cells) == 3, 'Quantity diagnostic cell coverage differs')
        avg = lambda members, key: statistics.mean(r[key] for r in members)
        time_readout.append(f"|{label}|≤{val_cells[0]['upper_inclusive']:g}|{avg(val_cells, 'time_nll'):.6g}|{avg(test_cells, 'time_nll'):.6g}|{avg(val_cells, 'NLL_sum_share_percent'):.2f}%|{avg(test_cells, 'NLL_sum_share_percent'):.2f}%|")
    time_readout += ['', 'Intermittent의 전체 TimeNLL 증가가 수량 ≤2 구간에 집중됐다는 위치는 확인했다. 시간 간격 분포 이동·조건별 분포 중심/퍼짐의 불일치·과신·수량 기준 checkpoint 선택은 가능한 설명이며, 이 집계 분해로 어느 하나가 근본 원인이라고 확정할 수 없다.', '',
        'Taxi는 Validation→Test의 절대 TimeNLL 평균이 감소해도 S2P2보다 나쁘다. 큰 수량의 시간 NLL이 거의 0인 것과 전체 시간 일반화는 구분해야 한다. 극소 NLL 분모에는 백분율 비교를 만들지 않았다.', '',
        '세 데이터의 고정 checkpoint, 원래 source117/123의 과학 파일 일치, 단위·출력부·정수 확률질량 NLL, 동일 target identity와 전체 충분통계는 검증됐다. 집계/단위가 달라서 생긴 차이라는 증거는 발견되지 않았지만, raw 시간 분포·조건별 예측 μ/σ 분석과 CPU 전체 checkpoint 재추론은 이 작업에서 수행하지 않았다.', '']
    (HERE / 'TIME_GENERALIZATION_READOUT.md').write_text('\n'.join(time_readout))
    base.write_csv(HERE / 'metrics_aggregated.csv', aggregate)
    lookup = {(r['dataset'], r['model'], r['seed'], r['split'], r['view']): r for r in rows}
    paired = []
    for dataset in base.DATASETS:
        models = [m for m in base.ORDER if m not in base.SIMPLE and (dataset, m, 42, 'test', 'overall') in lookup]
        for split in ['validation', 'test']:
            for view in ['overall', 'tail']:
                for comparator in models:
                    if comparator == CANDIDATE:
                        continue
                    for metric in base.METRICS:
                        deltas = [lookup[(dataset, CANDIDATE, seed, split, view)][metric] - lookup[(dataset, comparator, seed, split, view)][metric] for seed in [42, 52, 62]]
                        denominator = statistics.mean(lookup[(dataset, comparator, seed, split, view)][metric] for seed in [42, 52, 62])
                        paired.append({'dataset': dataset, 'split': split, 'view': view, 'model': CANDIDATE,
                            'comparator': comparator, 'metric': metric, 'delta_mean': statistics.mean(deltas),
                            'delta_sample_sd': statistics.stdev(deltas),
                            'baseline_mean': denominator,
                            'relative_change_percent': 100 * statistics.mean(deltas) / denominator if abs(denominator) >= 1e-6 else None,
                            'improved_seeds': sum(d < 0 for d in deltas), 'worst_delta': max(deltas),
                            **{f'seed{seed}_delta': delta for seed, delta in zip([42, 52, 62], deltas)}})
    base.write_csv(HERE / 'paired_candidate_deltas.csv', paired)
    prior_strict = ROOT / 'reports/titantpp_cnn_gru54_final_validation_comparison_20261006_v1/strict_validation_gate.json'
    adoption_source = e.read(prior_strict)['frozen_adoption_source']
    require(e.sha(ROOT / adoption_source['path']) == adoption_source['sha256'], 'Original adoption criteria SHA changed')
    dominance = []
    for dataset in base.DATASETS:
        for comparator in ['s2p2_matched_head', 'titantpp_history_mlp_width16']:
            failures = []
            for view, metric, strict in [('overall', 'qty_rmse', True), ('overall', 'qty_mae', False),
                    ('tail', 'qty_rmse', False), ('tail', 'qty_mae', False),
                    ('overall', 'time_nll', False), ('tail', 'time_nll', False)]:
                deltas = {str(seed): lookup[(dataset, CANDIDATE, seed, 'test', view)][metric] - lookup[(dataset, comparator, seed, 'test', view)][metric] for seed in [42,52,62]}
                bad = {s: delta for s, delta in deltas.items() if (delta >= 0 if strict else delta > 0)}
                if bad:
                    failures.append({'view': view, 'metric': metric, 'requirement': 'all3 strictly improve' if strict else 'no seed worse', 'failed_seed_deltas': bad})
            dominance.append({'dataset': dataset, 'reference': comparator, 'strong_dominance': not failures,
                'status': 'PASS_STRONG_DOMINANCE' if not failures else 'FAIL_TRADEOFF', 'failures': failures})
    e.write(HERE / 'strict_test_gate.json', {'scope': 'Test_previously_exposed', 'evidence_integrity_gate': 'PASS',
        'scientific_adoption_gate': 'PASS_STRONG_DOMINANCE' if all(r['strong_dominance'] for r in dominance) else 'FAIL_TRADEOFF',
        'frozen_adoption_source': adoption_source, 'criteria_source_sha256': e.sha(prior_strict),
        'dominance_results': dominance, 'checkpoint_reselection': False,
        'tiny_NLL_deltas_retained_in_exact_frozen_rule': True,
        'interpretation': 'Descriptions include absolute effects; microscopic tail NLL numerical ordering alone is not a practical claim.'})
    rankings = []
    for dataset in base.DATASETS:
        for split in ['validation', 'test']:
            for view in ['overall', 'tail']:
                groups = [r for r in aggregate if (r['dataset'], r['split'], r['view']) == (dataset, split, view)]
                for metric in base.METRICS:
                    ordered = sorted([r for r in groups if r[metric + '_mean'] is not None], key=lambda r: r[metric + '_mean'])
                    common = [r['model'] for r in ordered if r['model'] not in base.SIMPLE and r['model'] != 'deep_renewal_event_native_nb']
                    for rank, r in enumerate(ordered, 1):
                        rankings.append({'dataset': dataset, 'split': split, 'view': view, 'metric': metric,
                            'rank': rank, 'rank_scope': 'all_available_research_records',
                            'common_head_rank': common.index(r['model']) + 1 if r['model'] in common else None,
                            'seed_count': r['seed_count'],
                            'comparison_role': 'quantity_only_deterministic' if r['model'] in base.SIMPLE else 'research_only_native_NB_excluded_from_manuscript' if r['model'] == 'deep_renewal_event_native_nb' else 'learned_common_time_head',
                            'model': r['model'], 'mean': r[metric + '_mean'], 'sample_sd': r[metric + '_sample_sd']})
    base.write_csv(HERE / 'metric_ranks.csv', rankings)
    for split in ['validation', 'test']:
        text = [f'# {split.title()} 3seed · 고정 checkpoint 비교', '', '학습 모델은 같은 seed42·52·62의 평균 ± 표본 SD(ddof=1)다. 결정적 수량 기준선은 1회 계산이며 seed SD·TimeNLL이 없다. 작은 값이 좋다. 전체/큰 수량은 원래 Train 경계 > Taxi3449·Intermittent187·RAF200이다.', '', 'CNN+GRU54는 원래 수량 RMSE-selected checkpoint 9개를 고정했다. seed42에는 A100 및 Intermittent의 A100→5090 계보가 포함되고 seed52/62는 RTX에서 학습했다. 이번 결과는 3seed 탐색 비교이며 GPU 효율 또는 CNN/GRU 각각의 독립 기여 검증이 아니다.', '', 'Validation과 기존 노출 Test를 분리했다. Test는 미접근 독립 평가가 아니며, 이 표의 순위가 모델 채택 또는 논문 우월성 gate 통과를 뜻하지 않는다.', '']
        for dataset, label in base.DATASETS.items():
            for view, vlabel in [('overall', '전체'), ('tail', '큰 수량')]:
                n = by_group[(dataset, CANDIDATE, split, view)]['count_per_seed']
                full_n = by_group[(dataset, CANDIDATE, split, 'overall')]['count_per_seed']
                text += [f'## {label} · {vlabel}', '', f'동일 target {n:,}/{full_n:,}개 ({100*n/full_n:.2f}%) / seed.', '', '| 모델 | RMSE | MAE | TimeNLL |', '|---|---:|---:|---:|']
                order = sorted([r for r in aggregate if (r['dataset'], r['split'], r['view']) == (dataset, split, view)], key=lambda r: r['qty_rmse_mean'])
                for r in order:
                    def fmt(metric):
                        value, sd = r[metric + '_mean'], r[metric + '_sample_sd']
                        if value is None: return '—'
                        return f'{value:.6g}' + (f' ± {sd:.4g}' if sd is not None else '')
                    text.append('|' + base.LABELS.get(r['model'], r['model']) + '|' + '|'.join(fmt(m) for m in base.METRICS) + '|')
                text += ['']
        text += ['Deep Renewal native NB는 원고 제외 연구 기록으로 보존했다. 공통 lognormal 출력부를 사용한 외부 six와 likelihood 통제 비교로 합치지 않는다. 결정적 수량 기준선에는 seed SD·시간 점수가 없다.', '시간 간격의 관측 단위는 Taxi=시간, Intermittent=주, RAF=월이며, TimeNLL은 해당 기록 정수 간격 확률질량의 사건당 음의 로그 확률(nats/event)이다. 데이터 간 원시 RMSE·NLL을 합산한 순위는 만들지 않았다.', '']
        (HERE / f'{split.upper()}_TABLES.md').write_text('\n'.join(text) + '\n')
    summary = []
    for dataset, label in base.DATASETS.items():
        r = by_group[(dataset, CANDIDATE, 'test', 'overall')]
        s = by_group[(dataset, 's2p2_matched_head', 'test', 'overall')]
        vals = {metric: {'candidate': r[metric + '_mean'], 's2p2': s[metric + '_mean'],
            'relative_change_percent': 100 * (r[metric + '_mean'] - s[metric + '_mean']) / s[metric + '_mean']} for metric in base.METRICS}
        strong = all(lookup[(dataset, CANDIDATE, seed, 'test', 'overall')]['qty_rmse'] < lookup[(dataset, 's2p2_matched_head', seed, 'test', 'overall')]['qty_rmse'] and all(lookup[(dataset, CANDIDATE, seed, 'test', view)][m] <= lookup[(dataset, 's2p2_matched_head', seed, 'test', view)][m] for view, metrics in [('overall', ['qty_mae','time_nll']), ('tail', ['qty_rmse','qty_mae','time_nll'])] for m in metrics) for seed in [42,52,62])
        summary.append({'dataset': dataset, 'label': label, 'metrics': vals, 'strong_dominance_over_S2P2': strong})
    receipt = {'status': 'complete', 'scope': 'split_separated_Validation_and_previously_exposed_Test_3seed',
        'candidate_conditions': 9, 'new_fit_training_calls': 0, 'new_training': False,
        'benchmark_learned_conditions': len(selected), 'combined_learned_conditions': len(selected) + 9,
        'metric_rows': len(rows), 'Deep_Renewal_role': 'preserved research records; excluded from manuscript and common-head superiority claims',
        'checkpoint_reselection': False, 'raw_prediction_exports': False,
        'held_out_Test_access_authorized': True, 'independent_untouched_Test': False,
        'head_refit_stages_included': False, 'CPU_binary_reinference_audit': 'not_performed',
        'population_identity_verified': True, 'same_seed_pairing_verified': True,
        'summary': summary, 'aggregation': 'arithmetic mean; sample SD ddof1; within-seed metrics; no ensemble',
        'training_source_provenance': 'source117 reused seed42; source123 seed52/62; 117 scientific files identical',
        'reference_sources': sources, 'native_retrieval_receipt_sha256': e.sha(BUNDLE / 'retrieved/retrieval_receipt.json'),
        'native_validation_gate_sha256': e.sha(BUNDLE / 'retrieved/validation_gate.json'),
        'native_inference_completion_sha256': e.sha(BUNDLE / 'retrieved/inference_completion.json'),
        'actual_native_terminal_observation_sha256': e.sha(HERE / 'native_terminal_observation.json'),
        'supervisor_process_exit_sha256': e.sha(BUNDLE / 'retrieved/supervisor_process_exit.json'),
        'strict_test_gate_sha256': e.sha(HERE / 'strict_test_gate.json'),
        'execution_contract_sha256': e.sha(BUNDLE / 'execution_contract.json'),
        'created_utc': datetime.now(timezone.utc).isoformat()}
    receipt['outputs'] = {p.name: e.sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ['.csv', '.md', '.py'] and p.name != 'README.md'}
    e.write(HERE / 'comparison_receipt.json', receipt)
    print(json.dumps({'status': 'complete', 'metric_rows': len(rows), 'summary': summary}, ensure_ascii=False))


if __name__ == '__main__':
    run()
