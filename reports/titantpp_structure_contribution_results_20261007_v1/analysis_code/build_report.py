"""Summarize preserved Validation measurements; never evaluate or contact servers."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'search_artifacts/titantpp_architecture_capacity_handoff_20261006_v1/analysis'
PILOT = ROOT / 'search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1'
PILOT_REPORT = ROOT / 'reports/titantpp_contextual_retrieval_pilot_dual_20261007_v1/hourly/20261007T005659314481Z_1791334623041507000.json'
DS = {'yellow_trip_hourly': 'Taxi', 'intermittent_frozen_5000': 'Intermittent', 'raf_spare_parts': 'RAF'}
ARMS = ['P0_H0', 'P1_H0', 'P0_H1', 'P1_H1', 'PF_H1', 'P1_HC']
METRICS = ['qty_rmse', 'qty_mae', 'time_nll', 'tail_qty_rmse', 'tail_qty_mae', 'tail_time_nll']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def number(value):
    return f'{value:.3e}' if value != 0 and abs(value) < 1e-4 else f'{value:.6f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    receipt = read(OLD / 'completion_receipt.json')
    assert receipt['status'] == 'all54_Validation_originals_complete'
    assert receipt['canonical_conditions'] == 54 and receipt['local_originals_SHA_verified'] is True
    assert receipt['held_out_test_evaluated'] is False and receipt['selection_changed'] is False
    for name, digest in receipt['outputs'].items():
        assert sha(OLD / name) == digest, name
    assert all(r['local_originals_SHA_verified'] and r['actual_owner_processes_absent'] for r in receipt['originals'].values())
    s = read(OLD / 'Validation_summary.json')
    assert s['scope'] == 'full_Validation_only' and s['held_out_test_evaluated'] is False
    rows = [{**r, **r['metrics']} for r in s['rows']]
    assert len(rows) == 54
    assert len({(r['dataset'], r['arm'], r['seed']) for r in rows}) == 54
    grouped = {(g['dataset'], g['arm']): g for g in s['grouped_3seed']}
    assert len(grouped) == 18
    for ds in DS:
        for arm in ARMS:
            rr = [r for r in rows if r['dataset'] == ds and r['arm'] == arm]
            assert sorted(r['seed'] for r in rr) == [42, 52, 62]
            assert len({(r['validation_count'], r['tail_count'], r['tail_threshold']) for r in rr}) == 1
            for key in METRICS:
                values = [r[key] for r in rr]
                assert all(math.isfinite(v) for v in values)
                expected = grouped[ds, arm]['metrics'][key]
                assert math.isclose(statistics.mean(values), expected['mean'], rel_tol=1e-12, abs_tol=1e-12)
                assert math.isclose(statistics.stdev(values), expected['sample_SD'], rel_tol=1e-12, abs_tol=1e-12)
    by_id = {(r['dataset'], r['seed'], r['arm']): r for r in rows}
    for g in s['contrast_groups']:
        for key in METRICS:
            vals = [sum(weight * by_id[g['dataset'], seed, arm][key] for arm, weight in g['weights'].items()) for seed in [42, 52, 62]]
            expected = g['metrics'][key]
            assert math.isclose(statistics.mean(vals), expected['mean_delta'], rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(statistics.stdev(vals), expected['sample_SD_delta'], rel_tol=1e-12, abs_tol=1e-12)
            assert sum(v < 0 for v in vals) == expected['negative_seed_count']
    pilot = read(PILOT_REPORT)
    assert sha(PILOT_REPORT) == '3aa65d349891579ab5f9341dbc79be448c218aec477bda1935414fcea2cb09c1'
    assert pilot['held_out_test_evaluated'] is False and pilot['seed'] == 42
    qual = read(PILOT / 'qualification/5090/receipt.json')
    assert sha(PILOT / 'qualification/5090/receipt.json') == 'ef2e19f81f5578bdd03333c04f92665ad804c113d7d46dab5e9236082559c3f8'
    assert qual['held_out_test_evaluated'] is False and qual['status'] == 'passed'
    contrasts = []
    pairs = [('H_without_P', 'P0_H1', 'P0_H0'), ('H_with_P', 'P1_H1', 'P1_H0'), ('P_without_H', 'P1_H0', 'P0_H0'), ('P_with_H', 'P1_H1', 'P0_H1'), ('matched_P_FFN', 'P1_H1', 'PF_H1'), ('matched_H_current', 'P1_H1', 'P1_HC')]
    for ds in DS:
        for label, candidate, base in pairs:
            cm = grouped[ds, candidate]['metrics']; bm = grouped[ds, base]['metrics']
            row = {'dataset': ds, 'contrast': label, 'candidate': candidate, 'baseline': base,
                   'full_RMSE_improvement_percent_from_means': 100 * (bm['qty_rmse']['mean'] - cm['qty_rmse']['mean']) / bm['qty_rmse']['mean']}
            for key in METRICS:
                vals = [by_id[ds, seed, candidate][key] - by_id[ds, seed, base][key] for seed in [42, 52, 62]]
                row[key + '_delta_mean'] = statistics.mean(vals)
                row[key + '_delta_sample_SD'] = statistics.stdev(vals)
                row[key + '_improved_seeds'] = sum(v < 0 for v in vals)
            contrasts.append(row)
    data = {'scope': 'Validation development evidence only', 'selected_epoch': 'strict earliest finite full raw quantity RMSE minimum',
            'unique_conditions': 54, 'grouped_3seed': s['grouped_3seed'], 'contrasts': contrasts,
            'interaction': [g for g in s['contrast_groups'] if g['contrast'] == 'q11-q10-q01+q00'],
            'structural_support': s['structural_support'], 'pilot_as_of': pilot['hosts'],
            'new_evaluation_performed': False, 'held_out_access_performed': False}
    write_json(OUT / 'ANALYSIS.json', data)
    with (OUT / 'DERIVED_CONTRASTS.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(contrasts[0])); w.writeheader(); w.writerows(contrasts)
    lines = ['# Validation 상세 결과표', '', '원본 회수와 SHA 검증을 마친 54조건입니다. 모든 지표는 각 조건의 최초 최소 full Validation 수량 RMSE epoch에서 읽었습니다. 평균 ± 표본 SD는 동일 평가 모집단을 사용한 seed42/52/62의 모델 변동이며 독립 표본의 신뢰구간이 아닙니다.', '', '큰 수량은 Train 경계 초과의 Validation 행입니다. Taxi >3449: 79/8268(0.9555%); Intermittent >187: 1141/86285(1.3224%); RAF >200: 50/6690(0.7474%). 3seed로 평가 행 수를 세 배 늘리지 않습니다. near-zero TimeNLL은 과학 표기로 보존하며 결측값으로 처리하지 않습니다.', '']
    for ds, label in DS.items():
        lines += [f'## {label}: 3seed 평균 ± 표본 SD', '']
        tr = []
        for arm in ARMS:
            g = grouped[ds, arm]
            tr.append([arm] + [number(g['metrics'][k]['mean']) + ' ± ' + number(g['metrics'][k]['sample_SD']) for k in METRICS])
        lines += [table(['구조', '전체 RMSE', '전체 MAE', '전체 TimeNLL', '큰 수량 RMSE', '큰 수량 MAE', '큰 수량 TimeNLL'], tr), '']
    lines += ['## 같은 seed의 구조 차이', '', '아래 차이는 후보−대조입니다. 음수는 후보 개선입니다. RMSE 개선율은 평균끼리의 비율 `(대조 평균−후보 평균)/대조 평균×100`이며 개별 seed 개선율 평균과 다릅니다. 각 TimeNLL 차이도 수량 RMSE 선택 epoch를 유지합니다.', '']
    tr = [[DS[r['dataset']], r['candidate'] + ' vs ' + r['baseline'], f"{r['full_RMSE_improvement_percent_from_means']:+.4f}%", str(r['qty_rmse_improved_seeds']) + '/3', number(r['qty_rmse_delta_mean']), number(r['time_nll_delta_mean']), number(r['tail_qty_rmse_delta_mean']), number(r['tail_time_nll_delta_mean'])] for r in contrasts]
    lines += [table(['데이터', '후보 vs 대조', '전체 RMSE 개선율', '개선 seed', '전체 ΔRMSE', '전체 ΔTimeNLL', '큰 수량 ΔRMSE', '큰 수량 ΔTimeNLL'], tr), '', '## 54조건의 선택 epoch와 저장 epoch', '', 'host는 실제 실행 GPU입니다. Intermittent seed62의 여섯 구조는 모두 5080에서 실행했고 seed42/52는 5090에서 실행했습니다. 같은 seed 내부 비교는 같은 GPU/Runtime을 유지합니다.', '']
    tr = [[DS[r['dataset']], r['arm'], r['seed'], r['execution_host'], r['best_epoch'], r['completed_epochs'], number(r['qty_rmse']), number(r['time_nll'])] for r in sorted(rows, key=lambda r:(list(DS).index(r['dataset']), r['seed'], ARMS.index(r['arm'])))]
    lines += [table(['데이터', '구조', 'seed', '실행 GPU', '선택 epoch', '마지막 저장 epoch', '전체 RMSE', '전체 TimeNLL'], tr), '', f'54행·6지표의 반올림 없는 원본은 [Validation_per_seed.csv]({OLD}/Validation_per_seed.csv), seed별 모든 차이는 [Validation_paired_contrasts.csv]({OLD}/Validation_paired_contrasts.csv)에 보존되어 있습니다.', '']
    (OUT / 'VALIDATION_TABLES.md').write_text('\n'.join(lines))
    inputs = [OLD/'completion_receipt.json', OLD/'Validation_summary.json', OLD/'Validation_per_seed.csv', OLD/'Validation_paired_contrasts.csv', OLD/'summary_identity_mapping.json', PILOT_REPORT, PILOT/'qualification/5090/receipt.json', PILOT/'execution_contract.json', PILOT/'design_contract.json', ROOT/'search_artifacts/titantpp_architecture_contribution_dual_20261006_v1/execution_contract.json']
    receipt_out = {'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Validation_only_local_report', 'inputs': {str(p):sha(p) for p in inputs},
                   'originals': receipt['originals'], 'unique_old_conditions':54, 'independent_seeds':3, 'pilot_seed_count':1,
                   'verification': {'all54_unique':True, 'all18_group_means_and_sample_SD_recomputed':True, 'all21_contrast_means_SD_and_win_counts_recomputed':True, 'sealed_summary_output_SHA_verified':True},
                   'source_selection_changed':False, 'new_model_inference':False, 'new_training':False, 'Test_or_mixed_outcomes_accessed':False, 'CPU_binary_reinference_audit':'not_performed'}
    write_json(OUT/'PROVENANCE.json', receipt_out)
    print(json.dumps({'status':'report_tables_verified','conditions':len(rows),'groups':len(grouped),'derived_contrasts':len(contrasts)},ensure_ascii=False))


if __name__ == '__main__':
    main()
