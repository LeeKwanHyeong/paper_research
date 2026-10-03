"""Analyze the completed 5080 validation subset, without model execution."""
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / 'search_artifacts/titantpp_core_ablation_20260928_v1'
LEDGER = BASE / 'aggregation/20260928T232437308376Z/ledger.json'
OUT = Path(__file__).parent / 'completed_5080_20260929'
MODELS = ['titantpp', 'titantpp_local_detail', 'titantpp_history_mlp',
          'titantpp_level_only', 'titantpp_change_only', 'titantpp_no_static_lmm']
LABELS = dict(zip(MODELS, ['B (이력 보완 없음)', 'Full', '이력 MLP', '수준만', '변화만', '정적 검색 제거']))
DATASETS = {'yellow_trip_hourly': 'Taxi', 'intermittent_frozen_5000': 'Intermittent'}
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(PROJECT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def stats(values):
    return {'mean': st.mean(values), 'sample_sd': st.stdev(values)}


def main():
    ledger = read(LEDGER)
    contract = read(BASE / 'frozen_execution/execution_contract.json')
    assert ledger['evaluation_scope'] == 'validation_only'
    assert ledger['contract_sha256'] == 'eff125f587f7a9a8097d5d43b3d688eac481abb78bb6006b5aa4ab1b0ec2a88d'
    assert contract['comparison']['acceptance'] == {'body_mae_ratio_max':1.02,'each_nonempty_middle_bin_mae_and_rmse_ratio_max':1.02,'last30_rmse_mean_ratio_max':1.05,'last30_rmse_sample_sd_ratio_max':1.05,'overall_mae_ratio_max':1.01,'overall_raw_rmse_ratio_strictly_less_than':1.0,'recorded_time_nll_increase_max':.01,'tail_mae_ratio_max':1.02,'tail_rmse_ratio_strictly_less_than':1.0}
    terminal = read(BASE / 'monitor/20260928T195732Z/terminal_5080/terminal_audit.json')
    assert terminal['status'] == 'passed'
    assert read(BASE / 'baseline_audit.json')['status'] == 'passed'
    rows = [r for r in ledger['conditions'] if r['host'] == '5080']
    assert len(rows) == 36 and len({(r['dataset'], r['seed'], r['arm']) for r in rows}) == 36
    for row in rows:
        assert row['execution_status'] == 'complete' and row['audit_status'] == 'passed'
        assert not row['timing_interference_recorded']
        for ep in ['selected', 'last']:
            assert row[ep]['evaluation_scope'] == 'validation_only' and not row[ep]['held_out_test_evaluated']
        history = read(PROJECT / row['source'] / 'history.json')['history']
        row['first40'] = stats([h['val_qty_rmse'] for h in history[:40]])
    index = {(r['dataset'], r['seed'], r['arm']): r for r in rows}
    aggregates, contrasts = {}, []
    for ds in DATASETS:
        aggregates[ds] = {}
        for model in MODELS:
            rr = [index[ds, seed, model] for seed in [42, 52, 62]]
            ag = {k: stats([r['selected'][k] for r in rr]) for k in ['qty_rmse', 'qty_mae', 'time_nll']}
            ag.update(parameters=rr[0]['parameters'], fit_minutes=stats([r['fit_elapsed_seconds']/60 for r in rr]),
                      seconds_per_epoch=stats([r['fit_elapsed_seconds']/r['completed_epochs'] for r in rr]),
                      peak_mib=stats([r['peak_allocated_bytes']/1024**2 for r in rr]))
            for ep in ['selected', 'last']:
                ag[ep] = {k: stats([r[ep][k] for r in rr]) for k in ['qty_rmse', 'qty_mae', 'time_nll']}
                for group in ['body', 'tail']:
                    ag[ep][group] = {k: stats([r[ep][group][k] for r in rr]) for k in ['qty_rmse', 'qty_mae']}
                for group in ['quantity_cells', 'history_cells']:
                    ag[ep][group] = [dict(bin=i, count=rr[0][ep][group][i]['count'], **{
                        k: stats([r[ep][group][i][k] for r in rr]) if rr[0][ep][group][i]['count'] else None
                        for k in ['qty_rmse', 'qty_mae']}) for i in range(len(rr[0][ep][group]))]
            aggregates[ds][model] = ag
        for comparison in [m for m in MODELS if m != 'titantpp_local_detail']:
            paired = []
            for seed in [42, 52, 62]:
                full, ref = index[ds, seed, 'titantpp_local_detail'], index[ds, seed, comparison]
                x, y = full['selected'], ref['selected']
                checks = {'overall_rmse': x['qty_rmse'] < y['qty_rmse'],
                          'overall_mae': x['qty_mae'] <= 1.01*y['qty_mae'],
                          'body_mae': x['body']['qty_mae'] <= 1.02*y['body']['qty_mae'],
                          'tail_rmse': x['tail']['qty_rmse'] < y['tail']['qty_rmse'],
                          'tail_mae': x['tail']['qty_mae'] <= 1.02*y['tail']['qty_mae'],
                          'time_nll': x['time_nll'] <= y['time_nll']+.01,
                          'last30_mean': full['last30']['mean'] <= 1.05*ref['last30']['mean'],
                          'last30_sd': full['last30']['sd'] <= 1.05*ref['last30']['sd']}
                for cell in x['quantity_cells'][1:4]:
                    if cell['count']:
                        for metric in ['qty_mae', 'qty_rmse']:
                            checks[f'middle_bin{cell["bin"]}_{metric}'] = cell[metric] <= 1.02*y['quantity_cells'][cell['bin']][metric]
                paired.append({'seed': seed, 'rmse_full_minus_reference': x['qty_rmse']-y['qty_rmse'],
                               'rmse_full_reduction_pct': 100*(1-x['qty_rmse']/y['qty_rmse']),
                               'time_nll_full_minus_reference': x['time_nll']-y['time_nll'],
                               'guardrails': checks, 'all_guardrails_passed': all(checks.values())})
            fm, rm = aggregates[ds]['titantpp_local_detail'], aggregates[ds][comparison]
            contrasts.append({'dataset': ds, 'full_vs': comparison, 'pairs': paired,
                              'mean_rmse_reduction_pct': 100*(1-fm['qty_rmse']['mean']/rm['qty_rmse']['mean']),
                              'mean_mae_reduction_pct': 100*(1-fm['qty_mae']['mean']/rm['qty_mae']['mean']),
                              'rmse_wins': sum(p['rmse_full_minus_reference'] < 0 for p in paired),
                              'all_guardrail_seed_passes': sum(p['all_guardrails_passed'] for p in paired)})
    OUT.mkdir(exist_ok=True)
    payload = dict(evaluation_scope='validation_only', conditions=rows, aggregates=aggregates,
                   contrasts=contrasts, conditions_count=36, endpoint_records=72,
                   new_fits=24, reused_fits=12, contract_sha256=ledger['contract_sha256'],
                   thresholds=contract['comparison']['acceptance'], source_sha256=SOURCES,
                   statistical_limit='Three seeds; descriptive paired effects and sample SD, no significance claim.',
                   scope_limit='Completed 5080 subset only; Instacart/5090 campaign still incomplete.')
    (OUT/'comparison.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    fields=['dataset','seed','arm','reused','best_epoch','completed_epochs','optimizer_steps','parameters','fit_elapsed_seconds','peak_allocated_bytes']
    with (OUT/'conditions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields+['qty_rmse','qty_mae','time_nll']);writer.writeheader()
        for r in rows:writer.writerow({**{k:r[k] for k in fields},**{k:r['selected'][k] for k in ['qty_rmse','qty_mae','time_nll']}})
    md=['# 5080 완료 결과: Taxi·Intermittent', '',
        '평가 범위: validation만. 2데이터 × 3seed × 6모델 = 36조건/72 endpoint 기록(신규24, 재사용 B·Full12). 모든 조건을 포함했다. 5090의 Instacart 결과는 이 보고서에 포함하지 않는다.', '',
        '모든 selected 지표는 최초의 엄격한 validation 수량 RMSE 최솟값 checkpoint에서 계산한다. ±는 3seed 표본표준편차(ddof=1)이며 통계적 유의성이나 신뢰구간을 의미하지 않는다.', '']
    taxi=aggregates['yellow_trip_hourly'];intermittent=aggregates['intermittent_frozen_5000']
    imlp_gain=100*(1-intermittent['titantpp_history_mlp']['qty_rmse']['mean']/intermittent['titantpp_local_detail']['qty_rmse']['mean'])
    ilevel_gain=100*(1-intermittent['titantpp_level_only']['qty_rmse']['mean']/intermittent['titantpp_local_detail']['qty_rmse']['mean'])
    taxi_gain=100*(1-taxi['titantpp_local_detail']['qty_rmse']['mean']/taxi['titantpp_history_mlp']['qty_rmse']['mean'])
    md += ['## 먼저 읽을 결과와 해석', '',
        f'- **Taxi:** Full은 B보다 수량 RMSE가 12.96% 낮다. 다만 같은 추가 파라미터 수의 이력 MLP 대비 평균 이득은 {taxi_gain:.2f}%이고 3seed 중 2개만 이겼다. seed52는 MLP가 더 좋다. 이력 정보를 쓰는 이점과 Full 구조 자체의 필수성을 구분해야 한다.',
        f'- **Intermittent:** MLP가 Full보다 평균 수량 RMSE를 {imlp_gain:.2f}% 줄였고 3seed 모두 이겼다. 수준만 모델도 Full보다 {ilevel_gain:.2f}% 낮고 3seed 모두 이겼다. 이번 통제 비교는 Intermittent에서 현재 Full이 필요하다는 주장을 뒷받침하지 못한다.',
        '- **시간·구간 반례:** Taxi Full의 시간 NLL은 B보다 3seed 모두 나빠졌다. Intermittent MLP는 전체 수량 오차가 낮지만 긴 이력(>128)의 평균 MAE는 Full보다 높다. 전체 평균 개선을 모든 구간/시간 점수 개선으로 확장할 수 없다. Full 대 B/MLP의 계약상 보호 기준 전체 통과는 두 데이터 모두 0/3이다.',
        '- **정적 검색:** 제거 모델과 Full의 전체 RMSE 차이는 작고, 제거 모델의 평균 시간 NLL은 두 데이터에서 더 낮다. 이는 마지막 정적 검색의 필요성에 대한 약한 증거다. encoder의 persistent memory까지 제거한 실험은 아니다.',
        '- **Gate의 역할 — 해석:** 변화 경로를 그대로 더하는 것이 Intermittent에 불리할 수 있다는 가설과 양립한다. 상수 계수만 좋아지면 전체 변화 기여 조정으로 충분할 수 있고, 조건부가 상수보다 추가 개선하면 관측 이력에 따른 조정의 후속 근거가 된다. 현재 결과만으로 조건부 Gate가 문제를 해결한다고 예측할 수 없다.',
        '- **주장 한계:** Gate 설계는 이 결과를 보고 바꾸지 않는다. Gate4는 사전에 고정한 seed42 탐색 실행이므로 결과가 좋아도 3seed 재현성이나 held-out 일반화가 입증되는 것은 아니다.', '']
    for ds,label in DATASETS.items():
        md += ['## '+label,'','| 모델 | 수량 RMSE ↓ | 수량 MAE ↓ | 시간 NLL ↓ | 파라미터 | 평균 fit 분 | 평균 초/epoch |','|---|---:|---:|---:|---:|---:|---:|']
        for m in MODELS:
            ag=aggregates[ds][m]
            md.append('| '+LABELS[m]+' | '+' | '.join(f'{ag[k]["mean"]:.4f} ± {ag[k]["sample_sd"]:.4f}' for k in ['qty_rmse','qty_mae','time_nll'])+f' | {ag["parameters"]:,} | {ag["fit_minutes"]["mean"]:.2f} | {ag["seconds_per_epoch"]["mean"]:.2f} |')
        md += ['', '| 비교: Full 대 | Full 평균 RMSE 감소율 | Full RMSE 승리 seed | 보호 기준 전체 통과 seed |','|---|---:|---:|---:|']
        for c in contrasts:
            if c['dataset']==ds:md.append(f'| {LABELS[c["full_vs"]]} | {c["mean_rmse_reduction_pct"]:+.2f}% | {c["rmse_wins"]}/3 | {c["all_guardrail_seed_passes"]}/3 |')
        md += ['', '양의 감소율은 Full의 오차가 낮다는 뜻이다. 보호 기준은 원래 계약의 수량·시간·구간·last30 한도를 사후 변경 없이 각 seed 쌍에 적용했다. 중간 수량 구간은 bin1–3이다. 다른 비교의 탐색적 적용을 새로운 사전 검증 가설로 표현하지 않는다.', '',
               '| 모델 | tail RMSE | tail MAE | 이력 bin0 MAE | 이력 bin1 MAE | 이력 bin2 MAE |','|---|---:|---:|---:|---:|---:|']
        for m in MODELS:
            e=aggregates[ds][m]['selected'];vals=[e['tail'][k]['mean'] for k in ['qty_rmse','qty_mae']]+[c['qty_mae']['mean'] if c['count'] else None for c in e['history_cells']]
            md.append('| '+LABELS[m]+' | '+' | '.join(f'{v:.4f}' if v is not None else 'N/A' for v in vals)+' |')
        md += ['', '수량 tail은 train 기준 최상위 경계보다 큰 관측이다. 이력 구간은 ≤64, 65–128, >128이다. 구간별 표본 수와 전체 셀 원값은 comparison.json에 보존했다.', '']
    md += ['## 확인 범위와 한계', '',
           '- 신규24조건의 selected/last48 checkpoint는 종료 회수 시 byte SHA·텐서 SHA·메타데이터·strict loading 감사를 통과했다. B·Full12조건은 기존 감사 증적을 재사용했다.',
           '- 기존 수집기의 최초 선택·조기 종료·초기화·배치 prefix·표본 수·replay 대조 통과를 확인하고, 분석 원본 경로와 SHA를 comparison.json에 기록했다. 이번 분석은 새 GPU 평가를 수행하지 않았다.',
           '- fit 시간에는 trainer 준비·학습·validation·보고가 들어가며 별도 endpoint replay는 빠진다. 조기 종료와 실행 시점 차이가 있으므로 순수 구조 연산량, 추론 지연, 시간복잡도 이득으로 해석하지 않는다.',
           '- B·Full과 신규 모델의 같은 5080 사용은 비교에 도움이 되지만 독립적인 속도 benchmark는 아니다. peak 값은 프로세스 allocator 사용량이다.',
           '- 수준만·변화만은 Full보다 파라미터2,048개가 적다. 정적 검색 제거는 마지막 검색만 없애며 encoder persistent memory는 남는다.',
           '- validation은 checkpoint 선택에도 사용됐으며 Gate 설계에 이미 관측된 결과다. 아직 독립 held-out 일반화 증거가 아니며, n=3에서 통계적 우월성을 단정하지 않는다.', '',
           '## 남은 작업', '', '**Gate 4조건의 5080 사전 검증 재실행 — 승인 필요**',
           '- 3090 자동화 취소와 5080 이전·로컬26검증은 완료했다. CUDA 라이브러리 검색 경로 누락으로 합성 사전 검증에서 멈췄으며 본학습은 0조건이다. 경로 수정안과 실패 원본을 보존했고, 기존 자동 재실행 금지에 따라 1회 재검증 승인을 대기한다.',
           '- 승인 후 같은 seed42 상수·조건부 네 조건을 순차 실행한다. 기존 결과에 맞춰 모델·기준을 바꾸지 않으며, 통과 증적 없이 학습을 시작하지 않는다.', '',
           '**Instacart 완료 및 전체 주장 판단 — 외부 작업 대기**',
           '- 진행 중인 5090 결과까지 포함해 전체54조건을 감사한 뒤 데이터별 주장과 반례를 확정한다.', '']
    (OUT/'report.md').write_text('\n'.join(md))
    print(json.dumps({'report':str(OUT/'report.md'),'aggregates':{d:{m:{k:a[k] for k in ['qty_rmse','qty_mae','time_nll']} for m,a in mm.items()} for d,mm in aggregates.items()},'contrasts':contrasts},ensure_ascii=False))


if __name__ == '__main__':
    main()
