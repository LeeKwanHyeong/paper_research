"""Render the verified, additive Test diagnostic report without model selection."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
C = json.loads((ROOT / 'analysis_contract.json').read_text())
A = json.loads((ROOT / 'results/analysis.json').read_text())
V = json.loads((ROOT / 'results/verification.json').read_text())
I = json.loads((ROOT / 'independent_verification.json').read_text())
assert A['status'] == V['status'] == I['status'] == 'complete'
MLP = 'titantpp_history_mlp'
S2 = 's2p2_matched_head'
CURRENT = 'titantpp_current_only_param_matched'
AVAILABLE = 'titantpp_all_available_history_mlp'
NAMES = {MLP: 'TitanTPP MLP', 'rmtpp': 'RMTPP', 'thp': 'THP', 'nhp': 'NHP',
         'sahp': 'SAHP', S2: 'S2P2 (matched head)', 'attnhp_matched_head': 'AttNHP (matched head)',
         CURRENT: 'Current-only', AVAILABLE: 'All-available /8',
         'last_observed_quantity': 'Last quantity',
         'mean_quantity_in_same_observed_window': 'History mean'}
DATA = dict(zip([d['dataset'] for d in C['datasets']], ['Taxi', 'Intermittent', 'Instacart', 'RAF']))
MODELS = C['models'] + C['simple_models']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert I['analysis_sha256'] == sha(ROOT / 'results/analysis.json')
assert A['analysis_contract_sha256'] == V['analysis_contract_sha256'] == I['analysis_contract_sha256'] == sha(ROOT / 'analysis_contract.json')


def f(x, digits=4):
    return '—' if x is None else f'{x:.{digits}f}'


def metric(b, model, name='qty_rmse', sd=False):
    value = b['means'][model][name]
    if value is None:
        return '—'
    s = b['sample_sd'][model][name]
    return f(value) + (' ± ' + f(s) if sd and s is not None else '')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def label(b):
    d = b['definition']
    if d['label'] in ('all', 'early', 'middle', 'late'):
        return d['label']
    lo, hi = d['lower_exclusive'], d['upper_inclusive']
    if lo is None:
        return '≤' + f'{hi:g}'
    if hi is None:
        return '>' + f'{lo:g}'
    return f'({lo:g}, {hi:g}]'


def sse(b, model):
    rows = list(b['seed_metrics'][model].values())
    return sum(r['squared_error_sum'] for r in rows) / len(rows)


def human_date(value, dataset):
    if value is None:
        return '—'
    if dataset == 'insta_market_basket':
        return str(value) + '일'
    fmt = '%Y%m%d%H%M%S' if dataset == 'yellow_trip_hourly' else '%Y%m%d'
    dt = datetime.strptime(str(value), fmt)
    return dt.strftime('%Y-%m-%d %H시' if dataset == 'yellow_trip_hourly' else '%Y-%m-%d')


full = {d: views['all']['0'] for d, views in A['results'].items()}
parts = sum(len(row['parts']) for d in V['datasets'].values() for row in d['conditions'])
rows = sum(row['prediction_rows'] for d in V['datasets'].values() for row in d['conditions'])
targets = sum(d['target_count'] for d in C['datasets'])
reconciliations = sum(d['original_primary_reconciliation']['comparisons'] for d in V['datasets'].values())

lines = ['# 기존 Test의 전체·기간·수량·관측 이력 분석', '',
    '상태: **완료**. 네 데이터의 기존 Test 대상과 선택 checkpoint를 유지한 사후 설명적 분석이다. '
    '새 독립 평가 자료나 Test 재구성을 수행한 결과가 아니다.', '',
    f'- 입력: [기존 평가 원본]({C["source_bundle"]}/FINAL_REPORT.md), [동결 구간 계약]({ROOT}/analysis_contract.json).',
    f'- 학습 모델 9개 × seed42/52/62 × 데이터 4개 = 108조건, 결정적 기준선 8벡터. '
    f'Test 대상 {targets:,}건(조건 간 공통 대상), 저장된 예측 {rows:,}행을 사용했다.',
    '- 기존 전체 결과를 그대로 보존했다. 재학습·GPU 추론·원격 접속·원고 편집은 수행하지 않았다.',
    '- 이하 기본 수치는 **seed별 이벤트 가중 지표의 평균**이다. ±는 세 seed의 표본 표준편차이며 신뢰구간이 아니다. '
    'MAE·RMSE·시간 NLL은 낮을수록 좋다. 단순 기준선에는 시간 NLL·seed 표준편차가 없다.', '',
    '## 확인된 결과', '',
    table(['데이터', 'Test 대상', 'MLP RMSE', 'S2P2 RMSE', 'Current-only RMSE', 'All-available /8 RMSE'],
          [[DATA[d], f'{b["count"]:,}'] + [metric(b, m, sd=True) for m in [MLP, S2, CURRENT, AVAILABLE]] for d, b in full.items()]), '',
    '1. **Taxi:** 세 시간 구간 모두 MLP의 평균 RMSE가 S2P2보다 높다. 수량 구간에서도 5개 중 4개가 높고, '
    '수량 (1562, 3449]에서만 낮다. 따라서 불리한 전체 결과를 특정 기간 하나의 문제로 설명할 근거는 없다. '
    'All-available /8의 전체 RMSE가 MLP보다 낮다는 구조 대조 결과도 함께 유지한다.',
    '2. **Intermittent:** 세 기간 및 기존 세 이력 구간 모두 MLP의 평균 RMSE가 S2P2보다 낮다. '
    '다만 수량 (31, 46]에서는 5.1814 대 4.7987로 높다. '
    '고정 외부 학습 비교군 대비 전체 수량 이득은 유지되지만, All-available /8의 전체 RMSE 2.6602가 MLP 2.7612보다 낮다. '
    '시간 NLL은 MLP 15.3778로 THP 8.3263보다 높아 수량 이득을 시간 예측 우월성으로 확대할 수 없다.',
    '3. **Instacart:** 수량 ≤20의 두 구간에서는 MLP RMSE가 더 낮고, 수량 >20의 세 구간에서는 더 높다. '
    '큰 수량 구간에서 두 모델 모두 평균적으로 과소예측하며 MLP의 음의 편향이 더 크다. '
    '이는 오차의 위치를 보여 주며 구조나 손실함수가 원인임을 증명하지 않는다.',
    '4. **RAF:** 전체 MLP와 S2P2의 RMSE는 39.7951 대 39.7729로 가깝지만, 그 자체로 통계적 동등성을 뜻하지 않는다. '
    'Current-only와 단순 History mean의 전체 RMSE가 더 낮다. MLP는 S2P2보다 전체 MAE가 낮으며, 지표에 따른 차이를 모두 보존한다.', '',
    '## 기간별 결과', '',
    'Taxi는 기록 시간, Intermittent는 익명화된 기록의 고정 7일 좌표, RAF는 기록 월의 전체 Test 범위를 세 등분했다. '
    'Intermittent 좌표는 Unix epoch를 기준으로 나눈 7일 구간이며 ISO 주차가 아니다. '
    'Instacart는 사용자별 구매 시작 이후 경과일 3–123 / 124–244 / 245–365일이다. '
    '공통 달력 날짜나 사용자별 Test 진행률이 아니다. 표의 날짜는 각 구간에 실제 존재하는 기록의 최소·최대다.', '',
    'Taxi·Intermittent·Instacart의 기존 Test는 개체별 분할이며, RAF는 전역 달력 월 경계로 나눈 2001년 12월~2002년 12월 자료다. '
    '어느 데이터에서도 기간 간 개체 구성이 같다고 보장되지 않으므로 앞뒤 구간의 차이를 곧바로 분포 이동이나 미래 예측 강건성으로 해석하지 않는다.', '']
time_rows = []
for d, views in A['results'].items():
    for bi, b in views['time'].items():
        pop = I['datasets'][d]['cohort_descriptions']['time'][bi]
        time_rows.append([DATA[d], label(b), human_date(pop['recorded_date_min'], d) + ' ~ ' + human_date(pop['recorded_date_max'], d),
                          f'{b["count"]:,}', f'{pop["entity_count"]:,}'] + [metric(b, m) for m in [MLP, S2, CURRENT, AVAILABLE]])
lines += [table(['데이터', '구간', '실제 기록 범위', '대상 N', '개체 N', 'MLP RMSE', 'S2P2', 'Current-only', 'All-available /8'], time_rows), '',
          '개체 수는 구간 간 중복될 수 있어 합산하지 않는다. Intermittent의 기록 날짜는 사용자 확인에 따라 과거 기록을 일괄 이동한 날짜이며 실제 미래 관측을 뜻하지 않는다.', '',
          '## 수량별 결과와 제곱 오차의 비중', '',
          '수량 경계는 기존 학습 자료에서 고정된 값을 재사용했다. 실제 target 수량으로 나눈 사후 진단이며 '
          '예측 입력이나 모델 선택 규칙이 아니다. SSE 비중은 동일 모델·동일 데이터 내 구간의 seed 평균 SSE / 전체 seed 평균 SSE다.', '']
quantity_rows = []
for d, views in A['results'].items():
    for bi, b in views['quantity'].items():
        share = 100 * sse(b, MLP) / sse(full[d], MLP)
        quantity_rows.append([DATA[d], label(b), f'{b["count"]:,}', f(100*b['count']/full[d]['count'], 2) + '%']
                             + [metric(b, m) for m in [MLP, S2, CURRENT, AVAILABLE]]
                             + [f(share, 2) + '%', metric(b, MLP, 'bias'), metric(b, S2, 'bias')])
lines += [table(['데이터', '수량', 'N', '대상 비중', 'MLP RMSE', 'S2P2', 'Current-only', 'All-available /8', 'MLP SSE 비중', 'MLP 편향', 'S2P2 편향'], quantity_rows), '',
    '편향은 예측−실제 수량의 평균이다. 음수는 평균적인 과소예측이며 RMSE의 우열과 동일한 지표가 아니다.', '',
    '- Taxi의 수량 >3449인 83건(1.00%)이 MLP 전체 SSE의 30.88%를 차지한다. 이 구간 RMSE는 MLP 646.4226, S2P2 587.4488이다.',
    '- Instacart의 수량 >20인 66,665건(11.53%)이 MLP SSE의 54.96%를 차지한다. '
    '나머지 88.47%에서 얻은 제곱 오차 감소보다 이 세 구간의 증가가 크다.',
    '- RAF의 수량 >200인 39건(0.75%)이 MLP SSE의 78.05%를 차지한다. '
    '두 모델이 이 소수의 큰 수량에서 모두 큰 오차를 보인다. 이 대상들을 제외하거나 임의로 가중치를 낮추지 않았다.', '',
    '## 실제 입력 이력별 결과', '',
    '이력은 target을 제외한 실제 입력 사건 수이며 원래 관측창·최대 길이 제한이 적용된다. '
    '평생 구매 횟수나 경과 일수가 아니다. 아래 주 분석은 기존 이력 경계를 유지한다. '
    '빈 구간 N=0은 실패나 0오차가 아니라 지표 미정의(—)다.', '']
for view, title in [('history', '기존 이력 구간'), ('history_detail', '보조 세부 이력 구간')]:
    lines += ['### ' + title, '']
    if view == 'history_detail':
        lines += ['모든 데이터에 공통 경계 [1,3,7,15,31,63,127]을 사용했다. '
                  '새 결과를 계산하기 전에 기존 분기 활성화 기준에서 정했다. '
                  'RAF의 모든 대상이 기존 ≤64 구간에 포함되는 한계를 보완하는 설명용 표다.', '']
    view_rows = []
    for d, views in A['results'].items():
        for bi, b in views[view].items():
            view_rows.append([DATA[d], label(b), f'{b["count"]:,}'] + [metric(b, m) for m in [MLP, S2, CURRENT, AVAILABLE]])
    lines += [table(['데이터', '입력 이력', 'N', 'MLP RMSE', 'S2P2', 'Current-only', 'All-available /8'], view_rows), '']
lines += ['이력별 표는 서로 다른 대상 집단을 비교한다. 동일 대상의 이력 길이를 조작한 실험이 아니므로 '
          '“이력이 길어지면 모델이 나빠진다/좋아진다”는 인과 주장은 할 수 없다. '
          '특히 Taxi의 짧은 이력과 긴 이력 집단은 수량 규모가 같다고 보장되지 않는다.', '',
          '## 전체 비교군의 원래 Test 결과', '',
          '구조 대조와 단순 기준선까지 모두 보존한다. 아래 표는 새 구간을 골라 만든 결과가 아니라 기존 전체 Test 결과의 재현이다.', '']
for d, b in full.items():
    lines += ['### ' + DATA[d], '', table(['모델', 'MAE 평균 ± seed SD', 'RMSE 평균 ± seed SD', '시간 NLL 평균 ± seed SD'],
          [[NAMES[m]] + [metric(b, m, k, True) for k in ['qty_mae', 'qty_rmse', 'time_nll']] for m in MODELS]), '']
lines += ['시간 NLL은 고정된 이분산 lognormal 모형이 기록된 정수 시간 구간에 부여한 확률의 음의 로그다. '
          '이번 기존 Test 평가의 동결 소스와 점수 정의를 그대로 사용했다. 과거 v0.7의 clamped time loss와 혼합하지 않으며 새 likelihood나 효율 지표를 계산하지 않았다.', '',
          '## 해석 범위', '',
          '- 기존 Test를 이미 본 이후 수행한 탐색적 분석이다. 전체 결과와 모든 미리 정한 구간을 함께 공개하며, 유리한 구간을 새로운 주 평가로 대체하지 않는다.',
          '- 작은 수량·큰 수량 구간의 상반된 오차는 개선 과제를 정하는 근거다. 수량 구간을 바꾸거나 큰 수량을 제거해 성능을 올리는 근거가 아니다.',
          '- 세 seed의 표준편차는 학습 난수에 따른 변동이다. 여러 구간의 유의성, 통계적 동등성, 독립 평가의 일반화 신뢰구간을 새로 주장하지 않는다. 기존 전체 평가의 불확실성 결과는 원본 보고서에 유지한다.',
          '- 기존 비교 범위에 따른 Deep Renewal 제외는 결과 확인 이후의 사용자 결정이다. 연구 기록은 보존되며 이 표로 모든 수요 모델 대비 우월성을 주장하지 않는다.',
          '- 별도 TitanTPP B 평가와 A100 탐색 후보는 추가하지 않았다. validation 탐색 결과와 이번 Test 결과를 합쳐 모델을 고른 확증 결과로 표현하지 않는다.', '',
          '## 검증 및 산출물', '',
          f'- 합성 계약 테스트 32개 통과. 경계값·동일 시각·빈 구간·가변 표본 수·seed 집계·메타데이터 불일치 처리를 확인했다.',
          f'- 실제 108조건의 receipt·checkpoint/source 연결과 예측 {parts:,}개 SHA를 확인했다. '
          f'모든 모델·seed의 대상·정답·입력 이력이 동일함을 검증했다.',
          f'- 각 구간 분할의 N·절대 오차 합·제곱 오차 합·NLL 합·signed error 합이 전체로 복원된다. '
          f'원래 전체 seed 지표·평균·표본 SD의 {reconciliations:,}개 비교가 허용오차 rtol=atol=1e-10 내에서 일치한다.',
          f'- 독립 구현은 MLP·S2P2의 24개 예측 벡터와 8개 단순 기준선을 다시 계산했다. '
          f'{I["comparisons"]:,}개 수치 비교가 통과했으며 지표 최대 절대 차이는 {I["max_absolute_metric_difference"]:.3g}다. '
          '주 분석기의 시간 나눗셈·bincount 대신 명시적 상한·불리언 마스크·직접 합계를 사용했다.',
          '- 이번 검증은 저장 예측의 무결성과 집계 검증이다. 새로운 checkpoint binary 로딩·CPU replay 감사를 수행한 것으로 표현하지 않는다.', '',
          f'- [모든 구간·모든 모델 지표 CSV]({ROOT}/results/metrics.csv)',
          f'- [조건별 seed 지표와 오차 합 CSV]({ROOT}/results/seed_metrics.csv)',
          f'- [구간별 대상·개체·날짜 설명 CSV]({ROOT}/cohort_descriptions.csv)',
          f'- [주 집계 및 검증]({ROOT}/results/verification.json)',
          f'- [독립 수치 검증]({ROOT}/independent_verification.json)',
          f'- [전체 기계 판독 결과]({ROOT}/results/analysis.json)', '',
          '## 남은 작업 순서', '',
          '**검증된 구간 차이로 원고 해석을 보완한다 — 다음 작업**',
          '- 대상: 현재 LNCS 원고의 기존 Test 설명. 전체 결과를 유지하고 Intermittent의 이득, Instacart 큰 수량의 오차, Taxi·RAF의 한계를 함께 반영한다. 이번 작업에서는 원고를 수정하지 않았다.',
          '**추가 개선 후보는 validation 탐색과 연결해 검토한다 — 다음 작업**',
          '- 대상: 별도 A100 탐색 캠페인. 이번 Test 진단은 개선 가설을 세우는 자료이며 새 후보의 확증 평가와 구분한다. 추가 학습·Test 추론은 이번 작업에 포함되지 않는다.', '']

with (ROOT / 'REPORT.md').open('x') as handle:
    handle.write('\n'.join(lines))
cohort_rows = []
for d, evidence in I['datasets'].items():
    for view, bins in evidence['cohort_descriptions'].items():
        for bi, values in bins.items():
            cohort_rows.append(dict(dataset=d, split='test', view=view, bin_id=bi,
                                    bin_label=label(A['results'][d][view][bi]), **values))
with (ROOT / 'cohort_descriptions.csv').open('x', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(cohort_rows[0]))
    writer.writeheader()
    writer.writerows(cohort_rows)
receipt = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
           'split': 'test', 'scope': A['scope'], 'learned_conditions': 108, 'failed_conditions': 0,
           'test_targets': targets, 'prediction_rows': rows, 'prediction_parts_sha_verified': parts,
           'independent_learned_vectors': 24, 'deterministic_vectors': 8,
           'synthetic_tests_passed': 32, 'new_training': False, 'new_inference': False,
           'remote_access': False, 'manuscript_edited': False, 'checkpoint_cpu_replay_performed': False,
           'files': {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/'analysis_contract.json', ROOT/'results/analysis.json',
                     ROOT/'results/verification.json', ROOT/'independent_verification.json', ROOT/'results/metrics.csv',
                     ROOT/'results/seed_metrics.csv', ROOT/'cohort_descriptions.csv', ROOT/'REPORT.md',
                     ROOT/'analyze_strata.py', ROOT/'test_analyze_strata.py', ROOT/'verify_outputs.py', Path(__file__)]}}
with (ROOT / 'COMPLETION.json').open('x') as handle:
    json.dump(receipt, handle, indent=2, allow_nan=False)
    handle.write('\n')
print(json.dumps({'status': 'complete', 'report': str(ROOT/'REPORT.md')}))
