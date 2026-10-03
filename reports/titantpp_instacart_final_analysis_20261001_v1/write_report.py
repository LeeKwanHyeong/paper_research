"""Render the validated local analysis without changing the manuscript."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

O = Path(__file__).resolve().parent
R = O.parents[1]
x = json.loads((O / 'analysis.json').read_text())
M = ('qty_mae', 'qty_rmse', 'time_nll')
P = x['representative']
labels = dict(zip(x['models'], ['TitanTPP MLP', 'RMTPP', 'THP', 'NHP', 'SAHP', 'S2P2', 'AttNHP']))
groups = {(r['model'], r['endpoint']): r for r in x['three_seed']}
audit_path = R / x['new_5090_audit']['path']
audit = json.loads(audit_path.read_text())
collection = json.loads((audit_path.parent / 'original/collection_manifest.json').read_text())
kst = lambda t: datetime.fromtimestamp(t, ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
fmt = lambda r: ' | '.join(f"{r[m]['mean']:.6f} ± {r[m]['sample_sd']:.6f}" for m in M)
lines = ['# Instacart 최종 비교: MAE의 작은 이득과 RMSE의 열세', '',
         f"분석 완료: {x['completed_analysis_kst']}. 5090 학습 큐 종료: {kst(collection['status']['updated_unix'])}.", '',
         '**5090의 6조건 원본 회수와 CPU 검증을 완료했다. Instacart에서 TitanTPP MLP가 외부 최상위 모델보다 우수하다는 결론은 나오지 않았다.** '
         'S2P2·AttNHP보다 평균 MAE는 조금 낮지만 평균 RMSE는 높고, RMSE는 두 모델에 대해 세 seed 모두 열세다. '
         '7모델 중 TitanTPP의 평균 순위는 MAE 2위, RMSE 6위, 시간 NLL 5위다.', '',
         '대표 모델은 기존 `titantpp_history_mlp`다. 모든 값은 동일한 최초 최소 raw 수량 RMSE checkpoint의 validation 결과이며, '
         '평균과 표본표준편차는 seed42·52·62를 동일 가중치로 계산했다. 표본 수는 seed당 503,733개이고 '
         '같은 validation 표본을 세 번 독립 데이터셋처럼 합산하지 않는다.', '',
         '## 1. 7모델의 최종 3seed 비교 — 완료', '',
         '모든 지표는 낮을수록 좋다. 외부 모델은 동일 입력·head·loss·선택 기준으로 연결한 비교군이다.', '',
         '| 모델 | 수량 MAE | 수량 RMSE | 시간 NLL |', '|---|---:|---:|---:|']
for model in x['models']:
    lines.append(f"| {labels[model]} | {fmt(groups[model, 'selected'])} |")
lines += ['', '**MAE 최저는 RMTPP, RMSE와 시간 NLL 최저는 S2P2다.** '
          'TitanTPP는 RMTPP보다 MAE가 0.1604% 높고, S2P2보다 RMSE가 0.5210% 높다. '
          '작은 차이라는 점과 방향이 불리하다는 점을 함께 기록한다. 세 seed의 표준편차가 작거나 겹친다는 사실만으로 '
          '통계적 동등성·유의성을 판정하지 않는다.', '',
          '## 2. 추가 비교군 대비 차이와 seed별 일관성 — 완료', '',
          '개선율 = (비교군 평균 − TitanTPP 평균) / 비교군 평균 × 100. 양수는 TitanTPP 이득, 음수는 열세다. '
          '승수는 동일 seed에서 TitanTPP의 지표가 더 낮은 횟수다.', '',
          '| 비교군 | MAE 개선율 | RMSE 개선율 | MAE 승 | RMSE 승 | 시간 NLL 승 |', '|---|---:|---:|---:|---:|---:|']
for p in x['paired']:
    lines.append(f"| {labels[p['comparator']]} | {p['qty_mae']['plain_reduction_percent']:+.4f}% | "
                 f"{p['qty_rmse']['plain_reduction_percent']:+.4f}% | {p['qty_mae']['plain_wins']}/3 | "
                 f"{p['qty_rmse']['plain_wins']}/3 | {p['time_nll']['plain_wins']}/3 |")
lines += ['', 'S2P2 대비 평균 MAE는 0.1781%, AttNHP 대비 0.1057% 낮다. 두 비교 모두 MAE 승수는 2/3이다. '
          '반면 평균 RMSE는 각각 0.5210%, 0.3063% 높으며 승수는 모두 0/3이다. '
          '따라서 “추가 TPP 대비 수량 예측이 개선됐다”보다 “MAE와 RMSE에서 서로 다른 결과를 보였다”가 정확하다. '
          'NHP에 대해서는 두 수량 지표와 시간 NLL이 세 seed 모두 낮지만, 이것을 외부 비교군 전체의 우위로 넓히지 않는다.', '',
          '| 모델 | seed | 선택 / 종료 epoch | MAE | RMSE | 시간 NLL |', '|---|---:|---:|---:|---:|---:|']
for s in (42,52,62):
    for model in [P,'s2p2_matched_head','attnhp_matched_head']:
        r = next(r for r in x['rows'] if r['model']==model and r['seed']==s and r['endpoint']=='selected')
        last = next(z for z in x['rows'] if z['model']==model and z['seed']==s and z['endpoint']=='last')
        lines.append(f"| {labels[model]} | {s} | {r['epoch']} / {last['epoch']} | " + ' | '.join(f'{r[m]:.6f}' for m in M) + ' |')

lines += ['', '## 3. MAE와 RMSE의 방향이 다른 이유 — 관측된 오차 분포', '',
          '수량 구간은 원래 동결된 경계 8·20·25·35를 사용한다. `searchsorted(..., side="left")`에 따라 '
          '아래 구간의 상한이 포함된다. 수량은 Instacart에서 기록된 같은 사용자·활동일의 product-row 수로 정의한 basket-size proxy다.', '',
          '| 실제 수량 구간 | 표본 수 | 비중 | TitanTPP MAE / RMSE | S2P2 MAE / RMSE | AttNHP MAE / RMSE |',
          '|---|---:|---:|---:|---:|---:|']
q_labels = ['≤8','8 < q ≤20','20 < q ≤25','25 < q ≤35','>35']
for i,label in enumerate(q_labels):
    rr = [next(r for r in x['selected_strata'] if r['model']==model and r['partition']=='quantity' and r['bin']==i)
          for model in [P,'s2p2_matched_head','attnhp_matched_head']]
    n = rr[0]['count']
    lines.append(f'| {label} | {n:,} | {100*n/503733:.2f}% | ' +
                 ' | '.join(f"{r['qty_mae']['mean']:.4f} / {r['qty_rmse']['mean']:.4f}" for r in rr) + ' |')
low_share = 100*(247651+202534)/503733
high_share = 100*(27322+20190+6036)/503733
tail_share = 100*6036/503733
lines += ['', f'S2P2와 비교하면 TitanTPP는 수량 20 이하의 {low_share:.2f}%에서 평균 MAE·RMSE가 낮다. '
          f'그러나 수량 20 초과의 {high_share:.2f}%에서는 두 지표가 모두 높다. 가장 큰 수량 구간(>35)은 '
          f'{tail_share:.2f}%에 불과하지만 평균 RMSE가 TitanTPP 24.6419, S2P2 23.4686이다.', '',
          '저수량 두 구간의 이득은 전체 표본당 평균 제곱오차 차이에 −1.5882, 고수량 세 구간의 열세는 +1.9458을 기여한다. '
          '합계 +0.3576 때문에 전체 RMSE는 S2P2보다 높아진다. 반면 절대오차 합계에서는 저수량 이득이 조금 더 커 전체 MAE는 낮다. '
          '이 계산은 세 seed의 구간별 SSE를 전체 표본 수로 나누어 합한 값이며, 구간 RMSE를 단순 평균해 전체 RMSE를 만든 것이 아니다.', '',
          'AttNHP에 대해서는 가장 작은 수량 구간(≤8)에서 TitanTPP가 유리하고, 나머지 네 구간의 평균 수량 지표는 AttNHP가 유리하다. '
          '따라서 단순히 “대부분 구간에서 더 좋다”는 설명도 비교군에 따라 달라진다.', '',
          '이력 길이별 수치도 `analysis.json`과 `strata.csv`에 보존했다. S2P2는 모든 비어 있지 않은 이력 길이 구간에서 '
          '평균 RMSE가 낮다. 따라서 현재 결과만으로 긴 이력이 부족한 문제 하나를 원인으로 확정할 수 없다. '
          '위 구간 분석은 오차가 어디에서 발생했는지 설명하며, 모델 구조가 왜 그 오차를 만들었는지에 대한 인과 검증은 아니다.', '',
          '## 4. 활성 분기 정규화의 Instacart 결과 — seed42 탐색으로 분리', '',
          '이미 감사된 4090→5080 복구 결과를 재사용한다. 선택30/종료70epoch, MAE 3.990506, RMSE 5.871661, 시간 NLL 2.802719다. '
          '동일 seed의 plain MLP 대비 MAE 0.02495%, RMSE 0.12025% 감소했지만, S2P2·AttNHP의 같은 seed RMSE보다 여전히 높다. '
          '정규화의 3seed 우위 또는 대표 모델 교체를 뒷받침하는 결과로 쓰지 않는다. 보류 중인 seed52·62를 새로 실행하지 않았다.', '',
          '## 5. 선택·마지막 checkpoint와 실측 비용을 보존한다 — 완료', '',
          '종료 후 마지막 checkpoint도 별도로 남긴다. 주 결과는 위의 RMSE 선택 checkpoint이며, 마지막 지표로 유리한 결과를 바꿔 고르지 않는다.', '',
          '| 모델 | 마지막 MAE | 마지막 RMSE | 마지막 시간 NLL |', '|---|---:|---:|---:|']
for model in x['models']:
    lines.append(f"| {labels[model]} | {fmt(groups[model,'last'])} |")
lines += ['', '| 5090 조건 | 선택 / 종료 | 실측 fit 시간(h) | epoch 기록 합(h) | peak allocated(MiB) |', '|---|---:|---:|---:|---:|']
for r in x['recorded_cost']['conditions']:
    lines.append(f"| {labels[r['job']['arm']]} seed{r['job']['seed']} | {r['selected_epoch']} / {r['completed_epochs']} | "
                 f"{r['fit_elapsed_seconds']/3600:.4f} | {r['epoch_timing_seconds_sum']/3600:.4f} | {r['peak_allocated_bytes']/1024**2:.3f} |")
lines += ['', f"6조건의 `train_one` 실측 합은 {x['recorded_cost']['fit_elapsed_seconds_sum']/3600:.4f}시간이다. "
          '이는 epoch 기록 합·캠페인 전체 경과시간과 다르다. 준비·최종 재평가·대기 시간을 추정 보정하지 않았다. '
          '개인 5090 추가 cloud 임대료는 0달러이며 전기료는 측정하지 않았다. 다른 종료 epoch와 다른 모델의 이 시간을 '
          '기존 TitanTPP 대비 공정한 단독 속도 비교로 해석하지 않는다.', '',
          '## 6. 원본 회수·검증 범위 — 완료', '',
          f"- 5090 원본 {audit['verified_retrieved_files']}파일·동결 소스 {audit['verified_source_files']}개·checkpoint {audit['verified_terminal_checkpoints']}개를 회수·검증했다. 서버 원본은 변경하지 않았다.",
          '- 계약·동결 source closure·approval/permit·실제 native qualification·초기 tensor SHA·모델 경로를 대조했다. '
          'CPU strict loading, 모델 tensor SHA, optimizer 상태와 전체 step, RNG·shuffle 복원을 확인했다.',
          '- 전체 train/validation 노출, 기존 B/Full과 공통 배치 prefix, 최초 최소 RMSE 선택과 patience40 종료, '
          'selected/last 재평가 기록·구간별 합계를 확인했다. native 합성60updates는 실제 fit step에서 제외했다.',
          '- 추가 6조건의 optimizer step은 seed42 각1,742,384, seed52 각793,407, seed62 각1,166,775다. '
          '마지막 epoch와 종료 후 추가 validation의 exposure SHA를 구분해 검증했다.',
          '- 기존 Core 및 외부4종의 완료 감사와 선택/마지막 endpoint를 재사용했다. 7모델×3seed×2역할=42기록을 '
          '원래 동결 CSV 및 tensor SHA와 대조했다. 현재 checkout의 과학 코드로 결과를 다시 생성하지 않았다.',
          '- 회수 시 해당 캠페인 소유 프로세스는 없었다. 첫 CPU 도구 선택에서 번들 Python에 torch가 없어 '
          '기존 5080 감사와 같은 `/usr/local/bin/python3` 환경을 사용했다. 패키지 설치·원격 Runtime 변경은 없다.',
          '- 로컬 CPU 감사는 새 CUDA 예측 재현이 아니다. 저장된 CUDA RNG 바이트와 기존 native 검증 기록을 확인했다. '
          'raw 데이터와 held-out/test는 읽지 않았고 새 학습·재평가·GPU 측정은 실행하지 않았다.', '',
          '## 논문 해석과 남은 순서', '',
          '**Instacart 결과 분석 — 완료**', '',
          '현재 결론은 “TitanTPP는 Instacart에서 일부 비교군보다 MAE가 조금 낮지만, 최상위 비교군 대비 수량 정확도 전반의 우위는 확보하지 못했다”다. '
          'Taxi·Intermittent에서의 기존 개선은 별도 데이터 결과로 유지하고, Instacart는 고수량 오차와 지표 간 상충을 보여주는 결과로 포함한다.', '',
          '**원고와 공통 비교표에 최종 수치를 반영한다 — 다음 작업**', '',
          '이번 요청에 따라 분석을 먼저 완료했으며 원고와 기존 보고서는 변경하지 않았다. 다음에는 Table 4의 '
          'Instacart Pending 두 행을 이번 3seed 결과로 교체하고, “미완료” 문구·관련 해석·주장 근거를 함께 갱신한다. '
          '원고의 기존 수치가 오래된 상태라는 사실을 이 분석 완료와 혼동하지 않는다.', '',
          '**독립 평가 규칙을 고정한다 — 이후 작업**', '',
          '모델·비교군·선택 checkpoint·지표·평가 범위를 고정한 뒤 별도 승인 범위에서 독립 평가를 진행한다. '
          '현재 validation 결과만으로 최종 일반화·통계적 유의성을 주장하지 않는다.', '',
          '## 증거 파일', '',
          f'- [5090 원본 CPU 감사]({audit_path})',
          f'- [회수 영수증]({audit_path.parent / "retrieval_receipt.json"})',
          f'- [원본 파일 목록]({audit_path.parent / "original/collection_manifest.json"})',
          f'- [기계판독 분석]({O / "analysis.json"})',
          f'- [선택·마지막 조건별 표]({O / "conditions.csv"})',
          f'- [수량·이력 길이 구간별 표]({O / "strata.csv"})',
          f'- [검증 결과]({O / "verification.json"})',
          f'- [입력 파일 SHA 목록]({O / "source_manifest.json"})', '']
(O / 'report.md').write_text('\n'.join(lines))
(O / 'README.md').write_text('# Instacart 최종 회수·검증·분석 — 완료\n\n'
    '5090 추가 TPP 6조건의 원본 회수와 CPU 검증, 기존 TitanTPP 및 외부6종의 3seed 비교를 완료했다. '
    '현재 원고와 기존 전체 비교 보고서는 이번 작업에서 수정하지 않았다.\n\n'
    f'- [결과 분석]({O / "report.md"})\n- [기계판독 결과]({O / "analysis.json"})\n'
    f'- [검증]({O / "verification.json"})\n\n'
    '`build_analysis.py`와 `write_report.py`는 회수된 validation 원본으로 분석을 재생성한다. '
    '`collect.py additional`은 5090 읽기 전용 회수이며 기존 목적지를 덮어쓰지 않는다. '
    '`audit.py additional`은 CPU 상태 복원·기록 검증이며 forward를 호출하지 않는다.\n')
print(O / 'report.md')
