# Causal QKV Hard-LMM — 두 데이터셋 탐색적 후속 평가

2026-09-08, `paper_research / codex/hard-lmm-causal-qkv`.
기계 판독 계약은 `hard_lmm_causal_qkv_followup_v1.json`이다.
이 문서와 계약은 Taxi·Intermittent 후속 seed42 결과를 보기 전에 고정한다.

## 현재 기준선과 연구 범위

기존 Instacart seed42는 epoch80에서 정상 조기 종료했고 epoch40을 선택했다.
후보의 raw RMSE는 5.8772571642, B는 5.8722169305로 0.0858% 악화했다.
따라서 기존의 세 데이터셋 공통 raw RMSE 개선 기준은 통과하지 못했다.
기존 `hard_lmm_causal_qkv_screening_20260907`의 계약·결과·최종 판정은 수정하지 않는다.

사용자가 승인한 이번 범위는 남은 Taxi와 Intermittent의 탐색적 평가다.
Taxi는 RTX 5080, Intermittent는 RTX 5090에서 병렬로 실행한다.
두 실험은 데이터셋별 효과의 크기와 손익을 확인하기 위한 것이며, Instacart의
실패를 소급해서 통과로 바꾸거나 기존 공통 개선 주장을 회복하는 실험이 아니다.

## 유지하는 구조와 학습 계약

모델은 첫 encoder block의 event Q/K/V에 kernel3 causal residual을 추가한
`titantpp_hard_memory_causal_qkv`다. 모델 구현 기준은
`a7795355920d4bdc5fb63dc40deccbfcbd4daec0`이며, 기존에 추적하던 Python
소스 405개의 SHA-256을 별도 manifest에 고정했다. 실행을 위한 새 orchestration과
전용 테스트만 추가한다. 모델·loader·head·loss·selector는 변경하지 않는다.

- 두 데이터셋 모두 seed42, 최대300 epoch, minimum40, patience40.
- validation raw quantity RMSE의 가장 이른 엄격한 유한 최솟값을 선택한다.
  조기 종료도 같은 지표를 사용한다. 300은 최대치이며, 조기 종료를 해제하지 않는다.
- 기존 B와 동일한 log 수량 MSE, AdamW, batch128, lr0.001, h64,
  gradient clip1, time scale3과 legacy clamped time head를 유지한다.
- 기존 full-data e1 통과 증적을 재사용한다. 데이터·split·population·구간별
  표본 수·저장 복원 감사와 구현 소스 동일성을 검증한 후 연결한다.
- 각 서버에서는 실제 CUDA 단위 검사와 L256 synthetic 학습·finite·복원
  검사를 수행한다. 이 검사는 full-data e1 재실행이 아니다. 기존 5090의
  상대 비용 측정을 5080에서 측정한 비용으로 보고하지 않는다.
- 결과는 validation에 한정한다. 추가 seed와 held-out 평가는 이번 실행에
  포함하지 않으며, 자동으로 시작하지 않는다.

## 데이터셋별 기존 성능 기준

각 데이터셋의 선택 checkpoint를 같은 B 기준선과 비교하여 모두 보고한다.

| 지표 | 데이터셋별 통과 기준 |
| --- | --- |
| raw RMSE | B보다 엄격히 낮음 |
| 전체 MAE | B 대비 악화 1% 이하 |
| body MAE, 관측 수량 ≤train p95 | B 대비 악화 2% 이하 |
| tail MAE, 관측 수량 >train p99 | B 대비 악화 2% 이하 |
| legacy clamped time loss | B 대비 증가 0.01 이하 |

이 time loss는 현재 학습기의 legacy 값이다. 정상화된 다른 시간 likelihood와
숫자를 직접 비교하거나 proper Time NLL 개선으로 해석하지 않는다.

## 후속 검토 기준 — 실행 전 고정

이번에 승인된 두 결과가 모두 완료되고 감사가 통과해야 판단한다. 기존
Instacart 결과를 포함한 세 데이터셋에서 다음을 모두 충족하고,
Taxi·Intermittent 중 적어도 하나의 raw RMSE가 B 대비 1% 이상 개선되면
추가 seed 검토 자격만 부여한다.

- raw RMSE와 전체 MAE 악화 각각 1% 이하.
- body MAE와 >p99 MAE 악화 각각 2% 이하.
- legacy clamped time loss 증가 0.01 이하.

위 조건에 미달하면 이 후보의 탐색적 후속 평가를 종료한다. 충족하더라도
추가 seed를 자동 실행하지 않는다. 이 규칙은 계산 자원을 더 사용할지 검토하기
위한 사전 기준이며, 통계적 비열등성·유의성·최종 채택 판정이 아니다.
seed42 한 개의 작은 차이는 반복 실험의 일관성을 증명하지 못한다.
기존 공통 개선 기준의 실패 판정은 어느 경우에도 보존한다.

## 실행과 중단 규칙

각 서버는 배정된 데이터셋 한 개만 실행한다. 한 데이터셋의 성능 기준 미달로
다른 서버의 이미 승인된 탐색적 실행을 중단하지 않는다. 소스·데이터·CUDA·
학습·결과 감사에 오류가 발생하면 해당 서버의 실행을 중단하고 기록한다.
기존 출력 경로 덮어쓰기나 자동 재시도는 허용하지 않는다.

서버별 소스 archive와 data checksum, GPU/runtime, CUDA 테스트,
합성 학습 검사, full-data launch와 최종 감사 증적을 새 결과 디렉터리
`paper/results/hard_lmm_causal_qkv_followup_20260908`에 보관한다.
시간별 확인은 종료·오류·필요 조치 등 유의미한 변화가 있을 때 알리고,
두 서버가 모두 종료되면 모니터링도 종료한다.
