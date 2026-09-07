# Causal QKV Hard-LMM — 최종 seed42 screening 결과

**판정: 후보 미채택. Instacart raw RMSE의 엄격한 개선 조건에 미달했다.**

대상은 `paper_research / codex/hard-lmm-causal-qkv`의 실행 소스
`a7795355920d4bdc5fb63dc40deccbfcbd4daec0`이며, 모든 학습은 5090에서 실행했다.
첫 encoder block의 event Q/K/V에 zero-init kernel3 residual을 더하는
단일 후보를 평가했다. Head, loss, selector, 데이터 split과 채택 기준은 유지했다.

## 종료와 선택

- Instacart seed42: 2026-09-07 20:15:51 → 23:33:49 KST.
- 최대300/minimum40/patience40 계약에서 80 epoch를 완료했다.
- 가장 이른 validation raw-RMSE 최솟값은 epoch40이다.
- 이후 40 epoch 동안 더 낮은 값이 없어 정상 조기 종료했다.
- 실행·데이터·복원 감사는 통과했다. 성능 gate는 raw RMSE 하나만 실패했다.

## 동일 validation 모집단에서의 최종 비교

| 지표 | B 기준선 | 후보 epoch40 | B 대비 변화 | 고정 조건 판정 |
| --- | ---: | ---: | ---: | --- |
| Raw RMSE | 5.872217 | 5.877257 | +0.0858% | **실패: 엄격한 개선 필요** |
| 전체 MAE | 3.993781 | 3.995619 | +0.0460% | 통과: 악화 ≤1% |
| Body ≤train p95 MAE | 3.438602 | 3.428296 | -0.2997% | 통과: 악화 ≤2% |
| >train p99 MAE | 21.917795 | 22.190152 | +1.2426% | 통과: 악화 ≤2% |
| Legacy clamped time loss | 3.215523 | 3.215251 | -0.000272 | 통과: 증가 ≤0.01 |

Body와 시간 loss는 소폭 개선됐고 전체 MAE는 거의 같지만,
전체 raw RMSE는 B를 넘지 못했다. 이 작은 단일 seed 차이로 통계적 열등성을
확정할 수는 없으나, 사전에 고정한 공통 개선 gate를 통과한 결과도 아니다.
결과를 본 뒤 허용 오차나 selector를 변경하지 않았다.

## 감사와 보관

- 모든 80 epoch가 train target 1,991,192개를 처리했고 finite 검사를 통과했다.
- Validation target 503,733개, body 477,507개와 target identity/quantity
  SHA-256, train-only 경계 `[8,20,25,35]`, 구간별 표본 수가 B와 일치한다.
- Best/last checkpoint의 source와 route, canonical state hash,
  모델 strict restore, optimizer restore를 원격과 로컬에서 검증했다.
- 선택 모델과 last에 저장된 best model의 synthetic 출력이 완전히 일치한다.
- 새 Q/K/V lag1·lag2가 실제로 학습됐고, 미작동 경로로 인한 실패는 아니다.
- 내려받은 summary와 best checkpoint의 파일 SHA-256이 원격 감사 기록과 일치한다.
- `final_decision.json`에 정확한 값과 로컬 재현 감사 결과를 보관했다.
  Binary checkpoint는 로컬/5090 결과 경로에 보관하고 Git에는 hash manifest를 기록한다.

CUDA·비용 검사와 세 데이터셋 e1은 모두 통과했다. B 대비 synthetic step 비용은
약 7~19%, peak allocated memory는 약 5~7% 증가했다. e1은 실행 계약 검증이며,
Taxi·Intermittent의 성능 개선 증적으로 사용하지 않는다.

## 후속 실행 상태

Instacart primary gate 실패에 따라 Taxi·Intermittent seed42 screening,
추가 seeds, held-out test를 실행하지 않았다. 2026-09-08 00:16 KST 확인 시
5090 GPU compute process가 없었고 유휴 상태였다. 시간별 heartbeat를 삭제했다.
이번 campaign은 종료됐으며, 다른 후보의 구현이나 학습은 시작하지 않았다.
