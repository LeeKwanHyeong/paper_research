# Causal QKV Hard-LMM — 5090 실행 기록

- 저장소/브랜치: `paper_research / codex/hard-lmm-causal-qkv`
- 실행 소스: `a7795355920d4bdc5fb63dc40deccbfcbd4daec0`
- 비교 집계 수정: `46bf2bc` (body를 B와 동일한 ≤train p95로 고정)
- 서버: RTX 5090, Python 3.12.13, PyTorch 2.11.0+cu130
- 소스 경로: `/home/leekwanhyeong/workspace/paper_research_causal_qkv_a7795355920d_5090`
- 결과 경로: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/hard_lmm_causal_qkv_a7795355920d_5090_20260907`
- tmux: `hard_lmm_causal_qkv_a779535_5090`
- 시간별 확인 heartbeat: `hard-lmm-causal-qkv-5090`

## 완료된 검증

첫 encoder block의 event Q/K/V에 kernel3 인과적 channelwise residual을
추가했다. h64 기준 추가 파라미터는 576개이며 zero-init에서 B의 공통 상태,
출력과 gradient를 보존한다. 기존 head, loss, raw-RMSE selector와 static
top4 memory 계약은 유지한다.

로컬 관련 통합 테스트는 122개 통과, CUDA 전용 2개 skip이다.
실제 B 세 데이터셋 산출물의 metadata로 factory 재구성을 확인했으며,
복원 보강 후 관련 audit/campaign 18개 테스트를 다시 통과했다.
5090에서는 63개 테스트가 skip 없이 통과했다. 이 중 실제 CUDA device를
사용한 초기 동일성·gradient·학습 경로 검사가 3개다.

커밋 archive의 SHA-256을 양쪽에서 검증했다:
`2b3375e65ad473205828a113c0a07f5601822714a71f5834e01faa60cdeb423d`.

## CUDA 비용 검사 — 통과

같은 synthetic 입력 batch128/h64에서 B와 후보를 3회씩 순서 교대로 측정했다.
각 회는 warmup5, forward/backward/clip/AdamW 측정15 step을 포함한다.
아래는 각 모델의 반복별 중앙값을 다시 집계한 비율이다.

| 입력 길이 | 후보/B step 시간 | 후보/B peak allocated memory |
| --- | ---: | ---: |
| 8 | 1.1866 | 1.0543 |
| 64 | 1.1809 | 1.0665 |
| 256 | 1.0704 | 1.0497 |

세 길이 모두 시간 ≤1.5배, 메모리 ≤1.25배 기준을 통과했다.
이는 synthetic step 비용 증적이며 full-data epoch 시간이나 성능 향상 증적은 아니다.
Q/K/V의 current·lag1·lag2 gradient와 학습값이 모두 활성화됐고,
학습된 kernel을 0으로 바꾸면 최종 state·수량 예측이 변하며 원상 복원된다.

## 실행 순서와 판정

Campaign은 2026-09-07 20:11:02 KST에 시작했다.
비용 검사 후 Intermittent→Taxi→Instacart full-data e1을 검증하고,
모두 통과하면 Instacart→Taxi→Intermittent seed42 screening으로 이어진다.
Screening은 최대300 epoch, 최소40 epoch, patience40이며 B와 동일하게
validation raw RMSE의 가장 이른 최솟값을 선택한다.

한 데이터셋이라도 frozen B-relative gate를 놓치면 후속 데이터셋을 중단한다.
실행·표본·복원 감사 실패도 후속 학습을 차단한다. 추가 seed와 held-out test는
이 campaign의 실행 범위에 포함하지 않는다.

실제 최신 단계와 마지막 확인 시점은 함께 보관한 `campaign_status.json`을
기준으로 한다. e1 수치만으로 예측 성능의 개선 여부를 판정하지 않는다.

## 2026-09-07 20:16:26 KST 확인

세 데이터셋의 e1, 전체 처리 건수, target identity·quantity hash,
수량 구간/표본 수, finite 계산과 모델·optimizer 저장·복원 감사가 모두 통과했다.

| 데이터셋 | 처리한 train target | validation target | Body ≤p95 target |
| --- | ---: | ---: | ---: |
| Intermittent | 393,824 | 86,285 | 81,739 |
| Taxi | 38,393 | 8,268 | 7,888 |
| Instacart | 1,991,192 | 503,733 | 477,507 |

Instacart seed42 screening은 20:15:51 KST에 시작했고 확인 당시 실행 중이다.
이 단계의 성능 판정은 아직 없다. 종료 후 frozen gate를 통과할 때만
Taxi와 Intermittent를 이어서 실행한다. 시간별 heartbeat가 종료·오류·
판정 결과를 확인한다.
