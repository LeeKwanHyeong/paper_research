# BOUNDED-QK Hard-LMM 로컬 구현·검증

2026-09-08 · `paper_research / codex/hard-lmm-causal-qkv`

**Encoder를 실제로 변경하는 BOUNDED-QK 후보와, 선택된 Backbone을 동일한
정상화 시간 likelihood로 비교할 평가 경로를 구현했다.** 이번 기록은 로컬 CPU의
계약·저장·복원 검증이다. 새 후보의 벤치마크 성능, CUDA 실행 가능성 또는 비용
기준 통과를 뜻하지 않는다. 신규 GPU 학습과 held-out 평가는 실행하지 않았다.

## 현재 구현

- `models/TPPs/CountAwareTitanBoundedQK.py`: 첫 encoder block의 Q/K causal
  residual을 head별 원 projection RMS에 맞춰 제한한다. V에는 기존 FULL의
  causal residual 연산을 그대로 사용한다. hidden64에서 B보다 576개 파라미터가
  많고 FULL과 같은 수다. 새 kernel은 0이며 초기화 RNG를 추가 소비하지 않는다.
- 기존 FULL attention에서 projection 결합 연산만 hook으로 분리했다. 기존
  비영(非零) kernel을 사용한 전후 비교에서도 출력과 gradient가 bitwise 일치했다.
- 전용 backbone 이름은 `titantpp_hard_memory_bounded_qk`, 학습 역할은
  `hard_lmm_bounded_qk_candidate`다. 학습 runner에 등록하고 기존 raw-RMSE
  selector, log quantity MSE, legacy time objective 및 최대 300 epoch 계약을 유지했다.
- `paper/scripts/run_backbone_normalized_duration.py`: 선택된 B/FULL/BOUNDED의
  source·dataset·cache identity를 확인하고, 각자의 frozen hidden state로 동일한
  K=1 duration head만 학습한다. 실제 캠페인에서는 검증을 마친 aligned-B를
  재사용하며 FULL과 새 후보의 시간 head를 새로 평가한다.

고정 수식과 채택 기준은 이전 커밋 `80cc675`의
`paper/contracts/hard_lmm_bounded_qk_causal_v_v1.md`에 있다. 그 파일의
“구현 전” 상태는 당시 설계 동결 시점의 기록으로 보존했다. 이번 구현 상태는
이 보고서와 `verification.json`을 기준으로 한다.

## 검증한 범위

최종 결과는 **143 passed, CUDA 1 skipped, 이전 source SHA 검사 2 deselected**다.
명령, Python·PyTorch 버전 및 실행 종료 상태는
`verification.json`과 `local_tests.txt`에 기록했다. 시험 소스는 `80cc675`의
독립 임시 checkout에 이번 변경만 복사한 것이다. 작업 디렉터리에 있던 별도
`CountAwareTPP.py` 변경과 미추적 `scripts/`, `.agents/` 파일은 포함하지 않았다.
검증한 소스 파일별 SHA는 `source_overlay_manifest.json`에 기록한다.

첫 회귀 실행에서는 신규 단위 테스트가 임시 archive에 없는 `.git`을 읽는
문제 1개와, 이전 FULL 실행의 고정 source SHA 검사 2개가 실패했다.
신규 단위 테스트는 목적에 맞게 합성 revision을 주입하도록 수정했다. 이전
FULL의 두 검사는 바뀐 소스를 거절하는 것이 정상이며, 기존 계약은 수정하지
않고 최종 실행에서 명시적으로 제외했다. FULL의 연산 보존은 별도의 이전
소스 대조로 검사했다. 최초 실행 기록은 `initial_regression_tests.txt`,
`initial_regression_verification.json`에 보존한다.

1. **모델 계약**: B·zero-init FULL과 초기 출력/공통 gradient/kernel gradient/RNG
   일치, Q/K residual bound, V 연산 일치, causality, padding·미래 target 차단,
   batch 독립성, 짧은 이력, persistent token 경로, finite 계산을 확인한다.
   작은 합성 학습에서 kernel·gain·검색 similarity·수량 예측이 실제로 변하는지도
   검사한다. CUDA가 없는 로컬 결과를 CUDA 통과로 해석하지 않는다.
2. **실제 학습 runner의 합성 실행**: hidden64 모델을 합성 train/validation
   합계 84개 입력 행으로 2 epoch 학습한다. 1 epoch 중단 후 재개한 실행과 무중단
   실행의 최종 model·optimizer·history·loader RNG 및 선택 checkpoint를 비교한다.
3. **정상화 시간 평가**: source와 데이터의 교차 사용, 불완전 full-data 증적,
   selector·시간 학습 설정 불일치를 거절한다. 연속 density·정수 PMF·censored
   survival 계산과 head 학습·optimizer 재개를 검증한다. 전체 `run()` 경로는
   합성 parquet의 train 4개/validation 2개 target으로 hidden 추출→1 epoch
   head fit→선택 모델 독립 복원→완료 결과 재사용까지 실행한다.
   이 실행은 full-data 또는 full-fit 결과로 인정되지 않는다.
4. **기존 FULL 보존**: 이전 `80cc675` 소스와 eval/train × FP32/CPU BF16
   autocast의 4조건에서 출력·입력 gradient·모든 파라미터 gradient가 bitwise
   일치한다. 결과와 재현 코드는 `full_hook_parity.json`,
   `replay_full_hook_parity.py`에 있다.

전체 시간 평가 테스트에서 발견한 위치 임베딩 길이 누락은 FULL/BOUNDED의
encoder metadata에 `max_len`을 저장하도록 수정했다. 따라서 저장된 K=1
checkpoint도 올바른 길이로 모델을 재구성한 뒤 strict restore할 수 있다.

## 기존 증적 재사용

`aligned_B_recovery.json`의 74개 검사는 모두 통과했다. 5080의 기존
`aligned_k1_8213dbc_5080_20260907`에서 세 aligned-B checkpoint의 파일·state
SHA와 여섯 hidden cache의 digest·target identity·finite 조건을 확인했다.
checkpoint와 소형 metadata만 로컬로 회수했고 약 899.90 MiB cache는 전송하지
않았다. 원격 학습·모델 forward·코드 변경은 하지 않았다. **B head 재학습은 필요 없다.**

| 데이터셋 | 재사용 aligned-B normalized NLL |
|---|---:|
| Intermittent | 0.7873644338 |
| Taxi | 0.6956474254 |
| Instacart | 2.8129205141 |

위 값은 기존 결과이며 새 Backbone 성능이 아니다. 데이터셋 간 평균을 만들지
않는다. 기존 legacy time score와 수치상 직접 비교하지 않는다.

`source_compatibility.json`은 실제 B/FULL source checkpoint 6개를 strict
restore하고 새 K=1 builder를 통과시킨 결과다. 소스 파일·state·학습 설정·데이터
identity와 전체 epoch 이력의 earliest raw-RMSE minimum·조기 종료를 확인했다.
수량 보존은 **합성 입력만**으로 확인했고 trainable head는
130개다. 이 검사에는 임의의 합성 시간 초기값을 사용했으므로 실제 train 초기값,
Time NLL 또는 성능 평가로 사용할 수 없다. 재현 스크립트는
`paper/scripts/audit_bounded_qk_source_compatibility.py`다.

## 해석의 한계

Q/K residual 크기가 기존 일반화 실패의 원인이라고 입증된 상태는 아니다.
이번 후보는 V의 유용한 경로를 유지하면서 Q/K 혼합 강도를 제한하는 단일 가설이다.
V 연산 보존이나 초기 출력 일치가 학습 후 MAE·RMSE 보존을 보장하지 않는다.
추가 norm 연산은 O(Ld)이지만 attention 전체는 여전히 O(L²)다.

정상화 K=1 head는 Backbone과 분리해서 학습한다. 각 Backbone의 수량 경로는
그 head 학습 전후 bitwise 보존되어야 한다. K=1 head 자체의 개선을 Backbone
기여로 계산하지 않고, 동일한 head를 사용한 B/FULL/BOUNDED 비교로 구분한다.

## 남은 작업 순서

**다음 작업 / 로컬 → GPU 실행 준비 — 실행 소스와 manifest를 동결한다.**
- 이번 독립 source commit과 실제 서버·runtime·데이터·checkpoint·history를
  묶은 실행 manifest를 결과 생성 전에 저장한다. 이번 evaluator의 Git revision과
  runner SHA 검사는 전체 소스 배포 무결성 검사를 대신하지 않는다. 전송 archive의
  모든 소스 SHA 검증과 실행 전 receipt는 GPU launcher에서 완료해야 한다.
- 기존 비용 profiler를 B/FULL/BOUNDED 3행으로 확장하고 실행·중단 순서를 고정한다.
  현재 profiler의 기존 B/FULL 결과만으로 새 후보 비용을 판단하지 않는다.

**후속 작업 / GPU — CUDA·비용·실데이터 e1 계약을 먼저 검증한다.**
- 고정 source를 별도 경로에서 실행한다. batch128/hidden64/L8·64·256,
  warmup5/측정15/반복3으로 B와 FULL 모두에 대해 비용을 측정한다.
  B 대비 step≤1.5배·peak allocated memory≤1.25배 조건을 적용한다.
- 세 데이터셋 후보 e1에서 처리 건수·gradient·checkpoint/optimizer 복원을
  확인한다. 기존 e1을 새 후보 e1로 대체 인정하지 않는다.

**후속 작업 / GPU — 고정 후보 seed42와 FULL의 정상화 시간 기준선을 평가한다.**
- 계약 통과 후 새 후보만 최대300/min40/patience40으로 학습한다. 기존 B와
  FULL의 수량 학습은 재사용한다. FULL과 BOUNDED는 각각 자신의 hidden cache를
  사용해 동일 K=1 head를 평가하고 기존 aligned-B와 비교한다.
- 세 데이터셋 B 대비 raw RMSE 감소 및 MAE/time guardrail, FULL 이득 보존을
  설계 계약대로 판정한다. 기존 FULL의 Instacart 실패 기록을 유지한다.
  추가 seed와 최종 held-out은 screening 이후 별도 단계다.
- 이번 변경은 `paper_research / codex/hard-lmm-causal-qkv`의 독립 커밋이다.
  `paper_research/develop`, `paper_research/master` 병합 또는 push는 포함하지 않는다.
