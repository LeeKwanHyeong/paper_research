# Hard-LMM 경과시간 표현 후보 — 로컬 구현·계약 검증

**완료: 로컬 구현과 통합 테스트 122개 통과, CPU float16 검사 1개 제외.** 해당 검사는 승인된 5090 CUDA 단계에서 수행한다. 이 기록에는 새 후보의 실제 데이터 학습이나 성능 채택 결과가 없다.

대상은 `paper_research/master`, 구현 전 기준선은 `9155cd3`이다. 사용자가 승인한 계약·로컬 구현·독립 커밋·5090 CUDA/전체 e1 범위에서 진행했다. 기존 미추적 `scripts/`는 변경하거나 포함하지 않았다.

## 확정한 후보

`titantpp_elapsed_age_static_memory`는 기존 separate-key encoder의 attention 점수에 **causal prefix의 정규화된 경과시간과 사건 순서 위치 간 차이**를 더한다. 두 layer·네 head의 `encoder.elapsed_age_beta[2,4]` 8개 계수를 0으로 초기화한다. 기존 입력·사건 위치 embedding·K/V·top-4·temperature·`h+r` 결합·head·loss·optimizer·checkpoint 선택은 유지한다.

- 각 query가 자신의 관측 prefix만 사용한다. 첫 관측의 경계 바깥 gap은 제외하고, target·padding을 누적 계산 전에 제거한다. 잘못된 observed mask는 거부한다.
- 미래·padding·persistent key의 bias는 0이다. H≤2, zero span, 등간격 이력도 중립 geometry다.
- FP64 누적·비율 계산 후 FP32 geometry를 만들고, 기존 attention 점수 dtype에 맞춰 더한다. Beta=0에서도 gradient 경로를 유지한다.
- 별도 backbone·role·checkpoint identity와 fresh-run 방어를 사용한다. 모델·optimizer를 자동 변환하거나 기존 run을 재개하지 않는다.

전체 규칙과 사전 성능 기준은 [후보 계약](../../contracts/hard_lmm_elapsed_age_v1.md), [기계 판독 계약](../../contracts/hard_lmm_elapsed_age_v1.json), [CUDA/e1 실행 계약](../../contracts/hard_lmm_elapsed_age_cuda_e1_5090_v1.json)에 고정했다.

## 검증 결과

| 검증 | 결과 |
|---|---|
| 같은 seed의 공통 파라미터·RNG 상태 | CPU exact |
| Beta=0의 train/eval 출력 | Separate-key와 CPU exact |
| Geometry 독립 계산 및 기존 J 대응 | 통과 |
| Prefix 인과성·target·padding·persistent 처리 | 통과 |
| Beta gradient·encoder·검색·예측 변화 | 불규칙 합성 이력에서 통과 |
| 극단 유한 입력, CPU bfloat16 | 유한한 forward/backward 확인 |
| Checkpoint route·optimizer 다음 step 복원 | CPU exact |
| h64, 길이64/256 학습 형태 | 합성 forward/backward/AdamW step 통과 |
| 실행 도구의 합성 전체 e1·감사·패키징 | 통과 |
| 기존 separate-key·가중 검색·모델 역할 | 회귀 통과 |

전체 통합 결과는 **122 passed, 1 skipped**, 8.52초다. CPU float16 제외 사유는 [pytest.xml](pytest.xml)에 기록했으며 CUDA 단계에서는 fp16/bf16을 포함해 실패와 skip 없이 통과해야 한다. 실제 GPU 학습 형태는 batch128, h64, 길이64/256이다.

[실행 manifest](execution_manifest.json)는 변경 소스·계약·새 테스트 16개 파일의 hash와 테스트 로그·XML hash를 고정한다. [독립 검토](independent_review.json)는 정적 코드·계약 검토이며 별도 테스트 실행으로 표현하지 않는다.

## 다음 승인된 실행 순서

1. 이 변경을 `paper_research/master`에 독립 커밋하고, 해당 커밋만 패키징한다. 기존 원본 checkpoint는 무결성 감사용 참조이며 fresh 후보 학습에 사용하지 않는다.
2. 5090의 GPU·기존 Runtime·데이터 checksum을 확인한 후 새 snapshot에 전송하고 패키지·파일별 hash를 검증한다. 기존 checkout·서비스·Runtime은 유지한다.
3. CUDA 모델·계약 테스트를 통과하면 Taxi 전체 e1, Instacart 전체 e1 순으로 실행한다. 전체 train/validation 건수, beta·AdamW 갱신, peak 메모리, 저장·복원과 결과 무결성을 감사한다.

E1의 완료 조건은 정상 실행과 계약 준수다. 이후 별도 승인된 성능 평가에서는 원본의 기존 기준과 함께 Instacart의 separate-key 대비 body MAE 5% 개선, Taxi의 body MAE 악화 1% 이하, 두 dataset의 RMSE·tail MAE 악화 1% 이하와 Time NLL 증가 0.01 이하를 동시에 적용한다. 모델 채택과 e300 실행은 이번 로컬 검증으로 결정하지 않는다.
