# Matched frozen log-normal duration comparison (seed 42)

## 결론

Frozen-B에 정상화된 heteroscedastic log-normal duration head를 붙이는 경로는 기술 계약을 모두 통과했지만, **세 데이터셋의 공통 Time NLL 개선안으로는 채택하지 않는다.** Intermittent와 Instacart에서는 사전 기준인 `B <= A + 0.01`을 만족했으나, Taxi에서 B의 validation proper Time NLL이 A보다 `0.71011` 높았다.

이 결과는 수량 성능의 손상이나 head 구현 오류로 설명되지 않는다. B의 encoder, memory, quantity head와 수량 예측은 모든 validation target에서 보존되었다. 모든 모델의 epoch-0 proper Time NLL도 데이터셋별로 정확히 같았다. 같은 likelihood, 초기화, 최적화 조건에서 학습 후 격차가 나타났으므로, Taxi에서는 raw-RMSE로 선택된 B checkpoint의 frozen 표현에 시간 예측에 필요한 정보가 충분히 남아 있지 않다는 해석이 가장 직접적이다.

## Proper Time NLL

| Dataset | A | B | RMTPP | THP | B - A | `B <= A + 0.01` |
|---|---:|---:|---:|---:|---:|:---:|
| Intermittent | 0.93990 | 0.78736 | 미측정 | **0.51672** | -0.15253 | 통과 |
| Taxi | -0.44356 | 0.26656 | -0.42602 | **-0.62582** | +0.71011 | **실패** |
| Instacart | 2.81493 | 2.81683 | **2.81326** | 2.81439 | +0.00190 | 통과 |

음의 NLL은 연속분포의 density가 1보다 클 수 있기 때문에 유효하다. 데이터셋마다 시간 단위와 분포가 다르므로 NLL을 데이터셋 사이에서 합산하거나 평균내지 않았다.

## B 수량 성능 보존

| Dataset | A MAE | B MAE | 변화 | A RMSE | B RMSE | 변화 |
|---|---:|---:|---:|---:|---:|---:|
| Intermittent | 0.76209 | **0.60434** | -20.70% | 1.72240 | **1.49956** | -12.94% |
| Taxi | 51.76773 | **28.67402** | -44.61% | 181.53759 | **88.19500** | -51.42% |
| Instacart | 4.02653 | **3.99378** | -0.81% | 5.97416 | **5.87222** | -1.71% |

Frozen duration head 적합 전후의 B 수량 예측은 bitwise identical이다. 위 A/B 비교는 A가 historical joint-objective checkpoint, B가 사전에 고정한 raw-RMSE checkpoint라는 선택 차이를 포함한다. 따라서 proper Time NLL 결과는 각 논문용 수량 checkpoint에 남아 있는 frozen representation 정보의 비교이며, backbone architecture만의 순수 효과로 해석하지 않는다.

## 실행 계약과 범위

- 세 데이터셋에 동일한 130-parameter duration head, AdamW 설정, validation proper Time NLL selector를 사용했다.
- 시간 통계와 Instacart 30일 right-censoring 규칙은 train split에서만 확정했다.
- encoder, memory, quantity head와 cached hidden state는 고정했으며 held-out test는 열지 않았다.
- 신규 A/RMTPP/THP 8개 행은 CUDA e1 후 full 적합과 독립 replay 감사를 통과했다.
- 기존 B 3개 행은 `c04d32b` artifact의 source, checkpoint, non-time state를 다시 인증해 재사용했다.
- Intermittent RMTPP full-e300 checkpoint 파일이 유실되어 전체 12개 중 11개 행만 존재한다. 이 누락은 A 대 B 판정에는 영향을 주지 않지만 세 데이터셋 전체의 4-model 우위 주장은 허용하지 않는다.
- seed 42 공통 gate가 Taxi에서 실패했으므로 추가 seed는 실행하지 않았다.

## 재현 및 감사 증적

- Source commit: `7c35821522b044033fc1cf14016297eb5fd093b6`
- Contract SHA-256: `1152303afbe9152844038efc725164333eaec1ce1fcc3a02c539efd5437bf476`
- Transfer archive SHA-256: `79ec5aed61403bc0d59a49453dc09e18617225c970aa8d025d328700c502b3a5`
- CUDA contract tests: 46 passed
- 로컬 관련 테스트: 56 passed
- 실행 시간: controller artifact 시각 기준 약 610초
- 원격 artifact: `/home/leekwanhyeong/artifacts/matched_frozen_lognormal_duration_seed42_5090_20260906_7c35821`
- 로컬 compact artifact: `search_artifacts/matched_frozen_lognormal_duration_seed42_5090_20260906_7c35821`
- 재생성 가능한 feature cache 2.6 GiB는 로컬 회수에서 제외했다.
- 회수한 72개 파일은 원격과 파일별 SHA-256이 모두 일치한다.

정량 행은 `metrics.csv`, 실행·무결성 판정은 `validation_audit.json`, 회수 파일별 digest는 `compact_artifact_sha256.txt`에 기록했다.

## 다음 판단

현재 후보는 B의 수량 성능을 정확히 보존한다는 목적은 달성했지만, frozen `h`만 읽는 시간 head로 Taxi의 시간 성능을 회복할 수 없었다. 같은 수량 경로를 유지하면서 다음 후보를 검토한다면 B의 수량 경로에는 gradient를 보내지 않고, log-duration 이력만 읽는 작은 공통 causal time adapter를 별도로 두는 것이 최소 변경이다. 이 후보는 세 데이터셋에 같은 구조와 규칙을 적용하고, 먼저 train/validation-only Taxi 진단에서 B의 `h`보다 추가 시간 정보를 제공하는지 확인한 뒤 구현 여부를 정해야 한다.
