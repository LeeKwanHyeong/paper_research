# 수량 중심 논문 성립 여부 감사 — 2026-09-07

**수량 중심으로 논문 범위를 정하는 것은 가능하지만, B가 다른 모델보다 수량에서
공통으로 우월하다는 결론은 현재 성립하지 않는다.** 동일 raw-RMSE selector의
Instacart seed42 비교가 이미 완료됐으며 B는 RMTPP·THP 양쪽에 MAE와 RMSE가 높다.
최종 multi-seed/held-out 논문 증거도 아직 완성되지 않았다.

대상은 `paper_research / codex/aligned-k1-duration-head`, 감사 시작 HEAD는
`32a5401`이다. 로컬 B artifact와 계약을 검증하고, 로컬에 없던 baseline 증적은
5080에 보관된 5080/5090 실행 JSON 및 파일 해시를 읽기 전용으로 확인했다.
이번 감사는 학습·inference·held-out 평가를 실행하지 않았다.

## 동일 selector의 실제 완료 결과

Instacart validation targets는 모두 **503,733**개이며 target identity와 raw quantity
hash가 고정 계약에 일치한다. 전체 MAE/RMSE는 같은 raw-RMSE-selected checkpoint다.

| 모델 | 선택 epoch | 종료 epoch / 최대 예산 | Quantity MAE ↓ | Raw RMSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Hard-LMM B | 72 | 112 / 300 | 3.993781 | 5.872217 |
| RMTPP | 40 | 80 / 300 | **3.985841** | **5.836961** |
| THP | 50 | 90 / 300 | 3.991003 | 5.852342 |

B는 RMTPP 대비 MAE **0.1992%**, RMSE **0.6040%** 높고, THP 대비 각각
**0.0696%**, **0.3396%** 높다. 차이는 작지만 과거의 엄격한 seed42 gate는 실패다.
한 seed만으로 통계적 열등성·동등성·개선 불가능성을 주장하지 않는다.

Baseline source는 `9440609673cfdf7bcfec5c77d7f677affd04ae49`, B source는
`f75243473adc25d622319dbca9bda7e076d8240f`다. RMTPP는 5080, THP는 5090에서
실행됐다. 5080의 보관 사본을 현재 확인했으며 5090 장비 자체의 현재 상태를
감사한 것은 아니다. 실제 target·학습·selector 계약을 대조했다.

## 완료·누락 구분

| 항목 | Intermittent | Taxi | Instacart | 처리 |
| --- | --- | --- | --- | --- |
| B seed42 full fit | 완료 | 완료 | 완료 | 3개 재사용 |
| RMTPP·THP CUDA·e1 | 2개 완료 | 2개 완료 | 2개 완료 | 정상 실행 증적, 성능 결론에 사용하지 않음 |
| RMTPP·THP seed42 full fit | 2개 미실행 | 2개 미실행 | 2개 완료 | Instacart gate 실패로 후속 4개 중단 |
| 동일 selector seeds52·62 | 6개 미완료 | 6개 미완료 | 6개 미완료 | 완결 비교 시 총 18개 필요; 자동 실행하지 않음 |
| 최종 held-out | 잠금 | 잠금 | 잠금 | 완결 3×3×3 설계라면 27 checkpoint 평가 |

미실행/미완료는 감사한 frozen campaign 범위다. 모든 외부 경로에 대해 새 실행이
전혀 없다는 전역 탐색 주장은 하지 않는다. Controller의 종료 상태는
`stopped_gate_failed`, 완료 job 8개(6 e1 + 2 full fit), additional seeds/test false다.

따라서 **전부 재학습할 필요는 없다.** 기존 5개 full fit과 6개 e1을 재사용할 수
있다. 완결된 혼합 결과 논문으로 진행할 때 남는 학습은 seed42 4개 + 추가 seed
18개이며, 이는 최대 누락 inventory다. 호환되는 실제 checkpoint/trajectory가
추가 발견되면 다시 감사해 재사용한다. 과거 성공 조건에 따른 자동 후속 실행은
중단된 상태이므로 이 수치를 새 실행 승인이나 대기열로 해석하지 않는다.

## 기존 A/B 개선의 정확한 의미

| 데이터셋 | A MAE | B MAE | A RMSE | B RMSE |
| --- | ---: | ---: | ---: | ---: |
| Intermittent | 0.762085 | 0.604340 | 1.722403 | 1.499555 |
| Taxi | 51.767732 | 28.674020 | 181.537594 | 88.194997 |
| Instacart | 4.026535 | 3.993781 | 5.974160 | 5.872217 |

이 양의 결과는 seed42 validation에서 **같은 Hard-LMM의 selector/early-stopping
정책 변경**에 따른 관측이다. Baseline도 같은 selector로 바꾸었을 때의 상대적
우위와 구분해야 한다. 기존 joint-selector Taxi·Instacart 3-seed 표는
[이전 validation 표](../titantpp_v0_7_validation_freeze_20260905/paper_tables.md)에
보존하고 새 B 주 비교표에 섞지 않는다.

## 검증 결과와 한계

- 로컬 B 3개: checkpoint·summary·history SHA-256 일치, 선택 epoch 독립 재계산,
  train 전체 처리 건수, validation 수량 지표 대조 통과.
- Baseline 8개: complete status, passed receipt, 실행 source/계약/data/split hash
  binding, validation count 대조 통과. e1의 CUDA·복원 검증은 기존 receipt를 재사용.
- Instacart full fit 2개: checkpoint·resume·history·summary·launch 파일 해시 재확인,
  history의 가장 이른 raw-RMSE minimum을 독립 재계산했다. Validation target
  identity/quantity hash와 train 매 epoch 처리 건수가 일치한다.
- 5개 수량 구간의 count-weighted MAE와 squared-RMSE를 합산해 전체 지표를
  재계산했다. 원본 audit 허용오차 `rtol=1e-10, atol=1e-8` 안에서 일치한다.
  Remote snapshot의 strict `summary_metric_match=false`는 원본 값을 보존했다.
  History raw RMSE와 receipt 사이 약 1e-14 차이는 허용오차 내이며 비교 방향을
  바꾸지 않는다. 새로운 raw prediction inference는 하지 않았다.
- seeds52·62 불확실성이 남고, validation을 반복 사용한 탐색 결과다. 최종
  일반화 주장은 별도 잠긴 held-out 결과가 필요하다.

재현: 저장소에서 `python3 paper/results/quantity_focused_feasibility_audit_20260907/verify_evidence.py`.
[검증 결정 JSON](decision.json), [원격 receipt/status snapshot](remote_evidence.json),
[원격 독립 확인](remote_independent_checks.json)을 함께 보존한다.

## 확정한 계약과 남은 순서

**완료 / 로컬 — 모델·평가·허용 주장 범위 동결**

- [수량 중심 계약 v1](../../contracts/quantity_focused_model_claim_contract_v1.md)에
  B를 고정 quantity 모델로, 동일 selector RMTPP/THP를 비교군으로 명시했다.
- 세 데이터셋 모두 유지한다. Quantity MAE/RMSE를 주 표에 둘 수 있으나 기존
  시간 열화는 limitation/부록에 남긴다. 최종 논문 증거 완료와 출판 가능성은 미확정이다.

**다음 작업 / 로컬 — 혼합 결과를 포함한 완결 평가 계약 작성**

- 누락된 Taxi·Intermittent 비교를 포함하는 평가가 논문의 주장에 필요한 범위를
  정한다. Instacart 패배를 보존하고 기존 all-win gate 실패를 변경하지 않는다.
- 추가 seed·held-out의 새 실행 계약과 source 재사용 범위를 정한 뒤 필요한
  실행만 준비한다. 현재 세션에서 GPU 작업을 시작하지 않는다.

**보류 / 로컬 — 공통 K=2의 median·초기화·학습 가능성 검증**

- 사용자 요청의 수량 논문 확정 선행 조건이 충족되지 않았다. K=2 구현·테스트는
  아직 하지 않았다. 시간 head는 quantity를 고정하므로 이번 수량 순위를 바꾸지 않는다.
- 향후 진행 시 기여는 우선 시간분포 head 개선으로 구분한다. 이는 K=2가
  불가능하다거나 수량 중심 논문 자체가 불가능하다는 판정은 아니다.
