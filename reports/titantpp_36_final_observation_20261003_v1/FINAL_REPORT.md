# TitanTPP 기존 36조건·이전 A100 6조건 최종 관측

5090의 마지막 Instacart / Deep Renewal native NB / seed62가 2026-10-03T09:30:32+09:00에 완료됐다. 최종 저장은 86epoch, 선택은 46epoch다. 2026-10-03 09:42:20 KST의 단일 읽기 전용 관측에서 서버 종료 상태, 소유 프로세스·GPU PID 부재, terminal scientific_success, 소형 기록 SHA 및 selected/last validation replay 완료를 확인했다.

| 캠페인 | 완료 | 실패 | 진행 | 미시작 | CPU 감사 완료 |
|---|---:|---:|---:|---:|---:|
| 후속 36조건 | 36 | 0 | 0 | 0 | 34 |
| 이전 A100 cross-product/recent-four-mean | 6 | 0 | 0 | 0 | 4 |

완료는 학습 및 terminal·소형 기록·validation replay 확인을 뜻한다. 전체 연구 감사 완료를 뜻하지 않는다. 모든 학습이 종료되어 학습 ETA는 더 이상 적용하지 않는다.

## 관측 출처

| 서버 | 완료 | 실제 관측 시각 KST | 이번 회차 |
|---|---:|---|---|
| 5080 | 24/24 | 2026-10-02T19:53:10+09:00 | 종료 증거 재사용 |
| 5090 | 9/9 | 2026-10-03T09:42:20+09:00 | 새 관측 |
| pro4500 | 3/3 | 2026-10-03T05:58:04+09:00 | 종료 증거 재사용 |
| 이전 A100 | 6/6 | 2026-10-03T04:47:38+09:00 | 종료 증거 재사용 |

[최종 관측 분석](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/hourly_monitor/20261003T004220094498Z/analysis.json) · [5090 원관측](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/hourly_monitor/20261003T004220094498Z/5090/snapshot.json)

## 후속 36조건 전체 목록

모델명과 seed를 포함한 조건 ID를 유지한다. `best / 종료`는 선택 epoch와 총 완료 epoch이며, 전 조건 상태는 완료다.

| 조건 ID | GPU | best / 종료 | 원본 회수 | CPU 감사 |
|---|---|---:|---|---|
| `insta_market_basket__42__deep_renewal_event_native_nb` | 5090 | 13 / 53 | SHA 확인 | 통과 |
| `insta_market_basket__42__titantpp_all_available_history_mlp` | 5090 | 30 / 70 | SHA 확인 | 통과 |
| `insta_market_basket__42__titantpp_current_only_param_matched` | 5090 | 30 / 70 | SHA 확인 | 통과 |
| `insta_market_basket__52__deep_renewal_event_native_nb` | 5090 | 13 / 53 | SHA 확인 | 통과 |
| `insta_market_basket__52__titantpp_all_available_history_mlp` | 5090 | 15 / 55 | SHA 확인 | 통과 |
| `insta_market_basket__52__titantpp_current_only_param_matched` | 5090 | 15 / 55 | SHA 확인 | 통과 |
| `insta_market_basket__62__deep_renewal_event_native_nb` | 5090 | 46 / 86 | 미완료 | 미완료 |
| `insta_market_basket__62__titantpp_all_available_history_mlp` | 5090 | 35 / 75 | SHA 확인 | 통과 |
| `insta_market_basket__62__titantpp_current_only_param_matched` | 5090 | 1 / 41 | SHA 확인 | 통과 |
| `intermittent_frozen_5000__42__deep_renewal_event_native_nb` | 5080 | 113 / 153 | SHA 확인 | 통과 |
| `intermittent_frozen_5000__42__titantpp_all_available_history_mlp` | 5080 | 10 / 50 | SHA 확인 | 통과 |
| `intermittent_frozen_5000__42__titantpp_current_only_param_matched` | 5080 | 21 / 61 | SHA 확인 | 통과 |
| `intermittent_frozen_5000__52__deep_renewal_event_native_nb` | 5080 | 114 / 154 | SHA 확인 | 통과 |
| `intermittent_frozen_5000__52__titantpp_all_available_history_mlp` | 5080 | 11 / 51 | SHA 확인 | 통과 |
| `intermittent_frozen_5000__52__titantpp_current_only_param_matched` | 5080 | 60 / 100 | SHA 확인 | 통과 |
| `intermittent_frozen_5000__62__deep_renewal_event_native_nb` | pro4500 | 102 / 142 | SHA 확인 | 미완료 |
| `intermittent_frozen_5000__62__titantpp_all_available_history_mlp` | pro4500 | 67 / 107 | SHA 확인 | 통과 |
| `intermittent_frozen_5000__62__titantpp_current_only_param_matched` | pro4500 | 42 / 82 | SHA 확인 | 통과 |
| `raf_spare_parts__42__deep_renewal_event_native_nb` | 5080 | 259 / 299 | SHA 확인 | 통과 |
| `raf_spare_parts__42__titantpp_all_available_history_mlp` | 5080 | 9 / 49 | SHA 확인 | 통과 |
| `raf_spare_parts__42__titantpp_current_only_param_matched` | 5080 | 5 / 45 | SHA 확인 | 통과 |
| `raf_spare_parts__52__deep_renewal_event_native_nb` | 5080 | 300 / 300 | SHA 확인 | 통과 |
| `raf_spare_parts__52__titantpp_all_available_history_mlp` | 5080 | 5 / 45 | SHA 확인 | 통과 |
| `raf_spare_parts__52__titantpp_current_only_param_matched` | 5080 | 5 / 45 | SHA 확인 | 통과 |
| `raf_spare_parts__62__deep_renewal_event_native_nb` | 5080 | 244 / 284 | SHA 확인 | 통과 |
| `raf_spare_parts__62__titantpp_all_available_history_mlp` | 5080 | 19 / 59 | SHA 확인 | 통과 |
| `raf_spare_parts__62__titantpp_current_only_param_matched` | 5080 | 8 / 48 | SHA 확인 | 통과 |
| `yellow_trip_hourly__42__deep_renewal_event_native_nb` | 5080 | 300 / 300 | SHA 확인 | 통과 |
| `yellow_trip_hourly__42__titantpp_all_available_history_mlp` | 5080 | 163 / 203 | SHA 확인 | 통과 |
| `yellow_trip_hourly__42__titantpp_current_only_param_matched` | 5080 | 35 / 75 | SHA 확인 | 통과 |
| `yellow_trip_hourly__52__deep_renewal_event_native_nb` | 5080 | 300 / 300 | SHA 확인 | 통과 |
| `yellow_trip_hourly__52__titantpp_all_available_history_mlp` | 5080 | 59 / 99 | SHA 확인 | 통과 |
| `yellow_trip_hourly__52__titantpp_current_only_param_matched` | 5080 | 116 / 156 | SHA 확인 | 통과 |
| `yellow_trip_hourly__62__deep_renewal_event_native_nb` | 5080 | 300 / 300 | SHA 확인 | 통과 |
| `yellow_trip_hourly__62__titantpp_all_available_history_mlp` | 5080 | 298 / 300 | SHA 확인 | 통과 |
| `yellow_trip_hourly__62__titantpp_current_only_param_matched` | 5080 | 54 / 94 | SHA 확인 | 통과 |

## 이전 A100 6조건 전체 목록

Intermittent의 원래 2epoch 비용 gate 중단과 승인된 동일 fit 재개를 하나의 조건으로 집계했다.

| 조건 ID | best / 종료 | 원본 회수 | CPU 감사 |
|---|---:|---|---|
| `intermittent_frozen_5000__42__titantpp_history_cross_product` | 1 / 41 | SHA 확인 | 미완료 |
| `intermittent_frozen_5000__42__titantpp_history_recent4_mean` | 68 / 108 | SHA 확인 | 미완료 |
| `raf_spare_parts__42__titantpp_history_cross_product` | 53 / 93 | SHA 확인 | 통과 |
| `raf_spare_parts__42__titantpp_history_recent4_mean` | 16 / 56 | SHA 확인 | 통과 |
| `yellow_trip_hourly__42__titantpp_history_cross_product` | 111 / 151 | SHA 확인 | 통과 |
| `yellow_trip_hourly__42__titantpp_history_recent4_mean` | 95 / 135 | SHA 확인 | 통과 |

## 원본 회수·비용·삭제

두 Pod 모두 원본 SHA 회수 이후 삭제 및 목록 부재가 확인됐다. 아래 금액은 준비 시도와 저장료를 포함한 추정이며 개별 청구서 금액이나 계정 잔액 차감 귀속액이 아니다.

| 소유 Pod | 원본 파일 | 삭제 확인 KST | 총 추정 / 승인 상한 |
|---|---:|---|---:|
| `751nbij4r4cwm2` | 449 | 2026-10-03T04:48:04+09:00 | $15.05 / $150 |
| `eu8yh29yympuf4` | 270 | 2026-10-03T05:58:34+09:00 | $10.91 / $25 |

`751nbij4r4cwm2`: [원본 회수](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_mlp_candidates_a100_20261002_v1_retry1/retrieved/retrieval_receipt.json) · [삭제 증거](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_mlp_candidates_a100_20261002_v1_retry1/control/cleanup_receipt.json) · [비용 보고](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_mlp_candidates_a100_20261002_v1_retry1/cost_report.json)

`eu8yh29yympuf4`: [원본 회수](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_intermittent_seed62_runpodpro4500_20261002_v1_retry1/retrieved/retrieval_receipt.json) · [삭제 증거](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_intermittent_seed62_runpodpro4500_20261002_v1_retry1/control/cleanup_receipt.json) · [비용 보고](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_intermittent_seed62_runpodpro4500_20261002_v1_retry1/cost_observation_20261003_v1.json)

## 남은 감사 — 미완료

- 5090 Instacart Deep Renewal seed62: checkpoint 원본 회수 및 CPU 감사.
- PRO4500 Intermittent Deep Renewal seed62: 원본 회수 완료, CPU 감사 미완료.
- 이전 A100 Intermittent cross-product·recent-four-mean seed42: 원본 회수 완료, CPU 감사 미완료.

기존 감사 13조건, 추가 감사 22조건(후속 18 + A100 4), 구조 대조 최종 감사 3조건을 재사용했다. PRO4500 all-available seed62는 후속 감사 통과를 반영했다. 이번 회차에는 원격 학습·replay·추가 감사·원본 회수를 실행하지 않았다.

## 기록 보존과 모니터 종료 범위

- 최초 준비 오류, PRO4500 준비 실패, A100 Intermittent 비용 gate 중단·재개 이력은 보존한다. 최종 실패 0은 이 이력을 삭제하거나 없었던 것으로 간주한다는 뜻이 아니다.
- 5080 예약 claim의 FileExistsError는 승인된 이관 경계이며 새 학습 실패로 집계하지 않는다.
- Deep Renewal은 원고 비교에서 제외됐어도 학습·연구 기록을 보존한다. 이 보고서는 validation-only이며 held-out/test 또는 혼합 결과를 열람하지 않았다.
- 두 캠페인과 두 기존 Pod의 종료·회수·비용·삭제 조건을 충족했으므로 `titantpp-36`만 삭제한다. 실제 삭제 결과는 이 폴더의 `automation_deletion_receipt.json`에 남긴다.
- 새 A100 routing/placement 캠페인과 Pod `c88rn6dvhdafj0`, 다른 자동화, 기존 PAUSED 모니터, 채팅 archive 상태는 이번 종료 범위에 포함하지 않는다.

세부 조건별 snapshot SHA·CPU 감사 원본 경로와 validation 지표는 [기계 판독 보고서](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_36_final_observation_20261003_v1/final_observation.json)에 보존한다.
