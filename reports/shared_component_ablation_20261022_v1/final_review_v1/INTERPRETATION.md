# A–D 결과 해석과 남은 연구 판단

근거: [전체 Validation/Test·비용 표](/Users/igwanhyeong/PycharmProjects/paper_research/reports/shared_component_ablation_20261022_v1/final_review_v1/RESULT_SUMMARY.md) · [원본 파일과 split 연결](/Users/igwanhyeong/PycharmProjects/paper_research/reports/shared_component_ablation_20261022_v1/final_review_v1/EVIDENCE_INDEX.md) · [432개 seed 차이](/Users/igwanhyeong/PycharmProjects/paper_research/reports/shared_component_ablation_20261022_v1/final_review_v1/ALL_SEED_DELTAS.csv)

A=정적 prototype+이력 보정, B=이력 보정만, C=prototype만, D=공통 encoder. 모델 이름은 미정이며 원논문 MAC 비교는 계속 보류한다.

## 실제 완료 범위

학습·재사용 36조건(신규 30·기존 A 6), 선택본 Validation 36, 고정 Test 36, Train 공통 batch의 학습/평가 비용 216회가 완료됐다. 복구 중 과학 학습을 재시작하지 않았다. 기존 실패·checkpoint·원본 833개 보존 파일을 유지했다. 회수와 별도 비교 완료를 구분했고 비교는 순수 로컬 재계산으로 검증했다.

## 구성요소 효과는 데이터와 지표에 따라 갈린다

1. Taxi Test 전체 수량 RMSE는 A 108.043±12.836, B 118.881±18.674, C 174.065±34.662, D 160.180±25.494다. 평균 비교에서 A는 C보다 37.93%, B는 D보다 25.78% 낮아 이력 보정의 수량 예측 효용을 지지한다. 다만 Test recorded TimeNLL은 A 1.01692, B 1.21550, C 0.81411, D 0.79674로 같은 결론이 아니다.
2. prototype의 추가 가치는 일정하지 않다. RAF Test A–B RMSE 차이는 평균 0.104%에 그친다. Taxi와 Intermittent Validation에서는 B가 A보다 RMSE가 낮지만, 고정 Test 평균에서는 A가 B보다 각각 9.12%·1.53% 낮다. Validation/Test의 순위 차이를 감추거나 Test를 근거로 새로운 모델/seed를 선택하지 않는다.
3. RAF Test 전체 RMSE는 C 39.4167±0.1802와 D 39.4194±0.2067가 A 39.7611±0.1760·B 39.8024±0.1937보다 낮다. 반면 A/B의 recorded TimeNLL은 C/D보다 낮다. Intermittent Validation RMSE는 D 1.72449±0.04340가 가장 낮고, Test에서는 C 2.79947±0.13910가 낮다. A의 모든 데이터 우월성이나 두 요소의 보편적 필수성을 주장할 수 없다.
4. 불리한 결과 205개를 432개 전체 seed 차이와 함께 보존했다. Intermittent Test 큰 수량의 seed52는 A RMSE 21.2539, B 13.8946으로 A가 52.97% 높다. A의 큰 수량 평균 12.6550±7.8292도 변동이 크다. B의 Intermittent Test recorded TimeNLL 30.4833±11.8709는 A 13.3127±6.8946·D 13.1925±7.2968보다 높다.

## 비용은 GPU·seed·모드를 나눠 판단한다

prototype을 제거한 B는 A보다 Train 공통 batch 프로필의 최대 allocated 메모리가 RAF 54.39%, Taxi/Intermittent 40.41% 낮다. 세 seed의 값이 같았으며 다른 프로세스 예약을 제외한 해당 profiler의 allocator peak다. Intermittent B의 연산 batch 지연시간 감소율은 5080/seed42=31.83%, 5090/seed52=24.04%, 5090/seed62=23.26%다. 서로 다른 GPU의 절대 시간 평균을 만들지 않았다.
Intermittent 평가 모드 allocated peak는 A 389.2417MiB·B 389.2256MiB로 거의 같다. 학습 비용의 절감률을 평가 메모리 절감률로 옮기지 않는다. 전체 parameters는 RAF A 88,451/B 84,355/C 78,851/D 74,755, Taxi·Intermittent A 99,459/B 95,363/C 89,859/D 85,763이다.
비용 eval은 Test split 속도가 아니라 고정 Train batch의 평가 모드이며 joint loss 계산을 포함한다. train은 선택 checkpoint 복사본과 새 AdamW의 37step(예열 5·측정 32)으로, 원래 fit 재개가 아니다. 기술 반복 3회와 훈련 3seed를 구분한다. 실제 fit의 elapsed/epoch는 Validation·저장을 포함한 상각 시간이다. 과거 A의 병렬 fit 시간과 신규 단독 fit 시간은 직접 비교하지 않는다.

## 이 실험이 뒷받침하는 범위

현재 근거는 사건 수량·시간 예측에서 이력 보정과 정적 prototype의 정확도·비용 절충이 데이터에 따라 달라진다는 것이다. 두 모듈을 모두 추가하면 일관되게 더 정확하고 더 효율적이라는 주장은 성립하지 않는다. 공유 15차원·8분기 자체의 최적성·독창성, 외부 모델 전체 대비 우월성, 같은 정확도까지의 총학습비용 우위는 이 A–D 비교만으로 확정하지 않는다.
기존 외부 5seed는 원래 A의 근거로 보존한다. B/C/D의 외부 5seed 결과처럼 이름을 옮기지 않는다. 새 학습·단순 보정기·MAC·원고 작성은 추가 실행하지 않았다.

## 원본 split과 파일

- raf_spare_parts / validation: [A seed52 원본 receipt](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/shared_component_ablation_20261022_v1/retrieval/test_cost/5090/originals/run/raf_spare_parts__full__seed52/validation_receipt.json) · 다른 모든 arm/seed는 [전체 원본 인덱스](/Users/igwanhyeong/PycharmProjects/paper_research/reports/shared_component_ablation_20261022_v1/final_review_v1/EVIDENCE_INDEX.md)에 연결했다.
- raf_spare_parts / test: [A seed52 원본 receipt](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/shared_component_ablation_20261022_v1/retrieval/test_cost/5090/originals/run/raf_spare_parts__full__seed52/test_receipt.json) · 다른 모든 arm/seed는 [전체 원본 인덱스](/Users/igwanhyeong/PycharmProjects/paper_research/reports/shared_component_ablation_20261022_v1/final_review_v1/EVIDENCE_INDEX.md)에 연결했다.
- yellow_trip_hourly / validation: [A seed52 원본 receipt](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/shared_component_ablation_20261022_v1/retrieval/test_cost/5090/originals/run/yellow_trip_hourly__full__seed52/validation_receipt.json) · 다른 모든 arm/seed는 [전체 원본 인덱스](/Users/igwanhyeong/PycharmProjects/paper_research/reports/shared_component_ablation_20261022_v1/final_review_v1/EVIDENCE_INDEX.md)에 연결했다.
- yellow_trip_hourly / test: [A seed52 원본 receipt](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/shared_component_ablation_20261022_v1/retrieval/test_cost/5090/originals/run/yellow_trip_hourly__full__seed52/test_receipt.json) · 다른 모든 arm/seed는 [전체 원본 인덱스](/Users/igwanhyeong/PycharmProjects/paper_research/reports/shared_component_ablation_20261022_v1/final_review_v1/EVIDENCE_INDEX.md)에 연결했다.
- intermittent_frozen_5000 / validation: [A seed52 원본 receipt](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/shared_component_ablation_20261022_v1/retrieval/test_cost/5090/originals/run/intermittent_frozen_5000__full__seed52/validation_receipt.json) · 다른 모든 arm/seed는 [전체 원본 인덱스](/Users/igwanhyeong/PycharmProjects/paper_research/reports/shared_component_ablation_20261022_v1/final_review_v1/EVIDENCE_INDEX.md)에 연결했다.
- intermittent_frozen_5000 / test: [A seed52 원본 receipt](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/shared_component_ablation_20261022_v1/retrieval/test_cost/5090/originals/run/intermittent_frozen_5000__full__seed52/test_receipt.json) · 다른 모든 arm/seed는 [전체 원본 인덱스](/Users/igwanhyeong/PycharmProjects/paper_research/reports/shared_component_ablation_20261022_v1/final_review_v1/EVIDENCE_INDEX.md)에 연결했다.

평균±SD는 훈련 seed 3개의 표본SD(ddof1)이며, 통계적 유의성 검정이나 모든 배포환경으로의 일반화 판정이 아니다. 5080/cuDNN91900과 5090/cuDNN92000의 평가환경 차이도 함께 기록했다.
