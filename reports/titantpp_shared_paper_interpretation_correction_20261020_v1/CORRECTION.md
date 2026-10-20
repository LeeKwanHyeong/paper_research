# 논문 근거 해석 정정 — 2026-10-20

기존 연구자료의 수치·원본·선택 checkpoint·SHA manifest는 보존한다. 이 문서는 기존 해석 중 과도한 단정을 정정하는 후속 기록이며, 추가 2seed 학습·입력·손실·head·Runtime을 변경하지 않는다.

## 기존 효율 측정은 완료된 근거다

기존 `reports/titantpp_efficiency_5080_20260930_v1/report.md`에 RTX5080/batch128/maxlen256에서 History MLP와 Titans-MAC event adapter를 실제 Train 표본으로 비교한 비용 측정이 있다. 각 조건을 3회 반복한 24worker가 성공했고, 이번 읽기 전용 확인에서 terminal manifest의 118파일 SHA와 원시 batch 집계가 일치했다. 학습 step 가속은 Taxi 5.742배·Intermittent 5.581배, 평가 batch 가속은 8.494배·8.208배다. 학습 peak allocated는 약 75.5% 감소했지만 평가 peak allocated는 MLP 389.23MiB 대 Titans-MAC 97.83MiB로 증가했다.

따라서 '효율 실측을 진행하지 않았다'는 포괄적 표현은 잘못이다. 기존 History MLP 구조의 Titan 원본 대비 계산 시간 개선은 확보된 실증이다. 비교한 모델은 96,003 대 122,310 parameters의 당시 구조이며, 학습 checkpoint 정확도 비교가 아닌 fresh seed42 process의 실자료 예열 5/측정 32batch 비용 측정이다. 이 배수를 현재 Shared15(길이256에서 99,459 parameters)에 직접 붙이거나 MLP16 대비 추가 가속으로 바꾸지는 않는다. 기존 근거를 재사용할 수 있으며 새 측정을 논문 진행의 필수조건으로 두지 않는다.

## Backbone 전체 비교와 공유 효과의 분리 주장은 다르다

공통 입력·손실·head·학습/선택 조건에서 제안 backbone 전체의 성능과 비용을 비교하는 현재 목적에 '같은 파라미터 수의 추가 대조'를 필수로 요구할 수 없다. 공유만이 개선 원인이라고 분리해 주장할 때 용량 대조가 필요하다. 이번 기여는 제안 구조 전체의 수량 성능·구조 비용으로 평가한다. 공유와 용량 감소의 순수 기여를 분리하지 않았다는 사실을 이유로 backbone 비교 자체를 무효라고 하지 않는다. 기존 memory가 유지됐으므로 새 memory 방법의 기여로 부르지 않는다.

## 이번 공유형의 Test 사용은 확인된 실행 경계로 판단한다

원본 raw9 계약은 Train/Validation 입력 6파일만 포함하며 평가 범위는 validation_only, held_out_test_authorized=false, monitor=validation_raw_quantity_rmse다. Shared15 모델은 2026-10-19 21:58:04KST commit6ada96e9에 존재하고 SHA eee18d3abe199f0c71d80da35acf01cf19acd8f56291ff7cfd33fd86d8bc82a2로 고정됐다. 원본 5090 실행계약 생성 2026-10-20 10:01:01KST 이후 RAF/Taxi Test 결과가 13:44:51KST, Intermittent가 14:14:28KST에 만들어졌다. Test 평가 계약과 결과의 checkpoint_reselection=false를 확인했다. Intermittent 최종 선택도 5/5/11 및 canonical tensor가 Test 고정본과 같았다.

확인된 이번 실행은 Train 학습→Validation 선택→고정 Test 평가다. 평가 결과를 읽었다는 사실만으로 Test를 학습·하이퍼파라미터·checkpoint 선택에 사용했다고 판정할 수 없다. 현재 Shared15 설정이 Test에 맞춰 변경됐다는 근거를 제시하지 못한 상태에서, 과거 다른 실험의 탐색 설명을 이번 평가에 일괄 적용한 것은 과도했다. 기존 문서의 'Test를 참고해 구조를 선택했다/독립 평가로 볼 수 없다'는 포괄적 단정을 확인된 사실로 반복하지 않는다. 사용자는 이번 학습·실험 선정에 Test를 직접 사용하지 않았다고 명시했다. 이 확인만으로 전체 개발 과정에서 Test가 어떤 간접 영향도 주지 않았다고 입증한 것은 아니며, 새로운 미사용 데이터에 대한 전향적 일반화 검증으로 확대하지도 않는다.

## 남는 확인 범위

추가 2seed를 완료해 5seed 평균·변동과 외부 비교를 확정한다. RAF의 평균 미우세·시간 NLL의 반대 방향·일부 seed 역전은 실제 수치이므로 그대로 보고한다. 기존 속도 측정은 측정한 모델과 조건으로 인용한다. 기존 frozen 보고서/계약에 이전 해석 문구가 남아 있으면 원문은 보존하되 이 정정 기록을 현재 해석으로 함께 제시한다. 이 기록은 추가 학습·재측정·기존 결과 덮어쓰기를 시작하지 않는다.
