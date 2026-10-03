# TitanTPP 최종 평가의 모델 선택·분석 절차 동결

> **2026-10-03 해석 정정:** 이 문서의 신규·미접근 모집단 요건을 일반적인 test 평가나 투고의 필수 조건으로 확대하지 않는다. 과거 접근과 현재 모델의 선택에 사용한 사실은 구분하며, 현재 모델에 대한 후자의 인과관계는 이 접근 조사로 확인하지 않았다. [해석 정정](../titantpp_test_access_history_review_20261001_v1/report.md)을 우선한다. 아래 실행 준비 미완료·외부 대기는 설계 당시 상태이며, 기존 분할의 [108조건 평가](../titantpp_legacy_evaluation_20261003_v1/FINAL_REPORT.md)는 이후 완료됐다. 동결 JSON·선택 규칙은 보존한다.

작성일: 2026-10-03 KST. **설계 동결 완료 / 독립 평가 모집단 적격성·실행 준비 미완료.**

대표 모델, 비교 대상, 지표와 불확실성 분석 절차를 고정했다. 현재 clean 독립 평가 데이터를 확보했다고 판단하지 않았고, 평가 실행이나 held-out 결과 열람을 시작하지 않았다. 이는 validation 개발과 A100 탐색 결과를 본 뒤 작성한 로컬 설계 기록이다. 개발 전에 등록한 실험이나 공개 사전등록으로 표현하지 않는다.

기계판독 기준은 [design.json](design.json), 기존 selected checkpoint 식별자는 [checkpoint_manifest.json](checkpoint_manifest.json), 데이터별 독립성 판정은 [population_eligibility.json](population_eligibility.json)이다. 이 문서의 설계 동결은 실제 데이터·Runtime·예산까지 채운 실행 계약의 확정을 뜻하지 않는다.

## 현재 확인한 사실

- 원고의 validation은 구조·대체 후보를 개발하는 데 사용됐다. 현재 표를 최종 독립 평가 결과로 바꾸지 않는다.
- Taxi와 Instacart는 현재 test와 이용 가능한 과거 test의 사건 키가 각각 모두 일치한다. 새 [계보 기록](../titantpp_pakdd_extension_preparation_20261001_v1/lineage.json)은 과거 접근을 기록한다. 과거 실행 bytes와 모든 사람의 실제 열람·선택 사용까지 복원한 것은 아니다.
- Intermittent는 현재 `site_cd::part_no`와 과거 품목 매핑·원본 추출 계보가 해결되지 않았다. 문자열 키가 겹치지 않는다는 사실은 독립성을 입증하지 않는다.
- RAF는 제한된 과거 검색에서 test 접근이 확인되지 않았다. 이는 미접근의 증명이 아니며, 2001-12~2002-12를 자동으로 clean final test로 정하지 않는다.
- 이전 metadata 점검에서 held-out 수량 분포 집계가 우연히 노출됐다는 기록도 보존한다. 이번 작업에서는 held-out 수량·간격 열, 예측, 성능 또는 validation/test 혼합 결과를 읽지 않았다.
- 기존 비교 84조건의 selected epoch·tensor SHA는 고정돼 있다. 기록된 로컬 경로의 **48개 binary SHA를 다시 확인**했고, **36개는 해당 경로에 없다**. 다른 위치나 원격의 부재·유실을 뜻하지 않는다. 이번에는 checkpoint를 역직렬화하거나 CPU/GPU forward를 실행하지 않았다.
- 새로 끝난 구조 대조 3조건 감사와 기존21조건 감사를 연결하여 **구조 대조24개 selected binary의 SHA를 모두 재확인**했다. [구조 대조 동결 목록](structural_checkpoint_manifest.json)에 선택 epoch·tensor SHA·원본 소스·실행 GPU를 묶었다. 전체 평가 대상108 learned condition 중72개의 로컬 selected binary가 바인딩돼 있으며 남은36개는 외부 comparator다.

## 1. 대표 모델과 비교 대상을 고정한다 — 완료

대표 모델은 네 데이터 모두 `titantpp_history_mlp` 하나다. 64차원 현재 상태와 같은 직전 관측 상태를 연결하는 8개 bias-free 128→4→64 GELU 분기, 총 보정 파라미터 6,144개, **고정 /8**, output projection의 **0 초기화**, 기존 availability threshold `(1,2,4,8,16,32,64,128)`를 유지한다. 여덟 threshold는 서로 다른 과거 지연을 읽으라는 뜻이 아니다. withheld valid row는 보정 가용성을 초기화하고 padding은 초기화하지 않는 기존 규칙도 유지한다.

이미 학습한 seed42·52·62의 **최초 최소 validation raw quantity RMSE selected checkpoint**를 고정한다. 시간 NLL이나 MAE에 유리한 다른 epoch를 고르지 않는다. 소스 revision `1de2c31e8e3febda7ff20d0527402ea9959b4098`와 조건별 source closure, train-derived 입력·head·관측 코딩은 목록으로 연결했다. 현재 작업 디렉터리의 최신 모델 코드를 원본 소스 대신 사용하지 않는다.

**주 비교:** TitanTPP, RMTPP, THP, NHP, SAHP, S2P2 공통 head adapter, AttNHP 공통 head adapter × 네 데이터 × 세 seed = 84개 selected checkpoint다. 전체 표에 여섯 상대를 모두 남긴다. 기존 validation 평균 RMSE로 고정한 주 상대는 Taxi·Intermittent의 RMTPP, Instacart·RAF의 S2P2다. 평가 결과를 본 뒤 더 유리한 상대를 골라 주 상대를 바꾸지 않는다.

**보조 구조 비교:** 파라미터 수를 맞춘 current-only와 전체 가용 분기 고정 /8 대조군의 24개 selected checkpoint를 사용한다. 최신 감사의 정확한 24행 파일·tensor·source SHA를 실행 입력용 목록에 바인딩했다. 이 비교는 인접 상태 결합과 분기 availability 규칙을 분리하며, 좋은 결과를 낸 대조군을 데이터별 대표 모델로 바꾸지 않는다. PRO4500 이관 출처는 보존하며 효율 측정과 합치지 않는다.

**단순 수량 기준선:** 같은 관측 window의 마지막 raw 수량과 같은 window의 raw 수량 산술평균을 모두 고정한다. target·padding을 제외한 입력은 학습 모델과 동일하다. 두 방법은 deterministic이므로 세 seed의 독립 학습처럼 복제하지 않는다. 수량 점예측만 정의하므로 시간 NLL은 NA다. 빈 history의 임의 fallback을 만들지 않고 공통 next-event target 자격 규칙을 먼저 확인한다.

A100 cross-product와 recent-four-mean은 단일 seed 탐색으로 보존하고 대표 모델로 승격하지 않는다. Deep Renewal은 사용자 요청대로 원고 비교에서 제외하되, 결과가 존재한 이후의 범위 결정임을 숨기지 않는다. 이 문서는 성능이 더 좋았다는 이유로 관련 방법을 부적절하다고 설명하거나, 결과 이전에 정한 제외 원칙을 만들어내지 않는다. 원고의 범위는 **수치 이력·공통 head를 쓰는 TPP encoder 비교**이며, native 분포형 수요 예측기 전체에 대한 우월성을 주장하지 않는다. 연구 기록의 Deep Renewal 결과·실패·원본은 유지한다.

## 2. 독립성·코호트·평가 경로를 판정한다 — 외부 작업 대기

| 데이터 | 기존 test의 지위 | 새 독립 평가의 후보 경로 | 현재 미확정 사항 |
|---|---|---|---|
| Taxi | 과거 노출된 legacy test | 지금까지 사용한 모든 기간 이후의 원본 달력 구간, 동일 공간 셀·시간 집계 | 새 기간 원본·SHA 미확보. 활동량/coverage 필터를 cutoff 이전 자료만으로 정했는지 확인 필요 |
| Intermittent | 이전 노출·원본 매핑 미해결 | 모든 기존 site cutoff 이후의 새 추출, 또는 출처를 확인한 외부 site·품목 | 양쪽 raw source 부재, 5000계열 표집 기준·burst 처리 차이·사용권·상위 site 의존성 미확인 |
| Instacart | 과거 노출된 legacy test | 별도로 확보한 미사용 basket history | 로컬에 검증된 후속 release 없음. 재구성한 상대 날짜는 후속 달력 기간을 입증하지 못함 |
| RAF | 고정 시간 holdout, 전체 접근 이력 미확인 | 독립 부품 archive 또는 검증된 후속 release | 알려진 84개월 archive 이후 기간 없음. 새 데이터·part 계보 미확보 |

새 데이터는 파일명이 새롭거나 release가 달라진 사실만으로 적격하지 않다. 원본 event key와 source/customer/part 매핑을 이용해 기존 train·validation·test 및 그 파생본과 중복을 점검해야 한다. 기존 train/validation/test를 다시 섞거나, 이미 본 subset을 새 test로 명명하지 않는다.

**코호트 자격은 예측 cutoff 이전 정보만으로 정한다.** 전체 기간의 총 주문 수, 이후에도 활동한 고객 여부, 전체 기간 수량 분포, 완성된 시계열 길이, 미래 coverage, 이후 수요가 큰 품목을 이용해 cohort를 고르면 미래 정보에 의한 selection이 생긴다. 기존 Taxi의 `min_count=100`, `min_coverage=0.999`와 Intermittent 5000계열 층화표집은 이 관점에서 점검해야 한다. 지금은 이런 미래 의존이 없다고 검증한 상태가 아니다. source·생성 시점·필터를 결과 없이 확인하고 고정한다.

평가 대상은 기존과 같은 **다음 관측 사건**이다. RAF에 관측하지 않았던 모든 0수요 월을 추가해 다른 수요 예측 문제로 바꾸지 않는다. event 정의와 완전 관측기간·행정적 censoring 규칙은 결과 공개 전에 결정한다. 미래 사건이 실제 관측되어 정답이 존재한다는 평가 자격과, 미래 결과가 좋은 entity만 골라 쓰는 코호트 selection을 구분한다.

**재학습 없이 가능한 경우:** 새 자료가 기존 사건·수량·단위·관측 코딩·수치 입력 정의에 맞으면 현재 selected checkpoint를 그대로 적용할 수 있다. 동일 entity의 이후 시점은 시간 일반화, 새 site/customer/archive는 외부 전이 평가로 각각 명명한다. 새 모집단에서 성능이 나쁘거나 calibration이 달라도 결과를 보고 재학습하는 권한이 생기지 않는다.

**새 학습 설계가 별도로 필요한 경우:** 입력 차원·의미·단위를 기존 코딩으로 맞출 수 없거나, cohort/train 구성 자체를 바꾸는 연구 질문을 택하거나, selected 원본을 끝내 회수하지 못해 다른 학습으로 대체해야 하는 경우다. 이때는 현재 조건의 연속으로 표시하지 않고 새 계약·승인을 거친다. 이 작업은 새 모델 학습을 승인하거나 시작하지 않았다.

기존 test를 당장 사용하는 별도 경로도 있다. 다만 이름은 **동결 모델의 legacy holdout 재평가**이며 미접근 독립 확증으로 쓰지 않는다. 다른 archive의 새 part나 같은 공개 파일의 다른 이름으로 독립성 문제를 우회하지 않는다.

## 3. 입력과 예측 시점을 모든 방법에서 맞춘다 — 설계 완료

모든 learned method는 관측한 과거 gap·quantity의 고정 `log1p` 처리와 valid/observed mask, 동결 adapter에 필요한 chronology/position만 사용한다. entity ID, site/part map, timestamp와 target ID는 분할·인과성 검사·집계를 위한 bookkeeping이다. 이를 새 embedding이나 calendar covariate로 입력하지 않는다. item/category, target quantity, 예측 대상 gap, 미래 사건, test-fitted 정규화·clipping을 금지한다.

각 target은 `(dataset, original entity ID, seq, split, raw event key)`로 식별하고, 84+24 learned condition과 단순 기준선에서 동일한 target ID·수량 정답 hash를 사용한다. population을 방법마다 다르게 줄이지 않는다. chronological one-step protocol에서 앞선 평가 사건은 **먼저 그 사건을 예측하고 실제 관측한 뒤에만** 이후 history로 들어간다. 파라미터·optimizer·train 통계 갱신, test 기반 조기 종료, ensemble은 하지 않는다.

각 데이터의 window, max sequence length, train-based transform, gap 단위, 시간 scale과 likelihood 관측 규칙은 [dataset_freeze.json](dataset_freeze.json)을 따른다. Instacart의 상한 코드30도 유지한다. 다른 원본의 연속 gap을 같은 코딩으로 바꾸어 쓴다면 결과는 해당 recorded observation task로 설명한다.

## 4. 지표·seed·불확실성을 고정한다 — 설계 완료

주 지표는 **원단위 수량 RMSE**, 보조 지표는 같은 selected checkpoint의 MAE와 기록된 정수 시간 구간의 확률질량 NLL이다. continuous density NLL이나 시간·수량 joint likelihood로 바꾸어 적지 않는다. 지표는 각 seed의 모든 평가 target을 event-weighted로 집계한 뒤 세 seed의 산술평균과 표본 SD(ddof=1)를 낸다. seed MSE를 먼저 평균해 제곱근을 취하거나, 예측값을 평균한 ensemble로 바꾸지 않는다. 원단위가 다른 네 데이터의 RMSE를 합산하지 않는다.

차이는 `D = mean_seed(metric_TitanTPP − metric_comparator)`로 두고 음수가 개선이다. RMSE 상대 개선율은 `100 × (comparator − TitanTPP) / comparator`다. 분모가0이면 비율은 NA와 절대 차이를 보고한다. NLL은 상대 변화율로 해석하지 않는다.

**평가 반복의 기본 단위는 entity/series이며 겹치는 window나 사건을 독립 표본으로 취급하지 않는다.** Intermittent는 전체 품목계열, Instacart는 전체 사용자, RAF는 전체 part cluster를 묶는 paired bootstrap 10,000회다. cluster를 복원 추출했을 때 그 cluster의 모든 손실 합과 target 수에 같은 multiplicity를 적용하므로 원래 event-weighted 지표가 유지된다. cluster별 RMSE의 단순평균이 아니다.

Taxi는 공간 셀끼리 같은 시간의 충격을 공유하므로, 기존 10/1 설계의 **모든 셀에 공통으로 적용하는 168달력시간 원형 블록 bootstrap**을 주 분석으로 유지한다. 이미 생성된 손실을 재표집하며, 가짜 연결 history로 다시 추론하지 않는다. 24/336시간 블록과 전체 셀 cluster bootstrap을 사전 지정 민감도로 함께 보고하고 유리한 구간만 선택하지 않는다. 시간 grid의 빈 시간에 가상 target을 추가하지 않는다. timestamp가 없으면 target 순서를 대신 써서 구간을 계산하지 않는다.

Intermittent site, RAF source system처럼 상위 집단의 의존성이 확인되면 성능을 보지 않은 metadata로 상위 cluster 계획을 먼저 동결해야 한다. 개체 cluster가 이런 공동 충격을 해결했다고 표현하지 않는다. 계열이2개 미만 또는 Taxi의 grid가2개 주 블록 미만이면 계산 불가다. 독립 cluster 또는 주 길이 비중복 블록 수가20 미만이면 탐색 구간으로 표시한다. 이는 보고용 경계이며 타당성을 보장하는 정리가 아니다.

세 seed 모두 **같은 bootstrap draw**를 사용하고 seed 자체는 재표집하지 않는다. 각 draw에서 seed별 RMSE→seed평균→paired 차이를 계산한다. 단순 기준선은 한 개의 동일한 예측벡터를 세 모델 seed와 비교하는 것이며, 관측 수가 세 배가 되는 것이 아니다. seed SD는 학습 난수의 기술통계이고, bootstrap CI는 **현재 세 학습 결과에 조건부인 평가 표본의 불확실성**이다. 가능한 모든 학습 난수·개발 선택의 불확실성을 포함한다고 주장하지 않는다.

기존 난수 규칙을 유지한다: NumPy PCG64, seed20261001+데이터 index(Taxi, Intermittent, Instacart, RAF), 문자열 entity ID/시간순 정렬, 10,000회, linear quantile, paired 2.5/97.5 percentile **95% CI**. 실제 Runtime의 NumPy 버전과 resampling multiplicity SHA도 실행 전에 고정한다.

**다중 비교:** 네 데이터의 사전 고정 주 상대와의 RMSE 차이4쌍에는 0.625/99.375 percentile 구간도 함께 보고한다(각98.75%, Bonferroni family 보조 구간). bootstrap 가정이 타당한 범위의 근사 보정이며 독립 데이터 적격성을 만들지는 않는다. 다른 learned 상대, 구조 대조, 단순 기준선, 보조 지표와 strata의95% CI는 개별 기술적 구간이다. 이 중 유리한 것만 골라 전체 모델군 우월성·동등성 또는 유의성을 선언하지 않는다. p-value를 만들어내지 않는다. 모든 데이터의 방향, 크기, seed별 결과와 시간 NLL의 상충을 보고한다.

군집 bootstrap의 적합성은 의존성·모형 가정에 달려 있다([Field & Welsh, 2007](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2007.00593.x)). 시간 블록 방식은 기존 설계의 [Künsch, 1989](https://doi.org/10.1214/aos/1176347265) 참조를 유지한다. 반복 수·블록 길이·작은 표본 경계는 이 연구의 설계 선택이다. 결과를 전처리·선택에 이용하지 않는 원칙은 [scikit-learn 공식 지침](https://scikit-learn.org/stable/common_pitfalls.html)과 일치한다.

## 5. 공개 전 점검과 한 번의 결과 공개를 정의한다 — 설계 완료

1. 결과·target 분포를 열지 않고 출처, 사건 키, cutoff와 cohort selection을 확인한다. source/license, event-key·entity-key hash, eligibility 명세를 실제 파일에 바인딩한다.
2. selected 원본·source·입력·관측 코딩·Runtime SHA를 맞춘다. 현재 빠진36개 외부 comparator 원본의 위치/CPU 검증을 완료하고 바인딩된 구조 대조24개와 함께 전체 manifest를 검증한다. 서로 다른 epoch나 현행 소스로 대체하지 않는다.
3. 별도로 승인된 합성/기존 validation 검사로 evaluator의 인과성, 공통 target ID, 엄격 로딩, 단위, 지표·bootstrap을 검증한다. **이번 문서 검증은 실제 evaluator 통합 검증이 아니다.**
4. 실제 population·서버/CPU/GPU·비용 상한·마감·출력 경로·결과 열람 책임자를 기록하고 실행 및 held-out 예측/성능 열람의 명시적 허가를 받는다. 기존 학습 승인이나 이 설계 작업 승인을 새 평가 허가로 확대하지 않는다.
5. 평가 전 manifest와 분석 스크립트 SHA를 봉인한다. 승인된 한 batch가 모든 방법/seed와 실패 기록을 생성한다. 중간 성능을 보고 모델·상대·기간·seed를 골라 남기지 않는다.
6. 정해진 분석을 한 번 공개하고 모든 결과·실패·미실행 조건을 보존한다. 순수 실행 장애 재시도도 원인과 최초 기록을 남긴다. 모델·입력·선택 규칙을 바꾼 뒤 같은 cohort를 다시 본 결과는 개발/탐색이며 새 독립 확증에는 새 미사용 자료가 필요하다.

**즉시 중단 조건:** 미해결 사건 중복·출처, cutoff 이후 정보로 정한 cohort, SHA/config 불일치, selected 원본 부재, target 정렬 오류, NaN/Inf·누락 예측, 부재한 data rights, 승인하지 않은 Runtime/비용 또는 봉인 전 성능 노출이다. 유리한 공통부분집합만 남기는 오류 복구는 허용하지 않는다. 원래 모집단과 실패 이유를 기록한다.

## 남은 순서

**평가 데이터와 원본의 실행 적격성을 채운다 — 다음 작업 / 데이터 출처 확인 대기**

- 각 데이터에서 legacy 재평가와 새 독립 cohort 중 경로를 명시하고 필요한 source·기간·권한 목록을 확보한다. 새 cohort 미확보 상태에서는 독립성 주장을 하지 않는다.
- 외부 comparator36개 원본 바인딩은 cohort 조사와 병렬로 진행할 수 있다. 두 결과가 갖춰져야 실제 population과 전체 manifest를 실행 계약으로 고정한다.

**평가 실행기를 점검한다 — 다음 작업 / 위 바인딩 이후**

- 합성·허용된 validation으로 공통 target, 인과적 입력, 원본 소스/시간 코딩, paired bootstrap과 실패 처리 검증을 마친다. 실행기 구현·실측 CI는 아직 미완료다.

**정확한 실행 대상과 결과 열람을 승인한 뒤 평가한다 — 승인 필요**

- 승인 대상은 `design.json`의 `required_execution_approval_fields`다. TEST_SESSION_PROTOCOL 0.2·9절의 held-out lock과 현재 설계 작업 범위 때문에 실제 평가와 결과 열람이 별도 단계다. 현재 원격 학습·서버·Pod·자동화·원고는 이 설계 작업이 변경하지 않았다.

검증 기록: [verification.json](verification.json). 출처별 역할·SHA: [source_evidence_map.json](source_evidence_map.json).
