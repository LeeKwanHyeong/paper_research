# TitanTPP 독립 평가 기준

작성일: 2026-10-01 · 상태: **평가 설계 확정 / 실행 준비 미완료**

이번 문서는 네 데이터의 validation 개발을 마친 시점에서 모델·checkpoint·평가량·비교 절차를 고정한다. 독립 평가 성능은 아직 생성하거나 열람하지 않았다. 현재 원고의 결과는 계속 validation 결과로 표시한다.

## 1. 현재 기준선과 독립성 판정

대표 모델은 **TitanTPP (`titantpp_history_mlp`, 고정 /8)**다. 네 데이터 모두 같은 구조를 사용한다. 외부 비교군은 RMTPP, THP, NHP, SAHP, S2P2, AttNHP의 기존 공통 head 연결 구현이다. seed42·52·62를 유지한다. 네 데이터 × 7모델 × 3seed = **84개 selected checkpoint**가 대상이다.

현재 [checkpoint 목록](checkpoint_manifest.csv)은 84개의 선택 epoch·모델 tensor SHA·원본 경로를 연결한다. 그중 **48개는 로컬 binary의 파일 SHA를 이전 CPU 감사와 다시 대조**했다. 기존 세 데이터의 RMTPP·THP·NHP·SAHP **36개는 기록된 로컬 결과 경로에 binary가 없어 원본 회수와 CPU 검증이 남았다**. 이 36개는 이전 감사에서도 JSON 결과 기록 검증 대상이었다. 모델이 유실됐다고 단정하거나, 결과 집계 완료를 모든 binary 감사 완료로 바꾸지 않는다.

split 독립성은 다음과 같이 분류한다. [접근 이력](split_access_ledger.json)과 [데이터 식별자](dataset_manifest.json)에 출처와 SHA를 기록했다.

| 데이터 | 확인한 이력 | 현재 독립성 판정 |
|---|---|---|
| Taxi | 과거 test 이름의 결과 파일과 현재 split-manifest 경로를 참조하는 실험 기록이 있음 | 미접근 독립 test로 확정 불가. 당시 데이터 bytes·실제 열람·선택 사용 여부 미확인 |
| Intermittent | 이전 head_office 기반 intermittent 평가 이력이 있음. 현재 frozen_5000과 같은 manifest 경로의 직접 연결은 확인되지 않음 | 과거 cohort와 현재 target의 겹침·파생 관계가 미확인. 경로 불일치만으로 독립 판정하지 않음 |
| Instacart | 과거 test 이름의 결과 파일과 현재 split-manifest 경로를 참조하는 실험 기록이 있음 | 미접근 독립 test로 확정 불가. 당시 데이터 bytes·실제 열람·선택 사용 여부 미확인 |
| RAF | 현재 캠페인은 train/validation 파일만 배포. 표본 확인한 과거 2개 계약도 validation-only | 현재 캠페인의 test 미평가는 확인. 전체 과거 접근 이력은 아직 미확정 |

2026-09-27 기존 감사는 91개 historical manifest와 test 지표 이름의 파일 4,094개를 기록했다. 중첩 캠페인·복사본을 포함할 수 있으므로 이 수치를 독립 실험 수나 사람의 열람 횟수로 해석하지 않는다. 당시 감사 중 held-out target 분포의 집계가 잠시 노출됐다는 기존 기록도 보존한다. 이 문서에는 그 수치를 복사하거나 설계 선택에 사용하지 않았다. 이번 작업은 split 파일의 byte SHA와 허용된 metadata만 확인했다.

독립성은 두 축으로 판정한다.

1. **선택으로부터의 독립성:** 해당 평가 결과·예측·target 요약이 구조, hyperparameter, checkpoint, 제출 주장 선택에 사용되지 않았는가.
2. **예측 시점의 인과성:** 각 target 예측에서 그 target과 이후 사건을 입력·전처리·상태 갱신에 사용하지 않는가.

같은 고객·품목의 이후 시점을 평가하는 것은 가능한 평가 설계다. 다만 새 고객 일반화와 구분한다. 데이터 전처리의 전체 기간 활동성·계열 길이 기반 cohort 선택도 별도로 기록하고, 이 제한을 bootstrap으로 해결됐다고 보지 않는다.

**판정 규칙:** 접근 이력이 해소되지 않은 기존 test를 사용하면 명칭은 ‘고정 모델의 기존 test 재평가’다. ‘미접근 독립 평가’로 쓰려면 과거 노출·target 중복·cohort 계보가 해소되거나, 실제로 사용하지 않은 후속 기간/외부 데이터와 그 모집단·기간·전처리를 별도 동결해야 한다. 기존 train/validation/test 행을 다시 섞어 새 독립 test로 명명하지 않는다. 새 데이터의 평가 가능성 확인은 성능을 보지 않는 식별자·기간·중복 검사부터 한다.

## 2. 모델과 checkpoint를 고정하는 방법

- 선택 규칙은 **최초의 최소 validation raw quantity RMSE**다. 이미 선택된 `best_val_qty_rmse_model.pt`를 사용하며 test에서는 epoch를 다시 고르지 않는다.
- 각 파일을 `dataset/model/seed/selected_epoch/file SHA/tensor SHA/source revision/동결 계약`으로 식별한다. 원격 경로만으로 동일성을 인정하지 않는다. [기계판독 목록](checkpoint_manifest.json)을 실행 입력의 기준으로 삼는다.
- 과거 source revision과 정확한 모델·loader·head·전처리·단위·기록된 시간 확률질량 계산을 재사용한다. 현재 작업 디렉터리의 변경 코드를 과거 모델 대신 사용하지 않는다. [데이터 목록](dataset_manifest.json)의 설정을 그대로 사용한다.
- **Full, Gate, 활성 분기 정규화는 본 비교의 대표 후보가 아니다.** 현재 논문의 별도 탐색 결과로 남긴다. 데이터마다 성능이 좋은 변형을 골라 TitanTPP라는 한 이름으로 합치지 않는다.
- seed3개의 예측을 평균한 ensemble을 새로 만들지 않는다. train+validation 재학습, test 보정, test 기반 clipping·후처리·조기 종료를 하지 않는다.
- 원본 binary 회수 후 strict loading, tensor SHA, 모델 설정, 선택 epoch를 대조한다. 알려진 validation endpoint 재현은 별도 실행 계약의 사전 검사이며, 이번 작업에서 추론을 실행하지 않았다. 파일이 실제로 유실됐다면 다른 epoch로 대체하지 말고 해당 조건을 미완료로 보고한다.

본 비교의 고정 기준 상대 모델은 validation 평균 RMSE에 따라 **Taxi·Intermittent는 RMTPP, Instacart·RAF는 S2P2**로 정한다. 이는 test 결과를 보기 전의 기준이다. 결과표에는 여섯 비교군을 모두 남기고, test에서 가장 좋은 상대를 새로 골라 확인적 검정의 기준으로 바꾸지 않는다.

## 3. 평가 단위와 인과적 입력

대상은 기존 정의의 **다음 관측 사건의 수량과 기록된 사건 간격**이다. 관측되지 않은 모든 시각의 0수요를 채워 넣는 일반적인 다기간 수요 예측과 구분한다.

- 각 target에 `(dataset, 원본 entity ID, seq, split)`의 고유 ID를 부여한다. 순서·중복·평가 모집단을 모델 간 동일하게 검증한다. 서로 다른 seed도 같은 target을 평가한다.
- 평가는 chronological one-step 방식이다. 앞선 train/validation 사건은 기존 loader 규칙 안에서 context로 쓸 수 있다. 앞선 test 사건은 **그 사건의 예측을 완료하고 관측한 이후**에만 뒤 target의 history로 들어간다. 파라미터 업데이트는 없다. 모델을 미래까지 열린 history로 재구성하지 않는다.
- 수량/시간 변환, 표준화, 범주 경계, lookup과 window는 학습 때 고정된 값을 사용한다. 평가 집합의 분포로 재계산하지 않는다. Instacart의 시간 상한 코드30과 각 데이터의 시간 단위를 유지한다.
- 모델별 target 누락·중복·NaN·Inf가 생기면 우선 실행 오류로 표시한다. 유리한 공통 부분집합만 다시 만들어 비교하지 않는다. 실패 조건을 결과표에서 삭제하지 않는다.
- RAF 현재 배포본은 train/validation 전용이므로 그대로는 test 평가가 불가능하다. 독립성 판정 뒤 원본 parent split을 식별하고, 동결된 전처리를 적용하는 별도 실행 준비가 필요하다.

승인 후 남길 target별 파일 schema는 `dataset, entity_id, seq, split, target_id, model, seed, checkpoint_file_sha256, state_tensor_sha256, source_closure_id, raw_quantity, predicted_raw_quantity, recorded_gap, time_nll, unit, history_length`다. Taxi에는 `time_bucket`도 저장한다. 평가 target ID·정답의 해시가 모든 모델에서 같아야 한다. 원본값은 내부 보관하며 공개본에는 고객·품목 식별자를 직접 넣지 않는다. 이번에는 schema만 정의했다.

## 4. 지표와 집계

**주 지표는 원래 단위의 quantity RMSE**, 보조 지표는 동일 checkpoint의 quantity MAE와 recorded-time NLL이다. RMSE는 큰 절대 수량 오류에 더 큰 손실을 부여하려는 평가 선택이다. 모든 데이터가 고수요라는 가정이나 실제 재고 비용을 직접 측정했다는 뜻은 아니다.

모델 m, seed s, 평가 target i=1,…,N에 대해:

\[
R_{m,s}=\sqrt{N^{-1}\sum_i(\hat q_{m,s,i}-q_i)^2},\qquad
A_{m,s}=N^{-1}\sum_i|\hat q_{m,s,i}-q_i|,
\]
\[
L_{m,s}=N^{-1}\sum_i\ell^{time}_{m,s,i},\qquad
\bar M_m=\tfrac13\sum_{s\in\{42,52,62\}}M_{m,s}.
\]

시간 NLL은 기존 head의 **기록된 정수 구간 확률질량**에 대한 값이다. 연속시간 density NLL이나 수량과 시간의 joint NLL로 바꿔 적지 않는다. 수량 MSE 학습 손실과 보고용 raw RMSE도 구분한다.

각 데이터·모델별 세 seed의 지표 평균과 **표본표준편차(ddof=1)**를 보고한다. RMSE는 seed마다 먼저 제곱근을 취한 다음 평균한다. seed별 MSE를 먼저 평균한 뒤 제곱근을 취하거나 예측을 ensemble한 값으로 대체하지 않는다. 데이터별 원단위가 다르므로 네 데이터의 raw MAE/RMSE를 하나로 평균하지 않는다.

모델 차이는 `D = 평균(TitanTPP 지표 − 비교군 지표)`로 정의하여 음수가 개선이다. RMSE 상대 개선율은 `100 × (비교군 평균 RMSE − TitanTPP 평균 RMSE) / 비교군 평균 RMSE`다. 비교군 값이0이면 비율을 정의하지 않고 절대 차이만 제시한다. 시간 NLL에는 상대 개선율을 사용하지 않는다.

본문 결과표는 28행(4데이터×7모델), 세 지표의 평균±SD와 평가 target 수를 포함한다. 별도 표에는 고정 상대4쌍의 절대 RMSE 차이·상대 개선율·구간을 둔다. 모든 seed와 여섯 비교군의 paired 차이는 부록에 남긴다. 수량 구간·history 구간은 기존 **train 경계**를 재사용하고, 불리한 구간도 포함하는 탐색 분석으로 표시한다.

## 5. 불확실성 계산 절차

### 5.1 seed 변동과 표본 불확실성을 분리한다

세 seed의 SD는 학습 난수 변동을 보여주는 기술통계다. 표준오차나95% 신뢰구간으로 표시하지 않는다. 아래 bootstrap은 **현재 세 개의 고정된 학습 결과를 조건으로 하는 평가 표본의 불확실성**이다. seed와 target을 곱해 독립 반복3N개로 세거나, 이 구간이 가능한 모든 학습 실행의 불확실성을 포함한다고 해석하지 않는다.

### 5.2 paired resampling의 단위를 고정한다

| 데이터 | 주 resampling 단위 | 해석과 사전 검사 |
|---|---|---|
| Intermittent | `oper_part_no` 전체 계열 | 선택된 계열의 모든 평가 target을 묶는다. 같은 사업장 등의 상위 집단 공유 여부를 metadata로 확인한다 |
| Instacart | 사용자(`oper_part_no`) 전체 계열 | 사용자별 반복 구매의 의존성을 유지한다 |
| RAF | 부품(`oper_part_no`) 전체 계열 | 부품별 반복 수요를 묶는다 |
| Taxi | 모든 공간 셀에 공유하는 **연속168 달력 시간** 블록 | `time_bucket` 축에서 같은 시간의 모든 셀을 함께 선택하여 공간 간 공통 충격을 유지한다 |

계열 bootstrap은 G개 entity 중 G개를 복원 추출한다. 뽑힌 entity의 **모든 target 오차 합과 target 수**를 multiplicity만큼 더한 뒤 event-weighted 지표를 다시 계산한다. entity별 RMSE를 단순 평균하지 않는다. 계열 간 근사 독립성을 가정한 조건부 구간이며, 공통 사업장·공급 충격까지 제거하지는 않는다. 결과 공개 전에 상위 집단 구조가 확인되면 성능을 보지 않은 metadata 근거로 계약을 개정하고, 임의로 유리한 단위를 선택하지 않는다.

Taxi는 평가 범위의 연속 hourly grid 길이를 T라 하고, 시작 시각을 균일 복원 추출한 길이 L=168의 원형 블록을 연결해 T시간으로 자른다. 뽑힌 각 시각에 속한 모든 셀·모델·seed의 평가 target에 동일 가중치를 준다. 사건이 없는 시간에 가상의 0수요 target을 만들지 않는다. resampling은 이미 계산된 손실을 대상으로 하므로 블록 경계를 넘는 가짜 history로 모델을 다시 추론하지 않는다. 시간 국소 의존성과 근사 정상성을 전제한다. train-only 분석은 Taxi의 `time_bucket`과 시간 단위 seq의 대응을 확인했다.

Taxi는 L=24와336시간도 **사전 지정 민감도 분석**으로 함께 계산하되, 가장 유리한 구간만 고르지 않는다. 이 절차는 기존 validation용 ‘계열별168 target 블록/1,000회’ 계획과 다르다. 이번 기준은 달력 시간의 공동 블록이며, 이전 계획을 실행한 것처럼 보고하지 않는다.

**가용성 기준:** entity가2개 미만이거나 Taxi의 T가2L 미만이면 해당 구간을 계산 불가로 표시한다. entity 수 또는 비중복 길이-L 블록 수가20 미만이면 작은 표본의 탐색 구간으로 표시하고 유의성 근거로 쓰지 않는다. 20은 보고용 보수적 기준이며 통계적 타당성을 보장하는 정리가 아니다. timestamp가 없으면 Taxi를 임의의 target순서 bootstrap으로 대체하지 않는다.

### 5.3 난수·계산·출력 규칙

- 데이터별 **10,000회**, NumPy PCG64 seed20261001에 데이터 순서(Taxi, Intermittent, Instacart, RAF)의 index를 더한다. 구현 시 NumPy 버전·정렬 규칙·resampling index/weight 파일 SHA를 고정한다. entity는 문자열 ID의 오름차순, 시각은 오름차순으로 정렬한다.
- 같은 반복의 target multiplicity를 **모든 모델과 세 seed가 공유**한다. 메모리가 부족하면 entity/시간 단위의 `count, absolute_error_sum, squared_error_sum, time_nll_sum`으로 계산해 동일 결과를 얻는다.
- 반복마다 seed별 RMSE·MAE·NLL을 재계산하고 seed평균 차이 D를 구한다. 상대 RMSE 개선율도 그 반복의 seed평균으로 다시 계산한다. 고정 point estimate의 분모를 모든 반복에 재사용하지 않는다.
- 주 출력은 D와 RMSE 개선율의 **2.5/97.5 percentile 구간**이다. quantile은 linear interpolation을 사용한다. 무효·빈 반복은 숨겨 재추출하지 않고 원인·개수를 보고하며, 구간 계산 검증이 끝날 때까지 해당 구간을 확정하지 않는다.
- 고정 RMSE 상대4쌍에는 0.625/99.375 percentile 구간도 보조 보고한다(각98.75%, 네 비교의 Bonferroni 보정). 기본 bootstrap 근사의 유효성이 충족될 때에만 가족 단위 약95% 범위로 해석한다. 데이터 독립성이나 적은 cluster 문제를 이 보정이 해결하지 않는다.
- 여섯 상대×네 데이터, 보조 지표·구간 분석의95% 구간은 **개별 비교의 기술적 구간**이다. 이들을 골라 전체 모델군 우월성·동등성·p-value를 선언하지 않는다. 개선 폭의 크기, 방향, seed별 일관성과 시간 NLL의 상충을 함께 보고한다.

계열 bootstrap과 시간 블록 bootstrap의 전제는 각각 [Field & Welsh (2007)](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2007.00593.x), [Künsch (1989)](https://doi.org/10.1214/aos/1176347265)에 따른다. 반복 수·난수 seed·블록 길이·표본 수 경계는 본 평가의 사전 설계 선택이다. test를 모델 선택에서 분리하는 원칙은 [scikit-learn 공식 평가 지침](https://scikit-learn.org/stable/common_pitfalls.html)과도 일치한다.

## 6. 실행 전 완료 조건과 남은 작업

**평가 기준과 원본 식별자를 고정한다 — 완료**

- 대상84조건, 선택 규칙, 주·보조 지표, paired bootstrap, 누락 처리와 보고 범위를 이 문서와 JSON으로 기록했다. 기존 원고·실험 수치·학습 코드·스케줄러는 변경하지 않았다.

**split 접근 이력과 원본 binary 준비를 마무리한다 — 다음 작업 / 독립성 판단은 정보 대기**

- 과거 실제 test 열람·선택 사용에 대한 연구자 확인과 데이터 계보를 연결한다. 이력이 끝내 불명확하면 기존 test 재평가와 새 독립 평가를 구분하는 경로를 선택한다.
- 기존 외부4모델의36개 selected binary를 기록된 원격 원본과 대조하여 회수·CPU 검증한다. 회수만으로 split 독립성이 해결되지는 않는다. 학습을 다시 시작하지 않는다.
- 접근 이력 조사와 원본 회수는 서로 독립적으로 진행할 수 있다. 최종 평가 데이터 계약은 이 둘의 결과를 받은 뒤 현재 세션에서 순서대로 확정한다.

**평가 실행기를 검증하고 실행 계약을 확정한다 — 다음 작업**

- 합성 데이터와 기존 validation만으로 인과성·target ID 정렬·checkpoint strict loading·단위·지표·resampling 재현성을 검증한다. split 적격성,84파일 SHA,Runtime,서버,단일 평가 예산·마감·출력경로를 실제 실행 계약에 채운다.
- metadata 준비는 성능 열람과 분리한다. 이번 문서의 설계만으로 held-out 실행 허가나 GPU 예산이 생기지 않는다.

**고정된 평가를 실행하고 원고 주장을 대조한다 — 이후 작업 / held-out 실행·성능 열람 승인 필요**

- 위 조건과 TEST_SESSION_PROTOCOL 0.2·9절의 평가 경계를 충족한 뒤, 승인된 범위에서 한 번 평가한다. 원고의 validation 결과를 자동으로 test로 바꾸지 않고 두 근거를 구분해 갱신한다.
- 실험 조건을 바꾸어 다시 평가해야 하면 첫 결과와 실패 원인을 보존한다. 모델·규칙 변경 후 같은 test를 다시 본 결과는 후속 탐색으로 표시한다.
