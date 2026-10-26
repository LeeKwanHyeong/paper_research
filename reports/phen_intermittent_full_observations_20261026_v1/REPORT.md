# Intermittent 전체 과거 조건과 checkpoint 관측 확보

## 현재 결론

**전체 자료에서는 엄격하게 같은 과거 조건을 비교할 여지가 생겼지만, 모든 Validation 사건을 Train과 맞춰 비교할 수는 없다.** 기존 2,048건씩의 부분표본에서 최소 표본 기준을 충족한 정확한 조건은 0개였고, 전체 자료에서는 532개였다. 이 조건에 속한 Validation은 29,720건(34.44%)이다. 반대로 45,792건(53.07%)은 정의한 정확한 과거 조건이 전체 Train에도 없다.

따라서 부분표본이 작아서 비교하지 못한 영역과, 전체 자료에도 직접 대응하는 Train 조건이 없는 영역이 함께 존재한다. 이 결과는 후속 분석의 가능한 범위를 정한 것이며, Prototype의 효과나 시간 NLL 악화 원인을 판정한 결과는 아니다.

계획한 **1,920,436건의 관측 회수·검증을 완료**했다. 기존 best Validation 172,570건, v4에서 정상 종료한 seed42 best Train 393,824건, v6 seed42 last 480,109건, v8 seed52의 세 cohort 873,933건을 연결했다. 전체 target 누락·중복은 없으며 각 구간의 독립 QA와 최종 원본 재검산을 통과했다. 기존 실패와 원래 기준은 보존했다. 사용자가 별도로 승인한 cosine·log-MSE 절대 허용차 개정을 각각의 실행에 적용했고, 수량 예측·시간 NLL·Prototype 번호의 기준은 바꾸지 않았다. **관측 확보 완료와 악화 원인 규명은 구분한다.** 원인 해석은 다음 분석 단계에 남아 있다.

## 범위와 보존 원칙

대상은 Intermittent의 Train·Validation과 기존 PHEN checkpoint 네 개다. seed 42는 best epoch 1·last epoch 41, seed 52는 best epoch 5·last epoch 45를 사용한다. 원자료·checkpoint·기존 관측과 실패 기록을 보존하며, Test 열람, 재학습, optimizer step, checkpoint 덮어쓰기, 원고 수정은 하지 않는다.

후속 관측을 위한 GPU 실행은 별도 산출물 경로를 사용한다. 진행 중이던 작업과 보호된 정지 작업의 신원을 보존하고, 다른 작업의 종료·재개나 공용 Runtime 변경은 하지 않는다.

## 전체 자료에서 비교 가능한 과거 조건을 확인한다 — 완료

전체 Train은 **393,824건**, Validation은 **86,285건**이며 각각 5,000개 시계열을 포함한다. 관측 이력이 없는 각 시계열의 첫 사건 5,000건은 target에서 제외했다. 실제 모델과 같이 직전 관측 사건을 기준으로 520칸의 입력창을 구성하고, target 한 칸을 포함한 길이 제한에 따라 관측 이력은 최대 255건으로 제한했다. 시간상 이전에 관측된 Validation 사건도 다음 사건의 입력 이력에 포함했다.

주요 비교 조건은 **직전 수량·직전 간격·그 전 수량·정확한 이력 길이가 모두 같은 사건**이다. 이력이 한 건뿐이면 그 전 수량과 변화량을 미관측으로 처리했다. 조건과 최소 표본 기준은 전체 집계 전에 고정했으며 결과를 본 뒤 경계를 바꾸지 않았다.

| 전체 Validation의 정확한 과거 조건 | 사건 수 | 전체 86,285건 중 비율 |
| --- | ---: | ---: |
| Train에 같은 조건이 없음 | 45,792 | 53.07% |
| Train에 같은 조건은 있으나 최소 표본 기준 미충족 | 10,773 | 12.49% |
| 각 split에서 6건·2개 시계열 이상 | 29,720 | 34.44% |

최소 기준을 충족한 조건은 **532개**다. 이 중 각 split에서 20건·5개 시계열 이상인 강화 기준은 268개 조건, Validation 24,785건(28.72%)이다. 이력 길이를 사전에 정한 구간으로 묶은 보조 비교에서는 355개 조건, Validation 63,466건(73.55%)이 최소 기준을 충족했다. 구간 비교는 정확한 조건 맞춤을 대체하지 않는다.

**6건·2개 시계열은 서로 다른 두 검색 조합을 비교하기 위한 필요조건일 뿐이다.** 아직 이 집계에는 Prototype 검색 조합을 사용하지 않았으므로, 같은 조건 안에 실제로 서로 다른 조합이 존재하는지, 각 조합에 충분한 사건이 있는지는 별도로 확인해야 한다. 20건·5개 시계열 기준도 추정의 정밀도나 인과적 해석을 보장하지 않는다.

기존 부분표본 자체로는 정확한 조건의 최소 기준을 충족한 집단이 **0개**였다. 같은 2,048개 Validation 사건을 전체 Train·Validation의 조건별 개수로 판정하면 **725건(35.40%)**이 기준을 충족한다. 이는 표본 확대가 도움이 되는 영역이 있다는 근거다. 한편 45,792건의 미지원은 어디까지나 정의한 네 변수의 **정확한 조합이 Train에 없다는 뜻**이며, 그 사건을 예측할 정보가 전혀 없거나 모델이 반드시 실패한다는 뜻은 아니다.

원자료 metadata 480,109행과 조건별 집계는 별도 구현으로 전수 검산했다. 본 집계와 독립 QA 모두 실제 종료 코드 0이며 입력·코드·계약 SHA는 변하지 않았다. 근거는 [support 보고서](support/results_v1/REPORT.md), [전체 집계표](support/results_v1/coverage.csv), [독립 QA](POSTRUN_SUPPORT_QA.md)에 있다.

## 기존 best Validation 기록을 재사용한다 — 완료

seed 42·52 best의 기존 Validation 관측을 각각 86,285건씩, 총 **172,570건** 재사용했다. 원문 29개 필드의 값과 배열 순서를 Parquet로 보존했고, target 신원·정답 수량·간격·실제 이력 길이가 앞서 확인한 전체 metadata와 모두 일치했다. 누락·중복·불일치는 0건이다.

원 probe의 실제 child 종료 코드 0과 stdout/stderr byte·SHA, collection·host·job receipt, checkpoint binary·state·epoch의 연결을 확인했다. 형식 변환과 독립 원문 검산 모두 실제 종료 코드 0이다. 원본에는 과거 local gradient 진단이 포함되어 있지만 optimizer step과 checkpoint write는 0이며, 이번 재사용 변환에서는 forward·backward도 실행하지 않았다.

정밀도와 필드 가용성은 다음과 같이 구분한다.

| 항목 | 재사용 best Validation | 이번에 수집한 관측 |
| --- | --- | --- |
| 시간 분포 중심·폭 | 원본에서 float32로 내보낸 값 | native 시간 계산의 float64 값 |
| 시간 NLL | 원래 계산된 float64 값 | native recorded-interval NLL의 float64 값 |
| native 수량 log-MSE·학습 손실·head preactivation | 원본 미저장, null 유지 | 실제 계산값 저장 완료 |
| 최종 head 입력 은닉벡터 | 원본 미저장 | 두 best Train에 한해 저장 완료 |
| Prototype 번호·cosine similarity | 원문 top-4 순서와 값 보존 | native 검색 관측 저장 완료 |

float32 중심·폭으로 NLL을 다시 계산해 원래 float64 NLL을 대체하지 않았다. 원단위 예측을 역변환하여 없는 native log-MSE를 채우지도 않았다. 신규 관측과 합칠 때도 이 차이를 유지한다. 근거는 [재사용 보고서](reuse/REPORT.md), [원본·결과 계보](reuse/results_v1/PROVENANCE.json), [독립 QA](POSTRUN_REUSE_QA.md)에 있다.

## 누락된 전체 관측을 기존 checkpoint에서 확보한다 — 완료

네 checkpoint의 전체 Train과 두 last checkpoint의 전체 Validation, 합계 **1,747,866건**을 확보했다. 재사용 best Validation **172,570건**을 더해 전체 **1,920,436건**, checkpoint·split 조합 여덟 개의 수집과 검증을 마쳤다.

| checkpoint | 전체 Train 대상 | 전체 Validation 대상 | 현재 보고 상태 |
| --- | ---: | ---: | --- |
| seed 42 best, epoch 1 | 393,824 | 86,285 | Train 신규·Validation 재사용 모두 독립 QA 완료 |
| seed 42 last, epoch 41 | 393,824 | 86,285 | v6 전체 Train·Validation 회수와 독립 QA 완료 |
| seed 52 best, epoch 5 | 393,824 | 86,285 | v8 Train 신규·Validation 재사용 모두 독립 QA 완료 |
| seed 52 last, epoch 45 | 393,824 | 86,285 | v8 전체 Train·Validation 회수와 독립 QA 완료 |

Prototype 검색 정보, 수량 예측과 손실, 시간 분포의 중심·폭·NLL을 같은 관측에서 확보했다. 모델 상태와 가중치는 갱신하지 않았다. 이후 head 진단에 필요한 두 best Train의 최종 head 입력도 별도로 저장했으며, head 학습은 실행하지 않았다.

초기 소스 격리·입력 sidecar, UUID 접두어 비교, 프로세스 CUDA 경로 누락은 관측 수집의 운영·통합 오류였으며 별도로 수정했다. 원 모델·checkpoint·허용차와 실패 원문은 보존했다. 이를 과거 PHEN 학습 실패나 Validation 악화의 원인으로 해석하지 않는다.

이후 v4에서 **수치 검증 실패**가 확인됐다. seed52 last는 target97674의 cosine 두 값이 각각 3.5064e-7·3.0920e-7 차이로 기존 허용차 1.7777e-7·2.6192e-7을 넘었다. seed42 last는 target269443의 cosine 두 값이 1.3015e-7·2.1933e-7 차이로 허용차 1.1950e-7·1.0939e-7을 넘었다. 해당 실패 목록에서 Prototype 번호·순서, 수량 예측, 수량 log-MSE, 시간 NLL의 기준 초과는 없었다. 부동소수점 정밀도 차이와 맞는 양상이지만, 이를 근거로 기존 기준을 통과했다고 처리하지 않는다.

seed42 전체 dispatcher는 last 실패 때문에 actual exit1이다. 그 안에서 이미 끝난 best Train 자식은 actual exit0이며, 393,824행의 신원·중복·모든 chunk SHA·전체 집계·NLL·head 산술·원래 2,048개 참조값을 독립 검산했다. 신규 관측과 은닉벡터 청크의 저장 크기는 116,943,300bytes다. 실패 증거 archive230개 member의 byte/SHA와 후속 보호 프로세스 신원도 확인했다. [부분 수집 inventory](PARTIAL_COLLECTION.json)와 [부분 산출물 독립 QA](POSTRUN_FULL_V4_FAILURE_QA.json)에 전체 실패와 부분 통과를 구분했다.

승인된 cosine 기준을 적용한 v6에서는 seed42 last의 Train·Validation 사전 검증 총 4,096건이 독립 QA를 통과했고, 전체 **480,109건**도 관측된 collector 종료 코드 0(`actual_collector_exit=0`)·보호 작업 확인·회수·독립 QA를 모두 통과했다. 회수 파일 138개의 inventory를 확인했다. [5080 전체 관측 독립 QA](POSTRUN_FULL_V6_5080_QA.md)에 따라 검증된 범위는 **1,046,503건**으로 늘었다.

5090에서는 seed52 best Train 2,048건과 last Train 2,048건의 사전 검증이 통과했으나, last Validation의 batch 00005(0기반)에서 target 8,904의 수량 log-MSE 차이 **2.0489e-7**이 기존 허용차 **1.5138e-7**을 초과해 실제 종료 코드 1로 끝났다. 이번 실패의 기준은 log-MSE이며, 개정된 cosine 기준은 통과했다. 실패 원문은 [v6 5090 실패 기록](failed_gpu_qualification_v6/5090)에 보존했다. 사용자는 이후 log-MSE 절대 허용차만 1e-6으로 추가 개정하는 별도 검증을 승인했다. [별도 승인 기록](LOG_MSE_GATE_AUTHORIZATION_V8.json)에 따른 v8에서는 사전 검증 6,144건과 전체 관측 **873,933건**이 실제 종료 코드 0·보호 작업 확인·독립 QA를 통과했다. [5090 전체 관측 독립 QA](POSTRUN_FULL_V8_5090_QA.json)로 연결한다. v6의 실패 판정은 유지하며, 실행하지 않은 v7 자료는 inventory에 포함하지 않았다.

최종 집계는 **check-only → 별도 최종 inventory 생성 → check-only** 순으로 수행했고, 세 단계 모두 실제 종료 코드 0이다. 원본 연결 파일 817개와 여덟 cohort의 target·행 수·chunk SHA·상태 불변을 확인했다. 마지막 검사는 저장한 최종 JSON을 원본 재계산값과 생성 시각만 제외하고 대조했으며, 요약표도 같은 여덟 cohort에서 다시 만들어 일치함을 확인했다. [최종 inventory](FINAL_COLLECTION.json), [여덟 cohort 요약](FINAL_COLLECTION_SUMMARY.md), [마지막 재검산 종료 기록](FINAL_CHECK_02.receipt.json)에 완료 근거를 남겼다. 원래 [부분 수집 inventory](PARTIAL_COLLECTION.json)는 수정하지 않았다.

## 실제 저장 규모

| 저장 항목 | 파일 수 | 실제 bytes |
| --- | ---: | ---: |
| 전체 관측 Parquet | 434 | 134,948,477 |
| 두 best Train의 head 입력 NPZ | 194 | 187,080,460 |
| 합계 | 628 | **322,028,937** |

필요한 관측과 은닉벡터의 실제 저장량은 **307.11 MiB**다. archive·소스 사본·로그·manifest의 중복 저장은 이 합계에서 제외했다. 앞서 언급한 약 27GB는 이전 상세 벡터 출력 형식을 그대로 확장했을 때의 추정이었다. 이번에는 필요한 필드와 지정한 head 입력만 저장했으므로 해당 추정치를 실제 필요량으로 사용하지 않는다.

은닉벡터는 seed42·52 best Train 각 393,824건, 합계 **787,648건**이며 float32·64차원과 관측 target 순서 일치를 확인했다. 기존 best Validation **172,570건**의 native log-MSE·학습 손실·head preactivation은 계속 null이고 hidden은 미확보다. 이 필드들을 임의로 복원하지 않았다.

## 남은 작업 순서

**검증 완료된 전체 관측을 기준선으로 유지한다 — 완료**
- 기존 checkpoint 네 개의 Train·Validation 관측 1,920,436건과 두 best Train 은닉벡터 787,648건을 보존한다. 전체 target 누락·중복과 원본 상태 변경은 없으며, 재학습·optimizer step·Test 열람 없이 확보했다.
- v4 전체 dispatcher와 v6 5090 qualification의 실패 기록은 그대로 유지한다. 통과한 일부 산출물을 재사용하는 것과 과거 실행 전체를 성공으로 바꾸는 것은 구분한다.

**같은 과거 조건 안에 다른 검색 조합이 있는지 확인한다 — 다음 작업**
- 사전에 고정한 532개 정확한 조건부터 실제 Prototype 검색 조합별 사건·시계열 수를 확인한다. 비교 기준을 충족하지 못한 조건은 제외 사유와 함께 남긴다.
- 이후 과거 수량·간격만으로 설명되는 묶음인지, 같은 조건에서도 예측 오차와 연결되는 차이가 남는지 평가한다. 검색 조합의 차이를 곧바로 해석 가능성이나 인과 효과로 부르지 않는다.

**시간 예측의 중심 오차와 폭을 구분한다 — 다음 작업**
- 같은 과거 조건에서 Train·Validation의 시간 중심이 기록 구간을 얼마나 벗어나는지, 예측 폭이 그 오차를 얼마나 감당하는지 함께 확인한다.
- best·last 변화와 seed 차이를 비교하되, 저장되지 않은 시간 NLL 최저 epoch의 상태를 복원했다고 주장하지 않는다. 이 분석 후 시간 head 개선을 먼저 시험할 근거가 있는지 판단한다.
