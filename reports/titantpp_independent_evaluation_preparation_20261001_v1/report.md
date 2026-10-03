# TitanTPP 독립 평가 실행 준비

> **2026-10-03 해석 정정:** 과거 test 접근만으로 현재 모델의 test 기반 선택·평가 무효를 단정하지 않는다. 신규 자료는 현재 확인된 사실만으로 투고의 필수 조건이 되지 않는다. [접근 이력과 선택 사용의 구분](../titantpp_test_access_history_review_20261001_v1/report.md)을 우선한다. 아래 준비 상태·답변 대기·작업 순서는 작성 당시 기록이며, 이후 [108조건 평가 완료](../titantpp_legacy_evaluation_20261003_v1/FINAL_REPORT.md) 사실로 갱신해 읽는다.

현재 원고의 주 비교에 쓰인 **84개 selected checkpoint를 모두 로컬에 확보했다.** 이번에 누락된 외부 네 모델의 36개를 회수하여 CPU 검증을 마쳤고, 기존 48개의 감사를 재사용했다. 동결 모델과 loader를 사용하는 validation 표본 검사는 84조건 모두 통과했다. 실제 held-out 평가와 미접근 독립성 판정은 아직 완료하지 않았다.

## 1. 원본 회수와 모델 식별 — 완료

| 대상 | 신규 selected checkpoint | 회수 파일 | 회수한 파일의 byte 합 |
|---|---:|---:|---:|
| 5080: Taxi·Intermittent, RMTPP·THP·NHP·SAHP | 24 | 401 | 16,861,736 |
| 5090: Instacart, RMTPP·THP·NHP·SAHP | 12 | 354 | 10,951,210 |
| 합계 | 36 | 755 | 27,812,946 |

원격 파일을 읽어 전송 전 SHA와 크기를 기록하고, 로컬 원본 사본과 다시 대조했다. 회수 범위는 선택 checkpoint, 원래 계약·source 및 history·summary·노출·validation 재평가 기록이다. 서버 파일이나 실행 상태를 변경하지 않았다. 최초 `retrieval_receipt.json`의 `cpu_audit: pending`은 전송 완료 당시 기록이며, 후속 [checkpoint_audit.json](checkpoint_audit.json)이 현재 CPU 검증 결과다.

6개 과거 실행의 동결 source를 각각 격리해 모델을 만들고 strict loading, 유한 tensor, model tensor SHA, 선택 epoch, 최초 최소 validation raw 수량 RMSE 선택, 원본 기록의 일치를 확인했다. 학습 초기화는 저장된 native 검증 해시 연결을 대조했으며 CPU에서 GPU 초기화를 다시 재현했다고 해석하지 않는다. selected checkpoint는 모델 가중치와 optimizer 그룹 계약을 담는다. **이번 36개 검증은 optimizer 상태 binary나 last checkpoint의 추가 감사가 아니다.**

기존 48개와 합친 [evaluation_registry.json](evaluation_registry.json)은 네 데이터 × 7모델 × 3seed의 **84개 selected 파일**, 파일 SHA, tensor SHA, 선택 epoch, 원래 source·계약을 연결한다. 대표 모델은 `titantpp_history_mlp`이며 Full·Gate·정규화 변형은 주 비교에 추가하지 않았다. 원래 [선택 목록](../titantpp_independent_evaluation_protocol_20261001_v1/checkpoint_manifest.json)은 당시 기록으로 보존했다.

## 2. split 계보와 접근 이력 — 자료 검토 완료 / 독립성 미확정

[lineage_review.json](lineage_review.json)은 성능 파일을 열지 않고 실험 manifest, split·sampling 계약, 식별자 열과 파일 SHA를 대조한 결과다. 기존 감사가 찾은 과거 manifest 78개의 SHA는 모두 그대로였다. test 결과 생성 이력은 사람의 실제 열람이나 선택에 사용한 이력과 별개다.

| 데이터 | 확인된 사실 | 아직 확인되지 않은 부분 |
|---|---|---|
| Taxi | 과거 test 산출물의 존재와 현재와 같은 split manifest 경로 참조를 확인했다. | 과거 파일과 현재 파일의 byte 동일성 전체 연결, 실제 열람·선택 사용 여부. 경로 일치만으로 데이터 동일성을 확정하지 않는다. |
| Instacart | 과거 test 산출물과 같은 split manifest 경로 참조가 있다. | Taxi와 마찬가지로 과거 byte 계보와 연구자의 사용 이력이 필요하다. |
| Intermittent | 현재 5,000계열은 50사업장에 속한다. 과거 train의 23,387개 문자열 ID와 정확히 같은 ID는 없었다. | ID 구성 방식이 달라 0개 문자열 교집합이 사건 비중복을 증명하지 않는다. sampling manifest가 가리키는 원본 parquet도 현재 로컬에 없어 원본 SHA를 다시 확인하지 못했다. |
| RAF | 확인 범위의 과거 launch contract 25개가 `validation_only`, `held_out_test_evaluated=false`를 기록한다. 원본 workbook SHA와 전역 월별 split 계약을 연결했다. | 이 탐색은 모든 외부 세션·노트북·열람 행위를 포괄하지 않는다. 실제 사용 이력 확인은 남아 있다. |

Intermittent의 계열 선택 코드는 전체 관측기간의 사건 수를 이용하는 최소 길이 조건과 층화를 포함한다. 수량 분위수는 train 구간에서 계산하지만, 전체기간 사건 수를 사용한 코호트 구성은 별도로 공개해야 한다. 이는 모델의 미래 target 입력과는 다른 문제다. 현재 분할을 엄격한 전향적 모집단으로 표현하지 않는다.

RAF의 보존 계약은 train 58개월(1996-01~2000-10), validation 13개월(2000-11~2001-11), test 13개월(2001-12~2002-12)이다. 계약에는 test를 filtering·threshold·model selection에 사용하지 않았다고 명시되어 있다. 현재 학습용 배포 사본은 train/validation만 담는다. 이번에는 원본 workbook의 수요값이나 test target·성능을 읽지 않았다.

현재 **네 데이터 모두 미접근 독립 test라는 판정은 보류**한다. 연구자가 실제 test 결과를 보거나 설정 선택에 참고했는지에 대한 확인을 요청했고 답변 대기 중이다. 과거 접근이 있었거나 확인이 어려운 기존 split의 평가에는 `retrospective locked-test reevaluation`이라는 범위를 사용한다. 미접근 독립 평가가 필요하면 계보와 노출을 확인한 미사용 후기/외부 코호트를 별도로 고정한다. 기존 행을 다시 섞는 것으로 새 독립 평가를 만들지 않는다.

## 3. 동결 source를 사용하는 평가 경로 — CPU 표본 검증 완료

[evaluate_validation.py](evaluate_validation.py)는 9개 동결 source 묶음을 별도 Python 프로세스에서 불러온다. 현재 개발 중인 모델 코드를 대신 사용하지 않는다. 각 데이터에서 정렬된 validation 계열 두 개의 최대 8개 target을 사용했고, 원래 lookback·mask·전처리·head·시간 관측 계약을 유지했다. materialize하는 행은 train/validation으로 한정했다.

| 데이터 | 조건당 고유 validation target | 모델 × seed | 출력 행 |
|---|---:|---:|---:|
| Taxi | 8 | 21 | 168 |
| Intermittent | 8 | 21 | 168 |
| Instacart | 2 | 21 | 42 |
| RAF | 4 | 21 | 84 |
| 합계 | 22 | 84조건 | 462 |

- 84조건의 strict loading과 예측값 유한성, 모델·seed 간 동일 target ID·수량·간격 정렬을 확인했다.
- target 수량을 교란해도 수량 예측과 시간 loss가 유지되고, 관측 범위 안에서 target 간격을 교란해도 수량 예측이 유지됐다.
- batch와 단일 표본 출력이 허용 오차 안에서 일치했고, 추론 전후 모델 tensor SHA가 유지됐다.
- 첫 합성 교란 검사에서 Instacart의 관측 상한 30일을 넘는 간격을 넣어 계약 검사가 중단됐다. 교란값만 유효한 1일/2일로 고쳤다. 과학 source·관측 규칙·실제 데이터는 변경하지 않았다.

[validation_smoke.json](validation_smoke.json)에 통과 결과를 보존했다. **이 검사는 전체 validation endpoint 재평가나 새 성능 비교가 아니다.** 시간 NLL의 target 의존성은 정답의 확률을 계산하는 경로이며 수량 예측의 target 유입과 구별한다. 전체 모집단의 인과성·정렬과 GPU runtime 검증은 실제 실행 준비에서 더 확인해야 한다.

로컬 검증은 Python 3.12.10 / Torch 2.14.0 / NumPy 2.3.1의 CPU에서 했다. [5080 읽기 전용 inventory](runtime_5080_readonly.json)는 Python 3.12.13 / Torch 2.11.0+cu130 / NumPy 2.4.4와 대상 GPU UUID를 기록한다. 이는 native CUDA 검증이나 배포 완료 증거가 아니다. 관측 당시 compute process 목록이 비어 있었으며, 이를 이후 상태로 연장해 해석하지 않는다.

## 4. 지표와 불확실성 계산 — 구현·검증 완료

[paired_statistics.py](paired_statistics.py)는 seed별 event 가중 MAE·RMSE·시간 NLL을 계산한 후 seed 평균과 표본 표준편차(ddof=1)를 낸다. 예측을 ensemble하거나 서로 다른 데이터의 원단위 오차를 합치지 않는다. 비교는 TitanTPP−비교모델이며, 상대 RMSE 개선율의 분모도 각 bootstrap draw에서 다시 계산한다.

10,000회 PCG64 재표집에서 모든 모델·seed가 같은 표본 가중치를 쓴다. seed 자체는 재표집하지 않는다. 따라서 구간은 고정된 세 학습 모델에 조건부인 평가 표본 불확실성이며 학습 변동 전체를 대표하지 않는다. 일반 비교에는 pointwise 95% percentile 구간, 미리 정한 네 RMSE anchor 비교에는 98.75% 구간을 함께 낸다. 후자는 네 비교의 다중성을 조정하는 명목상 Bonferroni 수준이며, bootstrap의 유한표본 coverage를 보장하지 않는다. p-value는 계산하지 않는다.

| 데이터 | 기본 재표집 단위 | 보조 검사 |
|---|---|---|
| Taxi | 모든 공간 셀을 함께 묶은 168시간 원형 calendar block | 24·336시간 block |
| Intermittent | **사업장(site_cd) 전체 cluster** | 품목 계열 전체 cluster |
| Instacart | 사용자 전체 cluster | 기본 비교 유지 |
| RAF | 부품 전체 cluster | 기본 비교 유지 |

Intermittent는 이번 metadata 조사에서 50개 사업장이 확인되어, 기존 품목 단위 설계를 사업장 단위로 보완했다. 근거와 변경 시점은 [amendment.json](amendment.json), 반영된 설계는 [protocol_effective.json](protocol_effective.json)에 있다. 기본 단위로도 사업장 간 공통 충격까지 제거되는 것은 아니다. 결과를 보기 전에 정한 분석 규칙이며 모델·checkpoint 선택은 바뀌지 않는다.

14개 [단위 검사](test_evaluator.py)가 모두 통과했다. 검사에는 수작업 확장 표본과 가중 합의 일치, RMSE 집계 순서, seed SD, 동일 모델 차이 0, 결측·중복 target·비유한 값·미승인 split 거부, calendar 빈 시간 처리, 재표집 재현성, 작은 표본과 분모 0 처리가 포함된다. [JUnit 기록](unit_tests.xml)을 보존했다.

실제 validation 표본도 통계 경로에 연결했다. Instacart·RAF는 각각 10,000회 계산을 완료했지만 단위가 두 개뿐인 개발 검사다. Taxi는 시간 폭이 짧고 Intermittent는 사업장 하나만 포함되어 구간을 계산하지 않는 규칙이 작동했다. [표본 통계 검사](validation_statistics_smoke.json)의 값은 논문 성능 구간으로 사용하지 않는다. full-population 실행의 메모리·시간 상한은 아직 측정하지 않았다.

## 5. 실제 실행 계약 — 초안 / 실행 불가

[execution_contract_draft.json](execution_contract_draft.json)에 84개 모델 식별자, source, 지표, 준비 검증, 5080 runtime 후보를 연결했다. 평가 모집단과 적격성, 전체 target 목록, 배포 후 native 검증, 실제 시간·자원 상한과 마감은 미확정 항목으로 남겼다. 현재 실행기는 CPU validation 표본만 허용하며 held-out/GPU 실행 옵션이 없다.

원래 독립 평가 설계와 이미 완료된 Core·효율·validation 결과는 재사용한다. 이번 준비는 원고, 선택 epoch, benchmark 수치, 학습 코드와 계약을 바꾸지 않았다. [verification.json](verification.json)은 원본·코드·문서 연결과 준비 결과의 종합 검증 기록이다.

## 남은 작업 순서

**평가 데이터의 성격을 확정한다 — 외부 작업 대기**

- 연구자의 실제 test 열람·선택 사용 여부와 과거 원본 계보를 대조한다. 기존 split 재평가와 미사용 코호트의 독립 평가 중 어떤 결과를 만들지 확정한다. 이 판정이 평가 모집단과 원고의 표현을 결정한다.

**평가 모집단에 맞춰 실행 계약을 완성한다 — 다음 작업**

- 적격한 target 명세, 동결 source별 전체 평가 wrapper, resource 상한·마감·출력경로를 고정한다. 5080의 CPU/GPU validation 일치와 전체 범위의 정렬·시간·메모리를 확인한다. 공통 계약과 데이터 선택에 의존하므로 현재 세션에서 순서대로 진행한다.

**고정 평가를 수행하고 원고에 연결한다 — 이후 작업 / 실행·성능 열람 승인 필요**

- 검토 가능한 실제 실행 계약에 따라 평가한다. 실패·불리한 결과와 불확실성을 포함해 현재 validation 주장과 대조한 뒤 원고를 갱신한다. 새 학습이나 checkpoint 재선택은 이 평가에 포함하지 않는다.
