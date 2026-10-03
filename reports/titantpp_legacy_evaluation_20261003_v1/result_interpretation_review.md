# 기존 test 재평가 수치·주장 독립 검토

> **2026-10-03 해석 정정:** 새 미접근 자료를 확보하지 않았다는 사실과 기존 test 평가의 타당성은 같은 판정이 아니다. 과거 접근만으로 현재 모델의 test 기반 선택이나 평가 무효를 단정하지 않으며, 낮은 test 성능 역시 미사용의 증거로 쓰지 않는다. [접근 이력과 선택 사용의 구분](../titantpp_test_access_history_review_20261001_v1/report.md)을 우선한다. 아래 성능 비교·불확실성·주장 범위는 수치에 근거한 별도의 판단으로 유지한다.

검토일: 2026-10-03. 대상 split은 **기존 test**이다. 승인된 `analysis.json`과 `review/summary.md`의 기존 집계·시드 지표·구간만 읽었으며, 추론·학습·재표집·원격 실행을 하지 않았다. 백분율은 저장된 반올림 전 평균 지표로 계산했다. 원고와 봉인 파일은 수정하지 않았다.

원본: [analysis.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/analysis.json), SHA-256 `5041fa56d0bb7f40a721d4d78465682b8e651c5e422e8227cee7857e7d5d2488`. 표시 자료: [summary.md](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/review/summary.md). 아래 수치는 `results.<dataset>.<primary_panel>.means`, `seed_metrics`, `intervals`에서 가져왔다.

## 핵심 판정

- 고정 validation anchor와 test에서 관측된 최저 외부 모델을 구분해야 한다. S2P2는 이번 test의 네 데이터 모두에서 외부 6모델 중 평균 RMSE가 가장 낮다. TitanTPP는 그 S2P2보다 **Intermittent에서만** 평균 RMSE가 낮다. Taxi의 고정 RMTPP 대비 큰 감소를 test 최저 외부 모델 대비 우위로 표현하면 틀린다.
- Intermittent의 last-quantity 대비 MAE 우열은 기존 validation 서술과 반대다. 이번 test에서는 TitanTPP가 MAE와 RMSE 모두 낮다. validation 표와 과거 수치를 보존하되, 해당 서술의 split을 명시해야 한다.
- All-available은 원형보다 Taxi·Intermittent·Instacart의 평균 RMSE가 낮다. RAF에서는 current-only가 원형보다 낮고, window-mean의 RMSE 점추정은 전체 9학습 모델·2결정적 기준보다 낮다. 원형 branch-availability 규칙의 일반적 우위를 주장할 수 없다.
- TitanTPP의 평균 시간 NLL은 S2P2보다 네 데이터 모두 높다. 수량 결과와 함께 이 불리한 방향을 보고해야 한다.
- 이번 결과는 완료된 기존 test의 소급 재평가이며, 독립 미접근 평가를 새로 확보한 것이 아니다. 108조건 성공·SHA 검증은 실행 완전성과 연결성을 뒷받침하며, 이 한계를 없애지 않는다.

## 고정 비교와 관측 최저 외부 모델의 분리

이 표의 변화율은 `100 × (TitanTPP / 비교 모델 − 1)`이다. 음수는 TitanTPP 오류 감소, 양수는 증가다. 각 지표는 세 시드 지표의 산술평균이며 예측 앙상블 성능이 아니다.

| 데이터 | Titan RMSE | 고정 anchor / RMSE | 고정 anchor 대비 변화 | test 최저 외부 모델 / RMSE | 그 모델 대비 변화 |
| --- | ---: | --- | ---: | --- | ---: |
| Taxi | 118.623710 | RMTPP / 246.647279 | −51.905526% | S2P2 / 115.541513 | +2.667610% |
| Intermittent | 2.761218 | RMTPP / 4.211471 | −34.435780% | S2P2 / 3.215355 | −14.123991% |
| Instacart | 5.980156 | S2P2 / 5.946537 | +0.565345% | S2P2 / 5.946537 | +0.565345% |
| RAF | 39.795096 | S2P2 / 39.772935 | +0.055719% | S2P2 / 39.772935 | +0.055719% |

S2P2에 대한 paired seed별 Titan RMSE 우세는 Taxi 2/3, Intermittent 3/3, Instacart 0/3, RAF 1/3이다. Taxi에서 2/3이면서 평균은 불리한 것은 seed 42의 큰 차이와 양립한다. 따라서 paired-win 수와 평균 순위를 서로 대신 쓰면 안 된다. Instacart는 S2P2 대비 MAE가 0.267165% 낮지만 RMSE는 높아 지표별 방향도 구분해야 한다.

### 고정 네 anchor의 RMSE 차이 구간

차이는 `TitanTPP − anchor`이며, 98.75% 구간은 **기존 네 고정 RMSE 비교군**에 대한 Bonferroni 구간이다. Taxi 구간 미산출로 나머지를 세 비교군으로 재정의하거나, 관측 최저 모델로 anchor를 교체하지 않는다.

| 데이터 | 차이 | 개별 95% 구간 | 네 anchor 98.75% 구간 |
| --- | ---: | --- | --- |
| Taxi | −128.023568 | 미산출 | 미산출 |
| Intermittent | −1.450253 | [−1.743087, −1.158078] | [−1.827089, −1.085751] |
| Instacart | +0.033618 | [+0.029092, +0.038015] | [+0.027795, +0.039208] |
| RAF | +0.022161 | [−0.607894, +0.592478] | [−0.806270, +0.722236] |

Intermittent의 S2P2 대비 RMSE 차이는 −0.454136, 개별 95% 구간은 [−0.567595, −0.334250]이다. 이는 네 고정 anchor에 속하는 Intermittent 비교가 아니므로, 해당 구간에 네 anchor의 보정된 추론 범위를 부여해서는 안 된다. RAF의 구간이 0을 포함한다는 사실은 동등성 입증이 아니다.

모든 산출 구간은 **이미 학습된 세 시드에 조건부**인 paired cluster 재표집 구간이다. 시드를 재표집하지 않으므로 새 학습·새 seed 전반의 불확실성을 포함하지 않는다. Intermittent primary는 50개 site, Instacart는 196,615개 entity, RAF는 3,307개 entity이다. Taxi primary는 공유 달력 시간의 168시간 블록이며, 176개 시간단위에서 명목 비중첩 블록이 1개여서 미산출이다. 24시간 분석은 명목 7개로 기술적 민감도 분석이고, 336시간 분석은 미산출이다. 공간 셀 구간을 포함한 민감도 결과로 primary CI를 대체하지 않는다.

## Intermittent 단순 기준의 방향 반전

TitanTPP 대 last-quantity의 평균 MAE는 1.133524 대 1.263579로 **10.292608% 감소**, RMSE는 2.761218 대 3.206282로 **13.881002% 감소**한다. MAE 차이의 개별 95% 구간은 [−0.179705, −0.082375], RMSE 차이는 [−0.517236, −0.368366]이다. 세 Titan 시드 모두 두 지표가 결정적 last-quantity 기준보다 낮다.

따라서 “Intermittent에서 last-quantity가 낮은 MAE를 보이며 Titan의 장점은 RMSE에 한정된다”는 무조건적 설명은 이번 test에 맞지 않는다. “development validation에서는 …; 기존 test 재평가에서는 …”로 연결하고, 결과 변화의 원인을 이 집계만으로 단정하지 않는다.

Instacart의 window-mean RMSE는 5.980200으로 Titan의 5.980156과 매우 가깝다(Titan 감소 0.000742%; 차이 95% 구간 [−0.008345, +0.008408]). RAF window-mean은 39.357906으로 Titan보다 RMSE가 낮지만 MAE는 11.301169로 높다. Titan의 RAF RMSE 증가율은 1.110808%이며 차이 95% 구간 [−1.296244, +2.097403]은 0을 포함한다. 단순 기준 대비 일반적·명확한 RMSE 우위로 요약하면 안 된다.

## 구조 비교와 불리한 NLL

아래 구조 표의 변화율은 **all-available 기준을 원형으로 나눈 값** `100 × (all-available / original − 1)`이다. 위의 Titan 대 comparator 변화율과 분모가 다르다.

| 데이터 | All-available RMSE | 원형 대비 MAE 변화 | 원형 대비 RMSE 변화 | 원형 대비 NLL 변화 |
| --- | ---: | ---: | ---: | ---: |
| Taxi | 99.304085 | −13.644453% | −16.286479% | +15.117005% |
| Intermittent | 2.660208 | +0.074670% | −3.658185% | +37.854840% |
| Instacart | 5.973318 | −0.051696% | −0.114346% | −0.008643% |
| RAF | 39.998342 | +1.924763% | +0.510730% | −1.926932% |

Intermittent의 original-minus-all-available RMSE 차이 95% 구간은 [+0.029971, +0.164575]이나, MAE 차이는 [−0.018981, +0.017977]로 0을 포함한다. All-available NLL은 21.199041 ± 23.251413이고 세 seed 값은 9.313896, 6.292496, 47.990731이다. 따라서 NLL 평균 악화를 “모든 seed에서 악화”라고 표현하면 틀린다. RAF의 original-minus-all-available RMSE 구간 [−0.744650, +0.349663]도 0을 포함한다. Instacart의 구조상 RMSE 변화는 작으므로 조건부 구간의 방향과 실질적 크기를 함께 제시한다.

Current-only 대비 Titan RMSE 변화는 Taxi −17.722122%, Intermittent −5.073189%, Instacart +0.065980%, RAF +0.837978%다. RAF 차이의 95% 구간은 [+0.024008, +0.701678]이고, current-only가 세 paired seed 모두 낮은 RMSE를 보인다. 기존 validation의 “RAF에서도 직접 predecessor 경로가 작게 개선”을 test 결과에도 적용하면 틀린다.

| 데이터 | Titan NLL | S2P2 NLL | Titan − S2P2 | 변화율 | 차이의 primary 개별 95% 구간 |
| --- | ---: | ---: | ---: | ---: | --- |
| Taxi | 1.029177 | 0.690022 | +0.339154 | +49.151229% | 미산출 |
| Intermittent | 15.377800 | 13.673777 | +1.704022 | +12.461972% | [−0.321319, +3.579942] |
| Instacart | 2.831395 | 2.815437 | +0.015957 | +0.566779% | [+0.015590, +0.016307] |
| RAF | 4.046792 | 3.741535 | +0.305257 | +8.158608% | [+0.288312, +0.323141] |

이 NLL 백분율은 기록된 손실 값의 상대 변화이며, 보정 정도나 발생시간 예측 정확도의 백분율이 아니다. 원고에는 절대 NLL과 방향을 우선 제시하는 편이 명확하다. Intermittent는 RMTPP보다 NLL이 낮아도 S2P2와 THP보다 높다. 특히 THP NLL은 8.326340이며 Titan-minus-THP 구간은 [+6.236124, +7.922872]이다. 수량 우위를 시간 likelihood 우위로 확장하지 않는다.

## 원고 연결과 권장 영문 문장

검토 시점 `main.tex`의 abstract 56–57행, introduction 101행, limitations 772–774행, conclusion 827–829행, Appendix 1410–1411행의 last-quantity 설명은 validation 범위를 명시해야 한다. 원고가 동시 편집 중이므로 행 번호보다 문구를 기준으로 찾는다. 구조 결과 625–644행은 기존 validation 결과로 보존하고 test 구조 표·해석을 분리한다. “Execution remains pending” 784행 부근은 이번 소급 재평가 완료 사실로 갱신하되 독립 미접근 평가가 완료되었다고 바꾸지 않는다.

> In the retrospective reevaluation of the existing test partitions, TitanTPP had 14.12% lower mean quantity RMSE than S2P2 on Intermittent, whereas its RMSE was 2.67%, 0.57%, and 0.06% higher on Taxi, Instacart, and RAF, respectively. S2P2 had the lowest test RMSE among the six external encoders on each dataset; the validation-fixed anchors remained unchanged for the prespecified RMSE comparisons.

> On Intermittent, TitanTPP reduced MAE and RMSE relative to the last-quantity baseline by 10.29% and 13.88%, respectively, reversing the MAE ordering observed on development validation. Its mean duration NLL nevertheless exceeded that of S2P2 on all four test partitions, so the quantity results do not establish a general likelihood advantage.

> The all-available control attained lower mean RMSE than the original correction on Taxi, Intermittent, and Instacart, while current-only attained lower RMSE on RAF. These comparisons do not establish a general advantage for the original branch-availability rule. The reported intervals condition on the three fitted seeds, and the short Taxi test period did not support the primary 168-hour-block interval.

기존 validation 표를 교체하지 말고 데이터·표 제목·주장마다 split을 구분한다. 기존 test 접근 이력, DeepRenewal의 결과 존재 이후 사용자 요청에 따른 제외, source 문서 한계도 결과의 방향과 무관하게 유지한다. All-available이나 test 최저 외부 모델을 관측 이후 새 대표 모델·사전 선택 모델로 소급 승격하지 않는다.

## 완료 및 남은 작업

- **완료:** 기존 JSON 및 표시 자료의 핵심 수치, 시드 방향, 고정 anchor 구간과 원고 주장 범위를 검토했다. 표시 자료에서 위 핵심 수치의 불일치는 발견하지 못했다.
- **다음 작업:** 부모 작업에서 위 split 구분과 불리한 결과를 반영해 원고를 수정하고, 기존 validation 표 보존 및 삽입 표의 조판을 확인한다. 이 검토에서 추가 실험이나 새로운 선택은 요청하지 않는다.
