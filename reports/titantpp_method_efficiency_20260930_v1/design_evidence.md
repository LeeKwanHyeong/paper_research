# 설계 이유와 증거 — 정리 완료

## 논문에서 제안하는 방법

> **TitanTPP는 사건 간격과 수량을 인과적으로 인코딩하고, encoder 사이의 저차원 인접 이력 잔차와 학습된 정적 검색을 결합하여 다음 사건 수량을 예측한다. 추론 중 신경 메모리의 온라인 갱신 없이 관측 이력을 활용한다.**

이 문장이 방법의 출발점이다. 최종 구성은 `titantpp_history_mlp` 하나이며, Full과 Gate는 구성 비교 및 별도 탐색 결과다. 단순한 연속값 출력의 최초 제안, 온라인 메모리의 보편적 불필요성, 입증되지 않은 MAC 대비 가속을 C1에 포함하지 않는다. 기존 Titans 전체에서 모든 Memory를 제거했다는 설명도 실제 구현과 맞지 않는다.

## 설계 목적·확인 사실·미확인 효과를 구분한다

| 설계와 목적 | 확인된 근거 | 논문에서 주장할 범위 / 남는 질문 |
|---|---|---|
| 시간·수량 log1p 입력: 불규칙한 간격과 사건 크기를 함께 표현 | 동결 코드 식(1); 외부 TPP도 같은 관측 정보·head/loss로 비교 | 공통 문제 설정이다. 이 입력만의 독창성이나 두 feature 각각의 인과적 기여는 별도 입력 제거 실험 없이 주장하지 않는다. |
| causal encoder: 관측 이력을 인코딩하고 다음 사건 정보를 차단 | target row mask와 직전 관측 state 선택; causal key mask | 미래 target이 encoder에 들어가지 않는 구현이다. B도 이력을 인코딩하므로 B를 ‘이력 없는 모델’로 부르지 않는다. |
| encoder 사이 인접 이력 보완: 현재/직전 contextual state의 결합을 다음 block에 전달 | 식(3–4); 동일 프로토콜 B↔MLP 결과 | 모듈 추가 구성의 이득을 평가한다. 용량도 6,144개 증가하므로 모든 이득을 ‘인접 관계’ 하나에 인과적으로 귀속하지 않는다. |
| 폭4·8분기 residual: 보완 용량 제한 | 6,144개 parameter와 고정 /8 수식·CPU 동치 검사 | 구조적 용량 제한은 사실이다. 최적 rank·최적 분기 수 또는 Full 대비 parameter 절약은 입증되지 않았다. Full도 6,144개다. |
| 영 출력 초기화: 공통 초기 함수를 보존 | 잔차0·RNG 보존, 기존 초기화 감사와 이번 CPU 검사 | 초기 예측 보존의 수학적 성질은 설명한다. 일반화·학습 안정성 개선의 독립 효과는 주장하지 않는다. |
| persistent bank와 정적 top4 retrieval: 학습된 참조 표현을 공급 | 식(2),(5); 16개/64개 bank, 추론 중 쓰기 없음 | ‘메모리 전체 제거’가 아니라 ‘온라인 신경 메모리 갱신 생략’이다. Full 기반 no-static 결과로 MLP의 검색 필요성을 직접 입증하지 않는다. |
| 온라인 associative update 생략: 갱신·momentum 상태 없이 실행 | MLP forward 경로; Titans 원문과 MAC 구현 경로 | 연산 절차 차이는 확인됐다. 현재 MLP↔MAC 실제 속도·peak 메모리 우위는 별도 측정 대상이다. |
| 수량 회귀+시간 likelihood: 다음 사건의 크기·간격을 함께 학습 | 식(6–9); 공통 head/loss의 외부 비교 | continuous-valued point regression이며 수량의 연속 확률분포를 새로 정의한 것은 아니다. 시간 성능 동시 개선도 주장하지 않는다. |

## 이력 보완의 현재 실험 근거

같은 validation RMSE-selected checkpoint의 **3seed 평균**을 비교했다. 감소율은 `100 × (1 − MLP/B)`다. 표준편차와 개별 seed는 [기존 1차 분석](../titantpp_first_analysis_20260930_v1/report.md)을 함께 제시한다. 아래 감소율은 평균 간 대비이며 통계적 유의성 검정이 아니다.

| 데이터 | B → MLP MAE | B → MLP RMSE | MAE / RMSE 감소 | 시간 NLL B → MLP |
|---|---|---|---|---|
| Taxi | 28.7545 → 25.6237 | 90.5093 → 79.7111 | 10.89% / 11.93% | 0.72468 → 1.03867 |
| Intermittent | 0.75876 → 0.70052 | 1.78056 → 1.67303 | 7.68% / 6.04% | 0.38462 → 0.49711 |
| Instacart | 3.99317 → 3.99157 | 5.87957 → 5.88274 | 0.04% / −0.05% | 2.80647 → 2.80709 |

따라서 이력 보완의 가장 직접적인 결과는 **Taxi·Intermittent 수량 개선, Instacart 이득 제한, 시간 지표와의 상충관계**다. MLP는 Taxi에서 Full보다 평균 RMSE가 1.19% 높고, Intermittent에서는 7.77% 낮다. 대표 모델을 데이터별로 바꿔 선택하지 않는다.

외부 RMTPP·THP·NHP·SAHP 비교는 공통 head 아래의 encoder 경쟁력 근거다. 이 표가 원본 각 모델의 native head와 최적 튜닝까지 포함한 보편적 순위를 의미하지는 않는다. 기존 validation 분석 후 MLP를 채택했으며, 독립 평가 여부는 별도로 공개한다.

## Writing notes / claim–evidence map

사용자가 지정한 산출물: **“방법 절 초안·기호 정의표·구조 그림·코드와 수식의 대응표”**. 문서 분량이나 논문 전체 페이지 제한은 주어지지 않았다. 본 초안은 영어 Method 본문과 한국어 검토 자료로 구성했으며, 학회 제출용 최종 축약본은 아니다.

| Method 문단 | 핵심 주장 | 출처 / 확실성 |
|---|---|---|
| 1 | observed gap/quantity, target exclusion, next-event scope | core contract + `target_outputs`; 구현 사실 |
| 2 | causal blocks와 persistent K/V | 동결 backbone/memory 소스; 구현 사실 |
| 3 | lag-one bottleneck 수식·마스킹·영 초기화 | 동결 HistoryCorrection + CPU 식 동치; 수학/구현 사실 |
| 4 | static cosine top4 raw mean, shared head state | 동결 HardLocalMemoryMatcher; 구현 사실 |
| 5–6 | 수량 point regression·관측 정수 시간 질량·손실과 선택의 분리 | frozen head/likelihood/trainer/contract; 구현 사실 |
| 7 | 온라인 갱신 없는 구성과 Titans의 차이 | [Titans 원문](https://arxiv.org/abs/2501.00663), 로컬 MAC adapter; 메커니즘 비교 |
| 설계 목적 | 보완 표현·제한된 용량 | 의도와 실험 결과를 구분한 설명; 구성 요소별 독립 인과 주장 아님 |
| 효율 | 현재 MAC 대비 직접 속도비 미확정 | [비용 감사](efficiency_audit.md); 증거 한계 명시 |

영문 문장은 정의·계산·실험 해석의 확실성을 구분했다. `encode`, `combine`, `predict`, `retain`, `evaluate`처럼 실제 동작을 드러내는 동사를 중심으로 썼고, 수식 전후의 짧은 정의와 긴 조건 설명을 섞었다. 설계 의도를 성능 보장으로 표현하는 문장, ‘memory-free’, ‘first continuous-mark paradigm’, ‘6.86× faster TitanTPP’는 제외했다. 문단은 문제→입력→encoder→보완→검색→head→선택→주장 한계 순서다. 새로운 선행연구 전체 조사는 수행하지 않았으므로 독창성의 최종 문헌 검토는 남아 있다.
