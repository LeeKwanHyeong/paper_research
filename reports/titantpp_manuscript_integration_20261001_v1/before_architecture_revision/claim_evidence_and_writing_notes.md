# 원고의 주장–근거 대응과 작성 기록

## 요청 범위와 작성 제약

사용자 요청 원문:

> **방법·관련 연구를 원고로 연결한다 — 다음 작업**
> 완료된 수식·구조 그림·효율 자료를 통합합니다. 진행 중인 실험과 병행할 수 있습니다.
> 진행하자

- 산출물은 [새 영문 원고](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_history_mlp_manuscript_20261001_v1.md)다. 기존 v0.7 원고와 과거 방법·효율 보고서는 보존한다.
- 분량·페이지 수·학회 템플릿 지정은 없었다. 임의의 PAKDD 분량 제한을 적용하지 않았으며, 현재 초안은 제출 형식 편집 전의 전체 설명본이다.
- `oma-academic-writer`의 draft 절차, 근거표, reverse outline, hedge/과장 점검을 적용했다. 별도 에이전트·학습·원격 관측·held-out 열람·Notion 게시를 수행하지 않았다.
- 수치 기준은 기존 전체 비교의 2026-10-01 07:07 KST cutoff다. 이후 완료된 13조건 원본 감사는 무결성 수준만 갱신하며 성능 수치를 바꾸지 않는다. 현재 서버 상태를 새로 확인한 원고가 아니다.

## Claim–Evidence Map

| ID / 원고 위치 | 허용하는 주장 | 실제 근거 | 성격 / 제한 |
|---|---|---|---|
| C1 / 1·3절 | TitanTPP 한 구조: 두 인과 encoder 사이 저차원 이력 잔차와 정적 검색, 공통 시간·수량 head | [수식 대응표](../titantpp_method_efficiency_20260930_v1/code_equation_map.md), [기존 CPU 검사](../titantpp_method_efficiency_20260930_v1/verification.json); 회수된 동결 source의 관련 8파일 SHA 재대조 | 구현 사실. 기존 연산을 조합했다는 사실만으로 최초성·충분한 학술 독창성이 입증되지는 않음 |
| C1a / 식3·4 | 8분기 모두 같은 직전 관측을 읽고 고정 /8로 합산 | `HistoryCorrection`, `lag_source_indices`, 식 동치·padding/withheld·고정 분모 검사 | 다중 lag 검색이 아님. 활성 분기 정규화는 별도 탐색 변형 |
| C1b / 3.3절 | 6,144개 추가 파라미터, 출력 영 초기화로 공통 초기 함수 보존 | 같은 CPU 검사: RNG 보존·영 잔차·수식 최대 오차 5.20e-18 | 초기화 사실. 영 초기화 때문에 최종 성능이 좋아졌다는 인과 주장은 미검증 |
| C1c / 3.5·3.6절 | 수량은 log 공간 점 회귀, 시간은 기록 정수 구간 확률질량 | 동결 `CountAwareTPP.py`, `positive_integer_time.py`, `target_outputs` | 전체 목적을 joint marked-event NLL이나 수량 확률밀도라고 부르지 않음 |
| C1d / 5.1절 | B에 보완을 추가했을 때 Taxi·Intermittent 수량 지표 개선, Instacart 효과 제한 | [core 최종 조건](../titantpp_core_ablation_execution_20260928_v1/final_conditions.csv), [통합 비교](../titantpp_completed_external_comparison_20261001_v1/comparison.json) | validation 3seed 구성 비교. 추가 용량과 보완 경로가 함께 바뀌므로 용량 독립 효과·최적 rank/위치·임계값을 입증하지 않음 |
| C2 / 4·5절 | 외부 6개 공통 head 비교군 대비 Taxi·Intermittent 36개 같은 seed 쌍에서 MAE·RMSE가 모두 낮음 | 같은 통합 비교 및 [후속 5080 감사](../titantpp_completed_5080_audit_20261001_v1/report.md) | 완료된 2데이터에 한정. 유의성 검정, 원논문 native likelihood 순위, 전체 TPP SOTA 주장 아님 |
| C2a / 5.2절 | Instacart와 시간 NLL에서는 일관된 우위를 확보하지 못함 | 전체 통합 비교의 불리한 지표·seed 포함 | 유리한 지표만 취사선택하지 않음. 진행 중 Instacart 추가 비교의 3seed 평균은 미작성 |
| C2b / 5.2절 표4 | RAF는 S2P2보다 RMSE 0.13% 낮고 RMTPP보다 MAE 1.25% 높음; NHP의 시간 NLL이 가장 낮음 | [RAF24조건](../titantpp_raf_execution_20261001_v1/comparison.json), 완료된 CPU 감사 | 3seed validation의 적용 범위 결과. 정규화는 5.3절 표5에서 별도 비교 |
| C2c / 5.2절 | 짧은 이력과 약한 인접 관계가 Instacart 이득을 제한할 가능성 | [train 데이터 특성](../titantpp_data_characteristics_20260928_v1/README.md): 이력 중앙값5, 사용자 평균 제거 후 상관0.037 | 설명 가설. 비선형 신호 부재나 확정 원인으로 쓰지 않음 |
| C3 / 6절 | 같은 5080·입력에서 학습 step 5.58–5.74배, 평가 batch 8.21–8.49배의 MAC/TitanTPP 시간 비 | [효율 실측](../titantpp_efficiency_5080_20260930_v1/analysis.json), [검증](../titantpp_efficiency_5080_20260930_v1/verification.json) | 3회 시간 반복, 각 5warm-up+32batch. 학습 3seed·전체 epoch·수렴 시간·순수 serving latency 아님 |
| C3a / 표7 | 학습 peak allocated 약75.5% 감소, 평가 peak allocated 약3.98배 증가 | 동일 효율 실측과 원본 그림 | allocated/reserved 구분. 모든 GPU 메모리가 감소했다는 표현 금지 |
| RW1 / 2.1절 | recurrent·attention·continuous-time state-space TPP와 비교할 근거 | RMTPP·NHP·THP·SAHP·AttNHP·S2P2 1차 논문 | 원문 설명과 이번 common-head adaptation 설명을 분리 |
| RW2 / 1·2.2절 | 수치형 mark·간헐 수요를 다룬 선행 연구가 이미 존재 | Deep Renewal Processes(PLOS ONE 2021), FlexTPP(NeurIPS2025) | “최초 continuous mark 패러다임”으로 기여를 정의하지 않음. 두 모델을 새 학습 목록으로 자동 추가하지 않음 |
| RW3 / 2.3·6절 | Titans의 온라인 신경 메모리와 이번 추론 경로가 다름 | Titans(NeurIPS2025), 로컬 MAC adapter/직접 효율 증거 | 모든 memory 제거가 아님. 전체 속도 차이를 온라인 업데이트 제거 한 요인의 인과 효과로 귀속하지 않음 |

## 과장·누락을 막기 위한 편집 결정

1. **대표는 하나:** C1은 TitanTPP만 설명한다. 내부 ID `titantpp_history_mlp`는 추적 목적으로 남겼다. Full은 비교 대안이며 데이터별 대표 교체를 하지 않았다.
2. **데이터 의미:** Instacart는 같은 기록일의 상품 행 합계로, native 구매 단위 수나 반드시 주문 한 건이 아니다. Intermittent 상류 데이터의 실측/생성 경위와 배포권은 미확인으로 남겼다.
3. **분할의 한계:** 시계열별 약70/15/15 순서 분할과 전체 관측 길이를 참조한 cohort 선정은 별개로 설명했다. “전체 준비 과정이 train-only”라고 쓰지 않았다.
4. **선택 기준:** 큰 절대 수량 오차에 가중하는 목적과 RMSE 선택을 연결했다. 모든 수요가 크다는 가정, MAE 동시 최저 보장, 비대칭 재고비 최적화로 확대하지 않았다.
5. **비교 조건:** 공통 head 비교는 encoder 표현의 비교다. 입력/출력 교체와 S2P2의 scan 구현·AttNHP query 경계를 공개했다. 동일 epoch 상한이 동일 계산 예산이라는 표현은 피했다.
6. **효율:** 이전 방법 보고서의 “직접 효율 미측정” 서술을 새 원고에서는 실제 완료 측정으로 대체했다. 과거 문서는 소급 수정하지 않았다. 예전 MAC/B0 epoch 비율을 새 MLP 속도로 재사용하지 않았다.
7. **통계:** 본문 표의 ±는 sample SD다. 3seed 일관성에 유의성·동등성·모집단 보장을 부여하지 않았다. 시간 실측 3회와 성능 seed3개도 구분했다.
8. **연구 진행 상태:** 완료된 증거는 본문에, 미완료 비교와 독립 평가 요구는 7절에 분리했다. 초록·최종 결론을 성급히 확정하지 않았다.

## Reverse outline / 문단 흐름 검토

| 절 | 독자가 답을 얻는 질문 | 다음 절과의 연결 |
|---|---|---|
| 1 Introduction | 어떤 예측 문제이며 제안하는 연구의 범위는 무엇인가? | 수치 mark 최초성 대신 이력 구조·실측 비용·예측 상충관계로 문제를 좁힘 |
| 2 Related work | 이미 알려진 부분과 이번 설계의 비교 지점은 무엇인가? | 기존 TPP 표현과 Titans의 memory 경로를 분리해 방법의 역할 설명 |
| 3 Method | 실제 어떤 입력과 계산으로 다음 사건을 예측하는가? | 식1–9, 마스크·분기·초기화·head·loss·selector를 코드와 일치시킴 |
| 4 Evaluation protocol | 무엇을 같게 맞췄고, 비교에서 무엇이 달라졌는가? | 데이터 의미·split·공통 head·수정 범위·검증 범위를 정리 |
| 5 Completed evidence | 설계 목적 중 실제 결과로 지지되는 부분은 무엇인가? | B 대비 개선과 외부 비교를 구분하고 반례를 비용 주장과 분리 |
| 6 Computational cost | 어떤 비용을 얼마나 줄였으며 무엇은 줄이지 못했는가? | 같은 입력·실측 구간·cold/warm·메모리 반례를 수치로 명시 |
| 7 Limitations | 제출본 확정 전에 무엇을 더 확인해야 하는가? | RAF 통합 완료 / Instacart 종결 감사 → 최종 표 → 독립 평가 → 초록/결론 |

## Writing Notes

- 문체는 영문 방법·실험 원고이며, 한국어 설명은 본 기록과 README로 분리했다. 이름 없는 “새 패러다임”, “획기적”, “최초”, “보편적 우월성” 표현을 기여 문장에 사용하지 않았다.
- 확정된 구현은 단정형, train 특성과 성능의 관계는 가능성으로 썼다. 추정 원인을 결과처럼 제시하지 않았다.
- 관련 연구는 9개 1차 논문으로 제한하고 원문 문장을 길게 인용하지 않았다. 기존 초안의 오래된 상태·출처 오기를 그대로 복제하지 않았다. 특히 Deep Renewal Processes 첫 저자는 **Ali Caner Türkmen**으로 원문을 확인했다.
- 두 그림을 육안 확인했다. 구조도는 동일 lag1·고정 /8·잔차 위치·두 bank·head를 표시하며, 효율 그림은 평가 메모리 증가와 측정 단위를 함께 표시한다. 기존 PNG/PDF/SVG는 변경하지 않았다.
- 최종 제출 전에는 페이지 제한에 맞춰 구현 세부와 원본 추적표를 부록으로 옮기고, 로컬 사용자 경로와 upstream 추적 정보의 익명성 처리를 해야 한다. 이번에는 제출 템플릿 적용·번역·외부 게시를 하지 않았다.

## 아직 원고에서 확정하지 않는 내용

- Instacart 추가 비교군 전체 3seed 결과. RAF24조건의 최종 순위는 후속 통합에서 반영했다.
- 독립 held-out 결과, 통계적 유의성, 최종 초록·결론 수치.
- 각 구성요소의 필요성, 용량과 분리된 효과, 온라인 메모리 제거만의 인과적 속도 효과.
- 원래 모델들의 native likelihood 성능 우열이나 numerical-mark 연구 전반의 SOTA.
- PAKDD 채택 가능성 보장. 현재 산출물은 검토 가능한 원고 초안이다.

## 후속 문체 수정 — 2026-10-01

사용자가 제시한 피드백 중 적용한 문장:

> 서론(Introduction)과 방법론(Method)에서는 TitanTPP가 무엇을 해냈는지 자신감 있게 어필해야 합니다.

> "The contributions supported by the current development evidence are:"라는 표현은 매우 기계적입니다.

> 이 훌륭한 엔지니어링 성과가 왜 공급망이나 수요 예측(intermittent demand) 환경에서 중요한지, 그 'Motivation(동기)'을 서론에서 조금 더 매력적으로 포장할 필요가 있습니다.

이어진 요청: “이 부분 반영해서 수정하면되나?”에 따라 기존 영문 원고를 수정했다. 수치·비교 대상·실험 범위는 유지하며, 공급망 동기는 ‘다음 필요 시점과 크기의 예측’으로 설명했다. 측정하지 않은 재고비용 절감이나 시간 NLL 우위를 추가하지 않았다.

**수정 전 → 수정 후의 대표 사례**

| 위치 | 수정 전 | 수정 후 | 의도 |
|---|---|---|---|
| 기여 도입 | The contributions supported by the current development evidence are: | Our contributions are: | 불필요한 승인·면책 어조를 제거 |
| 수량 head | This is a nonnegative point prediction, not a fitted probability density over quantity. | Equation (6) produces a nonnegative quantity point estimate. | 방법의 출력 정의를 긍정형으로 서술 |
| 서론 동기 | An event sequence records both when an observation occurs and the numerical quantity associated with it. | Intermittent demand raises two linked forecasting questions: when will the next demand event occur, and how large will it be? | 데이터 형식 설명에 앞서 실제 예측 문제를 제시 |
| 결과 해석 | These observations bound the empirical claim rather than establish general superiority. | 기여2에서 동일 프로토콜·3seed validation과 Taxi/Intermittent RMSE 감소를 직접 설명 | 추상적인 주장 제한을 구체적 성과·비교 조건으로 대체 |

**배치 원칙**

- 서론: 수요 발생의 시점·크기 → 이력 표현의 계산 비용 → 작은 feed-forward 보완 경로 → 실제 quantity/효율 성과.
- 방법: 계산식과 마스크·분기·초기화·출력 정의·계산량을 설명한다. 수량 점 예측과 시간 확률분포의 구분은 유지한다.
- 실험: validation 개발, common-head adaptation, 같은 checkpoint의 지표, 측정 batch·hardware·warm-up은 해석에 필요한 조건으로 남긴다.
- 결과: Instacart 제한·시간 NLL 악화·평가 메모리 증가를 해당 표와 함께 유지한다.
- 7절: 용량과 설계 효과의 분리, 독립 평가, 비교군 범위, 비용 측정 일반화, 데이터 구성·수량분포·고정 horizon의 한계를 묶는다.

원문 전체는 [수정 전 사본](before_prose_revision/titantpp_history_mlp_manuscript_20261001_v1.md)에 보존했다. [34개 수정 블록](prose_revision_changes.json)은 원문과 개정문을 함께 담는다. 수식9개·모든 표 행·그림 링크·참고문헌 절의 동일성을 별도로 검증한다. 문서의 부정어 수 감소나 단어 수는 편집상의 변화이며, 연구 품질이나 ‘AI 작성 여부’를 판정하는 지표로 사용하지 않는다.

이 수정은 표현과 문단 배치에 관한 것이므로 기존 Claim–Evidence Map의 사실·근거·주장 범위는 유지된다. 7절의 제목은 간결하게 **Limitations**로 변경했다. 학습·평가 결과와 남은 작업 순서는 바뀌지 않는다.

## Contribution 초점 정리 — 2026-10-01

사용자 요청 “오케이 내용 반영해보자.”에 따라 C1을 인접 문맥 표현의 보완과 온라인 메모리 갱신 생략이라는 설계 원리 중심으로 수정했다. 분기 조건·영 초기화는 기존 방법 절에 유지했다. C2의 공통 프로토콜·3seed validation·수치·비교 범위는 그대로다. C3는 학습 step 시간과 메모리 효율을 설명하며, RTX 5080 기종은 서론에서 빼고 6절의 측정 설정에 유지했다. 정적 검색은 유지되므로 memory-free라는 표현은 쓰지 않는다.

[수정 전·후 3개 블록](contribution_revision.json)에 서론 연결 문장과 C1·C3 변경을 기록했다. 방법 근거는 3.2–3.4절과 기존 코드–수식 대응표, 효율 근거는 6절과 기존 동일 batch 계측이다. 2절 이후 본문·수식·표·참고문헌과 남은 작업 순서는 변경하지 않았다.


## RAF 결과 후속 통합 — 2026-10-01T09:20:16.927785+09:00

원본 감사와 [RAF8모델3seed 보고서](../titantpp_raf_execution_20261001_v1/report.md)를 방법 원고4.1·5.3절과C2의 데이터별 차이에 연결했다. 대표 MLP의 RMSE는S2P2보다0.13% 낮지만 MAE는RMTPP보다1.25% 높다. 정규화의 낮은 평균RMSE와 seed별 반례·시간NLL 악화를 함께 기록했다. 이전 원고·검증은 before_raf_integration/에 보존했다. 현재 숫자 확인은 통계적 유의성 검정이나 독립 held-out 평가가 아니다.


## RAF 강조 수준과 결과 배치 수정 — 2026-10-01

사용자 요청:

> 결과를 제거할 필요는 없고, 강조 수준과 배치를 조정하면 됩니다.
> 본문에는 RAF의 지표별 차이를 한 문단으로 설명하고, 정규화 변형의 상세 결과는 구성 비교 절이나 부록으로 옮기면 됩니다.

- **원문 → 개정문:** [수정 전 전체 원고](before_raf_repositioning/titantpp_history_mlp_manuscript_20261001_v1.md)를 보존하고, 현재 원고5.2절은 네 데이터 공통 비교,5.3절은 구조 대안 비교로 정리했다. [블록별 변경](raf_repositioning_changes.json)에서 원문과 개정문을 확인할 수 있다.
- **표4:** 대표 TitanTPP와 외부6모델 ×4데이터를 같은 형식으로 표시한다. 완료26그룹의 평균·표본표준편차는 동일 selected checkpoint 원본에서 재계산하며, Instacart S2P2·AttNHP 두 그룹은 Pending이다. 기존 개선율 표의10.13/15.85/21.72/34.39%는 본문에 보존한다.
- **RAF 해석:** RMSE 차이0.13%,2/3seed 승패,MAE1.25% 열세,NHP의 시간 NLL 우위를 한 문단으로 묶었다. 강한 두 데이터의 개선과 다른 환경에서의 지표별 차이를 설명한다.
- **표5:** 정규화 두 구조의 RAF 평균·표본표준편차와2/3seed 개선·시간 NLL 악화를 구성 비교에 유지한다. 데이터별 대표 모델 교체나 전체8구조 순위 은폐는 없다.
- **문체:** C2에서 RAF만 별도 성과처럼 덧붙인 문구를 줄이고 네 데이터의 적용 범위로 연결했다. 수치적 근접성을 통계적 동등성으로 쓰지 않는다. 방법의 시간 단위 설명에 기존 계약의 RAF6개월도 맞췄다.
- **검증 범위:** 수식·그림·참고문헌·효율표 보존, 공통 결과표26그룹과미완료2그룹, RAF변형 수치와배치를 확인한다. 실험·관측·원본 보고서·스케줄러 변경은 없다. 문장 점검은 과장·중복 방어문 제거와 근거 연결에 한정하며 AI 작성 여부 판정으로 사용하지 않는다.

**남은 순서:** 5090 Instacart 추가 비교 종료 확인·원본 감사 → 표4의 Pending 두 그룹 및 최종 해석 확정 → 별도 승인된 독립 평가와 제출본 편집.
