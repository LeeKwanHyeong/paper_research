# 기존 test 분할 재평가 최종 보고서

> **2026-10-03 해석 정정:** `retrospective`는 기존 test 분할을 재평가했다는 이력이며 평가 무효를 뜻하지 않는다. 과거 test 접근만으로 현재 대표 모델의 선택에 test를 사용했다고 단정하지 않는다. 이번 평가에서 validation으로 고정한 checkpoint를 재학습·재선택 없이 평가한 사실과 불리한 결과를 함께 보존한다. 신규 자료 확보를 투고의 필수 조건으로 제시한 설명은 [해석 정정](../titantpp_test_access_history_review_20261001_v1/report.md)에 따라 철회한다. 이번 수정은 설명 추가이며 아래 수치·실행 증거는 그대로다.

**평가·회수·독립 감사·통계 집계와 원고 갱신을 모두 완료했습니다.** 이번 결과는 기존 test 분할에 대한 고정된 사후 재평가(retrospective reevaluation)입니다. 독립적인 미접근 자료의 평가로 해석하지 않습니다.

**고정된 평가와 증적 확보 — 완료**

- 4개 데이터셋 × 9개 학습 모델 × 시드 42·52·62의 **108개 선택 체크포인트 평가가 모두 완료됐고 실패는 0개**입니다. 체크포인트는 validation 수량 RMSE가 최소인 최초 epoch로 고정했으며, 재학습·train/validation 재적합·test 결과에 따른 재선택은 하지 않았습니다.
- CPU 36조건, CUDA 36조건의 자격 검증과 대표 모델의 4개 데이터셋 전체 validation 재현을 통과한 뒤 test를 평가했습니다. 관측 이력만 사용하는지, 단일/배치 예측이 일치하는지, 파라미터가 불변인지 확인했습니다.
- test의 고유 대상은 총 **679,959개**이며, 27개 모델·시드 조합에 걸친 학습 모델 예측은 **18,358,893행·2,295개 Parquet 파일**입니다. 결정적 수량 기준 두 개는 데이터셋마다 각각 한 벡터로 집계했습니다.
- 원본 **3,094개 파일의 회수와 SHA 대조**를 완료했습니다. 별도 독립 감사는 184개 실행 영수증, 메타데이터·로그 651개와 고정 소스 1,203개의 해시를 확인했습니다. 체크포인트·선택 epoch·소스 계약·모집단 수·불변 검사에 불일치가 없었습니다. 이 독립 감사는 예측 파일 전체 재해시와 통계 재실행을 중복 수행하지 않았습니다.

근거: [완료 manifest](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/runs/test/attempt1/terminal_manifest.json), [자격 검증](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/qualification_gate.json), [회수 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/retrieval/receipt.json), [독립 감사](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/final_artifact_audit.json).

**수량과 시간 결과를 함께 해석 — 완료**

아래는 모두 **test** 결과입니다. TitanTPP는 고정 대표 모델 `titantpp_history_mlp`이며, 학습 모델의 값은 세 시드 지표의 평균 ± 표본 표준편차입니다. 모든 지표는 낮을수록 좋습니다. 시간 NLL은 고정된 이분산 lognormal 모형이 기록된 정수 시간 구간에 부여한 확률의 음의 로그입니다.

| 데이터셋 | 조건별 대상 수 | TitanTPP MAE | TitanTPP RMSE | 외부 6개 모델 중 test RMSE 최저 | TitanTPP 시간 NLL |
| --- | ---: | ---: | ---: | --- | ---: |
| Taxi | 8,327 | 37.2005 ± 4.9741 | 118.6237 ± 19.5696 | S2P2: 115.5415 ± 6.4023 | 1.02918 ± 0.27266 |
| Intermittent | 88,019 | 1.1335 ± 0.0195 | 2.7612 ± 0.0793 | S2P2: 3.2154 ± 0.2743 | 15.37780 ± 1.73309 |
| Instacart | 578,387 | 4.0637 ± 0.0030 | 5.9802 ± 0.0047 | S2P2: 5.9465 ± 0.0113 | 2.83139 ± 0.01045 |
| RAF | 5,226 | 10.0563 ± 0.0875 | 39.7951 ± 0.0750 | S2P2: 39.7729 ± 0.1609 | 4.04679 ± 0.41387 |

**S2P2는 네 데이터셋 모두에서 외부 모델의 실제 test RMSE 최저 모델입니다.** 이는 결과를 읽은 뒤의 기술적 순위이며, validation에서 고정한 anchor를 변경하지 않습니다. 고정 anchor는 Taxi·Intermittent의 RMTPP, Instacart·RAF의 S2P2입니다.

- **Taxi:** TitanTPP는 고정 RMTPP anchor보다 RMSE가 낮지만 S2P2보다 높습니다. All-available-history 구조 비교의 RMSE는 99.3041로 대표 모델보다 낮아, 대표 구조의 일관된 우월성을 주장할 수 없습니다.
- **Intermittent:** TitanTPP의 평균 RMSE는 외부 6개 모델보다 낮습니다. 다만 all-available-history 구조 비교는 2.6602로 더 낮습니다. 수량 결과와 별개로 **시간 NLL이 크게 악화됐습니다.** 동일한 대표 모델 seed 42에서 validation NLL 0.346835가 test에서 14.768278로 증가했습니다. test 세 시드 평균 15.37780도 THP의 8.32634보다 높습니다. 이 관측만으로 악화 원인을 확정하지 않습니다.
- **Instacart:** TitanTPP의 RMSE는 S2P2보다 높고, 고정 anchor 차이의 구간도 같은 방향입니다. 관측 이력 평균 기준의 RMSE 5.9802와 대표 모델의 RMSE는 반올림한 값이 같습니다. 강한 개선으로 표현하지 않습니다.
- **RAF:** TitanTPP와 S2P2의 RMSE 차이는 작고 신뢰구간이 0을 포함합니다. Current-only 구조 비교는 39.4644, 관측 이력 평균 기준은 39.3579로 대표 모델보다 평균 RMSE가 낮습니다. 이는 점 추정 비교이며, 동등성이나 비열등성의 입증은 아닙니다.

근거: [전체 test 집계 원본](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/analysis.json), [전체 모델·결정적 기준·구조 비교 표](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/review/summary.md). Intermittent의 seed 42 validation 비교값은 [고정 validation 재현 기준](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/validation_references.json)이며 v0.7 기준선과 섞지 않았습니다.

**고정 비교 기준의 불확실성 평가 — 완료**

차이는 **TitanTPP − 고정 anchor의 RMSE**이며 음수이면 TitanTPP의 오류가 작습니다. 각 시드에서 대상별 동일 가중치로 지표를 계산한 뒤 시드 지표를 평균했으며 예측 앙상블은 하지 않았습니다. 결정적 기준에는 시드 표준편차나 시간 NLL을 부여하지 않았습니다.

| 데이터셋 | 고정 anchor | RMSE 차이 | 개별 95% 구간 | 네 anchor 보정 98.75% 구간 |
| --- | --- | ---: | --- | --- |
| Taxi | RMTPP | -128.023568 | 미산출 | 미산출 |
| Intermittent | RMTPP | -1.450253 | [-1.743087, -1.158078] | [-1.827089, -1.085751] |
| Instacart | S2P2 | +0.033618 | [0.029092, 0.038015] | [0.027795, 0.039208] |
| RAF | S2P2 | +0.022161 | [-0.607894, 0.592478] | [-0.806270, 0.722236] |

- 산출 가능한 분석은 고정 난수 설정으로 paired bootstrap 10,000회를 수행했고 빈 재표집은 0회였습니다. 기본 단위는 Intermittent 50개 site, Instacart 196,615명 사용자, RAF 3,307개 부품입니다. 모델·시드에 같은 재표집 가중치를 적용하며 시드 자체는 재표집하지 않습니다.
- **Taxi의 기본 신뢰구간은 산출할 수 없습니다.** test 달력 격자는 176시간으로 기본 168시간 블록 두 개에 필요한 336시간보다 짧습니다. 24시간 민감도 분석은 명목상 7개 단위라 기술적으로만 해석하고, 336시간 민감도 구간도 미산출입니다. 131개 셀 재표집 결과는 기본 시간 의존성 구간을 대체하지 않습니다.
- 구간은 세 개의 고정된 학습 결과에 조건부인 평가 표본 불확실성입니다. 재학습 변동이나 독립 일반화의 불확실성을 뜻하지 않습니다. 98.75% 보정의 해석 범위는 네 고정 anchor의 기본 RMSE 비교이며, 다른 지표·구조 비교·민감도 분석을 같은 확증적 주장으로 확대하지 않습니다.

근거: [analysis.json의 데이터별 기본·민감도 분석](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/analysis.json), [고정 실행·재표집 계약](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/execution_contract.json).

**실행 복구와 해석 한계 기록 — 완료**

첫 배포에서는 빈 `sample_data` 디렉터리 12개가 파일 중심 패키징에서 누락돼 CPU 자격 검증 36조건이 import 단계에서 실패했습니다. 당시 예측·test 산출은 0개였습니다. 실패 배포를 `failed_deployment_1`에 보존하고 빈 폴더만 복원했으며 코드·데이터·체크포인트와 최초 24시간 마감을 유지했습니다. 보존 목록 115개 중 연구 증적 79개는 로컬 해시를 검증했고, 나머지 36개 matplotlib font cache는 명시적인 회수 제외 대상입니다. 성공 평가의 실패 0개와 이 초기 배포 실패 36개를 구분합니다. [보존 manifest](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/failed_deployment_1/archive_manifest.json), [복구 기록](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/deployment_layout_correction.json).

평가 freeze는 모델 개발과 탐색 이후에 정해졌습니다. Deep Renewal도 validation 결과가 존재한 뒤 사용자 요청으로 원고 비교에서 제외됐으며 연구 기록은 보존됩니다. 이를 사전 과학적 부적격 사유로 바꾸거나 native 분포적 수요 예측 방법 전반에 대한 우월성으로 해석하지 않습니다. 기존 validation 표와 v0.7 기준선은 별도로 보존합니다. [승인 범위와 한계](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/execution_contract.json).

**원고 본문·표·해석 연결 — 완료**

- [현재 원고](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd_2027_draft/main.tex)의 초록·서론·평가 방법·test 결과·논의·결론·Appendix A에 완료 결과를 반영했습니다. Appendix B의 last-quantity 설명에도 validation과 test의 순위 반전을 연결했습니다.
- 수치와 범위의 독립 검토를 받았으며, 두 단계 반올림으로 생긴 4개 표시값의 마지막 자릿수 차이를 원본 정밀도에서 직접 반올림한 값으로 수정했습니다.
- [표 대조 증적](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/manuscript/table_verification.json)에서 새 수치 272개, 학습 모델 36개 데이터·모델 그룹(108조건), 결정적 기준, 고정 anchor 구간을 확인했습니다. 기존 validation·계산 효율·훈련 특성·validation 구간별 분석 등 8개 표의 본문은 보존됐고, 중복 label이나 미정의 참고문헌·내부 참조는 없습니다.
- [내장 LaTeX 컴파일](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/manuscript/final_compilation.json)은 최종 소스에서 성공했습니다. 원고는 Codex 내장 편집기로 열도록 등록했습니다. 컴파일 성공과 PDF의 모든 페이지를 시각적으로 검수했다는 주장은 구분합니다.
- [원고 변경 기록](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/manuscript/update_receipt.json), [이전 원고 대비 변경분](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/manuscript/change.patch)에 갱신 범위와 소스 SHA를 남겼습니다. 이전 원고는 `manuscript/before.tex`에 보존합니다.

**남은 작업 순서 — 승인 범위 완료**

이번에 승인된 평가 범위 고정 → 전체 환경 검증 → 기존 test 평가·통계·원고 반영의 세 단계는 모두 끝났습니다. 이 캠페인에는 대기 중인 학습·평가 조건이 없습니다. 투고본 편집에서는 학회 형식·분량과 최종 PDF 조판을 별도로 점검할 수 있으나, 이는 이번 결과를 다시 선택하거나 추가 실험을 시작할 근거가 아닙니다.

분석 JSON SHA-256: `5041fa56d0bb7f40a721d4d78465682b8e651c5e422e8227cee7857e7d5d2488`.
최종 원고 SHA-256: `44189836334c73b75138689c9baa3496e53a650fdcff36ad9a75d534711aa331`.
