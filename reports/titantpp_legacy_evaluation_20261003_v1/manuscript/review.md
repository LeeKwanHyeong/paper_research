# 실제 결과 반영 원고 독립 검토

**판정: PASS — 표시 수치 오류 4곳과 검증 범위 문장 1곳이 모두 해결됐으며, 추가 차단 사항을 발견하지 못했다.**

- 검토일: 2026-10-03.
- 최종 대상: [main.tex](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd_2027_draft/main.tex).
- 최종 대상 SHA-256: `44189836334c73b75138689c9baa3496e53a650fdcff36ad9a75d534711aa331`.
- 수정 전 비교 대상: [before.tex](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/manuscript/before.tex).
- 결과 원본: [analysis.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/analysis.json), SHA-256 `5041fa56d0bb7f40a721d4d78465682b8e651c5e422e8227cee7857e7d5d2488`.
- 보조 근거: `execution_contract.json`, `review/summary.md`, `result_interpretation_review.md`. 모든 성능 검토는 승인된 기존 test 재평가 범위다.

## 발견 사항과 해결 확인

**[P2] 전체 관측 이력의 동일성을 검증했다는 표현을 요약 필드 검증으로 좁힘 — 해결.** 최초 [693행](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd_2027_draft/main.tex:693)의 `verified identical targets, observed histories`는 모든 history 배열의 내용·순서를 대조했다는 의미로 읽혔다. 실제 `evaluate.py` 25–29행과 `analyze.py` 26–28행의 `TRUTH_FIELDS` 및 후자의 251–267행은 target identity/truth와 `history_length`, `last_quantity`, `mean_quantity` 등 요약 필드를 비교한다. 전체 sequence를 독립 재구성한 감사는 아니다. 최종 원고의 `observed-history summaries` 문구와 위 SHA를 직접 확인했다. 이는 관측 이력이 다르거나 누수가 있다는 지적이 아니라, 완료된 검증의 범위를 정확히 표현하는 수정이다.

최초 검토 SHA `5fc9c12a57386429d65fe02af7925a91cc7aa2bed0c1e64061ae4eeb9e32089a`에서 NLL 표시 값 4곳이 원본 JSON을 직접 소수점 넷째 자리까지 반올림한 값과 달랐다. 이미 반올림된 중간 표시 값을 다시 반올림한 수준의 오류이며 결과 방향에는 영향이 없다. 원고 담당자가 수정했고, 아래 값과 최종 SHA를 다시 확인했다.

| 최신 원고 위치 | 대상 | 원본 값 | 수정 후 표시 | 상태 |
| --- | --- | ---: | ---: | --- |
| [739행](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd_2027_draft/main.tex:739) | Intermittent NHP NLL 표본 SD | 15.76364502107661 | 15.7636 | 해결 |
| [745행](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd_2027_draft/main.tex:745) | Instacart Titan NLL 표본 SD | 0.010450045035181788 | 0.0105 | 해결 |
| [1128행](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd_2027_draft/main.tex:1128) | 구조 표의 동일 Titan NLL 표본 SD | 0.010450045035181788 | 0.0105 | 해결 |
| [1130행](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd_2027_draft/main.tex:1130) | Instacart all-available NLL 평균 | 2.8311498070795427 | 2.8311 | 해결 |

초록의 `times lower` 표현도 [62행](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd_2027_draft/main.tex:62)에서 MAC이 TitanTPP 시간의 5.58–5.74배를 사용한다는 명시적 비율로 정리됐다. 기존 계산 시간비와 방향이 일치한다.

## 수치 및 표 대조

- 새 학습 결과 표 두 개의 **40개 표시 행**을 JSON의 평균·표본 SD와 직접 비교했다. 28개 본문 행과 12개 구조 행이며 Titan 행 4개가 의도적으로 중복된다. 고유 학습 그룹은 36개이고 세 시드가 108조건을 이룬다. 발견한 4셀 외 불일치는 없었고, 수정된 4셀도 원본과 다시 대조했다.
- 결정적 기준 표의 **8개 행**에 있는 MAE·RMSE가 원본과 일치한다. 결정적 기준에 seed SD나 시간 NLL을 부여하지 않았다.
- [815행](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd_2027_draft/main.tex:815)의 고정 anchor 표는 `Titan − anchor` 부호와 95%·98.75% 구간을 정확히 표시한다. Taxi의 미산출 표기도 맞다.
- 기존 `tab:main`, `tab:naive`, `tab:ablation`, `tab:extension`의 표 본문은 `before.tex`와 동일하다. validation 성능 수치에 test 값을 덮어쓰지 않았다.
- 중복 LaTeX label과 정의되지 않은 내부 `ref`/`eqref`는 발견하지 못했다. 이 검토에서는 컴파일이나 PDF 조판 검사를 수행하지 않았다.

## 주장 및 평가 범위

| 점검 대상 | 최신 위치 | 확인 결과 |
| --- | --- | --- |
| 초록·도입·결론의 중심 결과 | 46–66, 96–105, 971–988행 | Intermittent test 수량 개선, Taxi·Instacart의 S2P2 우세, RAF의 작은 점추정 차이와 0 포함 구간을 정확히 구분한다. 독립 미접근 평가 완료로 표현하지 않는다. |
| 고정 anchor와 test 최저 외부 모델 | 469–476, 698–708, 796–804행 | RMTPP가 Taxi·Intermittent 고정 anchor임을 유지한다. S2P2가 네 test에서 외부 최저 RMSE라는 관측 순위와 구분한다. Taxi의 RMTPP 대비 51.91% 감소를 전체 외부 모델에 대한 우위로 해석하지 않는다. |
| Intermittent 실제 변화율 | 699, 766–768행 | S2P2 대비 RMSE 14.12% 감소, last-quantity 대비 MAE/RMSE 10.29%/13.88% 감소가 원본과 일치한다. validation의 last-quantity MAE 우열 반전을 명시한다. |
| 불리한 시간 NLL | 711–716, 920–922행 | Titan Intermittent의 validation 0.4971 대 test 15.3778, test THP 8.3263, 네 데이터에서 S2P2보다 높은 Titan 평균 NLL을 드러낸다. 원인을 검증 없이 단정하지 않는다. |
| 조건부 불확실성 | 460–477, 796–811, 924–932행 | 세 고정 학습 시드를 함께 유지하고 seed를 재표집하지 않는 범위, 개별 95%와 고정 네 anchor 98.75%, 기타 비교의 기술적 범위를 구분한다. RAF의 0 포함 구간을 동등성 입증으로 쓰지 않는다. |
| Taxi primary 미산출 | 806–811행 | 176시간은 두 168시간 블록에 부족하다. 24시간 7개 명목 블록은 기술적 민감도, 336시간은 미산출이며 whole-cell 구간으로 primary를 대체하지 않는다. |
| 구조 비교 | 934–946, 1086–1107행 | current-only 대비 Titan RMSE 17.72%/5.07% 감소(Taxi/Intermittent), RAF·Instacart의 반대 방향, all-available의 원형 대비 16.29%/3.66%/0.11% 감소와 RAF 0.51% 증가가 맞다. all-available의 Taxi·Intermittent NLL 악화도 포함한다. |
| 단순 기준 한계 | 769–774행 | Instacart window-mean과 사실상 같은 RMSE, RAF window-mean의 전체 학습 패널보다 낮은 RMSE 점추정 및 높은 MAE를 함께 제시한다. 해당 구간의 부호와 값도 맞다. |
| 검증·test 분리 | 480, 551, 594, 998, 1086, 1614–1618, 1627–1635행 | 본문 validation 비교와 기존 Appendix A 수치, 별도 test 결과, validation strata의 범위를 구분한다. 과거 last-quantity 서술을 test에 적용하는 모순이 해소됐다. |
| Deep Renewal 제외 | 950–954행 | validation 결과가 존재한 뒤 원고 비교에서 제외되었음을 명시한다. 과학적 사전 제외나 원래 계획된 제외로 소급 포장하지 않고, 모델 범위 한계를 인정한다. |
| 날짜 이동 | 356–358, 961–965, 1149–1151행 | 사용자 확인 사실인 기밀·비식별 목적의 일정 offset 이동을 기재한다. offset 크기·공유 권한·독립 감사를 새로 주장하지 않고 upstream 문서와 재배포 문제를 별도로 둔다. 일정 offset의 시간차 보존 설명은 원래 달력 날짜나 달력 특성 보존 주장으로 확장되지 않는다. |
| 실행 단계·선택 고정 | 390–396, 452–458, 692–696행 | 새 학습·train-validation 재적합·test 전처리 적합 없이 기존 checkpoint를 재평가하는 계약과 일치한다. 완료된 기존 test 재평가와 아직 수행하지 않은 독립적·전향적 새 cohort 평가를 구분한다. |

이번 검토는 기록된 집계·계약과 원고 주장을 대조한 것이다. 실제 원격 실행과 108개 추론을 재현하거나 독립 데이터 접근 이력을 새로 감사한 것은 아니다.

## 완료 및 남은 작업

- **완료:** 변경부의 핵심 수치·부호·비교 모델·평가 split·조건부 CI·결과 이후 제외 고지·날짜 이동 표현 검토. 지적한 수치 오류와 검증 범위 문장 수정 확인. 부모 작업은 최종 SHA 기준 272개 수치·표 보존·참조 검증 재통과와 내장 컴파일 성공을 보고했다.
- **다음 작업:** 부모 작업에서 최종 산출물 기록을 마무리한다. 이 검토에 미해결 문장·수치 지적이나 추가 실험 요구는 없다.
