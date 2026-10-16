# Writing Notes and Claim–Evidence Map

## 2026-10-11 최신 개정: 완료 결과와 주장 정합 — 완료

이번 작업의 사용자 승인 범위:

> ㅇㅇ 그럼 Instacart랑 이력 보정 결과를 논문 초안에 반영하고, CNN+GRU54 중심으로 Results·Methods·주장 정합 작업부터 진행하자.

완료 Instacart9·보정27의 **Validation-only** 결과를 기존 CNN+GRU54 중심 초안에 반영했다. 기존 외부 비교 패널과 역사적 MLP4/MLP8 수치, Method 수식·Algorithm 1, 참고문헌 28개를 보존했다. 부록 C의 긴 표현 한 문장은 수치나 의미 변경 없이 줄바꿈 문제를 해결하도록 다듬었다.

| 현재 주장·위치 | 직접 근거 | 해석 범위 |
|---|---|---|
| 초록·5.1·결론: 기존 CNN+GRU54의 외부 비교 위치 | 기존 표·대표9/외부54 선택 기록, 이전 evidence와 byte 대조 | 별도 캠페인. 새 보정 full9 평균으로 교체하거나 합치지 않음 |
| 5.2·부록 G: 보정27의 선택/최종 epoch와 여섯 지표 | `titantpp_contribution_history_rtx_handoff_20261010_v1/analysis/selected_conditions.json` SHA `41ddf10b5379b2ba647e1c8500b7e02e2214bfa127e797d0ee293c1050668742`; completion SHA `656eef38d3192d57f89e86aaa2ed65e69ae5c256288668c1819db7e838c48362` | Taxi·Intermittent에서 모든 seed의 두 대조 대비 strict RMSE 개선. RAF52는 실패. Q1 개발 일관성 지지이며 유의성·독립 확인이 아님 |
| 5.2·Discussion·부록 G: 전체 TimeNLL11/18, 큰 수량 RMSE/MAE10/18 손해 | 같은 quantity-selected epoch의 원본과 seed별 full−control 차이 | 유리한 별도 epoch 선택 없음. 시간이나 드문 큰 수량의 손해도 보존 |
| 5.3·부록 H: Instacart9 구조 비교 | `titantpp_instacart_completion_recovery_20261010_v1/analysis/selected_conditions.json` SHA `b2cb366f3a7f26bc4ecd8706a2d58f3fcfbebb991b11d57c99e6c146a3ee04b3`; completion SHA `459ae9f15f6f4247584a8d47bd8e3901ab6e7ba91d6a98df3f64b09dde95cea6` | 평균 수량 이득은 작음. seed52 MLP12 우위, 평균 전체 TimeNLL 악화. 다른 파라미터 예산의 구조 비교 |
| Methods·§4·G/H: 실제 구현과 실행 범위 | frozen source129/Instacart126 closure, [Methods 대조](../../reports/titantpp_completed_results_integration_20261011_v1/methods_alignment.md) | full/current-only 활성25308 동일, 계산·깊이 차이와 current hidden의 prefix 이력 유지. bypass 등록 모듈은 동결·실행0. CNN/GRU·lag별 인과 분리 불가 |
| §4·Discussion: GPU·Runtime 한계 | A10015+RTX12 완료 provenance와 RAF42 재개 기록 | RAF42 full/current-only 물리 GPU 이관. seed 간 Runtime 차이. 병렬 wall time을 구조 효율로 해석하지 않음 |
| 5.4: prototype retrieval | 기존 설계·초기83항목 bitwise 검사와 후속 작업 순서 | 대기. 초기 검사만으로 전체 궤적 동일성 주장 없음. 새 RTX 공통 mean 계약 없이 A100 full 재사용 없음 |

수치는 원본 JSON에서 LaTeX로 생성하고 별도 QA가 평균·표본 SD 108개 셀과 paired 손해를 대조했다. 표본 SD는 3seed의 `n−1` 표준편차다. 각 fit의 최초 strict finite 최소 전체 raw 수량 RMSE epoch를 선택하고, 전체/큰 수량 MAE·RMSE·TimeNLL은 같은 epoch에서 가져왔다. Instacart 564개 완료 epoch에 대한 선택 점검과 보정27 독립 최종 감사는 새 추론과 구분한다.

[통합 ledger](../../reports/titantpp_completed_results_integration_20261011_v1/integration_evidence.json), [검토·검증 기록](../../reports/titantpp_completed_results_integration_20261011_v1/REVIEW.md), [수정 전 파일](../../reports/titantpp_completed_results_integration_20261011_v1/before_manifest.json)을 남겼다. CPU binary 재추론·Test·새 독립 평가는 미완료다. 학습·추론·SSH·자동화 변경은 수행하지 않았다. 새 `main.pdf`의 실제 검증은 이번 기록을 따르며 이전 18쪽 기록과 구분한다.

아래는 **2026-10-08 및 이전의 작성 이력**이다. 이전 절 번호·미완료 상태·63조건 계획·mean 재사용 설명은 당시 기록이며, 현재 상태나 실행 승인으로 사용하지 않는다.

## 2026-10-08 대표 모델과 Results의 일치 — 당시 완료

사용자 승인 범위는 “CNN+GRU54와 외부 비교군 → CNN+GRU54 보정 대조 → CNN+GRU54 검색 대조” 순서이며, 과거 MLP4·MLP8은 개발 동기와 부록 상세 근거로 남기는 것이다. Gemini 제안의 실증 연구 전환이나 제목 변경은 이번 수정에 포함하지 않았다.

**변경:** 기존 Results의 MLP4/MLP8 상세 표와 설명을 부록 B/C로 이동했다. 본문 Table 2는 완료 CNN+GRU54 9조건과 외부 18개 dataset/model 그룹으로 새로 구성했고, 본문 5.2/5.3은 대표 구조의 미완료 대조를 명시한다. 초록은 155단어이며 CNN+GRU54 수량 성능·시간 손해·RAF 예외를 먼저 설명한다. Methods/Algorithm 및 참고문헌은 변경하지 않았다.

**근거:** CNN+GRU54의 `analysis/Validation_comparison.json` SHA `0739b21347f57c97a059c41065dcf3881716c22b26c3a8517a8c164965d03829`, 완료 receipt와 9개 selected endpoint를 검증했다. 각 그룹의 6지표 평균·표본 SD 및 기존 외부 18그룹의 3지표를 재계산했다. 기존 MLP 표는 수치를 변경하지 않고 보존했다. [통합 근거](../../reports/titantpp_cnn_gru_results_reorganization_20261008_v1/evidence.json)를 참조한다.

| 현재 주장·위치 | 확인 근거 | 해석 범위 |
|---|---|---|
| 초록·5.1·결론: CNN+GRU54의 Taxi/Intermittent 평균 수량 오차 우위 | 완료 CNN9 endpoint와 기존 외부54 selected Validation 기록 | 공통 평가 계약의 개발 비교. 동일 초기화·GPU 또는 보정 경로의 인과 효과가 아님 |
| 5.1: RAF S2P2 RMSE 우위, 세 데이터 timing 한계, Intermittent 단순 규칙 우위 | 같은 선택 epoch의 지표, 보존된 단순 규칙 원본 | 불리한 결과도 유지. 수량·시간 공동 우월성을 주장하지 않음 |
| 5.2: 대표 보정의 추가 효과 | 신규 full/removal/current-only 27 fit 계약 | 미완료. MLP8의 18개 비교쌍 결과로 대체하지 않음 |
| 5.3: 검색 변경의 추가 효과 | 신규 weighted/fixed/general512 27 fit와 새 full9 mean 재사용 계약 | 미완료. 이전 521-parameter 파일럿과 구분 |
| 부록 A: 대표 모델의 선택 epoch와 큰 수량 지표 | 기존 CNN9 완료 원본 | seed42 재사용3+후속6. A100/RTX 경로를 공개하며 시간을 혼합하지 않음 |
| 부록 B/C: MLP4 외부·MLP8 내부 비교 | 이전 패널·54조건 Validation 근거 | 개발 동기와 별도 구조의 관측. 대표 CNN+GRU54 결과가 아님 |

아래의 절 번호·부록 위치는 앞선 개정 시점의 기록이다. 현재 구성은 위 표와 README를 따른다.

## 작성 기준

사용자 지시를 이 개정의 기준으로 삼았다.

> `/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd_2027_draft` 디렉토리에 있는 방식을 기준으로 잡고 draft 다시 작성해보자. 어차피 투고전에 내가 직접 수정할거니까, 현재 상황에서는 최대한 titantpp_pakdd_2027_draft 기존 작성한 main.tex 처럼 작성해. 또한, 너무 AI 스러운 문장이나 문법은 매우 지양해.

기존 원고에서는 문제 중심의 도입, 주제별 Related Work, 하나의 Method 절 아래 계산 순서를 설명하는 구성, 명시적인 기여 목록, LNCS 조판과 그림 표현을 참고했다. 과거 원고의 Test 수치·모델 폭·성능 주장은 복사하지 않았다. 새 원고는 CNN+GRU54를 대표 방법으로 설명하고, 완료된 MLP8 실험은 구조가 다른 선행 대조로 명명한다.

## 외부 비교군 Validation 결과 복원 — 완료

사용자가 전체 비교군 결과를 추가하도록 승인해 §5.1을 새로 작성했다. RMTPP/NHP/THP/SAHP/S2P2/AttNHP와 역사적 TitanTPP-MLP4의 네 데이터·3seed 비교이며, 이어지는 §5.2–5.3의 MLP8 ablation과 §5.4의 CNN+GRU54 증거 상태를 분리했다. MLP4는 폭4·8분기·명목 보정 파라미터6,144로, Instacart 폭16이나 RAF active-norm 대조가 아니다.

- **수치:** 84 selected 실행의 28그룹 평균·표본 SD를 재계산했다. MAE/RMSE/TimeNLL은 각 실행의 동일 quantity-selected Validation checkpoint에서 가져왔다. 불완전한 중간 통합표 대신 최종 Instacart 분석의 21실행을 사용했다. [수치·선택 epoch·SHA](../../reports/titantpp_external_panel_restoration_20261008_v1/panel_evidence.json)에 재사용한 완료 검증과 한계가 있다. 기존 binary 감사의 재사용이며 새로운 binary 재추론 감사는 아니다.
- **계약:** 과거와 현재 데이터 계약19쌍의 target identity/quantity SHA, window, head/loss, optimizer, Train 통계와 선택 규칙이 일치했다. 동결 source691개와 Methods manifest8개 raw SHA도 검증했다. [계약 감사](../../reports/titantpp_external_panel_restoration_20261008_v1/contract_audit.md)에 Runtime·소유 GPU·과학 revision의 차이를 명시했다. seed가 같다는 사실을 초기화나 학습 경로의 동일성으로 해석하지 않는다.
- **해석:** Taxi·Intermittent의 외부 여섯 모델 대비 수량 이득을 적되, RAF의 작은 RMSE 차이와 Instacart의 S2P2/AttNHP 우세를 보존했다. Timing NLL의 손해도 같은 선택 checkpoint 기준이다. 단순 수량 규칙8행을 추가했고, Intermittent 직전 수량 반복의 더 낮은 MAE를 명시했다.
- **편집:** 외부 모델은 numerical-input/common-head encoder adaptations라고 설명했다. MLP4/MLP8/CNN+GRU54를 초록부터 결론까지 구분했다. §3 Methods·Algorithm1·기존 수치 표와 부록의 수치는 유지했다. 기존 28개 References를 유지했다.
- **검증:** 현재 `main.tex`의 내장 편집기 컴파일을 확인했다. 로컬 `main.pdf`는 앞선 18쪽 출력본이며 이번 개정에서 별도 PDF를 생성하지 않았다. 현재 소스의 페이지 수·시각 검사는 이전 18쪽 검증 결과와 구분한다. [이번 검증 기록](../../reports/titantpp_external_panel_restoration_20261008_v1/validation.json)과 [수정 전 보존](../../reports/titantpp_external_panel_restoration_20261008_v1/before_manifest.json)을 남겼다.

## 앞선 도입·선행연구·부록 개정의 범위

사용자의 간헐 수요 문제 인식, 기존 Appendix 재사용, 누락 References 복원 요청을 함께 반영했다. 도입과 선행연구, 데이터 설명 부록을 확장했고 기존 Results·Discussion·Conclusion과 부록 A의 수치를 유지했다. 전체 학습 상태를 새로 조회하거나 이미 관찰한 결과를 독립 평가로 바꾸는 작업은 포함하지 않았다.

- **Introduction:** 간헐 수요의 발생 간격·수량 → 범주형 사건 유형과 수치형 수량 → 이력 인코더 위의 명시적 보정이라는 순서로 수정했다. 수량이 정수라는 사실과 categorical event type을 구분했다. RMTPP·THP를 대표 사례로 들고 Deep Renewal·FlexTPP를 함께 인용해 TPP 전체의 한계로 일반화하지 않았다.
- **Related Work:** 간헐 수요, 사건 인코더, 수량·시간 출력, 명시적 보정, 학습된 메모리의 다섯 단락으로 확장했다. 각 방법의 원래 설정과 이번 연구의 적용 조건을 구분했다. ReZero의 스칼라 게이트와 이 모델의 초기화, frozen-backbone adapter와 공동 학습, Titans의 적응형 contextual memory와 이 모델의 정적 prototype을 동일시하지 않았다.
- **References:** 기존 9개를 유지하고 19개를 복원했다. 28개가 모두 본문에 인용된다. [문헌별 출처·복원 및 제외 사유](../../reports/titantpp_submission_references_appendix_20261008_v1/reference_audit.md)에 1차 문헌의 확인 범위와 접근 한계를 적었다. 이는 표적 문헌 보완이며 관련 분야 전체를 망라한 참신성 검증은 아니다.
- **Appendices:** A의 완료 MLP8 6지표는 보존했다. B는 데이터 정의·Train 분포, C는 Train 사건 lag 상관, D는 수량·이력 길이별 진단 방법이다. 기존 폭4 MLP의 오차표·기여 수치는 현재 모델 결과로 복사하지 않았다. [부록 근거](../../reports/titantpp_submission_references_appendix_20261008_v1/appendix_evidence.md)와 [원본 SHA·계약 대조](../../reports/titantpp_submission_references_appendix_20261008_v1/appendix_evidence.json)를 남겼다.

## 앞선 재작성의 편집 기준

| 대상 | 개정 내용 | 이유 |
|---|---|---|
| 제목·초록 | 방법과 실제 확보한 결과를 먼저 설명 | 효율·외부 우월성 등 미확인 주장을 제목에 넣지 않음 |
| Introduction | 간헐 수요의 간격·수량 → 이력을 요약하는 인코더에 명시적 보정을 더하는 질문 → 설계와 3개 기여 | 실험 운영 계약보다 연구 동기가 먼저 읽히도록 함 |
| Related Work | 간헐 수요 / event encoder / quantity and timing / explicit correction / learned memory의 다섯 단락 | 선행연구의 방법과 이 연구의 질문을 직접 연결 |
| Method | 입력 → local/recurrent 경로 → retrieval → shared heads | 실제 계산 순서와 기호 정의를 일치시킴 |
| 그림 | CNN+GRU 흐름도, fixed/general512 비교도 | MLP 그림을 CNN+GRU에 잘못 재사용하지 않음 |
| Results | 완료 MLP54의 실제 표·seed 일관성·불리한 결과 | `pending` 표와 집필자용 지시문을 제거 |
| Discussion | 결과가 지지하는 범위와 남은 검증을 연결 | 반복적인 부정·면책 표현을 줄이고 한계를 해당 주장 옆에 둠 |
| 부록 | 기존 6지표와 Train 데이터 특성·lag 분석·진단 방법 | 수치는 평가 split과 모델 세대를 구분하고 모델에 독립적인 자료를 재사용 |

`This manuscript draft ...`, `The final version must ...` 같은 문서 관리 문구를 본문에서 제거했다. 상투적인 강조어와 과도한 수식어 대신 어떤 조건에서 무엇이 개선되거나 악화됐는지 썼다. 논문 전체에서 현재 상태를 숨기지는 않되, 미완료 고지는 Results와 한계 설명에 모았다. AI 탐지 우회 또는 탐지 불가능성은 주장하지 않는다.

## Claim–Evidence Map

| 위치·주장 | 근거 | 허용되는 해석 / 남은 한계 |
|---|---|---|
| §4.2·§5.1: 외부 모델 전체 비교 및 단순 규칙 | [패널 수치 감사](../../reports/titantpp_external_panel_restoration_20261008_v1/panel_evidence.json), [계약 감사](../../reports/titantpp_external_panel_restoration_20261008_v1/contract_audit.json) | MLP4 역사적 common-head Validation. 새 CNN+GRU54 결과, native model 전반의 우월성, 유의성·미노출 독립 평가가 아님 |
| §3 CNN+GRU54 경로, 마스킹, 25,308 parameter, static prototype64 | [동결 Methods 근거](../../reports/titantpp_contribution_writing_20261008_v1/WRITING_NOTES.md), [기존 방법 초안](../drafts/titantpp_contribution_methods_20261008_v1.md), [과학 검토](../../reports/titantpp_contribution_writing_20261008_v1/REVIEW.md) | 구현 설명. CNN/GRU/직전 상태 각각의 성능 기여가 입증됐다는 뜻은 아님 |
| §3 fixed/general512 | 같은 근거의 retrieval 수식·입력 요약·초기화 | mean/weighted 0개, fixed/general512 512개 추가 query parameter. 별도 정규화·고정 혼합의 대조이며 새 종류의 검색 연산이라는 주장은 없음 |
| §4 baseline parity | [83개 bitwise 검증](../../reports/titantpp_claims_and_parity_20261008_v1/evidence/attempt3/completion_receipt.json) | 초기화·합성 입력·공통 Train 배치 첫 업데이트의 제한된 정합. 과거 A100 전체 학습 동일성으로 확장하지 않음 |
| 초록·§5.2–5.3·MLP8 표: MLP54 결과 | [Validation_summary.json](../../search_artifacts/titantpp_architecture_capacity_handoff_20261006_v1/analysis/Validation_summary.json), raw SHA `13cc3a31cdd1f8472ea93db229e290e93eadf70d982e9532c2f41d8304b3a5b7`; [검산 파일](../../reports/titantpp_contribution_writing_20261008_v1/results_update_evidence.json) | 54개 완료 조건, 18개 dataset/arm의 6지표. 모두 quantity-selected Validation. CNN+GRU 효과, 유의성, 외부 우월성 증명 아님 |
| §5: correction 18/18 개선, matched controls Taxi/RAF 통과 | 위 검산 파일 `contrasts`, `historical_gate` | paired 방향 일관성. Intermittent matched control은 각 2/3승이며 이 예외를 보존 |
| §5: Taxi TimeNLL 증가·RAF 큰 수량 MAE/NLL 악화 | 위 6지표와 paired 결과 | 같은 수량 선택 epoch의 손해. 별도 시간 최적 checkpoint와 비교하지 않음 |
| §5: 과거 비용 측정 | 검산 파일 `qualification_cost`와 그 안의 source SHA | 3회 synthetic step 중앙값·검증 작업 포함. 새 warmup5/measure20 비용 gate와 구분 |
| §5.4: Instacart 단일 fit | [관측 JSON](../../search_artifacts/titantpp_contribution_instacart_20261008_v1/hourly_monitor/observation_5090_1791437887192863000.json), SHA `8db0339244646b0e230e8099cb11de373952490475c60ce8349a00e0af4dd9f3` | 2026-10-08 14:38:07 KST 보존 관측. MLP8/seed52/5090/E55 best15만 완료 관측; 최종9조건 회수·비교는 아님 |
| §4 신규 63조건 설계 | [실행 계획](../../reports/titantpp_contribution_execution_20261008_v1/execution_plan.json), [README](../../reports/titantpp_contribution_execution_20261008_v1/README.md) | 설계와 평가 규칙을 설명. 신규 조건이 완료됐다고 표현하지 않음 |
| Related Work 및 References | [문헌 대조 기록](../../reports/titantpp_contribution_writing_20261008_v1/LITERATURE_COMPARISON.md) 및 아래 원 논문 | 방법의 관계를 설명. 인용 논문의 성능을 다른 split·head의 수치와 섞지 않음 |

새로 추가한 기본 방법 인용은 원 논문에서 서지를 확인했다: [Transformer](https://arxiv.org/abs/1706.03762), [GRU](https://arxiv.org/abs/1406.1078), [AdamW](https://arxiv.org/abs/1711.05101). 이 3개는 앞선 9개 문헌에 포함됐으며, 이번 개정에서는 관련 선행연구 19개를 더해 총 28개를 구성했다. 본문은 문헌의 문장을 장문 인용하지 않고 해당 방법을 간단히 설명한다.

## 부록 재사용의 근거와 한계

| 추가 내용 | 근거 | 적용 범위 |
|---|---|---|
| 데이터 정의·Train 분포·이력 길이 | `reports/titantpp_dataset_appendix_20261002_v1/train_profiles.json` 및 현재 세 단계의 동결 데이터 계약 | 7개 단계/데이터 연결의 source ancestry, Train target identity·quantity SHA, 대상 수, lookback·cap·시간 단위·top-code 대조. 원시 데이터를 새로 계산한 결과는 아님 |
| 20개 lag 상관 셀과 고정 cohort 민감도 | `reports/titantpp_train_lag_heatmap_20261002_v1/analysis.json`, `figure.tex`, `verification.json`, 계산 코드 | Train 전체 계열의 기술 통계. loader window와 구분하고 lag별 pair 모집단 변화·RAF 긴 lag 부재를 표시. 보정 효과의 인과 증거 아님 |
| RMSE/MAE 및 strata의 전체 오차 기여 | 지표의 수학적 정의와 현재 계약의 구간 경계 | 수량 선택 checkpoint별 Validation 진단 방법. CNN+GRU54의 미산출 구간 성능을 수치로 제시하지 않음 |

RAF의 현재 Train/Validation 준비 파일은 과거 전체 split 파일과 raw SHA가 다르다. 부모 데이터 계보와 Train target identity·quantity SHA가 일치하는 것을 확인했다. 세부 값과 source SHA는 위 `appendix_evidence.json`에 기록했다. Train 집계와 메타데이터만 확인했으며 원시 parquet·예측 행·held-out 성과를 읽지 않았다.

## 검토와 보존

- 방법 검토에서 발견한 두 문제를 수정했다: 역사적 MLP8에 CNN Q/K/V 경로가 없다는 점, 아주 작은 TimeNLL의 평균·SD 표기.
- 영어·주장 검토에서 발견한 두 모호성을 수정했다: 서로 학습된 모델이 같은 hidden 값을 공유한다는 인상, Instacart 단일 관측의 시각 누락.
- 앞선 References/Appendix 개정 PDF의 18쪽을 렌더링해 확인했다. Train 표 글자와 lag 그림의 폭·글자 크기를 조정했다. 인용·교차참조·그림·표·수식 검토를 통과했고 LaTeX의 미정의 참조·overfull/underfull box가 없다.
- 이번 수정 전 12쪽 원고의 `main.tex`, `main.pdf`, `README.md`, `WRITING_NOTES.md`와 SHA는 [보존 디렉토리](../../reports/titantpp_submission_references_appendix_20261008_v1/before_revision/) 및 [manifest](../../reports/titantpp_submission_references_appendix_20261008_v1/before_revision_manifest.json)에 있다. 기준 원고 디렉토리는 변경하지 않았다. [검증 기록](../../reports/titantpp_submission_references_appendix_20261008_v1/validation.json)에 최종 PDF·소스 SHA와 보존 확인을 남겼다.
- 이 작업은 원고·문서만 수정했다. 학습, 추가 추론, Test 평가, 원격 서버 접속, 자동화 변경은 하지 않았다.

## 앞선 개정 당시의 저자 검토 메모

대표 모델의 완료 성능은 본문에 반영했고 MLP 상세는 부록으로 이동했다. 대표 구조의 보정·검색 대조가 완료되면 그 결과에 맞춰 기여와 결론을 갱신해야 한다. 데이터 출처와 재현성 정보, 문헌 범위, 독립 평가와 외부 비교는 최종 제출 전 보강 대상이다. 이 초안의 문장·수식·주장에 대한 최종 책임은 저자 검토 과정에서 확정한다. 기존 18쪽 출력본과 이후 확장된 현재 소스는 저자 검토용 초안이다. 최종 결과 반영 후 본문과 부록을 제출 분량에 맞춰 정리한다.
