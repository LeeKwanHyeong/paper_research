# Writing Notes와 주장–근거 지도

작성일: 2026-10-08. 범위: 연구 검토용 영문 Methods·실험 설계·관련 방법 설명. 새 학습·평가·원격 조회는 수행하지 않았다. Test/held-out 성과 및 Validation/Test 혼합 결과를 열람하지 않았다. 이 문서는 기존 계약을 바꾸는 문서가 아니다.

## 1. 사용자가 지정한 작성 기준

- “중심 질문을 **‘인코더의 이력 요약에 명시적 직전 상태 보정이 추가로 도움이 되는가?’**로 설명합니다.”
- “검색은 별도 보조 질문으로 두고, 기존 근거·새 검증·아직 미확인인 주장을 구분합니다.”
- “Methods와 실험 설계 초안을 먼저 작성하고, 남은 S2P2·관련 선행연구 대조를 보강합니다.”

이 기준을 우선했다. `oma-academic-writer`의 Writing Notes·Claim-Evidence Map 절차를 적용했으며, 본문에는 이미 관측한 성과와 결과를 기다리는 가설의 시제를 구분했다. 아래 기술 정정은 사용자 질문을 버리는 변경이 아니라 **실제 실험이 어느 범위의 질문에 답할 수 있는지** 명확히 하기 위한 것이다.

## 2. 원고의 핵심 판단

**주 기여 후보:** 이력 조건부 인코더 안에서 명시적 국소·순환 보정 경로가 추가로 유용한지, 보정 없음 및 같은 활성 파라미터 수의 현재 위치 전용 대조로 확인한다. CNN+GRU54는 lag-1 하나만 쓰지 않는다. Q/K/V의 current·lag1·lag2와 GRU의 누적 상태가 함께 바뀌므로, 성공하더라도 CNN·GRU·직전 상태 각각의 독립 기여로 분해하지 않는다.

**보조 기여 후보:** 시간·수량 query를 따로 정규화하고 고정 비중으로 결합하는 제약이 mean·weighted·generic512보다 유리한지 확인한다. 학습된 정적 prototype64/top4 자체나 두 view라는 이름을 신규성의 근거로 삼지 않는다. fixed가 통과하지 않으면 두 번째 기여를 성과 주장으로 승격하지 않는다.

**증거 체계:** 기존 MLP8 Validation은 동기, 새 3seed Validation은 대표 구조의 개발 검증, 미노출 자료에서의 평가는 별도 미완료다. 현재 source/initial-update parity와 과거 A100 checkpoint 재현도 구분한다.

## 3. 주장–근거 지도

| 본문 주장 | 유형·현재 상태 | 원본 근거 | 허용되는 해석과 제한 |
|---|---|---|---|
| 기존 MLP8 이력 경로의 수량 이득 | Validation 관측·완료 | [기존 표](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_structure_contribution_results_20261007_v1/VALIDATION_TABLES.md), [기여 판정](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_structure_contribution_results_20261007_v1/CONTRIBUTION.md) | H1/H0 18/18 쌍 전체 RMSE 개선. 용량 차이가 있어 대표 CNN+GRU54 기여 증거로 옮길 수 없음. |
| 수량 개선에 시간·꼬리 손해가 동반될 수 있음 | Validation 관측·완료 | 같은 표/기여 판정 | Taxi 전체 TimeNLL 평균 +53.17%/+69.64%. 같은 수량 선택 epoch 수치이며 일부 불리한 조건을 삭제하지 않음. |
| CNN+GRU54와 mean의 지정 초기화/초기 업데이트 동등성 | 기술 검증·완료 | [native attempt3 receipt](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_claims_and_parity_20261008_v1/evidence/attempt3/completion_receipt.json) | 5080 단일 Runtime의 83 bitwise checks. 전체 fit 또는 과거 A100 초기화·학습 동등성 증명 아님. |
| full의 국소·순환 보정 수식 | 동결 코드 정적 확인·완료 | 아래 M1–M4 | CNN은 첫 attention 전 Q/K/V, GRU는 첫 block 전체 출력에 적용. GRU가 현재 입력으로 갱신한 상태를 더함. |
| current-only의 25,308 active budget | 동결 코드 정적 확인·완료 | 아래 M3–M4, history 계약 | 파라미터 수 일치. activation·깊이·비용·함수 공간 동등성 아님. |
| fixed와 generic512의 공정한 예산 대조 | 설계·구현 확인 / 성과 대기 | 아래 M5–M6, retrieval 계약 | 각각 512개, 온도1. 이전 521개 generic·adaptive와 별개. |
| Q1/Q2 전체 실험 성과 | 미확인 | [실행 계획](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contribution_execution_20261008_v1/execution_plan.json) | 승인된63 fit은 9+27+27이며 history full9를 mean으로 한 번 재사용. 완료 수치로 쓰지 않음. |
| 논문 외부 비교의 공정성 | 일부 확인 / 추가 확인 필요 | 아래 S1–S3 | 현 S2P2 adapter 구현은 설명 가능. 과거 전체 source·Runtime·seed·checkpoint의 동등성은 미완료. 새63에 외부 모델 재학습 없음. |
| 학습 memory·시간/수량 모델링과 선행연구의 관계 | 6편 원문 방법 검토·완료 | [문헌 비교](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contribution_writing_20261008_v1/LITERATURE_COMPARISON.md) | 최초성·전체 문헌 부재를 보증하지 않음. |
| 새 독립 평가·통계적 유의성·외부 우월성 | 미확인 | 새 평가 receipt 없음 | 본문의 성과 주장으로 사용하지 않음. |

## 4. 동결 코드와 수식의 연결

아래 링크는 현재 작업 트리의 변경 가능한 모델보다 **실제 새 실험이 사용하는 동결 source**를 가리킨다. 모든 동결 source 파일의 raw SHA와 계약 canonical SHA를 확인한 결과는 `evidence_manifest.json`에 기록한다. 실행 없이 정적으로 확인한 사실이다.

- **M1 예측 시점:** [공통 target readout](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/paper/scripts/count_aware_tpp_backbone/core.py:60), [Titan target mask](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/models/TPPs/CountAwareTitanCNNGRU.py:264). Target는 최종 위치, readout은 바로 전 관측 위치다.
- **M2 공통 encoder:** [pre-norm block](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/models/Titan/backbone.py:45), [입력 embedding](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/models/TPPs/CountAwareTPP.py:435). 입력은 log1p 특징이며 Train z-score가 아니다.
- **M3 full correction:** [CNN/GRU 본체](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/models/TPPs/CountAwareTitanCNNGRU.py:85), [zero CNN initialization](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/models/TPPs/CountAwareTitanCausalQKV.py:170). CNN 576 + GRU path 24,732 =25,308. GRU latent width가54다.
- **M4 current-only:** [pointwise basis와 control](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/models/TPPs/CountAwareTitanCNNHistoryControls.py:98). CNN current projection은 첫 attention 전, GRU current hidden은 첫 attention/FFN 후이므로 둘을 같은 prefix-conditioned tensor라고 쓰지 않는다. 두 repeated-current step은 state carry를 없애면서 recurrent weights를 기능적으로 참여시킨다.
- **M5 prototype와 summary:** [static bank](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/models/Titan/common/memory.py:229), [32관측 summary 및 fixed](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_retrieval_20261008_v1/source/models/TPPs/CountAwareTitanContextualRetrieval.py:104). 각 view는 last/mean/population-std/last-minus-previous다. 시간 log(delta/s)와 수량 log1p를 Train 표준화한다.
- **M6 generic512·대수적 한계:** [제어 구현](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_retrieval_20261008_v1/source/models/TPPs/CountAwareTitanFixedRetrievalControls.py:46). Fixed half-sum은 정규화된 두 표현의 concat-dot/2와 같다. 이를 항상 literal concatenated cosine과 같다고 쓰면 zero/epsilon norm 예외를 놓친다. 원고는 정확한 concat-dot 등식만 쓴다.
- **M7 heads·loss:** [quantity](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/models/TPPs/CountAwareTPP.py:770), [integer duration law](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/models/TPPs/positive_integer_time.py:110). Quantity는 softplus log예측→expm1이며 Train 평균/표준편차 역변환 없음. 시간은 round/clamp 정수 질량. 기존 v0.7 density와 혼합하지 않는다.
- **M8 lookback와 sequence cap:** [window 선택](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/data_loader/event_seq_data_module.py:337), [dataset 설정](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/paper/scripts/count_aware_tpp_backbone/datasets.py:9). `lookback_weeks`는 dataset `seq` 단위로 사용되는 과거 변수명이다. Taxi168 hourly bins, Intermittent520 weekly bins, RAF84 monthly bins, Instacart52 daily bins이며, 마지막 관측좌표 기준 [seq−(W−1),seq]이다. Target 추가 후 sequence cap을 적용하므로 최대 history는 cap−1이다.

**S2P2 adapter의 별도 경계**

- **S1 입력/출력:** [adapter](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/models/TPPs/CountAwareAdditionalTPP.py:58). Categorical ID/embedding table 대신 gap·quantity 연속 특징을 `mark_embedding` 인자에 전달한다. `simple_mark=True`는 `num_marks=1`이라는 뜻이 아니다. 관측 직후 right state를 공통 head에 전달한다.
- **S2 target gap:** 공통 batch에는 target gap이 남지만 adapter는 causal scan의 target 이전 위치를 readout한다. Titan-specific mask 설명을 모든 external adapter에 똑같이 적용해 “target gap tensor 자체가 제거된다”고 쓰지 않는다.
- **S3 provenance:** 과거 외부 실행 배포 metadata와 비교하면 wrapper·vendor kernel·공통 caller는 일치하나 `CountAwareTPP.py`와 `CountAwareFactory.py`는 다르다. 따라서 일부 source 일치만 확인됐고 전체 학습 실행 동등성은 미확정이다. 원고는 S2P2 성능 수치를 새로 인용하지 않는다.

## 5. 작성 시점의 상태와 원고가 기다리는 증거

| 주제 | 상태 | 확인 범위 |
|---|---|---|
| MLP8 구조54 Validation 원본·집계 | 완료 | 2026-10-07 10:08:43 KST 완료 receipt. 대표 구조로 성과 이전 금지. |
| native CNN+GRU54/mean parity | 완료 | 2026-10-08 09:11:52 KST 완료 receipt. 제한된 기술 동등성. |
| Instacart 9 fits | 진행 중 | 로컬 보존 관측 2026-10-08 14:38:06–07 KST: 완료1/진행2/대기6/실패0/미확정0. |
| CNN+GRU54 보정27 | 다음 작업 | Instacart 종료·원본 검증 gate 이후. 이 문서에서 실행하지 않음. |
| 고정검색27 + mean9 재사용 | 다음 작업 | 보정 단계 종료·원본 검증 gate 이후. |
| CPU binary 재추론·새 독립 평가 | 미완료 | 본 작성 작업에 포함되지 않음. |

이 표는 [보존 상태 JSON](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contribution_execution_20261008_v1/STATUS_instacart_1791437887197402000.json)의 과거 관측이다. 새 시각으로 갱신하거나 현재 GPU 상태로 간주하지 않는다. 자동화·학습 계약·원격 작업에는 손대지 않았다.

## 6. 문장·인용·주장 품질 검토

- 본문에서 결과 대기인 Q1/Q2를 가설·판정 규칙으로 쓰고, 관측 결과는 MLP8과 parity의 한정된 과거형으로 분리했다.
- “최초”, “SOTA”, “유의하다”, “S2P2 외 모두 우월” 같은 미입증 문구를 성과 주장으로 쓰지 않았다.
- 과학적 범위가 다른 표현을 정정했다: 직전 단일 state→국소·순환 package, 온라인 사건 memory→정적 학습 slot, frozen encoder weights→고정 architecture와 end-to-end fit, 단일 categorical mark S2P2→연속 특징 adapter.
- 인용은 6편의 원 논문·저자 원문·출판처에 연결한다. 코드 사실은 논문 인용으로 대체하지 않고 동결 source에 연결한다. 문헌 사실과 해석은 문헌 비교표에서 구분했다.
- 영어 본문은 설명형 문단과 필요한 수식·비교표로 구성했다. 결과를 채워 넣기 위한 임의 수치·모조 결과표는 없다.
- 이 초안은 내부 검토용 AI-assisted 문서다. 현재 PAKDD2027 생성형 AI 정책을 만족하는 투고문이라고 표시하지 않는다. 정책에 맞는 저자 작성 과정은 별도 정리가 필요하다.

독립 문서 QA 및 파일 링크/해시 검증 결과는 `REVIEW.md`에 기록한다. 문서 검증은 학습·성능·재현성 검증을 대신하지 않는다.
