# Writing Notes and Claim–Evidence Map

## 작성 기준

사용자 지시를 이 개정의 기준으로 삼았다.

> `/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd_2027_draft` 디렉토리에 있는 방식을 기준으로 잡고 draft 다시 작성해보자. 어차피 투고전에 내가 직접 수정할거니까, 현재 상황에서는 최대한 titantpp_pakdd_2027_draft 기존 작성한 main.tex 처럼 작성해. 또한, 너무 AI 스러운 문장이나 문법은 매우 지양해.

기존 원고에서는 문제 중심의 도입, 주제별 Related Work, 하나의 Method 절 아래 계산 순서를 설명하는 구성, 명시적인 기여 목록, LNCS 조판과 그림 표현을 참고했다. 과거 원고의 Test 수치·모델 폭·성능 주장은 복사하지 않았다. 새 원고는 CNN+GRU54를 대표 방법으로 설명하고, 완료된 MLP8 실험은 구조가 다른 선행 대조로 명명한다.

## 주요 편집

| 대상 | 개정 내용 | 이유 |
|---|---|---|
| 제목·초록 | 방법과 실제 확보한 결과를 먼저 설명 | 효율·외부 우월성 등 미확인 주장을 제목에 넣지 않음 |
| Introduction | 수량과 발생 시점의 공동 예측 → 인코더 이후 명시적 보정 질문 → 설계와 3개 기여 | 실험 운영 계약보다 연구 동기가 먼저 읽히도록 함 |
| Related Work | event encoder / quantity and timing / learned memory의 세 단락 | 선행연구의 방법과 이 연구의 질문을 직접 연결 |
| Method | 입력 → local/recurrent 경로 → retrieval → shared heads | 실제 계산 순서와 기호 정의를 일치시킴 |
| 그림 | CNN+GRU 흐름도, fixed/general512 비교도 | MLP 그림을 CNN+GRU에 잘못 재사용하지 않음 |
| Results | 완료 MLP54의 실제 표·seed 일관성·불리한 결과 | `pending` 표와 집필자용 지시문을 제거 |
| Discussion | 결과가 지지하는 범위와 남은 검증을 연결 | 반복적인 부정·면책 표현을 줄이고 한계를 해당 주장 옆에 둠 |
| 부록 | 전체/큰 수량 6지표의 평균·표본 SD | 작은 NLL도 과학적 표기로 보존; 평균을 0으로 오독하지 않도록 함 |

`This manuscript draft ...`, `The final version must ...` 같은 문서 관리 문구를 본문에서 제거했다. 상투적인 강조어와 과도한 수식어 대신 어떤 조건에서 무엇이 개선되거나 악화됐는지 썼다. 논문 전체에서 현재 상태를 숨기지는 않되, 미완료 고지는 Results와 한계 설명에 모았다. AI 탐지 우회 또는 탐지 불가능성은 주장하지 않는다.

## Claim–Evidence Map

| 위치·주장 | 근거 | 허용되는 해석 / 남은 한계 |
|---|---|---|
| §3 CNN+GRU54 경로, 마스킹, 25,308 parameter, static prototype64 | [동결 Methods 근거](../../reports/titantpp_contribution_writing_20261008_v1/WRITING_NOTES.md), [기존 방법 초안](../drafts/titantpp_contribution_methods_20261008_v1.md), [과학 검토](../../reports/titantpp_contribution_writing_20261008_v1/REVIEW.md) | 구현 설명. CNN/GRU/직전 상태 각각의 성능 기여가 입증됐다는 뜻은 아님 |
| §3 fixed/general512 | 같은 근거의 retrieval 수식·입력 요약·초기화 | mean/weighted 0개, fixed/general512 512개 추가 query parameter. 별도 정규화·고정 혼합의 대조이며 새 종류의 검색 연산이라는 주장은 없음 |
| §4 baseline parity | [83개 bitwise 검증](../../reports/titantpp_claims_and_parity_20261008_v1/evidence/attempt3/completion_receipt.json) | 초기화·합성 입력·공통 Train 배치 첫 업데이트의 제한된 정합. 과거 A100 전체 학습 동일성으로 확장하지 않음 |
| 초록·§5·Table 2/3: MLP54 결과 | [Validation_summary.json](../../search_artifacts/titantpp_architecture_capacity_handoff_20261006_v1/analysis/Validation_summary.json), raw SHA `13cc3a31cdd1f8472ea93db229e290e93eadf70d982e9532c2f41d8304b3a5b7`; [검산 파일](../../reports/titantpp_contribution_writing_20261008_v1/results_update_evidence.json) | 54개 완료 조건, 18개 dataset/arm의 6지표. 모두 quantity-selected Validation. CNN+GRU 효과, 유의성, 외부 우월성 증명 아님 |
| §5: correction 18/18 개선, matched controls Taxi/RAF 통과 | 위 검산 파일 `contrasts`, `historical_gate` | paired 방향 일관성. Intermittent matched control은 각 2/3승이며 이 예외를 보존 |
| §5: Taxi TimeNLL 증가·RAF 큰 수량 MAE/NLL 악화 | 위 6지표와 paired 결과 | 같은 수량 선택 epoch의 손해. 별도 시간 최적 checkpoint와 비교하지 않음 |
| §5: 과거 비용 측정 | 검산 파일 `qualification_cost`와 그 안의 source SHA | 3회 synthetic step 중앙값·검증 작업 포함. 새 warmup5/measure20 비용 gate와 구분 |
| §5.3: Instacart 단일 fit | [관측 JSON](../../search_artifacts/titantpp_contribution_instacart_20261008_v1/hourly_monitor/observation_5090_1791437887192863000.json), SHA `8db0339244646b0e230e8099cb11de373952490475c60ce8349a00e0af4dd9f3` | 2026-10-08 14:38:07 KST 보존 관측. MLP8/seed52/5090/E55 best15만 완료 관측; 최종9조건 회수·비교는 아님 |
| §4 신규 63조건 설계 | [실행 계획](../../reports/titantpp_contribution_execution_20261008_v1/execution_plan.json), [README](../../reports/titantpp_contribution_execution_20261008_v1/README.md) | 설계와 평가 규칙을 설명. 신규 조건이 완료됐다고 표현하지 않음 |
| Related Work 및 References | [문헌 대조 기록](../../reports/titantpp_contribution_writing_20261008_v1/LITERATURE_COMPARISON.md) 및 아래 원 논문 | 방법의 관계를 설명. 인용 논문의 성능을 다른 split·head의 수치와 섞지 않음 |

새로 추가한 기본 방법 인용은 원 논문에서 서지를 확인했다: [Transformer](https://arxiv.org/abs/1706.03762), [GRU](https://arxiv.org/abs/1406.1078), [AdamW](https://arxiv.org/abs/1711.05101). 기존 6개 문헌의 확인 기록과 함께 참고문헌 9개를 구성했다. 본문은 문헌의 문장을 장문 인용하지 않고 해당 방법을 간단히 설명한다.

## 검토와 보존

- 방법 검토에서 발견한 두 문제를 수정했다: 역사적 MLP8에 CNN Q/K/V 경로가 없다는 점, 아주 작은 TimeNLL의 평균·SD 표기.
- 영어·주장 검토에서 발견한 두 모호성을 수정했다: 서로 학습된 모델이 같은 hidden 값을 공유한다는 인상, Instacart 단일 관측의 시각 누락.
- 최종 PDF의 12쪽을 렌더링했다. 검색 그림의 겹침과 역방향으로 보이던 화살표를 수정했고, 부록 제목·설명을 표 앞에 배치했다.
- 수정 전 `main.tex`, `main.pdf`, `README.md`와 SHA는 [보존 디렉토리](../../reports/titantpp_submission_rewrite_20261008_v1/before_revision/)에 있다. 기준 원고 디렉토리는 변경하지 않았다.
- 이 작업은 원고·문서만 수정했다. 학습, 추가 추론, Test 평가, 원격 서버 접속, 자동화 변경은 하지 않았다.

## 저자가 이어서 검토할 부분

대표 구조의 최종 결과가 들어오면 MLP 증거의 지면 비중을 조절하고 초록·기여·결론을 다시 맞춰야 한다. 데이터 출처와 재현성 정보, 문헌 범위, 독립 평가와 외부 비교는 최종 제출 전 보강 대상이다. 이 초안의 문장·수식·주장에 대한 최종 책임은 저자 검토 과정에서 확정한다. 제출 정책 관련 사항은 README의 사용 범위를 따른다.
