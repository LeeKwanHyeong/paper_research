# 고정 선택 모델의 행별 진단 추론

## Material Passport

Origin Skill: academic-research-suite / experiment-agent run→validate. 실제 원격 추론 진행 중이며 최종 Verification Status는 아직 미확정이다.

**선택 모델과 입력을 유지한다 — 현재 기준선**

- source131 CNN·CNN+GRU 18개 fit과 과거 공통 head 외부6개×3데이터셋×3seed 54개 fit이다. checkpoint 재선택·새 학습·통계 재추정·science 변경은 없다.
- 기존 외부 Test54의 개별 예측은 다시 추론하지 않고 읽기 재사용한다. 캠페인/source 차이, Intermittent42/52와62의 GPU/cuDNN 차이는 유지한다.
- Train은 고정 chronological loader의 최소1개 이력이 있는 next-event 대상이다. Taxi38,393/RAF25,779/Intermittent393,824이며 전체Train행을 대상으로 삼지 않는다.

**별도 경로에 진단 예측을 저장한다 — 진행 중**

- 신규 추론162개: 구성요소18×Train/Validation/Test54, 외부54×Train/Validation108.
- 5080은 기존cuDNN91900에서68모델150추론,5090은Intermittent42/52 CNN/FULL4모델12추론을 기존cuDNN92000에서 실행한다. GPU별 최대2개 worker가 함께 실행되며 이 시간을 단독 모델 효율 측정으로 쓰지 않는다.
- 각 host의 Validation 전체 종료·기존6지표/5수량구간 재현·실제exit0 뒤 Test gate를 기록한다. 같은host의 Train과 구성요소Test는 고정 계약 안에서 진행한다.
- 모든 worker는 eval/inference_mode, gradients0, before/after tensor SHA, source/input/checkpoint SHA, Runtime 전후 신원을 검증한다. target 가림·batch 일치도 검증한다.
- target_id·개체·seq·날짜·수량·gap·history last/mean/length 및 개별 raw/log 오류를 exclusive parquet로 저장한다. 과거 외부 Test의 truth에는 추가3필드가 있으므로 공통10필드로 대조한다.
- 실패/timeout은 원문·SHA·실제transport/child exit를 보존한다. 자동 retry·claim삭제·기존 결과 덮어쓰기·Runtime 변경은 없다.

**내부·외부 비교를 별도로 분석한다 — 다음 작업**

- 선택 모델 Train raw 적합과 Validation/Test 일반화를 구분하고, 고정 수량구간·history·gap·날짜·개체별 동일 행 오류를 비교한다.
- 개체 구성 변화·공통 개체의 비중 변화·개체 내부 오류 차이를 산술 분해한다. 인과적 기전이나 과적합 입증으로 확대하지 않는다.
- 모델 활성 용량·source/runtime 차이를 통제하는 새 학습은 이번 범위가 아니다. 사후 분석과 기존 Test 접근 이력을 명시한다.

**완료 증적과 결과를 검증한다 — 다음 작업**

- 두 pipeline의 실제 supervised OS exit0,162worker 실제exit0,예측행 누락/중복/모집단/SHA,기존 지표 재현 및 원본 봉인 보존을 확인한 뒤 최종 결과를 보고한다.
- 기존 원고(main.reviewed.tex/draft.tex 포함)와 다른 자동화는 변경하지 않는다.

신규 추론 도구는 paper_research 저장소 codex/contextual-retrieval-pilot-20261007 브랜치의709aa3fd 커밋이다. 원고의 기존 staged변경은 이 커밋에 포함하지 않았다.
