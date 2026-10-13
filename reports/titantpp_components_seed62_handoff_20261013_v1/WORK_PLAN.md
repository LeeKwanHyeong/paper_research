# CNN·GRU36 완료판정 수정과 미시작 seed62 병렬 인계

**현재 학습과 원본 봉인을 확인한다 — 완료**

- 대상: paper_research v5 원본 bundle, 5080·5090 격리 실험 root. 2026-10-13 08:27:20 KST의 원문 증적 기준 고유36개 중 완료27·진행1·대기8이다.
- 5080의 Taxi·RAF24개와 5090 Intermittent42 대조3개는 실제 worker exit0·terminal manifest 파일 SHA로 확인됐다. 5080 dispatcher 실제 exit0·최종 train_complete·owner/GPU PID 부재도 확인됐다. pipeline wrapper 자체 OS exit는 직접 관측하지 않았다.
- 기존 v5 source131, input7, operation4, local40, monitor seal, 원본 회차와 원고를 보존한다. 원격 학습 종료와 로컬 immutable 원본 회수·최종 Validation 집계는 별개다.

**완료 증적을 우선하는 검증기와 종료 캐시를 추가한다 — 진행 중**

- 대상: 새 terminal verifier·hourly v2와 별도 monitor revision. terminal manifest가 검증된 완료 상태를 이전 checkpoint의 exposure 검사로 덮는 오류를 수정한다.
- 원본 launcher·observer·봉인을 제자리 수정하지 않는다. 실제 exit·claim·manifest·qualification·PID·원관측시각을 검증한 캐시만 수용한다. 5080 원본 root의 종료 캐시는 새 실험 root의 전역 유휴 판정으로 사용하지 않는다.
- 원문 snapshot 재현과 누락·변조·nonzero exit·잔여 프로세스·시각 불일치 부정 테스트, 반복 캐시 조회 SSH0가 완료 조건이다.

**미시작 seed62의 네 구조 실행권을 분리한다 — 진행 중**

- 사용자 승인: “ㅇㅇ 그럼 완료판정·종료캐시 오류부터 수정하고 검증하자. 5080은 종료 캐시로 처리하고, 이후 진행 중인 5090 학습을 5080에서 병렬로 진행할 수 있도록 하자.”
- 대상: 기존 5090 Intermittent42 FULL과 seed52 네 arm은 자연 진행하며, 아직 미시작인 Intermittent62 네 arm만 새 5080 격리 root에 배정한다. 실행 중인 checkpoint를 옮기거나 resume하지 않는다.
- 원래 5090의 첫 seed62 NONE claim을 완성된 행정 예약으로 원자 게시한다. exact source/op/contract, claim-before-Popen 경로, 네 ID의 미시작, parent PID/argv/GPU를 확인해야 한다. 기존 claim과 경쟁하면 중단하고 원문을 보존한다.
- 이는 seed62 대기열 생성 배제 증명이다. 기존 5090 전체 프로세스가 해제됐다고 표현하지 않는다. seed52 종료 뒤 예상 FileExistsError·운영 exit1·failure·claim을 보존하며 과학적 수치 실패나 부모12 완료로 바꾸지 않는다.
- 동일 seed의 NONE/CNN/GRU3개 → FULL1개, B128/min40/max300/patience40, 원래 source131·데이터·초기화·loss·optimizer·selector·fresh fit·retry/resume0를 유지한다. host 절대 deadline을 늘리지 않는다.

**새 5080 root의 계약·Runtime을 확인하고 병렬 학습을 시작한다 — 다음 작업**

- 앞선 생성 배제 증명과 새 격리 계약의 네 ID를 결합한 뒤에만 deploy·native qualification/admission·train을 각 한 번 수행한다. 실제 고정 GPU UUID·Runtime·유휴·패키지 잠금·actual exit0를 확인한다.
- 과거 qualification을 새 계약의 qualification으로 바꿔 적지 않는다. 원본 5080 terminal cache는 보존하고 새 root를 별도 관측 대상으로 추가한다.
- 전체36개 고유 ID를 부모27/진행·대기와 인계4로 중복 없이 집계한다. Intermittent62의 5080/cuDNN91900과 42·52의 5090/cuDNN92000 차이는 평균·SD에 명시하고 시간은 host별로 분리한다.

**관측과 논문 근거를 순서대로 확정한다 — 다음 작업**

- 기존 자동화 cnn-gru36-checkpoint-eta의 한 시간 cadence와 사용자 보고 의도를 유지하고, 새 verifier·원본 종료 캐시·인계 root 관측을 적용한다.
- 학습 자연종료/계획된 운영 종료 확인 → 로컬 원본 회수 계약 → archive/manifest·선택/최종 binary 로컬 SHA → 모든 seed·대조의 여섯 Validation 지표 → Results·Methods·주장 정합 순서로 진행한다. Test·새 평가·검색·기존 원고 수정은 이번 작업에 포함하지 않는다.
