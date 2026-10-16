# Intermittent 시간 위치·분포 폭 분리 실행 전 독립 QA

## Review Result: PASS

- 검증시각: 2026-10-12T12:57:24.860170+09:00
- 상태: 로컬 사전 검증 완료. 실제 native qualification과 모델 평가·종료는 아직 미실행이며 이후 실제 원본으로 별도 판정한다.
- 변경 파일: 이 PRELAUNCH_QA.md만 작성. source/model/Runtime/main.tex/.agents 수정0, 원격 SSH·GPU 실행0.

### CRITICAL
- 없음.

### HIGH
- 없음. 준비 코드의 잘못된 dataset ID는 실제 실행 전 `intermittent_frozen_5000`으로 수정됐고9개 checkpoint 계약 검증을 통과했다.

### MEDIUM
- 없음. H1 QA의 단순 PASS 부분문자열 검사는 정확한 `## Review Result: PASS` 판정행1개 일치로 수정·검토했다. H1 분석 status/contract/실제18 exit0·pipeline OS exit0를 선행조건으로 확인한다.

### LOW
- 없음.

## 검토 범위 및 통과 조건

- [x] evaluator/analyzer/pipeline, prepare_bundle·prepare_transport·verify_prerequisite, remote health/setup/launch/wait_and_finalize/finalize, local retrieval verifier, launch.sh 검토.
- [x] evaluator 합성 CPU6개·pipeline 실제 작은 CPU 자식 프로세스4개·analyzer4개 검증 통과. 선택 실험 checkpoint를 테스트 fixture로 평가하지 않았다. Python AST13개 및 shell 문법 통과.
- [x] canonical `bc785c5ea669fe0f23d06604710dcc7ed9267b18996423298857279438d293c4`, input219개 전체 stdlib SHA 대조 및 evaluator.verify_contract() 통과.
- [x] source129 closure `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`, 원래 선택 checkpoint9개·Train/Validation reference18개 일치. frozen source·모델·초기화·손실·selector·원래 decode 불변.
- [x] H1 v2 native18 실제 종료·357개 archive/manifest·로컬 두 복사본·최종 분석·독립 FINAL_QA PASS가 먼저 확인됐다. H1 canonical `0c75bdd4d70adf8569ea2602c942e93f6b27b1206a4595711b2673997ee0758a`, QA rawSHA `05ea17243e3474fd3df9b5ada4214d2ae9db6f52f82966717fa2664fc142a1c5`. H2에 복사된 증적7개 SHA 대조 통과.
- [x] 사용자 승인 범위는9개 모델×Train/Validation18 model pass이다. 실행구성은 실제6개 joint OS worker이며 각 worker에 동일 seed의3모델을 함께 올린다. joint 동시성1, resident 모델3, shared PID를18개 개별 OS worker로 세지 않는다.
- [x] full/보정 제거/current-only의 정확히 같은 target/history에 대해 기존 원래6지표·state·target/truth SHA를 먼저 재현한다. μ·σ pairing은 RAM에서만 수행하며 per-event raw prediction arrays는 저장하지 않는다.
- [x] 원래 positive integer CDF score, scale3, native σ=0.001+softplus, topcodeNone·round/clamp를 유지한다. floor/τ/손실/optimizer 조정·fit·Test read/evaluation0.
- [x] 2대조×3seed×2split×4셀=48 paired score cell이며 full/full 중복을 제외한 새 cross24개. scale=fc−ff, location=cf−ff, interaction=cc−fc−cf+ff를 각각 보존한다. 음의 NLL 차이가 개선이며 상호작용을 누락하지 않는다.
- [x] 실제 native3 qualification 후 serial joint6을 실행하고 실제 exit/PID/PPID/argv·shared model receipts·pipeline OS exit0를 확인한 뒤만 exclusive 원본 회수. 이전 claim/failure와 원본 덮어쓰기·자동 retry0.
- [x] setup/launch는 H1 원격 terminal·원본archive/manifest SHA와 로컬검증/QA 증적을 확인한다. 승인된5080 GPU `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`, 기존 ai_env·cuDNN91900을 사용한다. 건강 receipt에서 reboot없음·GPUidle·패키지 프로세스없음·new root없음 확인.

## 검토한 실행 코드 SHA

- evaluator: `e749eedfa19703476bbd9caab6ee31253691f450b1a78f7b9eee7588caf3ec1e`
- pipeline: `c489b22b98ebadbc9382371d00279b4341b4a8ec6ede08ea8f713e13eae87e5b`
- analyzer: `93fec8c06591c3ac0553c434497e62e2cdf3ecbe9f99bd28032054f73f1ef3cf`

## 해석 및 남은 작업

**같은 target의 μ·σ 점수 교차를 관측한다 — 다음 작업**
- 이 PASS는 로컬 검토 결과다. operation 봉인 후 native3와 실제6 worker/18 model pass가 모두 끝나고 원본이 회수돼야 실제 완료로 판정한다.
- 교차 점수는 사후 계산이다. 실제 새 모델 성과, 보정 fit, CNN/GRU 인과 효과, 통계 유의성이나 새 독립 평가 증명으로 해석하지 않는다. Train·Validation 선택 모델 관측과 기존 Test 노출 사실을 구분한다.
