# 루프6 마지막 FFN 이력 결합 RAF6 — 실제 학습 시작 후

**현재 기준선과 승인 범위를 확인한다 — 완료**
- 사용자의 CNN backbone 반복 개선 사전승인, “기존 CNN기준선은 건들지마”, 최신 “1순위 진행하자”를 적용한다. 기존 CNN 모델·checkpoint·결과는 변경·재학습·재추론·재평가하지 않고 고정 저장 결과만 재사용한다.
- 프로젝트 `/Users/igwanhyeong/PycharmProjects/paper_research`의 TEST_SESSION_PROTOCOL.md 0·1·9·12, 이 디렉터리의 SCIENCE_SPEC.json·ACCEPTANCE.json·DESIGN_MANIFEST.json·START_RECEIPT.json·이 문서와 실제 latest/receipt를 먼저 읽는다. 예전 SCHEDULER_ACTIVE.md/LOOP_STATE.json의 구현 전 상태는 당시 기록이며 현재 단계로 쓰지 않는다. 완료된 루프1~5 및 모든 진단 helper를 재실행하지 않는다.
- 공통 입력/log1p·전처리·손실·head·메모리·학습예산·seed·선택 규칙을 유지한다. RevIN/GRU/head 변경으로 전환하지 않는다. 원고 main.tex/main.reviewed.tex/draft.tex, 기존 source·결과·실패·claim, 공용 Runtime·driver·auth, 다른 자동화와 삭제된 A100을 변경·접속하지 않는다.

**마지막 FFN 안에서 과거 내용을 결합하는 효과를 검증한다 — 학습 진행 중**
- 후보는 `titantpp_cnn_ffn_current_v1` 대조군과 `titantpp_cnn_ffn_temporal_v1`, 각각 RAF seed42/52/62, 총6개다. 마지막 attention 뒤 FFN의 W1 확장128차원에서 비선형 관측 잔차를 추가한다. 기존 Q/K/V 지역 필터576과 W1/W2는 그대로 존재하고 정상 학습한다.
- Current는 현재 prefix를 세 분기에서 사용하고 Temporal은 현재·직전·그이전 관측 prefix를 사용한다. 동일 가용 mask와 고정 분모3, 신규 a/b/c 3×128씩 총1152개, a=1·b=[-1,0,1]·c=0, 추가 RNG draw0이다. padding 건너뛰기와 withheld에서 새 국소 gather 초기화, absent 분기 bias 차단, nonfinite 거부를 유지한다. 기존 attention의 이력 규칙은 바꾸지 않는다.
- Current도 attention 이력과 가용성 정보를 포함한다. 같은 파라미터 수는 같은 함수공간·최적화·인과 분리가 아니다. 공통 가중치와 메모리 검색은 달라질 수 있다. Current만 통과하면 temporal 기여로 표현하지 않는다.
- 학습 구현 commit `7a0ff81a6088f668db6f70d78a77ed1de7666362`, branch `codex/contextual-retrieval-pilot-20261007`이다. source145를 동일 바이트로 상속하고 신규 모델/runner2개만 더한 source147 closure `b1821823bfbf842ef76bd6074e5ed5fa10ab39a40d9a03b64d7fe099a121dbf9`다. 현재 작업트리 부모 파일의 변경을 대신 상속하지 않는다.
- bundle `search_artifacts/titantpp_backbone_post_attention_ffn_raf6_20261016_v1`, contract rawSHA `1e5e5ea457da67337602e9db3119c17d168cdf1da75b29b1cccfc02a7b05bc23`, canonical `8a974e61beceb356db8fa6b3d22a2241ccb42dcd76e947245cd1e965324db5f9`, followup_integrity rawSHA `aefdd9d5f520e080edacc26a7823f65a5f234b3bec55a7ba02615d417c71c3c0`와1155개 봉인 파일을 유지한다. 모델48/운영109 검사와 독립 검토 통과, native 두 구조 및 실제3개 동시 합성 검증 통과다.
- 원격 root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_post_attention_ffn_raf6_20261016_v1_5090`이다. 기존 Python3.12.13/torch2.11.0+cu130/cuDNN92000/4threads/GPU-c9adc246-ef66-7906-bc07-6e152fa9952f를 유지한다. Current3 전부 종료 뒤 Temporal3, 최대3병렬이다. 이미 설치·qualification·학습을 시작했으므로 prepare/seal/deploy/start를 다시 실행하지 않는다.
- 실제 관측2026-10-16 07:51:48.499060 KST: Current42/52/62 진행3, Temporal42/52/62 대기3, 완료0·미확정0, 모두 저장전·selected미생성·timing0이다. GPU PID3057360/3057362/3057364와 worker/lease/pipeline/dispatcher/qualification/permit 신원을 확인했다. 이 시각을 현재 관측이나 유휴 증명으로 바꾸지 않는다.
- START_RECEIPT rawSHA `bba9370e7902d879e7e97d38734a4d4180b2a2e438a3824eaa2e85dc789d945c`, round `31119a239aabdfb1f1452e6f8a4c5fef7816e9d7168d8fc946e3e6337fd140a5`, summary `7dac3669de22bf3b4fd19fab2a3d06d86a6070ffd9cabc938318f347f180c6bb`, 최초latest `78c6c431c5f8d3b2748e650465b4a0d3f760ec6888e9d5213e1265b1e03278f0`다. 배포 transport exit0/stderr352bytes libtinfo 경고와 관측 원문은 보존한다. wrapper 자체 OS exit는 직접 관측하지 않았다.

**한 회차에 한 번 관측하고 완전 종료 뒤 원본을 회수한다 — 다음 작업**
- 학습 retrieval/final_receipt가 complete가 아니고 ACTION_REQUIRED/미완료 회수가 없으면 프로젝트root에서 회차당 정확히 한 번 실행한다:
`PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.advance_titantpp_post_attention_ffn_raf6 --integrity-sha aefdd9d5f520e080edacc26a7823f65a5f234b3bec55a7ba02615d417c71c3c0`
- helper는 모든 로컬 SHA를 원격 전에 검증한다. livehost SSH1, 별도 SSH/observer/반복 polling/동일 회차 helper재실행0이다. latest의 round/summary 경로와 stdlib rawSHA를 검증한다. 실제 PID/argv/start_ticks/lease/GPU가 확인된 fit만 진행으로 센다. 조회 실패를 학습 실패로 바꾸거나 미확정을 완료·유효 ETA로 바꾸지 않는다.
- 모든6 scientific worker·dispatcher·pipeline actualexit0, train_complete, terminal binary SHA, 소유 PID/GPU 명시적 부재 뒤만 원본을 한 번 회수한다. archive/manifest/모든 원본·선택·집계 SHA까지 확인한다. 완료 조건은 status=complete,new_fit_count6,baseline_reference_count3,seed_count3,held_out_test_evaluated=false,all_training_terminal_verified=true다. 완료 cache는 SSH0·원관측시각 유지이며 현재 GPU 전역 유휴 증명이 아니다.
- 조건36h와 상속 absolute host deadline1792679647.915156/원permitSHA b3faca44824ce90972a62a740bb338b2cf5f54b0dfeb0d3fb291e06bed2a7f5a는 연장하지 않는다. ETA는 각 fit 최근최대10 finite양수 완료epoch 시간 중앙값만 쓴다. best미갱신·속도유지 조건의 min(300,max(40,best+40))와300epoch 시나리오를 구분한다. 저장전·표본부족은 명시하고 전체큐 ETA 미확정, 다른fit 속도 차용0, 기한 초과 시 보장불가다.
- 실패/timeout/기존attempt·claim/신원미확정은 stdout/stderr/byte/SHA·실제 transport/worker/pipeline exit를 보존한다. 미확인 retry/resume/kill/claim삭제/원본덮어쓰기0이다. 승인된 안전한 국소 도구 수정이 필요하면 별도 버전에서 검증하고 기존 봉인을 덮어쓰지 않는다. 같은 미해결 오류의 원격 호출을 반복하지 않는다.

**고정 checkpoint의 Validation6 재현 뒤 Test6을 수행한다 — 평가 도구 준비 완료, 실제 평가는 대기**
- 평가 코드 commit `a2299ead096a17f351178c902aee337bd1f3d266`, EVALUATION_READY.json rawSHA `12573d1cf298f5a9d7cd7925eb53622c523138b5f4bac3055b59e66d68be9f67`와 내부36파일SHA를 먼저 검증한다. 작성자53검사·무작위100회, 독립53검사·무작위100회 actualexit0, 열린발견0이다. 독립QA rawSHA `275b40cfa8847e056e2d5327fc25cfa2483a36b5981e9b6f6f65df05f2987fbe`. 실제 prepare/nativeValidation/Test는 아직 미실행이다.
- 학습 회수 complete 및 종료/선택binary/tensor/source/input/metadata/archive/manifest 모든SHA가 검증된 뒤에만 로컬 prepare를 한 번 실행한다:
`PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.prepare_titantpp_post_attention_ffn_raf6_test --training-integrity-sha aefdd9d5f520e080edacc26a7823f65a5f234b3bec55a7ba02615d417c71c3c0 --registry-sha 57e88f4d5d338c6da7f54fe38d2b9492b06476fa93161331481de3c4ee1a6b49`
- 이미 유효 evaluation_v1/bootstrap_integrity.json이면 prepare를 생략한다. 실제 반환 bootstrapSHA를 보존하고 이어 한 번 실행한다:
`PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.advance_titantpp_post_attention_ffn_raf6_test --mode advance --integrity-sha <실제 prepare 반환 bootstrapSHA>`
- 미완료 bundle/attempt/claim/failure면 자동 반복하지 않고 원문을 보존·진단한다. 승인된 새 격리 평가경로 설치/Validation6/Test6/회수/로컬기록의 sandbox승격은 재질문하지 않는다. 실제 평가 root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_post_attention_ffn_raf6_20261016_v1_test_5090`이다.
- 전체 fullValidation rawRMSE 최초 strictfinite 최소epoch(동률첫epoch)를 고정한다. 전6 nativeValidation의 동일대상·여섯지표·실제 workerexit0가 모두 통과한 gate 뒤만 Test6을 실행한다. 평가 pipeline은 순차6이며 학습최대3병렬과 구분한다. Test 재선택0, 실제Test로 도구디버깅0, 기존CNN3/외부 재평가0이다. 평가 child만 고정timeout/소유group정리를 적용한다.
- 평가 완료는 evaluation_v1/final_receipt.status=complete,candidate_test_count6,baseline_reference_count3,comparison_model_count9,seed_count3,validation_replay_count6,actual_native_worker_exit0_count12,actual_pipeline_exit0_count2,external_comparison_complete=true,held_out_test_evaluated=true,historical_cnn_retrained=false,historical_cnn_reevaluated=false 및 모든 원본/종료/선택/결과 SHA까지다. reports의 evaluation_v1/THREE_SEED_VALIDATION·THREE_SEED_TEST·EXTERNAL_COMPARISON.json/.md를 재검산한다. 완료 cache도 빈 inventory·누락 원본을 거부하고 순수집계를 다시 확인한다. 평가 후 전역 GPU유휴/모든PID부재와 wrapper 자체OSexit를 별도 확인한 것으로 표현하지 않는다.

**성능과 방법론의 근거를 나누어 판단하고 기록한다 — 평가 후**
- 각 후보는 Validation/Test 각각 RMSE·MAE3seed평균이 기존CNN과 동일대상 외부지표별 최우수보다 모두 엄격히 낮아야 사용자 기준 통과다. 기존 외부등록부SHA는 위57e88...를 유지한다. 평균±표본SD(ddof1), 모든 seed의 Current−CNN/Temporal−CNN/Temporal−Current 절대·상대차이, 악화·예외, Train고정q>200 큰수량RMSE/MAE, 전체/큰수량 recorded positive-integer TimeNLL, 단순 기준선 TimeNLL N/A를 보존한다.
- Temporal−Current의 Validation RMSE/MAE가 모두 낮은지와 Test의 두 지표를 별도로 기록한다. 사용자 성능 통과와 이력 결합의 추가 근거를 혼동하지 않는다. 통과후보 선택은 Validation RMSE→MAE→정확arm 문자순이며 Test가 통과 자격에 쓰였음을 명시한다.
- 기존CNN/외부5080 cuDNN91900 대 신규5090 cuDNN92000 차이, 용량 증가·공동최적화·메모리 영향, 반복 Test 참고 탐색이라는 한계를 기록한다. 독립 일반화·원인 확정·통계 유의성을 주장하지 않는다.
- RAF 통과한 동일 구조만 Taxi5090와 기존Taxi 학습·평가·회수 완료+fresh유휴 확인된5080 Intermittent에 각최대3병렬 확장한다. 미달이면 결과보존→오류분석→중복되지 않는 근거 있는 단일 CNN 가설/대조→새격리 계약·구현·검증→5090RAF 다음 루프다. 근거 없는 무작위 탐색이나 다른 구조의 데이터셋별 성공을 공통 모델로 합치는 행위는 하지 않는다.
- Notion page3fabbe40-5613-80e5-91e9-e66ff02bb190에 각루프 시작·가설/구조·증적·결과·판정·실패·한계·다음 작업을 한국어로 누적하고 읽어 확인한다. 정상 진행·대기·변화없음은 조용히 유지하고 시작/주요전환/판정/완료/실패/필요 사용자조치만 알린다. 첫RAF 완료·미달로 자동화를 삭제하지 않는다. 동일구조3데이터셋 검증·최종기록 완료 또는 사용자중단 때 종료한다.
