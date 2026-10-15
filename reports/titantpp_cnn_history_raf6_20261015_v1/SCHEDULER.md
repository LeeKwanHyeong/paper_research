# RAF6 관측·회수·Validation 비교

사용자는 2026-10-15 “오케이 그러면 맞춰서 RAF 를 5090에서 바로 시험할 수 있도록 하자”라고 요청했고, RAF도 Taxi와 같은 두 후보×3seed를 검증 뒤 학습할지 묻는 질문에 “검증 통과 후 6개 학습 시작”이라고 승인했다. 5090 native 검증 actualexit0를 통과했고 학습은 이미 시작했다. 설치·검증·학습·tmux를 다시 시작하지 않는다.

프로젝트 /Users/igwanhyeong/PycharmProjects/paper_research의 TEST_SESSION_PROTOCOL.md 0·1·9·12와 이 문서를 먼저 읽는다. 대상은 RAF 이력 적응 CNN3개와 용량 대조군3개뿐이다. 기존 CNN3은 고정 기준선이다. Test·held-out·혼합 결과·원고 main.tex/main.reviewed.tex를 열지 않는다. 기존 Taxi 실험·다른 자동화·학습 source·계약·입력·손실·head·선택·Runtime·driver·인증·기한·원본·claim/failure를 변경하지 않는다. 추가 학습·자동 retry/resume·kill·관측코드 자동수정·재봉인0이다. 삭제된 A100에는 접속하지 않는다.

프로젝트 root에서 회차당 정확히 한 번 실행한다:
`PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.advance_titantpp_history_raf6 --integrity-sha ce6dda0a7dc53373651fd7988b70978ebb7ab30a89c90e2d1b607dd0f8b9153c`

현재 bundle search_artifacts/titantpp_cnn_history_raf6_20261015_v1의 followup_integrity rawSHA는 ce6dda0a7dc53373651fd7988b70978ebb7ab30a89c90e2d1b607dd0f8b9153c, contract rawSHA는 7f67baa9a6e67383d47353d7b6e563f0c25eed27d6849f9847be0a0960cd2ca2, canonical은 6b0e435cfa1dc1280a3cb3c409df20730dc4f2d5513b7ddc839feb7f347639ca, deployment manifest rawSHA는 461563e89ce855dd7d694098b9587142240ec7f213907bbd158890824b4c0b12다. 로컬 코드 commit d759b893이다. helper는 고정332개 로컬 SHA를 먼저 검증한다. mismatch는 추가원격0·덮어쓰기0으로 차단하고 필요한 조치를 보고한다.

5090 고정 root는 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_history_raf6_20261015_v1_5090이며 GPU UUID는 GPU-c9adc246-ef66-7906-bc07-6e152fa9952f, cuDNN92000이다. 기존 ai_env/runtime_env Python을 유지한다. helper는 live host를 status SSH1로 관측하고 모든6scientificworker 실제exit0·terminalbinarySHA·dispatcher/pipeline 실제exit0·train_complete·owner/GPU부재를 확인한 뒤 같은 연결에서만 원본을 회수한다. supervisor wrapper 자신의 OSexit는 직접 관측하지 않는다. 별도SSH/observer/반복polling/helper재실행0이다. 승인된 읽기전용SSH·별도증적·종료후원본회수에 필요한 sandbox승격은 기존승인을 재질문하지 않고 사용한다.

실제 기준관측 2026-10-15 19:32:19 KST는 완료0·진행3·대기3이다. 이력 적응 seed42/52/62 모두 저장1/best1과 실제worker/GPU가 확인됐다. 이 시각을 새로운 관측시각으로 표시하지 않는다. 이력 적응3개가 모두 종료한 뒤 용량 대조군3개를 시작한다. 최대동시3이다.

그 회차 round JSON/summary JSON/latest 포인터와 rawSHA를 재검증한다. 평소 건강한 진행이나 변화가 없을 때는 조용히 유지한다. 한 후보의3seed 완료·다음 후보 시작, 전체완료, 실패 또는 사용자 조치가 필요할 때만 알린다. 조회실패를 학습실패로 바꾸지 않는다. ACTION_REQUIRED/실패/미완료retrieval이 있으면 자동 재시도하지 않는다. 해당 사유와 필요한 사용자 조치를 한 번 보고하고 같은 미해결 실패 중 원격 작업을 반복하지 않는다.

ETA가 필요하면 해당fit의 최근최대10 finite양수 완료epoch elapsed_seconds 중앙값만 쓴다. best미갱신·속도유지 조건의 min(300,max(40,best+40))와300epoch 시나리오를 구분한다. 저장 전·표본 부족은 그대로 알리고 다른fit/GPU의 속도를 차용하지 않는다. 전체큐 ETA는 미확정이다. 조건36h·host168h 기한을 연장하지 않는다.

종료 후 새 retrieval 경로에 archive/manifest를 exclusive 보존한다. 기존 CNN3을 재사용하여 여섯 Validation 지표의3seed 평균·표본SD(ddof1)와 seed별 Adaptive−CNN/Capacity−CNN/Adaptive−Capacity 차이를 모두 기록한다. 전체 raw수량 RMSE와 MAE를 중심으로 보고하고 큰수량q>200 RMSE/MAE·전체/큰수량 recorded positive-integer TimeNLL의 악화도 포함한다. 선택은 fullValidation rawRMSE 최초 strict finite 최소epoch이고 동률은 최초epoch다. 기존 CNN=5080/cuDNN91900, 새후보=5090/cuDNN92000을 반드시 명시한다. 주 비교 Adaptive−Capacity는 같은 Runtime이지만 기존 CNN 대비 차이는 GPU/Runtime 차이를 포함한다. Test 개선·보편적 일반화·원인 확정으로 확대하지 않는다.

완료는 bundle retrieval/final_receipt.json의 status=complete,new_fit_count=6,baseline_reference_count=3,seed_count=3,held_out_test_evaluated=false 및 모든로컬SHA, reports/titantpp_cnn_history_raf6_20261015_v1/THREE_SEED_VALIDATION.json/.md의SHA까지 재검증했을 때다. 최종3seed 결과 보고 후 이 RAF6 자동화만 삭제한다. 부분완료에서는 유지한다.
