# 루프5 RAF Q-only/K-only CNN 학습·평가·후속 판단

## 현재 기준선 — 완료
기존 CNN3 및 외부 모델은 저장 결과만 재사용한다. 모델·checkpoint·결과 변경, 재학습·재추론·재평가0이다. 루프1~4 및 QK 분기 진단의 완료 helper를 다시 실행하지 않는다. 루프4는 네 사용자 성능 기준 중 Test RMSE만 통과했다. 공동 학습 가중치의 Q/K 잔차 끔 진단은 Validation RMSE−0.175854/MAE+0.143336이며 Q/K 제거의 채택 근거가 아니다. 개별 행의 AE/SE 변화는 같은 부호이고 집계 순위 차이는 서로 다른 행의 오류 크기에서 발생한다. 원인 확정이나 새로운 독립 검증으로 설명하지 않는다.

## 같은 용량으로 Q와 K의 적용 위치를 비교한다 — 준비 중
사용자의 CNN backbone 반복 개선 사전승인과 기존 CNN 제외 지시를 유지한다. SCIENCE_SPEC.json SHA75388ec70cd8a35bbf3bd866c94b00e0ee8d510691d860ba02bbc0414697c15f 및 EXPERIMENT_PLAN.md를 따른다. 후보는 titantpp_cnn_query_only_v1과 titantpp_cnn_key_only_v1 각각 RAF42/52/62, 총6개다. 각각 기존 자유선형 관측3tap 지역 필터192계수 한 개만 남긴다. 반대쪽 지역 필터와 V 지역 필터는 물리적으로 제거하며 공통 Q/K/V 투영은 정상 학습한다. 기존CNN576/QKOnly384와의 비교는 용량과 공동최적화 차이를 포함한다. 두 후보의 주대조는 KeyOnly−QueryOnly다. Q 필터는 메모리 key와의 raw 점수도 바꾸고 K 필터는 event key에 작용하지만 softmax 분모를 통해 메모리 비중도 바뀔 수 있다. 메모리와 독립적인 효과로 주장하지 않는다.

입력/log1p·전처리·손실·head·메모리·예산·seed·선택을 유지한다. 새 gate/norm/RevIN/GRU/비선형/더 긴 관측 범위/입력 변경0이다. 초기 공통 tensor/RNG는 원CNN과 동일하게 만들되 기존 학습 checkpoint를 불러오지 않는 fresh fit이다. 새CNN 기준선 학습0. 부모 source143은 byte-identical로 보존하고 새 모델/runner2만 추가한다.

## 5090 검증 후 처음 한 번만 시작한다 — 다음 작업
로컬 합성·운영 검사와 구현-명세 검토를 완료하고 신규 파일을 커밋한 뒤 prepare와 seal의 실제 반환 integritySHA를 사용한다. 검증된 진입점 paper.scripts.prepare_titantpp_qk_location_raf6 --mode prepare, --mode seal --tests-receipt <실제 증적> --model-tests-receipt <실제 증적>, paper.scripts.deploy_titantpp_qk_location_raf6 --integrity-sha <실제 반환>만 사용한다. 명령 예시는 SHA를 추측하거나 재시작할 허가가 아니다.

새 root는 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_qk_location_raf6_20261016_v1_5090이다. 기존 ai_env Python3.12.13/torch2.11.0+cu130/cuDNN92000/4threads/GPU-c9adc246-ef66-7906-bc07-6e152fa9952f를 유지한다. 실제 fresh GPU/소유 유휴, Runtime·입력·source·승인·계약SHA, 후보 두 구조의 합성 검사와 3개 동시 admission 통과 뒤만 시작한다. Query42/52/62 셋이 모두 종료한 뒤 Key42/52/62를 시작하며 최대3병렬이다. 원 absolute host deadline1792679647.915156 및 permitSHAb3faca44824ce90972a62a740bb338b2cf5f54b0dfeb0d3fb291e06bed2a7f5a를 상속하고 연장0이다. 조건별36h 제한을 유지한다. 이미 존재하는 root/attempt/claim/failure에서 자동 재시작·재설치·retry/resume0이다.

## 종료와 원본을 검증한다 — 학습 시작 후
실제 시작·integrity·현재 관측은 SCHEDULER_ACTIVE.md와 START_RECEIPT.json 및 latest.json을 확인한다. 학습 retrieval/final_receipt가 complete가 아니고 ACTION_REQUIRED/미완료 회수가 없을 때 회차당 정확히 한 번 paper.scripts.advance_titantpp_qk_location_raf6 --integrity-sha <실제 봉인SHA>를 실행한다. live SSH1, 별도 SSH/observer/반복polling/동일회차 helper 재실행0이다. 모든 stdlib SHA를 원격 전에 검증하며 mismatch시 추가원격0·덮어쓰기0이다. round/summary/latest rawSHA와 포인터를 확인한다. 실제PID/argv/start_ticks/lease/GPU/qualification/permit를 확인한 fit만 진행으로 센다. 조회 실패를 학습 실패로 바꾸지 않는다.

6 scientific worker 실제exit0·dispatcher/pipeline 직접관측exit0·train_complete·terminal binary SHA·명시적 소유PID/GPU부재 뒤만 원본을 한 번 회수한다. wrapper 자체OSexit는 직접 관측하지 않았다고 명시한다. 검증된 완료cache는 SSH0·원관측시각 유지이며 현재GPU 전역유휴 증명이 아니다. archive/manifest/모든원본SHA·선택 epoch·Validation 순수재집계까지 검증한다. 완료는 status=complete,new_fit_count6,baseline_reference_count3,seed_count3,held_out_test_evaluated=false,all_training_terminal_verified=true 및 모든 종료증거/SHA까지다.

ETA가 필요할 때 각 fit의 최근 최대10 finite양수 완료epoch elapsed_seconds 중앙값만 사용한다. best 미갱신·속도유지 조건의 min(300,max(40,best+40))까지와300epoch를 구분한다. 저장 전·표본 부족·신원 미확정을 명시하고 다른fit 속도 차용0, 전체큐 ETA 미확정이다. 기한 초과 시 보장 불가로 표시한다.

## 선택 checkpoint의 Validation 재현과 Test를 비교한다 — 원본 회수 후
평가도구는 별도 새 파일로 구현·합성검사·독립검토·봉인하고 실제 검증된 진입점/SHA만 사용한다. 준비가 완료되기 전 실행 명령을 추측하지 않는다. 전체fullValidation raw수량RMSE 최초 strict finite 최소epoch(동률 첫epoch)의 binary/tensor/source/input/metadata를 고정한다. 전6 nativeValidation의 동일대상·여섯지표·실제exit0 모두 통과한 gate6 뒤에만 Test6을 실행한다. Test 재선택0, 실제Test로 도구 디버깅0, 기존CNN/외부/이전후보 재평가0이다. 새 평가 child의 고정timeout·소유group 정리만 허용한다.

각 후보의 Validation/Test 각각 RMSE/MAE3seed평균이 기존CNN 및 동일대상 외부지표별 최우수보다 모두 엄격히 낮아야 사용자 기준 통과다. 평균±표본SD(ddof1), seed별 후보−CNN과 KeyOnly−QueryOnly 절대·상대차이, 악화와 예외, Train고정q>200 큰수량RMSE/MAE 및 전체/큰수량 recorded positive-integer TimeNLL, 단순기준선(TimeNLL N/A)을 모두 보존한다. 운영표는 후보6+CNN3=9개이며 과거QKOnly3는 해석참고만이다. 새5090/cuDNN92000과 기존CNN/외부5080/cuDNN91900 차이를 명시한다.

둘 다 통과하면 통과후보 중 Validation평균RMSE→MAE→arm문자순으로 확장후보를 결정하되 Test를 통과자격에 썼음을 밝힌다. 통과한 동일 구조만 Taxi5090와 기존Taxi 학습·평가·회수 완료 및 fresh유휴 확인된5080 Intermittent에 각최대3병렬 적용한다. 미달이면 원본보존→오류분석→중복되지 않는 근거 있는 단일CNN 가설/대조→새격리 계약·구현·검증→RAF5090 다음 루프다. 근거 없는 무작위 탐색0이다. 서로 다른 구조의 데이터셋별 성공을 공통모델로 합치지 않는다. 반복Test 참고 탐색이며 독립일반화·원인확정·통계유의성을 주장하지 않는다.

## 기록과 알림
Notion3fabbe40-5613-80e5-91e9-e66ff02bb190에 시작·가설/구조·증적·결과·판정·실패·한계·다음 작업을 한국어로 누적하고 읽어 확인한다. 정상 진행·대기·변화없음은 조용히 유지하고 시작/주요전환/판정/완료/실패/사용자조치만 알린다. 첫RAF완료·미달로 자동화를 삭제하지 않으며 동일구조3데이터셋 검증·최종기록 완료 또는 사용자중단 때 종료한다. 원고/기존 source·결과·실패·claim/공용Runtime·driver·인증/다른자동화/삭제된A100은 변경·접속하지 않는다. 승인된 새경로 설치·회수·증적의 sandbox승격은 재질문하지 않는다.
