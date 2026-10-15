# 루프4 RAF Q/K 전용 CNN — 승인된 운영 정책

## 범위
사용자의2026-10-15 CNN backbone 반복 개선 사전승인과 “기존 CNN기준선은 건들지마”, “그거 제외하고 하던거 마저해”를 적용한다. 이 문서는 새 RAF 후보1개×seed42/52/62,5090최대3병렬의 계약·검증·학습·회수·고정Validation/Test 비교 범위를 정한다. 기존CNN은 저장된 결과만 재사용하고 새 학습·재추론·재평가하지 않는다.

## 과학 조건
SCIENCE_SPEC.json의 Q/K-only 구조만 적용한다. Q/K 기존 관측3tap CNN과 공통QKV투영은 정상 학습하고 V 추가지역필터는 실제로 제거한다. 공통초기tensor와RNG를 유지한 fresh fit, 기존checkpoint warmstart0이다. localCNN576→384의용량감소도 포함한 구조축소실험이다. 기존 입력/log1p·손실·head·메모리·예산·seed·선택·padding/withheld경계를 유지한다. RevIN·GRU·head·gate·임의감쇠탐색0이다.

## 진입과 보존
TEST_SESSION_PROTOCOL.md0·1·9·12, 이문서,SCIENCE_SPEC,EXPERIMENT_PLAN,LOOP_STATE와실제receipt를 먼저 읽는다. 명령예시가아니라 실제LOCAL_READY·followup_integrity에봉인된 진입점과SHA만 사용한다. source141은 바이트동일상속하고 신규모델/runner2개만추가한다. 외부등록부SHA57e88f4d5d338c6da7f54fe38d2b9492b06476fa93161331481de3c4ee1a6b49,기존CNN3및이전모든원본·실패를보존한다.

새학습root는 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_qk_only_raf3_20261016_v1_5090 이다. 실제GPU-c9adc246-ef66-7906-bc07-6e152fa9952f,기존ai_env Python3.12.13/torch2.11.0+cu130/cuDNN92000/4threads를유지한다. 상속절대hostdeadline1792679647.915156과permitSHAb3faca44824ce90972a62a740bb338b2cf5f54b0dfeb0d3fb291e06bed2a7f5a,각조건36h/host168h한도는연장하지않는다.

모든로컬source/input/계약/승인/운영SHA와독립합성검증→freshGPU유휴·소유확인→실제nativeGPU/Runtime/input및후보3동시합성검증→exclusivepermit뒤딱한번학습한다. 이미존재하는root·intent·claim은재설치/재시작하지않는다. 자동retry/resume/kill/claim삭제0이다. 조회실패를학습실패로바꾸지않는다. 허용된별도경로작업의sandbox승격은기존승인을재질문하지않는다.

## 관측·회수
livehost는회차당1회,완료cache는SSH0·원시각유지다. 별도observer/반복polling/같은회차helper재실행0이다. PID/argv/start_ticks/lease/GPU와worker·dispatcher·pipeline실제exit를확인한다. 전체3종료·terminalbinarySHA·명시적소유PID/GPU부재뒤만host1회회수하고archive/manifest원본을exclusive보존한다. wrapper자신의OSexit는직접관측여부를구분한다. stdout/stderr원문·byte/SHA·transportexit/timeout을보존한다. 상태미확정에서완료/유효ETA로바꾸지않는다. ETA는각fit최근최대10finite양수완료epoch중앙값만쓰고best미갱신min(300,max(40,best+40))와300epoch시나리오를구분한다. 다른fit속도차용0,전체큐ETA미확정이다.

## 평가·판정
학습원본회수와모든SHA확인후선택checkpoint3개(전체fullValidation rawRMSE최초strictfinite최소,동률첫epoch)를고정한다. 별도신규평가도구합성/독립검증과봉인후nativeValidation3의동일target·여섯지표·actualexit0전부통과뒤만Test3을실행한다. 기존CNN/외부/과거후보재평가0,Test재선택0이다. Test로도구를디버깅하지않는다.

Validation/Test각각RMSE·MAE3seed평균이기존CNN및같은대상외부지표별최우수보다모두엄격히낮아야통과다. 평균±표본SD(ddof1),seed별절대/상대차이,악화,큰수량q>200/시간지표와단순기준선을모두기록한다. 기준CNN5080cuDNN91900대신규5090cuDNN92000차이와반복Test참고탐색한계를명시한다. 새독립일반화/원인확정/통계유의성/독창적원리로주장하지않는다.

RAF통과한동일구조만Taxi5090와기존Taxi작업완료및fresh유휴확인5080의Intermittent에최대3병렬로확장한다. 미달이면증적보존→오류분석→다음단일CNN가설/대조→새계약·구현·검증으로이어간다. 근거없는무작위탐색0,서로다른구조의성공을공통모델로합치지않는다.

## 기록과 종료
Notion3fabbe40-5613-80e5-91e9-e66ff02bb190에시작·가설/구조·증적·결과·판정·실패·한계·다음작업을한국어로누적하고읽어확인한다. 정상진행/대기/변화없음은조용히유지하고시작/주요전환/판정/완료/실패/사용자조치만알린다. 자동화cnn-backbone-raf는동일구조3데이터셋검증과최종기록완료또는사용자중단때만종료한다.

원고main.tex/main.reviewed.tex,기존source/결과/실패/claim,공용Runtime/driver/auth,다른자동화,삭제A100을변경·접속하지않는다. 루프1·2·3학습/평가/행추출/분기진단완료helper는재실행하지않는다. 과거종료cache는현재GPU유휴증명이아니다.
