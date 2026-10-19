# 손실 비교 직후 입력을 고정해 백본18을 학습한다

**기존 실행 순서를 보류한다 — 완료**

- 사용자2026-10-19 최신 승인에 따라 원시간 입력27개를 보류했다. 원시간 대기PID78887과 원시간 완료를 기다리던 구백본 대기PID5949의 시작시각·모듈·인자·자식부재를 확인한 뒤 SIGSTOP으로 멈췄다. 명시적인T상태와 원격bundle미생성을 DEFERRED_FOLLOWUPS.json에 보존했다.
- 원격 서버에 신호를 보내지 않았다. 현재 손실 학습과 기존 완료 처리PID69408, 기존CNN·checkpoint·원본·실패 기록은 변경하지 않았다. 정지된 외부Cap27도 그대로 둔다.
- 이전 준비·코드·기록은 당시 원본 그대로 보존한다. 이번새경로는 paper_research 저장소의 codex/contextual-retrieval-pilot-20261007 브랜치에서 독립 커밋하며 Push나 공용Runtime 배포는 하지 않는다.

**현재 손실 비교 완료를 검증한다 — 외부 작업 대기**

- 기존 literal실수량MSE9/MAE9 학습과 선택Validation18, 서버별원본회수, 저장log손실9개를 포함한27개 비교가 모두 끝나기를 기다린다. 기존 완료처리·collector·학습을 다시 실행하지 않는다.
- 최종receipt의18fit/18replay/9reference/27comparison/3seed/36scientificchildexit0/2dispatcherexit0/Testfalse, archive·manifest·모든원본·선택binary와 최초strictfiniteRMSE최소epoch·같은대상·여섯지표·시간head미학습증적을 검증한다.
- 기존 raw-time 관리도구의 로컬부모원본 검증 함수만 재사용한다. raw-time 학습·배포·관측·회수는 실행하지 않으며, raw-time 결과가 없어도 이번준비를 막지 않는다.

**입력은 유지하고 공통 손실만 고른다 — 규칙 고정 완료 / 실제 선택 대기**

- 시간입력과 수량입력은 모두 기존log1p로 고정하고 시간가중치는0이다. 원시간 입력으로 바꾸거나 입력조건을 탐색하지 않는다.
- log-MSE·실수량MSE·실수량MAE 세 손실의 데이터별3seed평균RMSE를 해당데이터의 고정log-MSE평균으로 나눈다. 세데이터의 비율을 같은가중치로 평균해 가장낮은 손실을 모든후보/데이터/seed에 공통 적용한다.
- 정확동률이면 같은방식의MAE비율, 고정손실순서 log_mse→raw_mse→raw_mae로 정한다. 세점수와27endpoint를 모두 보존하며 데이터별원단위오류를 평균하지 않는다.
- 조건선택은 backbone성공이 아니며 같은Validation을 설계·선택·비교에 사용한 탐색적검증이다. 입력효과·원인·독립일반화·통계유의성을 주장하지 않는다.

**동일 입력·손실로 두 백본을 비교한다 — 구현·합성 검증 완료 / 실제 학습 대기**

- 과학worker와모델은 이전에 검증한 동일바이트다. 현재관측후보 LMM(h2+P(e0))와 같은명목용량이력대조 LMM(h2+P(h2)), P는bias없는64→8→64GELU·추가1024가중치·출력0초기화다. 공통MLP16·head·메모리·초기공통tensor/RNG·padding/관측mask·예산을 유지한다.
- Cap Taxi·RAF·Intermittent×seed42/52/62×두후보=18개이며 이전MLP16은 선택손실의저장9개만 재사용한다. warmstart·기존CNN변경/재학습/재추론·Test는 없다.

|서버|배정|학습수|동시상한|
|---|---|---:|---:|
|5080|Taxi42/52/62·Intermittent42×두후보|8|3|
|5090|RAF42/52/62·Intermittent52/62×두후보|10|3|

- 기존960source와 모든부모봉인을 그대로 상속하고 새manager/runtime/selector3개만 추가한963source를 새묶음으로 준비한다. 후기경로worker와모델을 바꾸지 않는다.
- 현재손실의 부모종결을 검증한뒤 새root에서 GPU실제유휴·소유·Runtime·입력·두후보3동시합성검증을 통과해야 첫실데이터학습을 시작한다. 실제준비반환/봉인SHA만 사용한다.
- 기존Python3.12.13/torch2.11.0+cu130/4threads,5080cuDNN91900·5090cuDNN92000과GPU UUID를 유지한다. batch128·AdamW.001/WD.01/clip1·최대300/min40/patience40과fit36h 및host절대기한5080:1792888885.911408/5090:1792888885.815062를 연장하지 않는다.

**선택 Validation과 원본을 검증한다 — 학습 후 다음 작업**

- fullValidation 실수량RMSE의최초strictfinite최소epoch(동률첫epoch)를 고정하고 MAE도같은checkpoint에서 평가한다. 모든18학습과선택Validation18의actualexit0,dispatcher2의직접관측exit0,terminalbinarySHA와명시적소유PID/GPU부재 뒤 host별한번 회수한다.
- archive/manifest/선택/source/input/tensor/metadata·모든원본SHA를 검증한다. wrapper자신OSexit는 직접관측하지않았음을 유지한다. 실패·timeout·미완료attempt/claim·신원미확정이면 원문을남기고 자동retry/resume/재시작/claim삭제를 하지않는다.

**MLP16 대비 성능과 경로의 효과를 구분한다 — 회수 후 다음 작업**

- 저장MLP16 9개와후보18개,총27개에서 데이터별RMSE/MAE3seed평균±표본SD(ddof1), 모든seed 관측−MLP16/이력−MLP16/관측−이력의절대·상대차이와악화를 기록한다.
- 전체/상한도달 수량RMSE/MAE·recordedpositiveintegerTimeNLL과 상한초과빈집단N/A를 보존한다. 시간head는미학습이므로시간지표는진단값이다. 메모리검색·gradient·함수공간차이도포함되므로 관측정보소실원인으로 확정하지않는다.

**기존 손실 완료 뒤 한 번만 이어지도록 연결한다 — 다음 작업**

- 국소합성·계약·독립검토·커밋·준비봉인후 새유한후속프로세스를 한 번 연결한다. 이전보류프로세스는 재개하지 않으며 반복스케줄러를 만들지 않는다.
- 완료는 새retrieval/final_receipt complete,18fit/18replay/9reference/27comparison/3seed/36childexit0/2dispatcherexit0/Testfalse와모든로컬SHA까지다. 대기/준비완료를실제GPU학습시작이나결과완료로 보고하지 않는다.
