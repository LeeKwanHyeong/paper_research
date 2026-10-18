# MLP16 시간 가중치0 대조군 — 대기 큐 등록

## 검증과 큐 등록 — 완료

2026-10-18 14:09:08 KST 실제 관측에서 두 서버의 CPU 대기 큐가 살아 있음을 확인했다. 신규 α0의9개는 모두 대기 중이며 실제 GPU 검사·학습·Validation은 아직 시작하지 않았다.

| 서버 | 배정 | 새 학습 수 | 현재 상태 |
|---|---|---:|---|
|5080|Taxi42·52·62, Intermittent42|4|기존 시간18 작업 종료 대기|
|5090|RAF42·52·62, Intermittent52·62|5|기존 시간18 작업 종료 대기|

로컬 통합58tests+13subtests와 별도 독립58+13 검사가 통과했다. 기존 source944개는 그대로이며 신규 실행3파일만 더한947개와 Train/Validation 입력6개를 별도 경로에 설치했다. 실제 설치 archive에서도 새모듈 import와 동일 MLP16 생성이 통과했다. 이는 새 실제GPU 검증의 완료를 뜻하지 않는다.

## 수량 전용 대조의 의미 — 현재 기준선

시간 입력과 전체 MLP16·메모리·head·수량log-MSE·초기화·학습예산·선택을 유지하고 목적함수만 기존 수량loss tensor로 구성한다. 시간 전용head는모든실제 optimizer step에서 grad None·parameter 불변·optimizer state 미생성을 검사한다. 시간NLL은 시간손실로 학습하지 않은head의 진단값으로 보존하며 수치우위가 있으면 그대로 보고한다.

기존 CNN·기준MLP16·선택checkpoint·이전실패를 변경하거나 재실행하지 않는다. 기존0.1/10 작업은 계속 진행하며 외부Cap27의dispatcher/worker 각4개도정지T 상태를 유지했다. 새 반복 자동화나Test를 추가하지 않았다.

## 부모 큐 종료 뒤 GPU 검사·학습 — 다음 작업

서버별 기존시간18의 전체train/Validation·dispatcher 실제exit0와원본·소유PID/GPU부재를 확인한뒤 새native검사로 진입한다. 통과하면 서버당최대3개씩 α0 학습9개와선택checkpoint Validation을실행한다. 기존절대host기한과fit36h를유지한다. 실패·미완료attempt에서자동retry/resume을하지않는다.

## 원본 회수·네 가중치 비교 — 다음 작업

종료·원본 회수·SHA 검증후0/0.1/1/10 총36개 모델의동일seed Validation RMSE·MAE·시간NLL을 비교한다. 데이터셋별3seed평균·표본SD와seed별차이·악화·예외를 모두보존한다. 시간손실의기여를보는대조이며backbone개선이나원인확정으로해석하지않는다. 세부실행순서는PIPELINE_GUIDE.md,현재원문과SHA는START_RECEIPT.json이다.
