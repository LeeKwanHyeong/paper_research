# Cap 공유형 이력 MLP18 실행 상태

**실제 학습 시작 — 진행 중**

2026-10-19 22:23:37 KST 실제 관측: 5080 RAF 공유형 seed42·52·62가 GPU에서 진행 중이다. 각각 저장2epoch, best2·2·1이며 완료0·진행3·대기15·미확정0이다. 이 관측 시각을 현재 시각으로 바꿔 사용하지 않는다.

실제 GPU·Python3.12.13/torch2.11.0+cu130/cuDNN91900/4threads, 두 후보 각각3개 동시 native 합성 검증을 통과했다. qualification 실제exit0 및 합성child6개 exit0를 확인했다. 기존 중지된 외부 비교군 프로세스와5090 학습은 변경하지 않았다.

시작 전 조회 오류 두 건은 execution_v1/execution_v2의 원문·계약·root에 보존했다. 실제 학습0개인 상태에서 수정했으며 과학 조건은 바꾸지 않았다.

**남은 순서**

1. RAF6→Taxi6→Intermittent6을 최대3병렬로 학습한다. 총18개다.
2. 선택 checkpoint18개를 별도 Validation에서 재현한다.
3. scientific worker36개·dispatcher 성공 종료, 소유PID/GPU부재와 binary/manifest SHA 확인 뒤 원본을 한 번 회수한다.
4. 기존 MLP16 저장9개와 데이터셋별3seed RMSE·MAE 평균±표본SD, 모든 seed 절대·상대차이와 악화를 비교한다.

유한 완료 절차 PID 16494의 실행을 로컬에서 확인했다. 원격 종료 대기→회수→비교이며 반복 개선 자동화가 아니다. 기존 log1p 입력·log-MSE·시간 가중치0·head·예산·선택을 유지한다. Test는 평가하지 않는다.
