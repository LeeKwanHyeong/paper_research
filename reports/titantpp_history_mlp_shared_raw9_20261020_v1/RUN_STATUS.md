# 원본 공유형 MLP9 실행 상태

**원본 실험 준비·검증 — 완료**

- 공유형 MLP 하나 × RAF·Taxi·Intermittent × seed42·52·62 = 9개를 승인했다.
- 상한 적용 전 원본 데이터, 시간·수량 log1p, log 수량 MSE + 시간 손실 가중치1이다. 기존 원본 MLP16의 입력·head·Train 통계·학습 예산·선택을 유지한다.
- 원본 Train/Validation 세 데이터셋의 대상·수량 통계·시간 통계를 재계산해 통과했다. 기존 MLP16 저장9개를 결속했으며 재학습·재추론하지 않았다.
- 신규 코드133개 검사와 macOS 후속 대기 수정28개 검사, 총161개 로컬 검사가 통과했다. 실제 GPU 합성 검증은 앞 실험 종료 뒤 진행한다.

**현재 Cap18의 검증·회수를 기다린다 — 진행 중**

- 2026-10-20 09:11:16 KST 유한 후속 queue_v2를 실제 등록했다. PID59057의 실제 명령·생존과 WAIT_STARTED, 오류 로그0을 로컬로 확인했다.
- 등록 시점에는 Cap18 finalreceipt가 없고 원본9의 원격 시도·학습은0이다. 마지막 원격 Cap 관측08:57:37 KST는 학습15완료·Intermittent대조군3진행·최종Validation0이다. 이 과거 관측을 새로운 시각으로 갱신하지 않는다.
- 기존 Cap finisher PID16494는 계속 실행 중이다. 현재 학습과 finisher를 수정하거나 다시 시작하지 않았다.
- 이전 base QUEUE_FAILURE는 macOS Python 실행 경로 표기로 생긴 로컬 시작 전 실패다. 원격 실행·과학fit0이며 원문을 보존했다. 활성 경로는 queue_v2다. 상태를 읽을 때 CURRENT_EXECUTION.json의 우선순위를 따른다.

**원본9 학습·Validation·회수 — 다음 작업**

- Cap18 train18/Validation18·원본회수·최종비교·모든 SHA 완료를 먼저 검증한다.
- fresh GPU/소유·Runtime·입력과 후보3개 동시 합성 검증 후 RAF3→Taxi3→Intermittent3을5080에서 최대3병렬로 처음 한 번 시작한다. 부모 기한은 연장하지 않는다.
- 선택 checkpoint9의 Validation 재현 후 새 경로에 한 번 회수하고 기존 원본 MLP16과 전체/큰수량 RMSE·MAE·TimeNLL의3seed 평균±표본SD 및 모든 seed 절대·상대차이를 비교한다.
- 새 Test·추가 대조군·기존CNN 변경·기존외부 재개·반복개선 자동화는 없다. 실패·미확정이면 원문을 보존하고 자동 재시작하지 않는다.
