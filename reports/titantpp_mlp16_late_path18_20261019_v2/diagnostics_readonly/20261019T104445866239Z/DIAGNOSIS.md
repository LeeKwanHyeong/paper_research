# 5090 읽기 전용 확인

실제 관측: 2026-10-19 19:44:46 KST. SSH 실제 종료0·stderr0. 원격 계약과 source963 SHA를 확인했다. 새학습·신호·원본변경·Test는 모두0이다.

- GPU compute process는 비어 있다. GPU 메모리 사용량은 22 MiB / 32607 MiB다.
- RAF6은 학습 worker exit0 기록이 있다. Intermittent3은 시작 기록과 저장 history가 있으나 실제 프로세스와 종료 기록이 없다. 나머지 Intermittent1은 시작 기록이 없다.
- dispatcher도 현재 프로세스가 없고 실제 종료 기록은 없다. 원격 status.json의 running 표시는 과거 파일이며 현재 실행으로 해석하지 않는다.
- 이전에 정지한 외부 비교군 dispatcher/worker4개도 현재 /proc에서 없어 기존 상태 도구의 paused identity/state 검증이 실패한다. 왜 사라졌는지는 이번 관측만으로 확정하지 않는다.
- 10개 모두 선택 Validation 재현 시작 기록이 없고 최종 회수 receipt도 없다. 이 진단을 정상 종료·회수 gate 통과로 사용하지 않는다.

|데이터|seed|후보|관측 상태|저장 epoch|best epoch|
|---|---:|---|---|---:|---:|
|RAF|42|관측 전달|학습 exit0 기록|64|24|
|RAF|42|용량 대조|학습 exit0 기록|64|24|
|RAF|52|관측 전달|학습 exit0 기록|43|3|
|RAF|52|용량 대조|학습 exit0 기록|43|3|
|RAF|62|관측 전달|학습 exit0 기록|52|12|
|RAF|62|용량 대조|학습 exit0 기록|52|12|
|Intermittent|52|관측 전달|현재 프로세스·종료 기록 없음|15|11|
|Intermittent|52|용량 대조|현재 프로세스·종료 기록 없음|13|7|
|Intermittent|62|관측 전달|현재 프로세스·종료 기록 없음|13|12|
|Intermittent|62|용량 대조|미시작|N/A|N/A|

기존 정상 종료6개와 학습 원본·실패 기록은 보존한다. 프로세스가 사라진3개를 성공으로 처리하거나 checkpoint를 임의로 선택하지 않는다. 기존 완료 대기 SSH 실패는 reports/titantpp_mlp16_late_path18_20261019_v2/followup_v1/failure.json에 별도로 남아 있다.
