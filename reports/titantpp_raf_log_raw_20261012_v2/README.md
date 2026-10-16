# RAF log/raw 진단의 합성 검증 수정 실행

**실데이터 평가 전 오류를 보존한다 — 완료**

- v1은 cuDNN의 eval RNN backward 제한으로 첫 합성 검증에서 실제 exit1이었다. 실데이터 평가0개이며 claim·실패·봉인 원문은 v1에서 보존한다.
- 이 v2는 합성 gradient fixture에만 cuDNN을 잠시 비활성화하고 복원한다. 실제 eval forward·source129·선택 checkpoint9·Train/Validation 대상·출력·Runtime은 같다. 새 학습·Test0이다.

**승인된 RAF18 평가와 원본·분석을 확정한다 — 완료**

- 합성 native3 검증 뒤 Train9+Validation9를 실행하고 실제 exit0·원본 archive/manifest·로컬 SHA·독립 QA를 확인한다. 실패했던 root를 덮어쓰거나 자동 retry하지 않는다.
- 전체·수량 bin·큰 수량의 실제 per-target 로그 목표와 raw 오차를 묶는다. 결과는 제한된 관측이며 objective 변경 효과·최적화 실패의 증명이 아니다.

**완료 RAF를 기준선으로 Intermittent에 인계한다 — 완료**

- v2의 실제18 종료·원본·분석·QA를 고정한 뒤만 Intermittent18 평가를 시작한다. 공통 GPU 실행과 회수는 직렬이다.

**확인된 결과를 제한된 주장으로 연결한다 — 완료**

- 실제 pipeline 종료12:51:45 KST, Train9+Validation9·worker18 실제 exit0·native3 PASS·357원본 archive와 두 로컬 복사본 SHA·독립 최종 QA PASS다.
- Train 큰 수량은 전체의0.7952%이며 full 수량 log 손실5.93–6.78%와 raw SSE80.43–81.48%를 차지한다. 큰 수량에서 log/raw 모두 단순 평균·직전 값보다 세 seed 모두 나쁘다. Validation 큰 수량은 직전 값보다 log 오차는 좋지만 raw 오차가 나쁘고, 평균 기준선보다 둘 다 나쁘다.
- [최종 원본 분석](../../search_artifacts/titantpp_raf_log_raw_20261012_v2/analysis/RAF_LOG_RAW_REPORT.md)과 [독립 QA](FINAL_QA.md)를 사용한다. 전체 log 개선을 tail 성공이나 목적 변경 효과의 증명으로 확대하지 않는다.
