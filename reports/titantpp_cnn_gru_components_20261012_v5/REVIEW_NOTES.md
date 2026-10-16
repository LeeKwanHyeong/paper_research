# v5 NVML 종료 반영 대기 독립 QA

상태: 로컬 독립 QA·봉인·실제 native PASS. 최초3개 scientific fit 실행 및 세 조건의 실제 checkpoint 저장 확인. QA source 수정/SSH/GPU 실행0. v5 코드·회귀 및 봉인 확인 전 PASS를 부여하지 않는다.

v4 실제 실패 snapshot `control/status/5080/e79c461d1d77487cb87965583f0200ec.json` raw SHA는 `7db5db07091435f853465039fa7eaab31bdf1efc8806dee60c5026a9b66d2a81`이다. embedded raw SHA를 모두 확인했다. PID40079의 `/proc`가 소멸했으나 두 차례 즉시 NVML 재조회에서 계속 나타나 `GPU PID identity unresolved after two fresh inventories`로 차단됐다. 실제 qualify PID39969/PPID39953 exit1, 관측 GPU PID0/checkpoint0이며 과학 fit0이다. v4 실패 원본을 보존한다.

v5 검토는 최근·현재 소유 witness 조건을 유지하면서 최대25회, 0.2초 간격, monotonic5초 내의 NVML 반영 대기로 제한한다. 안정된 외부 argv, unknown PID, PID 재사용, permission/query failure를 정상 종료로 면제하지 않는다. source131/모델/초기화/손실/데이터/selector 및 최초5080 절대 lease는 불변이어야 한다. 5090은 별도 승인 전 모든 원격0이다.

독립 고유44개 검증 PASS 및 별도 helper mock PASS. 미해결 CRITICAL/HIGH/MEDIUM/LOW 없음. 실제 native와 최초 fit 시작은 NATIVE_QA.md에서 별도 확인했다. 전체 학습 성공/완료는 아직 미확인이다. 상세 범위는 PRELAUNCH_QA.md에 기록했다.

QA 변경 파일: PRELAUNCH_QA.md, NATIVE_QA.md, REVIEW_NOTES.md. source/코드/bundle/main.tex/.agents 수정0, 원격/GPU 직접 실행0.

18:32:30 KST 최종 초기 확인에서 세 조건 모두 실제 checkpoint SHA PASS. 마지막 저장/best는 NONE5/4, CNN5/4, GRU2/2이며 전체24완료0이다. 초기 확인 종료 후 추가 실행은 하지 않았다.
