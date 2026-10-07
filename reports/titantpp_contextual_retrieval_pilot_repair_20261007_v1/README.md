# CNN+GRU54 검색부 파일럿 운영 수정 — 2026-10-07

## 목적과 승인
사용자는 “문제 해결해서 진행하자”라고 명시적으로 승인했다. 기존 5080 시각 검사 차단과 5090 조건 7200초 중단을 보존하면서 미완료 실험을 한 번 이어간다. 과학 source126, 데이터, 모델, 손실, optimizer, batch128, min40/max300/patience40 및 최초 strict 최소 Validation 수량 RMSE 선택은 변경하지 않는다. Test는 열람·평가하지 않는다.

## 수정 내용 — 완료 / 양 GPU 학습 진행 중
- 5080: 서버 시각이 로컬 수신보다 약 2초 앞선 경우를 정상 수집으로 처리한다. 로컬 수신 시각·raw SHA·원본 qualification receipt를 결합하고 허용 시각차는 절댓값 10초다. 오래된 기록·변조는 거부한다.
- 5090 fixed: 원래 E148/best120의 checkpoint, optimizer·RNG·loader 상태와 148epoch history/exposure/timing을 SHA로 대조한 뒤 새 운영 root에서 E149부터 이어간다. 기존 run/claim/failure를 변경하지 않는다.
- 5090 mean E120/best80와 weighted E99/best59는 완료 결과를 재사용한다. adaptive/generic 두 조건과 5080 다섯 조건은 fresh fit이다.
- 새 운영 한도는 조건당 21600초(6시간), 서버당 129600초(36시간)다. 새 lease와 permit에 별도로 기록하며 원래 예산을 덮어쓰지 않는다. 한도는 ETA가 아니다. 추가 자동 재시도는 없다.

## 집계와 평가
5개 과학 설정 × 동일 Taxi seed42 × 두 GPU의 10실행이다. 두 seed 또는 독립 평가로 합산하지 않는다. 재사용2 + 재개1 + fresh7을 각 원래 execution ID로 한 번만 집계한다. 5090이 탐색 주 비교이며 5080은 같은 seed의 GPU 반복이다. 시간·비용은 GPU별로 분리한다.

주후보 adaptive가 5090 mean/fixed/generic 각각보다 전체 및 큰 수량 RMSE 모두 strict 개선하고, 기존 T64/T256 비용 gate를 만족할 때만 탐색 성공으로 판단한다. weighted 변화와 같은 epoch MAE·TimeNLL 손해도 필수 보고한다. 이는 단일 데이터·seed의 Validation 개발 증거이며 참신성·독립 Test·외부 우월성을 확정하지 않는다.

## 코드 검증
시각 처리, 실패 intent 재실행 거부, 계약·소유권·prefix·observer gate 회귀 테스트와 frozen native trainer의 synthetic CPU E148→E160 재개를 확인했다. 이 probe는 연구 데이터를 읽지 않고 실제 optimizer/RNG/loader 복원·best120 보존·patience40 E160 종료를 검증한다. 실제 GPU와 부분 binary의 startup gate는 별도로 확인한다.

## 보존 및 완료 기준
원래 실패·claim·부분 checkpoint, 원래 과학 계약과 네 operation 파일을 보존한다. 세 소유 root의 scientific terminal·실제 프로세스 종료와 immutable binary manifest/archive SHA 회수를 모두 확인한 뒤 최종 Validation 보고를 작성한다. CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료다.

## 다음 작업
1. **학습과 시간별 관측 — 진행 중:** 5090 fixed E150/best120, 5080 mean E1/best1의 실제 저장을 확인했다. 다음 순서의 adaptive/generic 및 5080 나머지 네 방식은 대기다.
2. **최종 원본 회수와 Validation 비교 — 학습 완료 후:** 10실행 및 세owner 종료가 검증되면 manifest/archiveSHA를 회수하고 동일 선택 epoch의 비교를 작성한다.

실제 진입 증거는 [15:25 KST 상태 보고서](STATUS_20261007_1525KST.md)와 JSON을 참조한다. 68 tests·30 subtests PASS, 기존 완료2를 재사용하며 7edc5f1은 운영 코드 커밋이다.
