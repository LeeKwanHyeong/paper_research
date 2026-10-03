# 기존 TitanTPP B의 추가 Test 평가

사용자 승인: 2026-10-03 “오케이 B 관련해 Test 진행해보자”.

**원본 연결 — 완료**

- 기본 B는 `titantpp`이며 추가 인접 상태 보정 MLP가 없는 static Hard-LMM이다.
- Taxi·Intermittent·Instacart × seed42/52/62의 원래 선택 체크포인트 9개를 재사용한다.
- 원래 파일·tensor SHA, 선택 epoch, validation 재평가, frozen source/loader/statistics를 연결했다.
- RAF의 과거 B는 시간 헤드와 joint-objective 선택 기준이 달라 이번 동일조건 비교에서 제외한다.

**평가 의미와 범위 — 고정**

- 이미 완료된 MLP Test 결과를 본 뒤 사용자가 요청한 탐색적 추가 비교다.
- 기존 Test 분할의 사후 재평가이며 새로운 미접근 자료 평가가 아니다.
- B와 MLP·외부6모델의 기존 동일 대상 예측을 비교한다. 기존 대표를 자동 교체하지 않는다.
- 모델/전처리 재학습, 선택 epoch 변경, 데이터 재분할, 새 유료 임대는 없다.
- 기존 Test 결과와 기록은 보존하고, 이번 결과는 이 폴더에 추가한다.
- MAE·RMSE·시간 NLL의 3seed 평균과 표본 표준편차를 함께 보고한다.
- 불확실성은 고정된 학습 seed에 조건부인 paired bootstrap이다. 이번 추가 비교는 탐색적 95% 구간을 사용한다.
- Taxi의 168시간 기본 블록 CI는 기존 Test 기간 부족으로 산출할 수 없다. 민감도 구간으로 대체하지 않는다.

**실행 환경과 검증 — 진행 중**

- 기존 연구용 5080 전용 평가 root 안의 새 report 하위에만 추가한다.
- 기존 Runtime·데이터·동결 source bundle을 재사용하며 다른 작업/프로세스는 변경하지 않는다.
- 단일 추론 worker, GPU 12GiB/RSS 32GiB, 조건 3시간·전체 24시간, 출력 15GiB 제한을 유지한다.
- 기존 평가기 `evaluate.py`의 바이트를 그대로 재사용한다.
- 9개 모든 조건에서 CPU 1batch/CUDA 4batch의 인과성·가중치 불변 검사를 하고, 전체 Validation 지표가 원래 선택 기록과 일치해야 Test를 허용한다.
- 작업 전용 프로세스가 자원 제한을 위반하면 감독자가 해당 프로세스 그룹만 중단하고 실패 증거를 보존한다.
- 원격 완료 후 원본 예측/receipt를 SHA 검증하여 회수하고, 기존 비교군과 target/truth/history 일치를 검증한다.

**진행 상태**

실제 실행·완료 여부는 `pipeline_status.json`, `runs/test/attempt1/terminal_manifest.json`,
`retrieval/receipt.json`, `analysis.json`, 최종 보고서를 각각 확인한다.
