# RAF log/raw v2 실제 종료 및 수치 독립 QA

## Review Result: PASS

- 검증시각: 2026-10-12T12:55:45.383994+09:00
- 상태: 완료. 선택 checkpoint9개 × Train·Validation18회, 실제 worker18 exit0 및 pipeline OS exit0를 확인했다. 신규 학습0, 신규 Test 평가0.
- 변경 파일: 이 FINAL_QA.md만 작성. source/model/checkpoint/Runtime/main.tex/.agents 변경0, 별도 SSH/GPU 평가0.

### CRITICAL
- 없음.

### HIGH
- 없음. v1 synthetic native qualification의 cuDNN eval backward 실패는 v1 FINAL_QA와 실패 원문에 보존했다. v1 선택 checkpoint 평가0이며 v2 완료로 덮어쓰지 않았다.

### MEDIUM
- 없음.

### LOW
- 없음.

## 확인한 완료 조건

- [x] canonical `0c75bdd4d70adf8569ea2602c942e93f6b27b1206a4595711b2673997ee0758a`, operation `ac834cd38fbb5f2ce693148711e7c187117c190923e78562eb5cb3975bfd8cd1` 일치.
- [x] input213/source129·원래 checkpoint9·Train/Validation reference18 봉인 SHA 일치. 기존 과학 계약과 실제 저장 모델 추론 경로 유지.
- [x] native synthetic3 PASS, 실제 launch/exit argv·PID·PPID 연결 확인. synthetic receipt 자체에 PID가 없으므로 그 파일 단독으로 PID를 입증했다고 표현하지 않는다.
- [x] native admission 이후 실제18 worker argv(launch는 Python 포함, receipt는 제외)·PID·PPID·선택 state·target/truth·원래6지표 재현. actual18 exit0, pipeline OS exit0, 마지막 phase complete 및 owner/GPU PID 부재의 봉인 원본 확인.
- [x] archive357개 각 member SHA, manifest, materialized·verified 두 로컬 복사본 전부 재대조. archive `4e6dc2ca32ec70a2e042e01b46d3ad4acdc3599442ce047a0a73c79b62134acf`, manifest `003dae66b668b01876c257a4b43229e35b2ee14faf3c99d611f6db0a58b09536`.
- [x] 분석 collect()를 stdlib 읽기 전용으로 독립 호출해18개 검증. 충분통계 각 bin 합계·native log loss·raw SSE/MAE·모집단 일치 확인.
- [x] 최종18행·24개 full−대조 대비·6개3seed 요약의 평균/표본SD를 원본으로 별도 계산해 정확 일치 확인. 분석 completion rawSHA `0a7978a5cb49cda538380bc6e32455c8560d3a97581b341069fd5f91a86eb753`와 분석2파일 SHA 일치.
- [x] scalar 평균 분해/로그평균 추측0, actual per-target native float32 log_qty_loss 관측, 동일 평가대상, raw 예측행 영구저장0.

## 수치와 해석의 범위

Train full의3seed 평균 전체 raw RMSE45.872441·수량 log MSE0.623400, Validation34.171127·0.580691이다. 큰 수량(true quantity>200) RMSE는 Train463.119714·Validation327.623692, raw bias는−356.569067·−280.781646이며 대상은 각각205·50개다. Validation 전체 full RMSE는 두 대조보다 낮지만 큰 수량 RMSE는 더 높다. 과거 다른 split의 RAF 악화를 이번 Validation 전체의 악화라고 바꾸지 않는다.

전체와 큰 수량은 다른 모집단이다. 같은 큰 수량에서도 log 오류가 큰 경우를 raw/log 목표 상충의 직접 증거로 과장하지 않는다. 손실 비중은 선택 모델의 eval 상태 수량 항이며 joint 시간+수량 학습 gradient 기여가 아니다. Train 오차만으로 용량 부족/최적화 실패를 확정하지 않는다.

실제 저장 모델과 synthetic forward는 원래 cuDNN eval 경로다. 추가 synthetic gradient fixture만 ATen(cuDNN 일시 비활성화·복원)을 사용했다. 실제 저장 모델 gradient·학습은 실행하지 않았다. Test 내용은 이번 검증에서 열람하지 않았다. 새 독립 평가·통계 유의성·CNN/GRU 인과 필요성의 증명이 아니다.

## 남은 작업 순서

**Intermittent 위치·분포 폭 분리의 동일 모델18회 평가 — 다음 작업**
- 이 H1 종료 원본·분석·본 QA를 봉인 선행조건으로 고정한 뒤 H2 계약과6개 joint worker/18 model pass를 검증한다. 신규 학습과 Test 평가는0개를 유지한다.
