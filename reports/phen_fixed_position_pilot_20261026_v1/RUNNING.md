# RAF·Taxi 고정 위치 PHEN seed 42 실행 현황

2026-10-26 14:57 KST 확인 기준. 실험은 완료되지 않았다.

## 사전 검증 — 완료

- 동결 원본 소스 987개는 변경하지 않았다. 추가 adapter와 제어/시각화 파일을 포함한 소스 991개 및 Train/Validation 입력 4개의 SHA를 확인했다.
- 로컬 35 tests actual exit 0 및 독립 QA PASS.
- 최초 synthetic 실행은 CUDA 라이브러리 환경 전달 누락으로 학습 진입 전에 실패했다. 원문·exit·SHA 및 원격 경로를 보존했다.
- 환경 전달을 복구한 attempt2의 RAF·Taxi synthetic/capacity 4개 프로세스는 모두 actual exit 0이었다. 각 검사의 stderr는 0 byte였다.

## 5080 최초 학습 — 진행 중

- RAF seed 42는 14:56:53 KST에 시작했다. 14:57:04 KST에 epoch 2 기록이 생성됐음을 확인했다.
- 각 epoch 종료 모델의 Train 25,779건과 Validation 6,690건을 같은 eval 함수로 평가한다. 추가 평가의 RNG·가중치 불변 검사를 통과한 기록이다.
- Taxi seed 42는 RAF의 학습·Validation 재현·실제 종료 검증 후 시작한다. 대상은 Train 38,393건 및 Validation 8,268건이다.
- 최대 300 epoch, 최소 40 epoch, Validation 수량 RMSE best 이후 patience 40 조건은 기존과 같다.
- 두 실행 모두 위치 표현만 고정 sin/cos로 바꾸고 PHEN 이력 보정·Prototype·head·loss·loader 조건을 유지한다.
- 5090에 접속하거나 Instacart를 변경하지 않았다. MAC은 보류한다.

## 곡선 생성과 회수 — 다음 작업

서버의 tmux 부모는 두 학습이 정상 종료되면 `curves/`에 RAF·Taxi별 PDF·PNG와 `epoch_metrics.csv`를 자동 생성한다. 다섯 지표는 수량 RMSE·MAE·log1p MSE, recorded time NLL, 결합 손실이다. online Train 손실도 별도로 남긴다.

원격 첫 학습 진입까지만 확인했으며 지속 polling은 하지 않았다. 완료 확인 후 현재 프로젝트 root에서 다음 순서로 회수·검증한다.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 reports/phen_fixed_position_pilot_20261026_v1/transport.py status --attempt 2
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 reports/phen_fixed_position_pilot_20261026_v1/transport.py collect --attempt 2
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 reports/phen_fixed_position_pilot_20261026_v1/analyze.py --check-only
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 reports/phen_fixed_position_pilot_20261026_v1/analyze.py
```

`collect`는 COMPLETE 및 모든 실제 child 종료·manifest SHA 확인 전에는 진행하지 않는다. 같은 회수 명령을 반복해 기존 파일을 덮어쓰지 않는다. 회수된 그림은 가독성을 확인하고, 기존 learned-position Validation 기준선과 비교한 결과는 `attempt2/analysis_v1/`에 생성한다.

현재 지표는 극초기 기록이다. 수량/시간 일반화 개선이나 과적합 해소 여부는 아직 판단하지 않는다.
