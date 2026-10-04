# 폭8·12의 고정 체크포인트 Test 평가

**실행 계약과 평가 코드 준비 — 완료**
- 대상: `paper_research`, Taxi·RAF의 폭8·12 × seed42·52·62, 학습 조건 12개.
- 각 원본 terminal manifest에 연결된 selected checkpoint의 binary SHA, tensor SHA, 초기화, source revision, 최초 전체 Validation raw quantity RMSE 최솟값을 검증한다.
- `evaluate.py`는 기존 폭16 평가기의 파생본이다. 동결 source의 모델·loader·`target_outputs`·`QuantityMetrics`를 재사용한다. per-target 예측과 private target metadata를 파일로 저장하지 않는다.
- focused synthetic tests는 연구 데이터·원격 서버·GPU에 접근하지 않는다. 원래 학습과 기존 보고서는 수정하지 않는다.

**전체 Validation 재현 후 동일 체크포인트 Test 평가 — 외부 작업 대기**
- root 담당 세션이 선택한 5080 전용 폴더를 계약에 고정한 후, `prepare.py`가 batch별 immutable 바인딩을 만든다.
- `pipeline.py`는 batch의 모든 Validation 평가가 통과해야 Test를 시작한다. 기존 target/truth SHA, 원래 loader와 전체·tail 수량 MAE/RMSE, 정수 간격 Time NLL을 확인한다. 허용 오차는 실행 전에 고정한 상대·절대 `1e-5`이며 mismatch 후 완화하지 않는다.
- 기존 결과 폴더가 있으면 자동 재시도나 덮어쓰기를 하지 않는다. 실패 기록을 보존한다.

```bash
python3 reports/titantpp_width8_12_test_20261004_v1/prepare.py \
  --output reports/titantpp_width8_12_test_20261004_v1/taxi_raf \
  --remote-root /absolute/approved/dedicated/root \
  --dataset-root /absolute/prior/approved/dataset/root

python3 reports/titantpp_width8_12_test_20261004_v1/pipeline.py \
  --root /absolute/approved/dedicated/root \
  --campaign /absolute/approved/dedicated/root/reports/titantpp_width8_12_test_20261004_v1/taxi_raf \
  --device cuda
```

위 명령은 계약 형식 예시이며 별도 원격 실행 승인이 아니다. `--dataset-root`를 생략하면 dataset manifest의 기존 상대 경로를 유지한다.

**Intermittent 6개 조건의 후속 Test — 외부 작업 대기**
- 대상: 폭8·12 × seed42·52·62. 여섯 학습 모두 완료한 뒤 별도 batch 계약을 준비한다.
- `prepare.py --datasets intermittent_frozen_5000 --training-contract <derived-contract> --original-root <original5090-run> --original-root <handoff5080-run>`로 parent와 승인된 derived 계약의 원본을 연결한다.
- dataset·arm·seed가 과학 조건의 identity다. 운영 host는 원본 provenance에 기록한다. 동결 source closure와 selected tensor identity는 일치해야 한다.

**기존 비교 결과와 별도 보고 — 다음 작업**
- 폭4·16과 외부 비교군의 기존 Test는 재실행하지 않는다. 원본 집계 `reports/titantpp_completed_validation_test_20261004_v1/results/metrics_per_seed.csv`를 `comparison_receipt.json`의 SHA와 대조해 Test 행만 재사용할 수 있다.
- 원래 Validation에서 선택한 체크포인트를 기존 Test split에서 평가하는 작업이다. 새로운 독립 Test로 주장하거나 Test 지표로 폭·epoch를 다시 선택하지 않는다.
- 현재 원고는 수정하지 않는다. 수치의 split, 조건 수, seed 평균과 표본 표준편차를 구분해 기록한다.

예상 평가 시간은 동일 5080의 폭16 실측에 기반한 추정이다. Taxi·RAF 조건당 한 split 약 2.6–3.4초, Intermittent 약 10.8–11.2초였으며, 초기 24 population 평가 약 1–3분과 후속 12 population 평가 약 1–3분을 예상한다. 이 값은 폭8·12의 실제 시간이나 GPU 간 효율 비교가 아니다.
