# 폭8·12의 고정 체크포인트 Test 평가

**Taxi·RAF 12개 조건의 Validation·Test 평가와 기존 비교군 집계 — 완료**
- 5080 세 번째 시도에서 전체 Validation 12개를 원본 endpoint와 재현한 후 같은 checkpoint의 Test 12개를 완료했다. 24 population 평가 시간은 96.514초였으며 원시 예측은 저장하지 않았다.
- [결과 보고서](results_taxi_raf/REPORT.md), [Test 전체](results_taxi_raf/TEST_TABLES.md), [Test tail](results_taxi_raf/TEST_TAIL_TABLES.md), [새 Validation 재현](results_taxi_raf/VALIDATION_NEW_TABLES.md), [수치·범위 JSON](results_taxi_raf/summary.json).
- 기존 Test 162지표행·77 aggregate receipt를 target/truth·loader·SHA로 검증해 재사용했다. 폭4·8·12·16과 모든 기존 비교군의 Test 전용 비교는 186지표행이다. [최종 검증](results_taxi_raf/final_verification.json).
- Taxi 폭12의 평균 Test RMSE는 103.916 ± 9.105로 폭4의 118.624 ± 19.570보다 낮지만 seed52에서는 악화했다. RAF 폭8·12의 평균 RMSE는 각각 39.927 ± 0.412, 40.011 ± 0.245로 폭4의 39.795 ± 0.075보다 소폭 높았다. Test에 따른 폭·epoch 재선택은 수행하지 않는다.

**실행 계약과 평가 코드 준비 — 완료**
- 대상: `paper_research`, Taxi·RAF의 폭8·12 × seed42·52·62, 학습 조건 12개.
- 각 원본 terminal manifest에 연결된 selected checkpoint의 binary SHA, tensor SHA, 초기화, source revision, 최초 전체 Validation raw quantity RMSE 최솟값을 검증한다.
- `evaluate.py`는 기존 폭16 평가기의 파생본이다. 동결 source의 모델·loader·`target_outputs`·`QuantityMetrics`를 재사용한다. per-target 예측과 private target metadata를 파일로 저장하지 않는다.
- focused synthetic tests는 연구 데이터·원격 서버·GPU에 접근하지 않는다. 원래 학습과 기존 보고서는 수정하지 않는다.

**전체 Validation 재현 후 동일 체크포인트 Test 평가 — 완료**
- root 담당 세션이 선택한 5080 전용 폴더를 계약에 고정한 후, `prepare.py`가 batch별 immutable 바인딩을 만든다.
- `pipeline.py`는 batch의 모든 Validation 평가가 통과해야 Test를 시작한다. 기존 target/truth SHA, 원래 loader와 전체·tail 수량 MAE/RMSE, 정수 간격 Time NLL을 확인한다. 허용 오차는 실행 전에 고정한 상대·절대 `1e-5`이며 mismatch 후 완화하지 않는다.
- 기존 결과 폴더가 있으면 자동 재시도나 덮어쓰기를 하지 않는다. 실패 기록을 보존한다.
- 첫 두 시도는 자료 경로와 프로젝트 경로 표시 파일 문제로 실제 추론 전에 중단됐다. [실패 기록](preinference_failure_notes.json)과 원본 계약을 보존했으며, 세 번째 시도는 경로·SHA·ancillary marker·staged import preflight를 통과했다.
- `historical_code/prepare_v1.py`는 초기 세 시도의 seal에 대응하는 원본이다. 현재 `prepare.py`의 후속 계약 비교는 서버 이동에 따른 `data.path`·`split_manifest.path`만 정규화하며 모델·전처리·손실·split·byte SHA는 그대로 일치해야 한다.
- `historical_code/pipeline_v1.py`도 초기 세 시도의 원본 SHA와 일치한다. 후속 Intermittent 파이프라인에는 조건별 GNU timeout과 외부 종료 신호 시 소유 평가 프로세스 정리를 추가했다. 완료된 Taxi·RAF의 계약·코드 seal·원격 기록은 보존한다.

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
- `followup.py`의 준비 hook은 로컬 원본 6개가 모두 완료·검증되어야 새 계약과 deployment request를 만든다. 승인된 hourly driver가 원격 원본 회수·preflight·단일 worker 실행을 담당한다. 부분 완료·기존 시도·실패 조건에서 자동 재학습이나 재시도하지 않는다.

**Intermittent 평가 완료 후 세 데이터 비교를 별도 집계한다 — 다음 작업**
- 폭4·16과 외부 비교군의 기존 Test는 재실행하지 않는다. 원본 집계 `reports/titantpp_completed_validation_test_20261004_v1/results/metrics_per_seed.csv`를 `comparison_receipt.json`의 SHA와 대조해 Test 행만 재사용할 수 있다.
- 원래 Validation에서 선택한 체크포인트를 기존 Test split에서 평가하는 작업이다. 새로운 독립 Test로 주장하거나 Test 지표로 폭·epoch를 다시 선택하지 않는다.
- 현재 원고는 수정하지 않는다. 수치의 split, 조건 수, seed 평균과 표본 표준편차를 구분해 기록한다.
- 이번 Taxi·RAF 집계는 보존하고, `analyze.py --campaign <taxi_raf_attempt3> --campaign <intermittent> --output <new_report_folder>`로 새 결과를 만든다. 전체·tail RMSE/MAE와 정수 간격 Time NLL을 함께 기록한다.

Focused synthetic 검증은 18개 통과했다. Validation 실패 시 Test 차단, 전체 gate 순서, 원본 입력 변경·기존 출력 보호, 순수 집계 저장, Intermittent 일부 완료 차단, deployment path preflight, seed 통계, operational path와 과학 조건 구분, 조건별 시간 제한과 외부 종료 신호의 소유 프로세스 정리를 확인했다. 연구 자료와 GPU는 테스트에서 사용하지 않았다.

예상 평가 시간은 동일 5080의 폭16 실측에 기반한 추정이다. Taxi·RAF 조건당 한 split 약 2.6–3.4초, Intermittent 약 10.8–11.2초였으며, 초기 24 population 평가 약 1–3분과 후속 12 population 평가 약 1–3분을 예상한다. 이 값은 폭8·12의 실제 시간이나 GPU 간 효율 비교가 아니다.
