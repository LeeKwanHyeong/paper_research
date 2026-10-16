# Q3-capped 수량 비교 데이터

기존 Intermittent·RAF·Taxi의 이벤트 수량을 데이터셋별 Train Q3로 제한한
별도 비교용 데이터다. 원본 오류를 정정한 자료가 아니라 예측할 수량의 상한을
정한 실험 자료이며, 기존 원수량 결과와 분리해 사용한다.

## 적용 기준

`q_cap = min(q_original, Q3_train)`을 입력 이력과 예측 정답에 공통으로 적용한다.
Q3는 **각 데이터셋 Train의 전체 이벤트 행**에서 75백분위(linear interpolation)로
계산한다. 개체별 Q3나 Validation·Test의 Q3를 사용하지 않는다.

| 데이터셋 | Train 전체 이벤트 | 적용 상한 | 원본 기준 |
|---|---:|---:|---|
| Intermittent | 398,824 | 20 | 고정된 5,000개 시계열 |
| RAF | 30,779 | 10 | 기존 양의 수요 이벤트 |
| Taxi | 38,524 | 65 | 기존 격자·시간별 승차 수량 |

앞선 대화에서 Intermittent Q3가 21이었던 것은 각 개체의 첫 관측을 제외한
393,824개 **예측 대상**만 집계했기 때문이다. 이번 자료는 첫 관측도 포함한
전체 Train 수량 기준으로 20을 사용한다. RAF와 Taxi의 두 기준은 각각 10과 65로 같다.

현재 모델이 사용하는 이벤트 테이블을 변환한다. RAF의 0수요 월별 패널이나
원시 Excel, Taxi의 집계 이전 개별 승차 기록을 다시 구성하는 작업은 포함하지 않는다.
기존 이벤트·개체를 삭제하거나 새로 표본 추출하지 않는다.

## 파일 구성

- `data/intermittent/`, `data/raf/`, `data/taxi/`: 기존 파일 prefix를 유지한
  전체 분할 파일, Train, Validation, Test, Train+Validation 파일과 새 split manifest.
- `manifest.json`: 원본·생성 파일의 경로와 SHA, 적용 정책 및 검증 결과.
- `COMPARISON.json`, `COMPARISON.md`: 변환 전후 수량 분포와 변경 행 수.
- `CAP_POLICY.json`: 변환 기준과 후속 비교의 고정 조건.
- `VERIFICATION.json`, `VERIFICATION.md`: 원본과 변환본의 독립 검산 결과.

Parquet 데이터는 저장소의 기존 규칙대로 로컬에 보존하고 Git에는 넣지 않는다.
정책·manifest·보고서·생성 코드는 버전 관리한다.

## 데이터 정합성

`demand_qty`와 이에 종속된 `log_qty`, `log10_qty`, `raw_order`, `mark`,
`scale_residual`, `z`를 함께 갱신한다. 호환용 로그의 밑은 기존대로 Intermittent·RAF는
2, Taxi는 10이다. 모델 내부의 자연로그 `log1p`와 이 호환용 칼럼을 구분한다.
수량 관련 칼럼 외의 개체·행 순서·시각·간격·분할·부가 정보는 보존한다.

Test에는 이미 Train에서 결정한 변환만 적용한다. Test 성능 평가·모델 추론·학습은
수행하지 않으며 Test 수량 분포로 상한을 다시 정하지 않는다. 원본 수량을 별도
모델 입력 칼럼으로 추가하지 않는다.

## 데이터 생성과 검증 — 완료

3개 데이터셋의 670,942행에서 상한 변환과 파생 수량식을 검산했다.
15개 Parquet와 3개 분할 manifest의 정합성 검사, 원본 SHA 대조 및
12개 테스트가 통과했다. 모델 학습·추론·성능 평가는 아직 수행하지 않았다.

프로젝트 root에서 실행한다. 이미 생성된 파일을 덮어쓰지 않는다.

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m benchmark_data.scripts.build_q3_cap
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m pytest -q benchmark_data/tests/test_q3_cap.py
```

## 후속 비교 — 다음 작업

이 자료로 학습할 때는 모든 비교 모델이 동일한 Cap 데이터·분할·예산·선택 규칙을
사용해야 한다. 수량 평가의 정답도 `q_cap`이다. 기존 원수량 CNN·외부 모델의 점수를
Cap 성능표에 그대로 옮기지 않는다. 기존 CNN 모델·checkpoint·결과는 보존한다.

Cap용 입력 SHA와 Train 통계를 사용한 새 실험 계약을 작성한 뒤 학습 여부와
비교 모델 범위를 결정한다. 현재 진행 중인 원수량 Cross Attention pre-check의
입력·계약·프로세스는 변경하지 않는다.
