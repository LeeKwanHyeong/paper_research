# 최종 독립 QA

## Review Result: PASS

고정 checkpoint 행별 진단의 실행 증적, 원시 예측, 통계 집계와 최종 해석을 검토했다. CRITICAL·HIGH·MEDIUM 문제는 없다. 수량 필드 이름에 관한 LOW 메모 하나를 남긴다. 검증 통과는 과적합이나 GRU의 인과효과가 증명됐다는 뜻이 아니다.

## CRITICAL

- 없음.

## HIGH

- 없음.

## MEDIUM

- 없음.

## LOW

- `reports/titantpp_frozen_row_diagnostics_20261014_v3/finish.py:21` — `new_inference_job_count=162` 및 65행의 `worker_count=162`는 캐시14를 포함한 고유 조건 수다. 필드 이름만 보면 실제 v3 신규 실행 수와 혼동할 수 있다. 계약과 검증은 정확하며 `EXECUTION.md`가 136/12/14로 구분한다. 봉인된 파일은 수정하지 않고 향후 읽기 전용 보고 어댑터에서 다음처럼 표기한다.

```python
counts = {
    'unique_diagnostic_conditions': 162,
    'fresh_5080_v3': 136,
    'completed_5090_v2': 12,
    'reused_validation': 14,
    'existing_external_test_reused': 54,
}
```

## 실행과 원본 검증 — 완료

| 구분 | 독립 확인 결과 |
|---|---:|
| 5080 v3 실제 신규 worker | 136, 모두 exit0 |
| 5090 v2 완료 worker 재사용 | 12, 모두 exit0 |
| 5080 v2 Validation 캐시 | 14, 원래 receipt·exit·계약 SHA 유지 |
| 고유 진단 조건 | 162 |
| 기존 외부 Test 재사용 | 54, 재추론 없음 |
| 최종 모델/분할 | 216 = 72모델 × 3분할 |
| 원시 예측 part | 2,064개 SHA·수치 검증 |
| 전체 예측 기록 | 15,859,464행; 같은 대상을 모델별로 반복한 기록 |
| native 회수 manifest 파일 | 2,626개 SHA 검증 |
| 분석이 결속한 원본 파일 | 2,687개 SHA 재검증 |

두 host의 pipeline·supervisor 실제 OS exit0, worker의 job/split/contract 일치, 모델 tensor 전후 SHA, 고정 선택 epoch·source·입력, Runtime·GPU와 평가 가드를 확인했다. 5080 회수 archive는 921,627,487bytes이며 SHA는 `4f802b80469e04b023d73c611cc87c270cee59bd5d089624863d88e5c0c8c1eb`다. 실제 transport exit0·timeout=false다.

직접 저장된 resolved Python 경로 필드는 없다. 실제 worker argv와 SHA가 고정된 평가 코드의 추론 전후 경로 검사, 실제 exit0를 연결한 증거로 구분한다. 원래 v1·v2 실패 증적과 claim, 성공한 캐시, 기존 과학 source를 보존했다. 원고 두 파일은 전달받은 기준 SHA와 독립 해시가 일치한다.

## 수치와 통계 검증 — 완료

분석기 함수를 재사용하지 않고 원시 행에서 예측−정답, 제곱오차·절대오차·편향·시간손실 합계, 전체 및 고정 수량 구간의 지표를 재계산했다. 각 데이터셋/분할의 공통 10개 정답·시간 필드와 대상 지문도 모델 간 일치했다.

- 216개 모델/분할의 raw 지표와 저장 지표가 일치한다.
- 모델 요약 360개 필드의 평균·표본SD가 일치한다. 표본SD는 3seed에서 `ddof=1`이다.
- 189개 쌍 비교의 부호와 RMSE·MSE·MAE·편향 차이가 일치한다.
- 2,079개 구간 분할에서 대상 수와 SSE·절대오차 합산이 전체 값을 보존한다.
- 63개 개체 구성·가중치·개체 내 오류 분해를 실제 원시 행으로 독립 재계산해 일치시켰다.
- 합성 검증은 평가기11개·분석기14개, 총25개가 통과했다.

평균 RMSE를 제곱해 MSE로 사용하지 않는다. MSE 기여율의 분모가 Test 격차인지 Validation→Test 격차 변화인지 구분한다. 부호가 다른 구간의 상쇄 때문에 특정 구간 기여가 전체 격차의100%를 넘을 수 있다.

## 최종 해석 검토 — 완료

CNN+GRU의 평균 Train·Validation 이득이 Test 평균으로 이어지지 않은 사실은 확인된다. 그러나 Taxi62·RAF42·Intermittent52는 Test RMSE에서 CNN+GRU가 더 좋다. RAF Train RMSE도 42·52에서는 CNN+GRU가 더 높다. 평균 반전을 모든 seed의 일관된 실패로 확대하지 않는다.

Test의 CNN+GRU−CNN model log MSE 평균도 Taxi +0.03345094, RAF +0.00765915, Intermittent +0.00102795다. 따라서 log 목적과 raw 지표 차이만으로 전체 평균 반전을 설명할 수 없다. Intermittent62의 log 이득/raw 손해와 다른 seed의 반대 예외를 보존했다.

Taxi 내부 비교의 손해는 target>last 조건의 CNN+GRU 과소 예측에서, CNN 대 S2P2·AttNHP 비교의 손해는 target<last 조건의 CNN 과대 예측에서 크게 나타난다. 서로 다른 양상을 하나의 원인으로 합치지 않는다. Intermittent는 중간 수량·하락 조건의 과대 예측을 확인했으며 극단적 큰 수량만의 문제로 한정하지 않는다. RAF 내부 비교는 공통 개체에서도 반전되지만, CNN 대 S2P2의 전체 순위 개선은 개체 구성·가중치 변화가 큰 부분을 차지한다. 최종 Markdown의 해당 숫자와 분모를 JSON 및 독립 재계산 결과에 대조했다.

11개 통계 오류 항목을 모두 검토했다. 비붕괴성은 해당 없음, 교란·평균 회귀·다중 구간 탐색·분석 경로 선택·인과 해석은 주의가 필요하다. 세 seed와 과거 Test 접근 이력, 서로 다른 Runtime·과거 외부 캠페인, 모델 용량 차이, target 기반 사후 구간, 개체 내 다른 대상·이력이 남으므로 통계적 유의성이나 근본 인과를 확정하지 않는다.

## 승인 조건 체크리스트

- [x] 고정 선택 모델·입력·source·Runtime·대상 모집단 연결 확인
- [x] 실제 실행과 성공 캐시 및 기존 외부 Test 재사용 구분
- [x] 모든 원시 파일 SHA, 대상 일치, 수치 재합산 확인
- [x] 3seed 평균·표본SD와 불리한 seed·지표 예외 유지
- [x] 내부 비교와 외부 여섯 비교군을 분리하고 누락 없이 분석
- [x] 통계 분해의 합산 보존과 인과 해석 한계 확인
- [x] 실제 Test를 이용한 도구 디버깅·checkpoint 재선택·새 학습 없음
- [x] QA의 원격 호출·모델 추론·checkpoint 역직렬화·source 변경 없음

## 검토 결속

- 실행 receipt: `5e5ba273ab545aee222d51c38804f6a6c3d44c2835c76f111e8819f7348ff157`
- ROW_ANALYSIS.json: `50edf6f15d0d8471b9d9ad36edb3582fa1818e211af447bffad4dde6ced7e82d`
- 최종 ROW_ANALYSIS.md: `9e79c861b9b99ffeddb79d4f742ae0d48f12611b4d403926b7a24e82baf9cc2e`
- 독립 재계산한 216개 지표와 상세 파일 SHA는 `FINAL_QA.json`에 보존했다.

## 변경 파일과 남은 작업

QA는 이 디렉터리의 `FINAL_QA.json/md`만 추가했다. source와 `.agents/`는 수정하지 않았다. 다음 작업은 root가 검증된 수치와 제한된 해석을 사용자에게 보고하는 것이다. 본 QA는 추가 모델 실행이나 원고 변경을 수행하지 않는다.
