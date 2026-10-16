## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-13T13:59:22.572795+09:00
- Verification Status: ANALYZED
- Version Label: taxi_raf_validation_history_v1

## 범위와 확인 근거

같은 5080 GPU/Runtime의 Taxi·RAF 각각4구조×3seed, 총24개를 저장된 full Validation history에서 비교했다. 기존 이력 보정27·A100·Instacart와 합산하지 않았다. 원격 학습24 actual worker exit0와 terminal 파일312개의 원격 SHA가 확인된 2026-10-13 08:27:20 KST 종료 snapshot을 사용했다. 이번에는 SSH·학습·추론·Test·원본 회수·원고 변경을 하지 않았다.

Snapshot: /Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_components_20261012_v5/control/status/5080/1ecf99983b614d9b8f3984cad020c2fe.json

Snapshot rawSHA: 352b65c7f5c55a11fae7720b67c26c437b6999f662cfa0b8297fe0b2c33556f2

24 history payload의 rawSHA와 terminal manifest 결속, 최초 strict finite 최소 전체 raw수량 RMSE 선택epoch를 확인했다. 아래 세 지표는 같은 선택epoch의 값이다. 평균은 seed42/52/62의 산술평균, SD는 표본SD(n-1)다. 현재 결과는 저장history 기준이며 최종 endpoint 재추론 결과표로 부르지 않는다.

## 세 지표의 평균 ± 표본SD

| Dataset | 구조 | 전체 수량 RMSE | 전체 수량 MAE | TimeNLL |
|---|---|---:|---:|---:|
| Taxi | NONE | 89.305 ± 1.151 | 28.683 ± 0.652 | 0.851 ± 0.079 |
| Taxi | CNN | 86.650 ± 2.099 | 27.793 ± 0.500 | 0.782 ± 0.082 |
| Taxi | GRU | 84.478 ± 1.646 | 26.710 ± 0.150 | 1.060 ± 0.128 |
| Taxi | FULL | 81.923 ± 5.027 | 25.816 ± 1.135 | 1.318 ± 0.733 |
| RAF | NONE | 34.224 ± 0.118 | 9.264 ± 0.059 | 3.595 ± 0.112 |
| RAF | CNN | 34.431 ± 0.073 | 9.218 ± 0.028 | 3.497 ± 0.139 |
| RAF | GRU | 34.211 ± 0.226 | 9.159 ± 0.025 | 3.578 ± 0.160 |
| RAF | FULL | 34.182 ± 0.234 | 9.181 ± 0.039 | 3.505 ± 0.097 |

NONE=보정 없음, CNN=CNN만, GRU=GRU만, FULL=CNN+GRU. 세 지표 모두 낮을수록 좋다. TimeNLL은 수량 RMSE로 선택한epoch의 시간 점수이며 별도의 시간 최적 checkpoint가 아니다.

## seed별 전체 수량 RMSE

| Dataset | seed | NONE | CNN | GRU | FULL | 최저 구조 |
|---|---:|---:|---:|---:|---:|---|
| Taxi | 42 | 90.515 | 84.241 | 86.377 | 86.900 | CNN |
| Taxi | 52 | 89.174 | 88.082 | 83.593 | 76.846 | FULL |
| Taxi | 62 | 88.224 | 87.628 | 83.464 | 82.022 | FULL |
| RAF | 42 | 34.353 | 34.420 | 34.077 | 34.124 | GRU |
| RAF | 52 | 34.120 | 34.364 | 34.471 | 34.440 | NONE |
| RAF | 62 | 34.199 | 34.509 | 34.085 | 33.983 | FULL |

## 수치가 말하는 범위

- Taxi: FULL의 평균 RMSE는 NONE 대비8.266%, CNN 대비5.456%, GRU 대비3.025% 낮다. NONE보다 세seed 모두 낮지만 CNN/GRU보다 seed42에서 높다. FULL의 평균 MAE는 NONE 대비9.998% 낮다. FULL TimeNLL 평균1.318은 NONE0.851 및 CNN0.782보다 높고, seed52 FULL2.164가 주요 손해다. FULL RMSE 표본SD5.027은 GRU1.646/CNN2.099/NONE1.151보다 크다.
- RAF: FULL 평균 RMSE34.182는 NONE34.224 대비0.121%, GRU34.211 대비0.084% 낮다. seed별 최저는42 GRU,52 NONE,62 FULL이다. MAE 평균은 GRU9.159가 FULL9.181보다 낮고, TimeNLL 평균은 CNN3.497이 FULL3.505보다 낮다. 큰 차이나 일관된 결합 이득으로 확대하지 않는다.
- 계약 joint_path_support: FULL이 NONE/CNN/GRU 각각보다 모든seed에서 strict개선인 dataset이 최소2개여야 한다. Taxi42와 RAF42/52의 반전으로 두dataset 모두 미충족이다. 현 선택history 값이 유지되면 Intermittent가 단독으로 이를 만족해도 두dataset 기준은 채울 수 없다. 평균 이득이 없거나 모델이 무용하다는 뜻은 아니다.

## 미확보와 주장 한계

큰 수량 RMSE·MAE·TimeNLL은 이 로컬 snapshot의 history에 없다. terminal manifest에서 endpoint_replays.json/summary.json의 원격SHA는 확인됐지만 원문은 아직 회수하지 않았다. val_tail_aux_loss/count를 큰 수량 성과로 대체하거나 val_tail_count=0을 큰 수량 대상이 없다는 뜻으로 해석하지 않는다. Binary 로컬 원본회수와 CPU/endpoint 재현도 이번에 수행하지 않았다.

활성 파라미터는 NONE89,859/CNN90,435/GRU114,591/FULL115,167이다. FULL은 단독, 나머지세arm은 동시에 실행되므로 순수용량통제 시너지 또는 공정한 구조별 실행시간 효율을 증명하는 비교가 아니다. 새 독립 평가·외부 기준선 우월성·통계유의성을 주장하지 않는다.

## 통계 오류 점검

11/11 항목을 점검했다. 아래 미확인/해당 없음은 확인 완료와 구분한다.

| 항목 | 상태 | 해석 경계 |
|---|---|---|
| Simpson | not_established | Dataset/seed-specific reversals retained alongside means. |
| Ecological fallacy | scope_limited | Aggregate errors do not establish improvement for every series. |
| Berkson | not_checked | Population-selection impact is not identified by these history files. |
| Collider | not_applicable | No conditional association or adjustment analysis. |
| Base rate neglect | not_applicable | Regression errors, not diagnostic prevalence rates. |
| Regression to mean | not_applicable | Not an extreme-group pre/post comparison. |
| Survivorship | bounded | All designated Taxi/RAF 24 fits included; not the complete36 outcome. |
| Look elsewhere | caution | Adverse seeds and TimeNLL retained; large-quantity metrics unavailable. |
| Forking paths | caution | Repeatedly viewed Validation is development evidence. |
| Correlation/causation | scope_limited | Active parameter counts and execution load differ; no capacity-controlled synergy proof. |
| Reverse causality | not_applicable | Fixed architecture comparison, no reverse-direction correlation claim. |

## 재현 상태와 후속 작업

독립 로컬 추출·선택·mean/표본SD 재계산 PASS. 새 모델 재추론은 하지 않았으므로 Verification Status는 ANALYZED이며 CPU 재현 VERIFIED가 아니다. 이후 immutable 원본과 endpoint/summary를 회수·SHA 확인하면 큰 수량 세지표까지 포함한 여섯 지표를 같은 선택epoch로 확정할 수 있다. Intermittent 진행과 현재 모니터는 그대로 유지한다.
