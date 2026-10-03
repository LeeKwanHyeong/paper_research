# TitanTPP B: 9조건 원본 연결

## 확인 범위 — 완료

기존 B는 `titantpp`이며 인접 상태 보정 MLP가 없는 static Hard-LMM 모델이다. Taxi·Intermittent·Instacart의 seed42/52/62 selected checkpoint 9개를 로컬에서 확인했다. 새 학습·추론·원격 조회는 수행하지 않았다.

9개 모두 기존 원격 수집 manifest의 파일 SHA와 현재 로컬 SHA가 일치한다. CPU에서 역직렬화하여 tensor SHA·모델 키·seed·선택 epoch·encoder/interface metadata를 원래 validation summary 및 endpoint replay와 대조했다. 기존 baseline_audit의 strict model load 감사는 재사용했고, 이번에 새 model 생성/load 또는 validation replay를 수행한 것으로 표시하지 않는다.

| 데이터 | seed | 선택 epoch | Validation RMSE | 원본 수집 서버 |
|---|---:|---:|---:|---|
| yellow_trip_hourly | 42 | 46 | 90.89626422 | 5080 |
| yellow_trip_hourly | 52 | 19 | 89.91723961 | 5080 |
| yellow_trip_hourly | 62 | 27 | 90.71428885 | 5080 |
| intermittent_frozen_5000 | 42 | 71 | 1.75560079 | 5080 |
| intermittent_frozen_5000 | 52 | 11 | 1.83127207 | 5080 |
| intermittent_frozen_5000 | 62 | 43 | 1.75481737 | 5080 |
| insta_market_basket | 42 | 30 | 5.86320170 | 5090 |
| insta_market_basket | 52 | 15 | 5.87921920 | 5090 |
| insta_market_basket | 62 | 35 | 5.89629690 | 5090 |

## 재사용할 source와 데이터 계약 — 완료

기존 108조건 평가 registry에 이미 있는 동일 과학 계약의 frozen source bundle 5개를 재사용할 수 있다. 계약 canonical SHA, source closure, source 파일 전체 SHA를 다시 확인했다. 로더·학습 통계·관측 시간 law·원자료 identity·원래 validation 지표를 조건별 인벤토리에 보존했다.

로컬 checkpoint는 `search_artifacts/titantpp_core_ablation_20260928_v1/baselines/{5080,5090}/{dataset}/seed_{42,52,62}/titantpp/best_val_qty_rmse_model.pt`에 있다. 원래 원격 경로와 수집 manifest 경로는 JSON에 별도 기록했다. 원격 파일의 현재 존속을 새로 확인한 것은 아니다.

**Instacart seed62 B는 원래 5090 replication 캠페인 소속**이다. RMTPP seed62의 remaining-model recovery bundle을 따라 연결하면 안 된다. 원래 B의 source/계약 SHA와 일치하는 `local_detail_replication_5090_20260924_v1`을 사용한다.

## RAF 제외 — 범위 확정

RAF B의 과거 3seed checkpoint는 있으나 legacy clamped RMTPP 시간 head와 joint-objective 선택을 사용한다. 현재 observed-time lognormal 시간 head와 validation raw-RMSE 선택 조건으로 학습한 B와 같지 않으므로 이번 9조건에 넣지 않는다. 20261001 RAF 캠페인에도 기본 B는 없었다.

## 남은 작업 — 다음 작업

평가 실행 담당이 이 인벤토리로 실행 계약을 고정하고 원래 source를 통한 strict load 및 허용된 validation 재현을 검증한다. 이후 사용자 승인 범위의 기존 Test 분할 평가를 수행한다. 기존 Test 결과를 보고 추가한 탐색적 B 비교라는 이력을 보존하고 기존 MLP 결과를 대체하거나 삭제하지 않는다.
