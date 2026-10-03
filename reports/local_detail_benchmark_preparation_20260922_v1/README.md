# 이력 보완 TitanTPP와 5개 대조 모델의 공통 조건 비교 준비

상태: **구현·합성 CPU 통합 검증·실행 계약 작성 완료. 새 GPU 실행은 미승인·미실행.**

2026-09-22 사용자가 요청한 비교 준비 기록이다. 기존 B, 이력 보완 TitanTPP,
RMTPP, THP, NHP, SAHP를 세 데이터셋에서 비교할 18개 학습을 준비했다.
이력 보완 후보는 `titantpp_local_detail`이며 시간·수량 gate를 추가하지 않는다.
현재 진행 중인 local-gate 실험, 해당 실행의 종료 상한, Scheduler는 변경하지 않았다.

## 1. 비교 조건 고정 — 완료

| 항목 | 동결 조건 |
| --- | --- |
| 모델 | B(`titantpp`), 이력 보완(`titantpp_local_detail`), RMTPP, THP, NHP, SAHP |
| 데이터 | Taxi, Intermittent, Instacart의 기존 동결 train/validation 분할 |
| 학습 수 | seed42 × 6개 모델 × 3개 데이터셋 = 18개, 모두 새 초기화 |
| 학습량 | 최대 300 epoch, batch128, AdamW(lr 0.001, weight decay 0.01), gradient clip 1.0 |
| 조기 종료 | 최소 40 epoch, validation 원단위 수량 RMSE가 40 epoch 동안 엄격히 개선되지 않으면 종료 |
| checkpoint | 유한한 validation 원단위 RMSE의 엄격한 최솟값. 동률이면 가장 이른 epoch 유지 |
| 시간 head | 현재 heteroscedastic lognormal duration head와 기록 단위의 확률을 계산하는 likelihood |
| 수량 head·목표 | 현재 `count_only_log_regression`, 시간·로그 수량 공동 목표, 수량 비중 1, raw 보조항 없음 |
| 초기화·표본 | B/이력 보완의 공통 초기 tensor 및 생성 후 난수 상태 일치. 모델별 같은 seed, 같은 batch 순서 |
| 평가 | validation만 사용. 선택 checkpoint와 마지막 checkpoint 각각 평가, 총 36회 |

시간 관측 규칙은 Taxi 시간 단위, Intermittent 주 단위, Instacart 일 단위와
30일 상단 코딩을 그대로 사용한다. 시간 scale·초기화 통계는 동결된 train 값이며
학습 전 데이터와 재계산 결과가 일치하는지 확인한다. 기존 clamp 시간 손실로
학습한 과거 benchmark300 결과를 새 비교표의 행으로 합치지 않는다.

조기 종료는 `epoch >= 40`이고 `epoch - best_epoch >= 40`인 첫 epoch에서
작동한다. 첫 최적값이 epoch 1이면 가장 빠른 정상 종료는 epoch 41이다.
**최대 300이라는 뜻이며 모든 모델을 반드시 300까지 돌린다는 뜻은 아니다.**
조기 종료로 실제 학습량은 달라질 수 있다. 공통으로 완료한 epoch까지의 batch
순서를 대조하며, 실제 종료 epoch·표본 수·optimizer step을 함께 보고한다.
최대 optimizer step 합계는 34,081,200이다.

구조가 다른 여섯 encoder의 모든 초기 tensor가 같다고 주장하지 않는다.
NHP·SAHP 등도 공통 head를 붙인 encoder 비교이므로 각 논문의 원래 전체 모델을
그대로 재현한 성능 비교와 구별한다. 이력 보완의 추가 파라미터 효과도 B와의
비교만으로 완전히 분리되지 않으므로 파라미터 수·메모리·처리 시간을 함께 남긴다.

## 2. 실행기 연결과 저장·평가 검증 — 완료

- 별도 실행기를 추가하고 기존 실행기와 진행 중인 실험 source를 보존했다.
- 57개 테스트가 통과했다. 여섯 모델의 공통 head·목표, 실제 batch 순서,
  엄격한 동률 선택·조기 종료, 모델·optimizer·난수·loader·selector 복원을 확인했다.
- 세 데이터셋 형식의 합성 데이터로 실제 학습 어댑터를 사용해 18개 조건을
  각각 40 epoch 실행하고, 선택/마지막 checkpoint 평가 36회를 검증했다.
- 별도의 CPU 비용 검증에서 여섯 모델·두 길이에 대해 총 60회 optimizer 갱신,
  B/이력 보완 초기 출력 일치, target·padding 인과성, 유한 gradient를 확인했다.
- 배포 묶음만 임시 경로에 풀어 91개 source 파일의 체크섬과 계약을 검증했다.
  이 별도 경로에서도 CLI 및 60회 합성 CPU 갱신이 정상 작동했다.
- 서버별 실제 CUDA 동작·메모리·처리 시간은 아직 검증하지 않았다.
  CPU 시간은 GPU 학습 시간으로 환산하지 않는다.

증적: [verification.json](verification.json), [tests.xml](tests.xml),
[cpu_cost_final.json](cpu_cost_final.json), [bundle_smoke.json](bundle_smoke.json).

## 3. 서버 배정과 실행 시간 — 계약 작성 완료

| 서버 | 순서와 대상 | 새 학습 수 |
| --- | --- | ---: |
| 5080 | Taxi 여섯 모델 → Intermittent 여섯 모델 | 12 |
| 5090 | Instacart 여섯 모델 | 6 |

각 데이터셋 안에서는 B → 이력 보완 → RMTPP → THP → NHP → SAHP 순서다.
동일 데이터셋의 모델을 같은 서버에서 비교해 서버 환경 차이가 그 비교에
섞이지 않도록 했다. 양쪽 서버는 병렬 실행하고 각 서버의 해당 실험 worker는
하나만 둔다. 외부 GPU 작업은 관찰하며 종료하지 않는다. 시작 시 가용 VRAM은
최소 2 GiB, 디스크 여유 공간은 최소 20 GiB여야 한다.

**2026-09-22 21:03 KST 읽기 전용 점검**에서 두 서버의 Runtime·CUDA 라이브러리·
데이터 체크섬이 계약과 일치했고 새 실행 경로는 없었다. 5080의 기존 실험은
완료되어 해당 worker/tmux가 없었다. 5090에서는 기존 Instacart 수량 gate가
14/120 epoch, 217,798 step으로 진행 중이었다. 새 실행 시 이 상태를 다시 확인한다.
서버별 기존 실험의 정상 종료와 worker/tmux 종료를 확인하기 전에는 새 작업을 시작하지 않는다.

### 시간 산정의 근거와 불확실성

아래는 과거 모델별 실측 epoch 시간과 현재 시간 head를 사용하는 인접 Titan
실험의 epoch 시간을 이용한 **가정별 계산**이다. 고정 비용을 더하는 방식과
비율을 곱하는 방식의 두 계산 결과 범위이며, 신뢰구간이나 새 실행의 실측 ETA가 아니다.
배포·대기·CUDA 검증·마지막 재평가·디스크 지연은 제외했다.

| 모든 조건이 종료되는 epoch에 대한 가정 | 5080 학습 합계 | 5090 학습 합계 | 양쪽이 함께 시작할 때 전체 시간 |
| --- | ---: | ---: | ---: |
| 모두 100 epoch에 종료 | 약 21~24시간 | 약 55~67시간 | 약 55~67시간 |
| 모두 300 epoch 수행 | 약 64~71시간 | 약 165~202시간 | 약 165~202시간 |

조기 종료 시점은 아직 알 수 없고, 새 head를 적용한 benchmark의 실제 GPU 비용도
미측정이다. 특히 Instacart NHP 비용이 크다. 원본과 계산은
[cost_estimate.json](cost_estimate.json)에 있으며
[reproduce_cost_estimate.py](reproduce_cost_estimate.py)로 재계산할 수 있다.
승인 후 CUDA 검증 및 실제 완료 epoch 시간으로 추정을 갱신하되 계약의 상한은 자동으로 늘리지 않는다.

### 새 비교에 제안한 자원·시간 상한

- 공통 종료 상한: 첫 CUDA 검증 직전에 생성할 **단일 start permit부터 240시간(10일)**.
- 개별 조건: 학습과 선택/마지막 평가를 합쳐 최대 120시간.
- CUDA 사전 검증: 서버별 최대 1,800초, 합성 batch128·길이64/256·총 60회 갱신.
- 동시에 서버당 이 실험 worker 1개, 두 서버 합계 GPU 시간의 최대치 480시간.
- 서버별 산출물 16 GiB, 개별 파일 64 MiB 상한.

**240시간은 예상 소요시간이 아니라 새 실행의 종료 상한이다.** 현재는 시작
permit이 없어 시계가 시작되지 않았다. 기존 local-gate 실험의 9월 24일 08:15 KST
종료 상한은 변경하지 않는다. 상한 초과·오류 시 이 실험이 소유한 worker만
종료하고 증적을 보존한다. 자동 retry/resume, 서버 간 이관, 추가 seed는 허용하지 않는다.

## 4. 실행 승인 후 절차 — 승인 필요

1. 아래 계약과 두 서버의 CUDA 검증·18개 비교 학습에 대한 사용자 승인을 기록한다.
   승인 JSON은 `approved: true`, 계약 canonical SHA, 두 host, 실제 사용자 지시를
   포함하며, 승인 전에는 작성하지 않는다.
2. 가용성·기존 실험 종료를 다시 확인한 뒤 고정 묶음을 두 서버의 새 경로에만 배치한다.
   기존 파일을 삭제하거나 공용 Runtime을 수정하지 않는다. 배포 전후 묶음과 source
   체크섬을 대조하고 동결 Python·환경·데이터 identity를 확인한다.
3. `start-permit` 명령으로 첫 CUDA 검증 직전에 공통 permit을 한 번 생성한다.
   양쪽 `qualify`에 같은 permit을 사용한다. 두 서버의 native CUDA 검증이 모두
   통과해야 하며, 실패하면 학습을 시작하거나 자동 재실행하지 않는다.
4. 양쪽 검증 receipt를 `training-permit`으로 결합한다. 원래 시작·종료 시각을
   유지하며 두 서버 각각 `train`을 한 번 실행한다. 각 host의 정확한 Python,
   새 root, tmux 경로는 계약의 `hosts` 항목을 사용한다.
5. 실제 완료 epoch 기반으로 진행을 기록한다. 종료 후 18개 조건의 조기 종료
   규칙·실제 step·공통 batch 구간·평가 36회·checkpoint identity를 감사한다.

실행 진입점은 `paper/scripts/run_local_detail_benchmark.py`다. 공통 인자는
`--contract`, `--approval`, `--permit`, 서버 실행 인자는 `--host`다.
`start-permit`에는 `--output`, `training-permit`에는 `--output`과
`--qualification-5080`, `--qualification-5090`을 지정한다.
명령 예시는 실행 승인이 아니다. 실행기 자체도 source/승인/permit identity,
현재 시간, Runtime, 양쪽 CUDA 검증 증적을 검사한다.

## 5. 해석 기준과 후속 순서

**공통 조건의 비교 학습 — 승인 필요 / 서버 간 병렬 가능**

- 현재 후보·head·목표를 유지한 seed42 validation 비교다. 준비 검증 통과는 성능 개선 증거가 아니다.
- 양쪽 CUDA 검증 이후 5080과 5090을 병렬로 사용한다. 5090은 기존 실험 완료를 기다린다.

**전체·구간별 성능과 비용 통합 — 학습 완료 후 다음 작업**

- 모든 선택 지표는 같은 raw-RMSE checkpoint에서 보고한다. 시간 최적값을 별도
  checkpoint에서 가져와 수량 최적값과 합치지 않는다.
- 이력 보완 대 B가 주 비교이며 네 benchmark 대비 결과도 각각 보고한다.
  전체 RMSE 개선, 전체 MAE 비율 ≤1.01, body/tail MAE 비율 ≤1.02,
  비어 있지 않은 중간 구간 MAE/RMSE 비율 ≤1.02, tail RMSE 개선,
  기록 시간 NLL 증가 ≤0.01, 마지막 30 epoch RMSE 평균·표본 표준편차 비율
  ≤1.05의 기존 기준을 그대로 남긴다. 기준별 통과·실패를 숨기거나 사후 완화하지 않는다.
- 서로 다른 조기 종료 길이에서 마지막 30 epoch의 위치도 달라진다. 고정된
  처음 40 epoch 통계와 전체 history를 함께 제공하고 계산량 차이를 밝힌다.
- 특정 데이터 특성에서 어떤 이력 정보가 도움이 됐는지 논의할 수 있지만,
  단일 seed validation으로 보편적 우월성이나 논문의 최종 성능을 확정하지 않는다.

## 고정된 산출물

- 계약: [local_detail_benchmark_observed_time_v1.json](../../paper/contracts/local_detail_benchmark_observed_time_v1.json)
- 배포 묶음: [source_bundle.tar.gz](../../search_artifacts/local_detail_benchmark_preparation_20260922_v1/source_bundle.tar.gz)
- 준비 receipt: [preparation_receipt.json](../../search_artifacts/local_detail_benchmark_preparation_20260922_v1/preparation_receipt.json)
- 서버 사전 점검: [host_readiness.json](host_readiness.json)
- 계약 canonical SHA256: `229bc1a5326934c7b37e77feaa79619241dbd979abef30c4e9147f65198ef738`
- source closure SHA256: `6c92e1790c8cac8016ae40e0c398be0cf2d93a38f444b951926d92489daa623a`
- 묶음 SHA256: `b6cdb72672a43763ede7f46aead59b4cb2c824d275c275f684ecd6e4fc7e97ae`

이번 작업에서는 새 GPU workload·held-out 평가·원격 배포·Scheduler 변경·커밋·Push·외부 게시를 수행하지 않았다.
