# Shared15 논문 재현 패키지 범위 감사

2026-10-21 / 읽기 전용 코드·문서 감사. 모델 import·checkpoint load·학습·평가·SSH·Git 변경·실험 파일 삭제는 하지 않았다. 작업 산출물은 이 파일 하나다. 새 checkout에서 재현 실행을 검증한 보고서가 아니다.

## 재현 패키지의 목적과 기존 기준 — 확인 완료

재현 패키지는 실험 증거를 지우는 작업이 아니라, 독자가 어떤 모델·데이터·설정·진입점을 따라야 같은 실험을 구성할 수 있는지 분명히 하는 작업이다. 현재 연구 저장소와 공개 배포 패키지의 파일 수가 같아야 할 필요는 없으나, 동결 실행본을 공개 패키지 모양에 맞춰 수정하면 안 된다.

- `TEST_SESSION_PROTOCOL.md:14–69`: 현재 코드와 과거 source revision을 구분하고 원본 결과·checkpoint는 보존한다.
- `DEVELOPMENT.md:8–18,34–43`: 개발 모델/운영 도구/실험 원본/보고서의 역할이 다르다. SHA로 연결된 원본은 이동·이름 변경·일괄 수정하지 않는다.
- `reports/repository_cleanup_20261016_v1/README.md:22–38`: 2026-10-16 정리는 코드·보고서 인덱스와 보존 커밋이지 과거 후보 코드 삭제가 아니었다. 보류된 Moment CNN도 보존했다.
- `paper/reproducibility/README.md:3–21`: 2026-10-04 상태를 담은 역사적 인덱스다. 당시 MLP16 기준선이나 진행 상태를 현재 Shared15 기준선으로 읽으면 안 된다. snapshot manifest/objects는 바이트 복원 자료이며 SHA 목록은 데이터 백업이 아니다.
- 루트 `README.md`의 v0.7은 별도 역사적 validation freeze다. 현재 Shared15 논문의 유일한 실행 안내로 쓰기에는 현재 조건과 차이가 있다. 이번 감사에서 루트 README를 읽으며 역사적 v0.7 표 일부가 함께 보였지만 현재 실험 metric 파일은 열지 않았고 수치는 분석하지 않았다.

## 현재 Shared15 과학 경로를 설명한다 — 확인 완료

| 목적 | 현재 필요한 파일과 근거 | 재현 패키지에서 분명히 할 내용 |
|---|---|---|
| 모델 ID·이력 보정 | `models/TPPs/CountAwareTitanSharedHistoryMLP.py:93–163` | `titantpp_history_mlp_shared15_v1`; 공통128→15 GELU 입력 변환, 분기별15→64 출력8개, 분모8, 가용 문턱과 즉시 직전 관측 source, zero-output 초기화 |
| 상속된 전체 경로 | `models/TPPs/CountAwareTitanCoreAblation.py:43–84`, `models/TPPs/CountAwareTitanMultiLagDetail.py:77–114,209–238` | shared correction이 Encoder1과 Encoder2 사이에 들어간다. 두 attention block과 후단 static Hard-LMM을 유지한다. source는 불규칙 관측 순서 기준이며 padding skip·withheld reset을 보존한다. |
| 공통 입력·두 head·손실 | `models/TPPs/CountAwareTPP.py:114,278–408,435–463,775`, `models/TPPs/positive_integer_time.py` | 관측 gap/quantity log1p, quantity softplus-log location→expm1, heteroscedastic lognormal duration 및 recorded positive-integer likelihood. 구식 v0.7 clamped head와 섞지 않는다. |
| attention·메모리 | `models/Titan/backbone.py:29–91,103`, `models/Titan/common/memory.py`, `models/Titan/common/configs.py`, `models/Titan/__init__.py` | hidden64·두 block·persistent16·static prototype64/top4 등 정확한 contract. 테스트 시 온라인 가중치 갱신이 없는 현재 구조와 원 Titans-MAC를 구분한다. `CountAwareTPP.py`가 import하는 key_value_memory/elapsed_age/titans_mac/tpp_gated_memory도 현재 import 의존성을 고려한다. |
| next-event target 차단 | `paper/scripts/count_aware_tpp_backbone/core.py:24–107` | 목표 수량0 처리와 memory_write_mask로 목표행 쓰기 차단, 현재 관측 마지막 state에서 다음 사건 예측. 단순 [B,T,C] 밀집 시계열로 재작성하지 않는다. |
| 학습 엔진 | `paper/scripts/count_aware_tpp_backbone/training.py:15–49,52–72,638–1045`, `observed_time.py`, `datasets.py`, `constants.py` | AdamW·clip·early stop·Validation 전수 평가·선택 binary/RNG/history 보존을 포함한다. core의 기본 joint selector가 아닌 worker가 설정한 raw quantity RMSE selector가 현재 기준이다. |
| loader·seed | `paper/scripts/run_taxi_quantity_interface_ablation.py:175–184,289–315`, `data_loader/event_seq_data_module.py:222–399`, `simple_lab_test/search/common/runner.py` | random/NumPy/torch/CUDA seed와 dedicated train generator, 정렬·chronological_split·lookback·max_seq_len·target split을 함께 재현해야 한다. 이들 helper도 현재는 과거 이름의 스크립트에 의존한다. |

위 목록은 사람이 이해하는 핵심 경로다. 실제 배포 가능 파일 전체의 완전한 최소 import closure로 검증한 목록은 아니다. 현재 동결 source983/987개를 이 표만 보고 지워서는 안 된다.

## 원본 데이터와 분할을 고정한다 — 확인 완료 / 공개권한 확인은 별도

필요한 데이터 문서는 `benchmark_data/README.md`, `benchmark_data/contracts/raf_spare_parts_v1.json`, `paper/scripts/count_aware_tpp_backbone/datasets.py`, 그리고 현재 campaign의 `SCIENCE_SPEC.json`·`FROZEN_CONDITIONS.json`·원 실행 `execution_contract.json`이다. generic dataset registry보다 해당 캠페인의 실제 입력 SHA·정확 target population·loader 조건을 우선한다.

- RAF: `benchmark_data/scripts/audit_and_prepare_raf.py`, `verify_raf_model_input.py`, `common.py`와 계약/manifest를 연결한다. 월 단위 원자료→공통 event table→고정 chronological split 과정을 명시한다. 원 workbook 재배포 권한은 확인되지 않았으며 `benchmark_data/README.md`도 redistribution not cleared로 명시한다. 공개 코드에 raw workbook을 넣는 것으로 재현성을 해결하면 안 된다.
- Intermittent: `paper/scripts/build_intermittent_frozen_subset.py`와 `simple_lab_test/notebooks/preprocessing/tpp_split_utils.py`, 원본 접근/동결 5000 선정·split manifest를 연결한다. 파일명이 비슷한 v2 registry와 기존 frozen5000을 자동으로 같은 데이터라고 취급하지 않는다.
- Taxi: `benchmark_data/scripts/audit_auxiliary_frozen.py`, `verify_benchmark_model_inputs.py`는 기존 auxiliary frozen split의 SHA/대상 검증 경로다. 원 pickup 원자료 취득/집계 경로와 원자료 권한은 별도 provenance가 필요하다. 이 두 감사 스크립트만으로 raw Taxi 전체 생성 재현이 완성됐다고 주장하지 않는다.
- admitted Train/Validation 적재: `paper/scripts/run_multilag_detail_execution.py:565–588`→`paper/scripts/quantity_comparison_data.py` 및 `run_quantity_comparison.py`의 frozen statistics binding. core의 `load_train_validation_frame`은 Test 행을 필터 후 materialize한다.
- 현재 Shared15 실험은 `original_uncapped`, 시간·수량 log1p, time_alpha1이다(`run_titantpp_shared_history_raw9.py:56–62`). `benchmark_data/Cap`은 별도 연구 데이터 변형이다. 공개 재현 기본값으로 조용히 대체하지 않는다.
- 원자료·derived parquet·checkpoint·개별 예측은 기존 Git ignore 정책과 접근권한을 유지한다. 공개 패키지는 다운로드/준비 안내, Train 기준 선정규칙·split·파일 SHA·schema를 제공하고 적법하게 배포 가능한 자료만 포함한다.

## 현재 학습·평가 진입점을 연결한다 — 확인 완료

**기존3seed 학습:**
`paper/scripts/run_titantpp_shared_history_raw9_5090.py`는 독립 새 모델 구현이 아니다. `run_titantpp_shared_history_raw9.py`의 전체 바이트 SHA를 먼저 검증하고 host/schema 문자열을 정확히 바꿔 실행한다. raw9는 `run_titantpp_shared_history_cap.py`와 `run_titantpp_mlp_cap36.py` helper를 재사용한다. Cap라는 파일명만 보고 현재 모델의 데이터가 Cap이라고 설명하면 틀린다.

- 원본9 설계·설정: `reports/titantpp_history_mlp_shared_raw9_20261020_v1/SCIENCE_SPEC.json`, `execution_5090_v1/SCIENCE_SPEC.json`, `EXPERIMENT_PLAN.md`, `PREPARED.json`와 승인된 bundle 계약. 이 감사에서는 계약의 구조와 코드 조건만 확인하고 현재 결과표를 읽지 않았다.
- 현재 seed42/52/62, 공통3데이터, log 수량MSE+시간NLL coefficient1, hidden64, batch128, AdamW lr.001/wd.01, clip1, maximum300/minimum40/patience40 조건은 `run_titantpp_shared_history_raw9.py:302–365`에서 강제한다.
- 선택: `run_titantpp_shared_history_cap.py:272–283`가 full-population raw quantity RMSE의 최초 strict finite minimum을 선택한다. 동률이면 첫 epoch다. MAE/시간 지표는 같은 checkpoint에서 산출한다. Test 최솟값으로 epoch·seed·모델을 다시 선택하지 않는다.
- 공통 seed/RNG·초기 tensor 보존/소스 SHA·입력 SHA·checkpoint tensor digest와 binary SHA·epoch·정확 target identity를 함께 내보내야 한다. seed만 같으면 재현된다고 보장하지 않는다.

**현재 추가2seed:**
`paper/scripts/run_titantpp_shared_external_extra42.py`는 Shared15+외부6종×3데이터×seed72/82, 총42개의 별도 worker이다. 원본 worker·준비 helper·Shared 모델 SHA를 하드코딩 검증한다. `prepare_titantpp_shared_external_extra42.py`, `titantpp_shared_external_extra42_runtime.py`, `manage_titantpp_shared_external_extra42.py`, `advance_titantpp_shared_external_extra42_recovery_v2.py`는 각기 계약/dispatcher/회수/짧은 종료 확인 역할이다. 논문용 새 단순 진입점과 역사적 서버 운영 제어를 같은 파일로 덮어쓰지 않는다.

**기존3seed 고정평가:**
`evaluate_titantpp_shared_raw_raf3_frozen.py`→`evaluate_titantpp_shared_raf_taxi6_frozen.py`→`evaluate_titantpp_shared_inter3_interim_frozen.py`는 검토된 앞 파일을 바이트 SHA로 연결하는 evaluator 계보다. 각각 허용 scope와 고정 selection binding이 다르다. 현재 소스명을 하나로 바꾸거나 원 evaluator를 정리하면 뒤 adapter도 깨진다. 결과가 확정된 뒤 public release에서는 원 설정·weights와 동등한 새 평가 진입점을 별도 버전으로 만들 수 있으나, 원 evaluator와 원 결과는 그대로 둔다.

**Validation/Test 순서:**
추가42 runtime은 Train42→Validation42 전수 재현/실제 exit0 gate→같은 고정 checkpoint Test42 순서다(`titantpp_shared_external_extra42_runtime.py:108–115,244–260`). `run_titantpp_shared_history_raw9.py:490–496`는 Test route 자체를 금지한다. 별도 승인된 고정평가 worker가 Test를 처리한다. 공개판도 split 접근 및 선택 단계가 명시돼야 한다. 현재 사용자 해석 정정(`reports/titantpp_shared_paper_interpretation_correction_20261020_v1/CORRECTION.md`)을 존중하고 근거 없이 Shared15가 Test를 사용해 학습·선택됐다고 쓰지 않는다.

**결과 집계:**
`paper/scripts/compare_titantpp_shared_external_five_seed.py`를 별도 결과 고정 경로의 집계 진입점으로 설명한다. 기존3seed를 덮어쓰지 않고 source/target/seed/checkpoint/worker exit/원본 SHA가 검증된 추가2seed를 연결한다. 집계 코드와 실제 model inference를 구분한다. 이 감사에서는 비교기 결과metric을 열거나 재계산하지 않았다.

## 외부 비교군과 MAC 대조 의존성을 보존한다 — 확인 완료

공통 head 비교에는 아래 파일과 upstream license/고정 revision 근거가 필요하다.

- `models/TPPs/CountAwareTPP.py`: `CountAwareRMTPP`·`CountAwareTHP` 및 공통 head.
- `models/TPPs/NeuralHawkesTPP.py`: CountAwareNHP.
- `models/TPPs/SelfAttentiveHawkesTPP.py`: CountAwareSAHP.
- `models/TPPs/CountAwareAdditionalTPP.py`: S2P2/AttNHP 공통head adapter. 원 논문 native marked NLL 재현이라고 부르지 않는다는 모듈 설명과 custom affine scan 범위를 유지한다.
- `models/TPPs/vendor/s2p2/{models.py,initializers.py,__init__.py,LICENCE,NOTICE}`; `models/TPPs/vendor/attnhp/{xfmr.py,__init__.py,LICENSE}`. adapter만 남기고 vendor/license를 제거하면 안 된다.
- `models/Titan/common/titans_mac.py`와 MAC event adapter/해당 campaign 실행 계약은 MAC 대조 기여를 재현하는 경로다. 진행 중 `reports/titans_mac_accuracy9_20261020_v1`와 source/checkpoint/result 원본은 별도 증거로 그대로 둔다.
- 추가42 worker가 외부모델에 `CountAwareFactory.build_count_aware_model`을 그대로 쓰며, Shared15만 별도 후보 factory hook을 사용한다(`run_titantpp_shared_external_extra42.py:93–114`). 각 backbone의 role·state identity 검증도 필요하다.

## legacy 파일을 지금 단순 삭제하면 안 되는 이유 — 확인 완료

AST만 읽어 확인한 `models/TPPs/CountAwareFactory.py`의 최상위 `models.TPPs.*` import는24개다. Shared raw9 worker는 이 파일의 `with_time_metadata`를 가져오며 학습 엔진도 이 factory를 최상위 import한다. 따라서 선택하지 않은 모델 파일도 Python import를 완료하는 데 필요할 수 있다.

직접 import되는 예: InterLayerMemory45행, MemoryFiLM51행, CausalQKV57행, BoundedQK63행, LevelHistoryQKV69행, ValueNorm75행, SlotMemory81행, SuccessorMemory87행, NonlinearEpisodeMemory95행, TaskRoutedMemory103행, PairMessage111행, StateTransport116행, LocalGate120행, CoreAblation121행, ActiveBranchNorm122행, MLPCandidates123행, HistoryControls124행, DeepRenewalEvent125행, MultiLagDetail126행. 이 목록은 단순 삭제 위험을 보이는 좁은 근거이며 모든 저장소 import closure를 완전히 열거한 결과는 아니다.

앞으로 public release에서 선택 모델만 lazy import하는 factory나 독립의 작은 factory를 만들 수 있지만 이는 import 구조 변경이다. 별도 release 경로·동등성 검사·clean checkout import/합성검사를 거쳐야 하며 기존 봉인 실행 source를 편집해서 달성하면 안 된다. 학습 source의 byte SHA와 현재 source가 같다는 기존 보장은 그 순간 깨진다.

## Runtime·requirements·shell 진입점을 정리한다 — 다음 작업

- 현재 tracked 개발 파일 검색 범위에서 root `requirements*.txt`, `pyproject.toml`, conda env lock 또는 Shared15 전용 requirements 한 벌을 찾지 못했다. 이는 저장소 모든 ignored 환경파일의 부재를 증명하지 않는다.
- 실제 최소 import 경로에는 torch, NumPy, Polars가 있고 `simple_lab_test/search/common/runner.py`의 최상위 matplotlib import도 따라온다. upstream adapter까지 포함한 완전한 dependency/version lock은 별도 clean 환경 검사로 확정해야 한다. 임의 최신버전을 pinning하지 않는다.
- 추가42가 강제하는 Runtime은 Python3.12.13 / torch2.11.0+cu130 / CUDA13.0 / cuDNN91900 / intra4·interop1이다(`titantpp_shared_external_extra42_runtime.py:64–67`). 기존 Shared3seed 학습5090의 cuDNN92000과 평가/추가학습5080의91900 등 실행 provenance를 기록한다. 서로 다른 GPU를 같은 Runtime이라고 줄여 쓰지 않는다.
- 현재 핵심 orchestration은 Python `-m paper.scripts...`이고 host-only adapter 및 실제 계약을 필요로 한다. 저장소 `paper/scripts`·`scripts` 검색에서 Shared15 전용 간단 shell wrapper는 확인되지 않았다. 과거 `paper/results/.../run_vnc_instacart_seed42_e300.sh`는 현재 Shared15 재현 진입점이 아니다.
- 향후 공개판에는 dataset/model/seed/config/output을 받는 작은 train/evaluate/aggregate 진입점과 명확한 CLI 예시를 제공하는 편이 좋다. 이들은 아직 생성·검증하지 않았으며, 기존 approved server 명령을 그대로 public 사용자에게 복붙하도록 권하지 않는다. SSH host, permit, 소유PID, 절대기한은 당시 운용 증거이고 새 실행 허가나 과학적 hyperparameter가 아니다.

## 시점을 나눠 정리한다 — 남은 순서

1. **현재 저장소 반영과 안내 보강 — 다음 작업:** 사용자가 승인한 develop 반영 범위에서 작업 파일을 보존하고 현재 Shared15·외부비교·MAC·효율성의 authoritative 계약/진입점 인덱스를 만든다. 미완료 실험을 완료로 표시하지 않는다. 이 단계는 삭제 없이 수행할 수 있다.
2. **실험 종료·회수·결과 고정 — 외부 작업 대기:** 5080 추가42와5090 MAC9의 기존 source/Runtime/학습/선택/checkpoint/완료/실패/자동회수 원본을 보존한다. 이 감사는 진행 상태를 새로 조회하지 않았다. 원본·재현·종료 증거가 갖춰진 결과를 고정하고 논문 표와 연결한다.
3. **투고 전 재현 실행 경로 검증 — 다음 작업:** 별도 새 경로에서 config·dependency pin·data rights/provenance·train/evaluate/aggregate 사용법과 작은 합성/clean checkout 동작을 검증한다. 과학적 설정과 target/selection을 바꾸지 않는지 확인한다. 기존 GPU 실험 재실행이나 결과교체를 자동 포함하지 않는다.
4. **public release 축소 — 결과 고정 후 별도 작업:** 정해진 논문 범위에 필요한 독립 dependency closure와 허용 데이터/체크포인트/라이선스만 배포본에 포함한다. 원 저장소·태그·frozen snapshot·실패이력·연구 증거는 보관한다. legacy 이동/삭제나 import refactor는 사용자가 실제 삭제 범위를 정하고 별도 검증할 때 진행한다.

공개판의 최소 파일 수와 연구 저장소의 전체 보존량은 다른 문제다. 지금 할 수 있는 가장 안전한 정리는 명확한 입구·계약·출처를 제공하는 것이고, 과거 실험 파일을 다 지워서 한 모델만 남기는 작업은 아니다.
