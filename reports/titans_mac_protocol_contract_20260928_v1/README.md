# MAC를 현재 공통 프로토콜에 연결하는 비교 계약

작성일: 2026-09-28. 대상: `paper_research / codex/hard-lmm-causal-qkv`.

후속 기록: 같은 날 사용자 승인 범위에서 별도 소스의 wrapper·전용 역할 구현과 CPU 합성 검증·동결을 완료했다. 현재 준비 상태는 [MAC 실행 경로 구현 기록](../titans_mac_execution_preparation_20260928_v1/README.md)을 따른다. 아래 설계 검토 시점의 기록과 설계 계약 SHA는 보존한다.

**계약 준비와 로컬 근거 검증을 완료했다. 현재 학습의 종료를 기다리는 동안 독립적으로 준비할 수 있는 작업이다. 새 학습·배포는 승인하지 않았고 실행하지 않았다.**

이 문서와 [설계 계약 JSON](../../paper/contracts/titans_mac_observed_time_comparison_v1.json)은 검토 가능한 과학적·운영적 설계다. 새 production wrapper, 최종 소스 묶음, 실행 계약 SHA, native CUDA 검증은 아직 없다. 설계 파일을 기존 runner에 넣어 실행해서는 안 된다. `approved=false`, 시작·종료 시각은 `null`이다.

## 비교 목적과 해석 범위 — 확정

검증할 질문은 **“동일한 관측 이력·시간/수량 head·학습 및 선택 규칙에서, 현재 TitanTPP 계열과 온라인 neural memory를 쓰는 Titans-MAC의 성능·비용은 어떻게 다른가?”**다.

- 주 비교는 **B 대 MAC**다. Full 대 MAC, 이력 MLP 대 MAC도 사전에 함께 정한다. core 결과를 보고 유리한 하나만 골라 비교하지 않는다.
- MAC의 기존 코드와 clip1 안정화 정책을 재사용한다. 모델명은 **공통 head를 사용한 Titans-MAC 이벤트 적응 모델, inner-clip1**로 표기한다. 원 논문의 설정을 그대로 재현했다거나 clipping까지 원본 메커니즘이라고 쓰지 않는다.
- MAC는 구간 단위 attention, neural memory, 파라미터 수가 현재 B와 다르다. 따라서 이 비교는 **백본 구성 전체의 대조**다. 온라인 메모리 한 요소의 인과적 기여나 메모리와 Full의 상호작용을 단독 입증하지 않는다.
- MAC가 이겨도 Full을 자동 폐기하거나, MAC가 져도 현재 모델의 논문 기여를 자동 확정하지 않는다. 수량·시간·구간별 반례와 비용을 함께 보고한다.

기존 결과와의 관계는 [재사용 감사 보고서](../titans_mac_reuse_audit_20260928_v1/report.md)에 근거한다. 과거 MAC의 legacy 시간 손실·joint 선택 결과와 새로운 관측 시간 NLL·raw-RMSE 선택 결과를 한 3-seed 표에 섞지 않는다. 과거 실패도 보존한다.

## 실행 수와 재사용 범위 — 확정, 실행 승인 전

| 서버 제안 | 데이터 | seed | 새 학습 | selected/last 재평가 | 실행 순서 |
|---|---|---|---:|---:|---|
| 5080 | Taxi | 42, 52, 62 | 3 | 6 | 먼저 수행 |
| 5080 | Intermittent | 42, 52, 62 | 3 | 6 | Taxi 종료 후 |
| 5090 | Instacart | 42, 52, 62 | 3 | 6 | 독립 수행 |
| 합계 | 3개 | 3개 | **9** | **18** | 서버별 한 작업 |

[9조건 실행표](execution_matrix.csv). 같은 checkpoint가 selected와 last여도 평가 역할 두 개를 기록한다. 새 9조건은 모두 처음부터 학습하며 과거 MAC checkpoint로 warm start하지 않는다.

현재 B·Full 18조건, RMTPP·THP·NHP·SAHP 36조건을 재학습하지 않는다. 진행 중인 core의 새 36조건은 기존 계약으로 완료·감사한 뒤 연결한다. 모두 완료됐을 때 고유 조건은 `18 + 36 + 36 + 9 = 99`, endpoint 기록은 198이다. 현재 core의 표 54개에 B·Full 18개를 다시 더하지 않는다. 이것은 조건 수이며 통계적으로 독립적인 데이터셋 수가 아니다.

## 공통 비교 규칙 — 확정

| 항목 | 고정할 내용 |
|---|---|
| 입력 | 기존 시간·수량 관측 이력. target 수량 및 미래 이력 접근 금지 |
| 데이터 | core 계약의 parquet/split SHA, target identity·수량·시간 hash, train/validation 표본 수 그대로 |
| loader | batch128, train shuffle, validation 고정 순서, drop_last=false, worker0 |
| head | `heteroscedastic_lognormal_duration` + `positive_integer_round_clamp_v1` 관측 확률질량 |
| 수량 손실 | 기존 `count_only_log_regression`, log1p 수량 MSE, 가중치1, tail 보조손실0 |
| 학습 | AdamW lr0.001, wd0.01, 외부 gradient clip1, AMP 없음 |
| 선택/종료 | validation raw 수량 RMSE의 earliest strict finite minimum, max300/min40/patience40 |
| 초기화 | 같은 seed와 고정 train 통계. 공통 head tensor 동일성을 서버에서 검증. 다른 encoder의 전체 tensor/RNG 소비까지 동일하다고 주장하지 않음 |
| 배치 비교 | 실제 배치 hash를 B/Full과 겹치는 epoch 전체에서 대조. MAC 때문에 배치를 버리거나 작게 바꾸지 않음 |
| 재평가 | 같은 raw-RMSE-selected checkpoint의 모든 수량·시간·구간 지표, last는 별도 기록 |
| 튜닝/평가 | HPO0, validation-only, held-out 잠금 유지 |

| 데이터 | 최대 sequence / lookback | 시간 scale / 단위 / 상한 | train / validation target | epoch당 optimizer step |
|---|---|---|---:|---:|
| Taxi | 256 / 168 | 1 / hour / 없음 | 38,393 / 8,268 | 300 |
| Intermittent | 256 / 520 | 3 / week / 없음 | 393,824 / 86,285 | 3,077 |
| Instacart | 64 / 52 | 7 / day / 30 | 1,991,192 / 503,733 | 15,557 |

최대 sequence 길이에 target이 포함된다. JSON에 기존 데이터 통계와 SHA를 복사해 잠갔으며, 이번 작업에서 실제 데이터 파일이나 held-out 성능을 열지 않았다. 손실과 loader를 맞춰도 모델별 파라미터 수·GPU시간까지 같은 실험은 아니다.

## MAC에서 유지할 메커니즘 — 확정

- hidden64, 2 layers, 4 heads, FF128, dropout0.1, persistent tokens16, segment16. 정적 Hard-LMM 및 Full/MLP 보완 모듈을 추가하지 않는다.
- 각 sample/window에서 새 neural memory 상태로 시작한다. batch·epoch·train/validation 사이에 온라인 상태를 이어붙이지 않는다. 같은 series라도 별도 window이면 새 상태다.
- segment 시작 상태를 읽고 인과적인 예측 상태를 만든 후, 관측된 유효 event를 이후 segment용 memory에 쓴다. target과 padding은 쓰지 않는다.
- 내부 clip1은 각 sample의 네 associative gradient tensor를 묶은 L2 norm에 적용하고, momentum·forgetting 갱신보다 앞에 둔다. 외부 optimizer clip1과 별개다.
- 기존 최적화된 state-only scan(batch128/chunk16)을 사용하되 참조 구현과 값·gradient·state 동등성 검증을 요구한다. Dynamo 제한64/512는 해당 worker 안에서만 설정한다. 전역 Runtime 패키지 변경이나 실패 후 자동 eager 전환은 없다.
- validation `eval/no_grad`에서도 관측 이력의 분석적 메모리 갱신은 수행한다. 외부 가중치는 바뀌지 않으며 replay마다 상태를 다시 만든다.

관측 이력이 첫 segment 안에만 있으면 이전 segment 쓰기를 읽는 기회가 없다. 이 제약을 숨기지 않는다. MAC 자체 진단에는 관측 길이 `<=15`, `16–31`, `32–63`, `64–127`, `>=128`을 추가한다. 기존 다른 모델에 같은 구간 지표가 없다면 이를 교차 모델 차이로 발표하지 않는다. 그 이유로 B/Full의 새 GPU replay를 자동 추가하지 않는다.

## 실제 변경할 wrapper와 소스 경계 — 설계 완료, 구현 전

현재 공통 factory는 modern 시간 head와 MAC clip1을 함께 생성할 수 있다. 반면 과거 `build_count_aware_titantpp_mac_primary`는 legacy head만 허용하고, 현재 observed-time role 목록에는 MAC가 없다. 따라서 과거 wrapper의 제한을 풀거나 기존 core 역할을 넓히는 접근은 사용하지 않는다.

| 대상 | 후속 구현 범위 |
|---|---|
| 새 `titans_mac_observed_time_contract.py` | MAC 하나·clip1·관측 시간 head·raw-RMSE selector·9조건·새 승인을 검증 |
| 새 `titans_mac_observed_time_runtime.py` | 공통 factory 호출 후 기존 semantic optimization 적용. train/replay 모두 같은 builder와 metadata 사용 |
| 새 `run_titans_mac_observed_time.py` | 독립 worker, 새 출력 root, 상태·예산·source/Runtime 감사, 18회 endpoint replay를 관리 |
| 새 사본의 `observed_time.py`, `constants.py` | `observed_time_titans_mac_clip1_v1` 전용 역할 등록. 기존 역할 허용 목록과 검증 규칙 보존 |
| 기존 factory / MAC·clip 구현 / 공통 trainer | 계산식과 소스 유지. 새 worker 내부에서 model builder를 명시적으로 연결하고 종료 시 복원하는 범위를 검증 |
| 기존 primary wrapper / core runner / 97개 원본 | 수정하지 않음. 과거 실행 명령을 MAC 새 실행으로 재사용하지 않음 |

후속 구현은 **별도 campaign source 사본**에 두고 원본 대비 변경 목록을 남긴다. 현재 작업에서는 이 구현을 만들거나 remote에 배포하지 않았다. 지금 비교 가능한 source 기준점은 기존97개 SHA와 추가 optimized scan/Dynamo wrapper의 SHA다. 구현 후 실제 import closure 전체를 새로 동결해야 하며, 아직 없는 실행 source SHA를 완료 값처럼 제시하지 않는다.

현재 원본97개는 검사 전후 모두 core 계약 hash와 일치했다. [검사 기준](source_before.json), [출처 SHA](sources.json), [설계 검증](verification.json).

## 검증 상태와 남은 검증 — 구분

**로컬 기존 검증 재실행 — 완료**
- 인과성, target write mask, padding, sample 독립성·series reset, 내부 clip, 참조/최적화 scan, checkpoint 복원, 관측 시간 질량 계산: **83 passed, 1 skipped**. [로그](existing_cpu_tests.log).
- skipped 1개는 CUDA 경로다. CPU 통과를 GPU 검증으로 표시하지 않는다.

**현재 factory의 연결 가능성 — 완료**
- [합성 probe](probe_existing_factory.py)는 실제 공통 hidden64와 세 데이터의 head 설정, seed42, CPU batch2/length34로 검사했다. 원자료 학습이나 real-data replay가 아니다.
- clip1+관측 head의 유한 forward/backward, 참조/최적화 출력·gradient 일치, 미래/target/padding 변경의 예측 불변성, strict state 복원 후 출력 일치, B와 공통 head 초기 tensor 일치를 확인했다.
- 파라미터 수: Taxi/Intermittent 122,310, Instacart 110,022. 이전 legacy head의 수치와 다르다. [결과](factory_probe.json).
- 과거 primary wrapper가 modern head를 거부한다는 사실도 확인했다. 이 probe는 production wrapper가 구현됐다는 의미가 아니다.

**새 wrapper/계약 통합 검증 — 다음 작업**
- 새 역할의 잘못된 head/clip/모델/seed/selector 거부, 구 역할 동작 보존, 학습·replay 동일 생성 경로를 검증한다.
- train mode/dropout 및 3seed 초기화, 최대 길이64/256, sample 수128 및 마지막 작은 batch, 여러 segment와 target 위치 경계, padding 방향/값·미래 시간 변경을 확인한다.
- 실제 모델과 독립적인 작은 oracle로 earliest strict RMSE 선택·tie·40epoch 최소·patience40 최초 종료·실패 보존을 검증한다. 선택 후 다른 checkpoint의 시간/tail 지표를 섞지 못하도록 한다.
- claim 중복, 비어 있지 않은 root, source/Runtime 불일치, deadline/OOM/NaN/진척 정지, 잘못된 checkpoint·split의 차단 경로를 검증한다.
- 데이터 파일을 열지 않는 설계 검사와 실제 native 실행 검증을 분리한다.

**서버 native 검증과 실데이터 첫 epoch — 실제 GPU 승인 필요**
- 기존 학습 종료·증적 회수 후 정확한 GPU/Runtime/library SHA를 다시 확인한다. old native qualification을 새 head의 qualification으로 대체하지 않는다.
- 서버별 최대30분·60 synthetic updates를 배정한다. 해당 서버의 head/context와 compiled/eager 동등성, clip·memory state·checkpoint 복원을 확인한다.
- 모든 새 fit의 첫 full train+validation epoch를 검사한다. 별도 학습을 버리고 다시 시작하는 pilot이 아니라 **공식 fit의 epoch1**로 이어진다. 저장·초기화·step/표본·유한성·내부 정책 불일치 시 epoch2로 넘어가지 않는다. 총9조건을 초과하지 않는다.
- 향후 시간 추정은 실제 완료 epoch의 기록으로 갱신한다. 현재 CPU probe는 full-batch GPU 안정성 증거가 아니다.

## 서버와 예산 — 검토용 제안, 승인 전

서버별 **한 worker**로 제안한다. 기존 core에서 관측한 15.5% 동시 실행 향상을 MAC에 적용하지 않는다. MAC의 compiled scan·메모리 사용은 별도이며 새 동시 실행 진단을 이 계약에 넣지 않았다.

| 항목 | 5080 | 5090 |
|---|---|---|
| Python | `/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12` | `/opt/miniconda3/envs/ai_env/bin/python3.12` |
| tmux 실행파일 | `/usr/bin/tmux` | `/opt/miniconda3/envs/ai_env/bin/tmux` |
| Runtime 기준 | torch2.11.0+cu130, CUDA13, cuDNN91900, numpy2.4.4, polars1.39.3 | torch2.11.0+cu130, CUDA13, cuDNN92000, numpy2.1.3, polars1.39.3 |
| 새 root 이름 | `titans_mac_observed_time_5080_20260928_v1` | `titans_mac_observed_time_5090_20260928_v1` |
| 서버별 전체 상한 | 216시간 | 264시간 |

새 root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/` 아래 제안 경로이며, 아직 생성·존재 여부를 확인하지 않았다. GPU UUID·환경 변수는 JSON에 명시했다. 기준 Runtime은 저장된 core 계약에서 가져온 값이며 이번 작업에서 서버를 다시 조회하지 않았다. 실제 시작 시 불일치가 있으면 패키지를 바꾸지 않고 중단한다.

과거 안정화 MAC의 **동일 서버 완료 fit 시간 ÷ 완료 epoch 수**를 비용 근거로 사용했다. 개별 epoch 계측치가 아니라 준비/컴파일/평가 시간이 섞인 평균 환산치다. 구 head·selector·Runtime의 수치이므로 현재 MAC의 ETA가 아니다.

| 데이터/예정 서버 | 과거 기록 | epoch당 평균 환산 | 새3seed 모두40epoch 가정 | 모두300epoch 가정 |
|---|---|---:|---:|---:|
| Taxi / 5080 | seed62: 46epoch, 0.778시간 | 60.9초 | 2.0 GPU시간 | 15.2 GPU시간 |
| Intermittent / 5080 | seed62: 254epoch, 42.895시간 | 608.0초 | 20.3 GPU시간 | 152.0 GPU시간 |
| Instacart / 5090 | seed52: 77epoch, 18.307시간 | 855.9초 | 28.5 GPU시간 | 214.0 GPU시간 |
| 합계 | 같은 속도라는 조건부 시나리오 | — | **50.8 GPU시간** | **381.2 GPU시간** |

[원본을 대조한 비용 계산](cost_estimate.json). 이 범위는 확률적 예상 구간이나 완료 보장 범위가 아니다. 새 head로 조기 종료 시점이 달라지고 새 qualification/replay·대기 시간이 추가된다. 과거 종료 epoch까지 같을 것이라고 가정하지 않는다.

검토용 전체 상한은 **480 GPU시간 = 5080 최대216시간 + 5090 최대264시간**이다. 위 300epoch 환산값보다 약26% 여유를 둔 운영 상한으로 제안했다. **이전 480시간 제안이나 현재 core의240시간 승인을 재사용하지 않는다.** 새 계약에 대한 별도 승인 전에는 자원을 예약하거나 실행하지 않는다.

두 서버를 같이 시작할 때 calendar 강제 상한은 **최대264시간, 11일**이다. 480 GPU시간을 한 서버에서 순차 소요하는 20일이라는 뜻이 아니다. 기존 실험이 끝날 때까지의 대기는 시작 전 외부 작업 대기이며, 위 상한을 시작하지 않는다. 실제 완료 ETA는 현재 산정하지 않는다.

- 승인된 시작 때 T0·전역 마감 `T0+264h`를 잠근다. 각 서버는 `min(자체 시작+216/264h, 전역 마감)`을 적용한다. native qualification부터 준비·학습·재평가·중간 대기를 모두 센다. 늦게 시작했다고 마감을 늘리지 않는다.
- 조건별 fit+두 replay 상한은 Taxi8시간, Intermittent64시간, Instacart84시간이다. **조건 상한보다 서버/전역 잔여 상한이 우선**한다. 모든 개별 상한을 동시에 끝까지 쓸 권리를 보장하지 않는다.
- 기존36시간 상한을 MAC에 적용하면 과거 Intermittent의 약43시간 완료 기록조차 담지 못한다. 따라서 더 긴 **새 상한을 명시적으로 검토**한다. 이는 기존 실험 연장이 아니다.
- 조기 종료는 성능 규칙에 따라서만 결정한다. 예산에 맞추려고 epoch·batch·표본 수를 줄이지 않는다. 상한에 걸리면 해당 조건을 미완료로 보고하고 자동 재시작하지 않는다.
- 출력16GiB/서버, 파일64MiB, free disk20GiB 이상. 시작 free VRAM12,000MiB 이상과 foreign CUDA compute process 부재를 확인한다. 다른 프로세스·화면 서비스를 종료하지 않고 조건 미충족이면 대기한다.
- compile 단계30분 상한, 실제 batch 진척5분 없으면 stack 기록·30분 없으면 소유 worker만 중단한다. timer heartbeat를 batch 진척으로 위장하지 않는다.

## 실패 처리·최종 결과 기준 — 확정

- 새 approval/permit/계약·소스 hash가 없으면 시작하지 않는다. root/claim/worker 흔적이 있으면 중복 시작·resume·claim 삭제를 금지한다. 이전 캠페인 관리자나 자동화를 변경하지 않는다.
- NaN/OOM, source/Runtime·데이터·초기화 불일치, replay 불일치, deadline은 해당 서버 후속 실행 중단 사유다. 다른 서버가 정상이면 승인된 범위를 계속 관찰한다. 자동 retry, 실패 batch/불리한 seed 삭제는 없다.
- SSH 응답 유실은 상태 확인 실패로 기록하고 claim/tmux/PID를 읽는다. 학습 실패로 단정하거나 다시 dispatch하지 않는다.
- 완료는 새9조건과18회 replay의 receipt, strict 선택·조기 종료·exposure·배치 prefix, checkpoint byte/state SHA, 원본 보존 회수, 소유 process/tmux 종료까지 확인한 상태다.
- 보고는 각 데이터의3seed 평균·표본 SD(ddof=1), 같은 seed 차이, 수량 전체/MAE/body/tail/bin, 관측 시간 NLL, first40/last30, selected/last, 실제 steps/표본·파라미터·시간·메모리를 포함한다. raw RMSE를 서로 다른 데이터 단위끼리 평균내지 않는다.
- 기존 Full/B gate는 바꾸지 않는다. 기존 임계값을 B·Full·MLP 대 MAC의 보조 guard로 적용할 때는 `candidate/MAC` 방향과 NLL 절대 차이를 명시한다. 빈 구간·0분모·표준편차 퇴화 구간은 NA이며 통과로 처리하지 않는다. **실행 정상 완료와 성능 gate 통과는 별도 판정**이다.
- 3seed·반복 validation 탐색만으로 일반화나 통계적 유의성·논문 채택을 확정하지 않는다. 외부 최신 비교와 held-out은 별도 범위다.

## 남은 작업 순서

**현재 core 학습·재평가 및 최종 취합 — 진행 중 / 외부 작업 대기**
- 대상: 기존 5080·5090 캠페인. 기존 승인과 heartbeat가 담당한다. 이 MAC 문서 때문에 중지하거나 설정을 바꾸지 않는다.
- 완료 조건: 기존54조건108기록의 종료·감사·성능/비용 및 주장 판단 보고.

**새 MAC wrapper와 제한된 역할을 구현하고 검증 — 다음 작업**
- 대상: `paper_research`의 새 campaign-local source. 위 변경 표의 범위만 구현한다.
- 현재 GPU 학습과 독립적으로 로컬 구현·합성 검증을 진행할 수 있다. 공통 계약과 평가 경로를 맞춰야 하므로 이 세션에서 직렬로 검증한다.
- 완료 조건: 새 역할/계약/실패 경로·인과성·메모리·head·selector 테스트와 소스 차이 감사, 실제 import closure 및 실행 계약 SHA 동결.

**실제 GPU 범위 승인과 native 검증 — 승인 필요**
- 대상: 제안한 5080 6조건·5090 3조건, 9 fits/18 replays, 위 Runtime·예산·새 경로.
- 구현 증적과 현재 core 결과를 함께 제시한 뒤 새 학습 승인을 받는다. 지금 `approved=false` 상태에서 자동 실행하지 않는다.

**MAC 비교 학습과 결과 판단 — 승인 후 다음 작업**
- 서버별 순차, 서버 간 독립 실행. 불리한 조건·실패를 보존하고 기존 결과와 중복 없이 취합한다.
- 그 결과로 백본 비교 주장의 지원·반례를 판단한다. Commit·Push·MR·공용 Runtime 재배포·외부 게시를 이번 범위에 추가하지 않는다.

## 재현 자료

- [설계 계약](../../paper/contracts/titans_mac_observed_time_comparison_v1.json), [9조건 CSV](execution_matrix.csv), [비용 근거](cost_estimate.json).
- [factory probe](probe_existing_factory.py) / [결과](factory_probe.json), [기존 테스트 로그](existing_cpu_tests.log).
- [계약 생성·일관성 검사 스크립트](build_contract.py), [검증 receipt](verification.json), [입력 source hash](sources.json).

이번 변경은 새 계약 JSON과 이 보고서 폴더에 한정한다. 학습 코드·원본97개·원격 Runtime·자동화·과거 결과는 변경하지 않았다.
