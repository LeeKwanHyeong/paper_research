# TitanTPP J/Q/T 실행기 및 현재 checkpoint 진단 — 2026-09-10

**J/Q/T 실행기의 CPU 구현·검증과 B/dual-timescale의 train-only gradient 진단을 완료했다. 현재 결과로 시간·수량의 완전 분리를 확정하지 않는다.** Intermittent의 선택 checkpoint에서는 두 모델 모두 clipping이 없었고, Taxi 후보에서는 공동 gradient clipping이 더 자주 발생했다. 단일 원인이 세 데이터셋을 공통으로 설명한다는 근거는 확보되지 않았다.

새 연구 데이터 학습과 GPU 연산은 실행하지 않았다. CPU 학습은 합성 데이터 테스트에서만 수행했다. 기존 `Intermittent 수량 평가 점검` scheduler(`vnc-hard-lmm-5090-screening`)는 PAUSED다.

**현재 코드 기준선 — 완료**

- 저장소: `/Users/igwanhyeong/PycharmProjects/paper_research`, 브랜치 `codex/hard-lmm-causal-qkv`, HEAD `1de2c31e8e3febda7ff20d0527402ea9959b4098` 위의 별도 experimental 파일이다.
- 기존 모델·baseline runner·기본 cap30 설정은 편집하지 않았다. 이 진단은 frozen B에 대응하는 legacy cap300을 명시한다. 시간 점수는 clamped time loss이며 정규화된 Time NLL로 해석하지 않는다.
- 기존 dirty worktree, 결과표와 checkpoint를 보존했다. 이번 작업은 커밋하거나 원격 저장소에 동기화하지 않았다.

**J/Q/T 진단 실행기와 재현 계약 — 완료, CPU 검증 범위**

- [엔진](../../paper/scripts/time_quantity_diagnostic.py)은 공동 J, 수량 전용 Q, 시간 전용 T를 제공한다. Q/T는 비활성 head의 forward를 호출하지 않고 optimizer에서도 제외한다. 목적 함수는 원래의 time loss와 가중치 1의 log-MSE를 사용한다.
- [CLI](../../paper/scripts/run_time_quantity_diagnostic.py)는 기본적으로 계약을 검증하고 manifest만 쓴다. `--execute`가 있어야 학습한다. 기본 검증도 train·validation 행을 읽으며, 데이터 없는 dry-run은 아니다. held-out 행은 materialization 전에 제외한다.
- 모든 arm에 같은 초기 state·독립적으로 재설정한 loader RNG·batch 순서·고정 epoch budget을 적용한다. 각 epoch의 global step, 표본 수, batch-order SHA를 최종 감사한다. early stopping은 없다.
- J에는 raw-RMSE와 legacy-time selector를 모두 저장한다. Q는 수량, T는 시간 selector만 사용한다. strict finite minimum을 적용해 동률이면 최초 epoch를 유지한다.
- 비교는 **J의 수량 선택 checkpoint ↔ Q의 수량 선택 checkpoint**, **J의 시간 선택 checkpoint ↔ T의 시간 선택 checkpoint**다. J의 서로 다른 두 checkpoint에서 얻은 최적값을 하나의 모델이 동시에 달성한 성적으로 합치지 않는다. 같은 epoch/step의 history 비교도 함께 남긴다.
- 마지막 epoch state를 원자적으로 저장하고 model·optimizer·Python/NumPy/Torch RNG·loader RNG·generator 공유 관계·source/runtime·초기화·계약 identity를 검증한 뒤 복원한다. epoch 중단은 마지막 완료 epoch부터만 재개한다.
- CUDA/MPS 학습은 현재 엔진이 거부한다. CPU에서 검증한 재현성을 GPU로 확대해 주장하지 않는다.
- [테스트 로그](pytest.txt): **60 passed**. 초기 공동 objective와 모든 parameter gradient 일치, 비활성 head 불변, target/padding 인과성, 동일 step/batch 비교, strict earliest tie, 연속 실행과 resume의 일치, RNG 변조 거부 등을 검증했다. 새 파일의 Ruff도 통과했다.
- [실제 데이터 계약 검증](contract_validation_receipt.json): 세 데이터셋의 data/split SHA, 정확한 train·validation target population, quantity hash와 초기화 manifest가 일치했다. 이 경로에서 모델 학습·validation 성능 계산은 하지 않았다.

**현재 B와 후보의 train-only gradient·clipping 진단 — 완료**

[고정 probe 계약](contracts/frozen_gradient_probe_v1.json), [실행 상태](gradient_probe_v1/status.json), [요약 JSON](gradient_summary.json), [요약 재계산 코드](summarize_probe.py)를 함께 보존했다.

- 각 데이터셋의 canonical train target에서 seed `20260910`으로 2,048건을 비복원 추출해 128건 × 16배치로 고정했다. B와 후보는 동일 target·순서·배치를 사용한다. 배치별 dropout seed는 `424242 + batch index`다. 구조가 다르므로 같은 seed가 같은 dropout mask를 보장한다고 주장하지 않는다.
- train 모드의 **고정된 선택 checkpoint**에서 gradient만 계산했다. optimizer update는 0회다. 전체 학습 궤적이나 실제 과거 optimizer의 update를 재현한 결과가 아니다.
- frozen source `ca8823e688a510e64e7fb81c9dc9d158d4bdfc7c`의 실제 import 경로·파일 SHA를 검사했다. 6개 모델 모두 checkpoint byte/state SHA와 선택 epoch를 확인하고, 첫 배치에서 frozen core와 새로운 공동 objective·모든 parameter gradient가 정확히 일치함을 확인했다. 실행 후 state·원본 checkpoint도 변하지 않았다.
- 실제 runtime은 Python3.12.10/PyTorch2.14.0 CPU, NumPy2.3.1, CPU thread4다. 과거 CUDA runtime과 동일한 수치 실행 환경이라는 주장은 하지 않는다. probe 본체 실행 시간은 약 70.61초다.
- train 행만 scan 단계에서 필터링한 뒤 수집했다. validation/test target·성능은 이 probe에 들어가지 않았다. B·후보 checkpoint를 선택한 과거 validation 절차 자체는 그대로 유지한다.

| 데이터셋·모델 | 선택 epoch | encoder cosine 중앙값 | 음수 cosine 배치 | 공동 clipping 배치 | 수량만 clipping 배치 | 공동/Q clipping 계수 비율 중앙값 |
|---|---:|---:|---:|---:|---:|---:|
| Intermittent B | 77 | -0.0299 | 10/16 | 0/16 | 0/16 | 1.000 |
| Intermittent 후보 | 37 | -0.0699 | 11/16 | 0/16 | 0/16 | 1.000 |
| Taxi B | 45 | +0.0399 | 6/16 | 4/16 | 0/16 | 1.000 |
| Taxi 후보 | 83 | +0.0278 | 4/16 | 13/16 | 0/16 | 0.653 |
| Instacart B | 72 | -0.0213 | 10/16 | 6/16 | 3/16 | 1.000 |
| Instacart 후보 | 72 | +0.0088 | 7/16 | 6/16 | 3/16 | 1.000 |

공동 clipping 계수는 encoder와 두 head를 포함한 전체 공동 gradient의 norm으로 계산한다. Q 계수는 같은 parameter 위치에서 수량 gradient만 남겼을 때의 반사실적 계수다. 실제 clipping이나 optimizer step을 수행하지 않았다. 두 계수의 비율은 수량 gradient 성분에 적용될 추가 축소를 설명하지만 AdamW update 크기나 이후 성능 개선량은 아니다.

확인된 사실과 해석:

- **Intermittent:** 선택 checkpoint의 현재 train 표본에서는 global clipping으로 후보의 수량 성능 악화를 설명할 근거가 없다. 작은 음수 cosine과 일부 배치의 반대 방향은 관찰되지만, 공유 표현이 실패 원인이라는 인과 증거는 아니다.
- **Taxi:** 후보의 공동 clipping은 13/16배치이고 Q-only는 0/16이다. 따라서 이 checkpoint에서 시간 objective를 더한 공동 gradient가 수량 성분도 축소할 수 있다는 구체적인 관찰이 있다. 후보의 epoch83과 B의 epoch45는 학습 시점이 달라, 구조 변경의 인과 효과로 해석할 수 없다.
- **Instacart:** B·후보의 공동 clipping은 모두 6/16, Q-only는 모두 3/16이다. 수량 gradient 자체의 크기도 포함해 진단해야 한다. Intermittent와 같은 원인을 가정하지 않는다.
- 세 데이터셋 모두 task cosine 중앙값이 0 부근이다. 간섭이 전혀 없다는 결론도, 공통의 강한 간섭이 있다는 결론도 여기서는 내리지 않는다. 한 seed·한 checkpoint·중복 가능한 이력을 가진 표본의 기술 통계이며 유의성 검정이나 일반화 증거가 아니다.
- **다음 판단:** 동일 초기화에서 다른 objective를 제거했을 때의 학습 결과를 J/Q/T로 직접 비교할 필요가 남는다. 지금 dual encoder를 최종 backbone으로 확정하지 않는다.

**동일 budget 학습의 설계와 비용 한도 — 동결, GPU 실행 연결은 다음 작업**

[실행 정책](execution_policy.json), [비용 계산 근거](cost_basis.json), `contracts/jqt_cpu_*_v1.json` 3개에 seed42, 각 arm 120 epoch, batch128, AdamW lr0.001/weight-decay0.01/clip1, 같은 data·population·62개 source 파일을 고정했다. 120 epoch는 이전 B의 완료 구간을 참조한 1차 진단 예산이며 수렴 보장은 아니다. J/Q/T 연구 학습을 시작하기 전에 예산을 고정했으며, 신규 학습 성능을 보고 군별 예산을 조정하지 않는다. 이 파일만으로 별도의 시각 인증을 갖춘 사전 등록을 주장하지는 않는다.

| 데이터셋 | train target/epoch | step/epoch | step/arm | J/Q/T 총 step | 과거 5090 시간으로 환산한 3개 arm |
|---|---:|---:|---:|---:|---:|
| Intermittent | 393,824 | 3,077 | 369,240 | 1,107,720 | 6.56 GPU시간 |
| Taxi | 38,393 | 300 | 36,000 | 108,000 | 0.74 GPU시간 |
| Instacart | 1,991,192 | 15,557 | 1,866,840 | 5,600,520 | 14.83 GPU시간 |
| 합계 | — | — | — | 6,816,240 | 22.14 GPU시간 |

22.14시간은 **과거 5090 B run의 전체 경과시간/완료 epoch × 120 × 3**이다. Q/T·5080에서 측정한 시간이 아니며, 데이터 로딩·검증·저장과 runtime 차이도 있다. 실제 5080 소요시간과 금전 비용은 미측정이다. 검토용 실행 상한은 총 72 GPU시간, 동시 작업1개, 연구 arm9개, arm당120 epoch로 정했다. timeout·실패 때문에 군별 budget이 달라지면 비교를 완료로 처리하지 않는다. 자동 재시작·상한 증액은 포함하지 않는다. 이 상한을 적용할 GPU launcher/job은 아직 생성하지 않았다.

[5080 읽기 전용 점검](5080_runtime_readonly_v2.json)에서는 RTX5080 16GB, Python3.12.13, package metadata 기준 torch2.11.0/NumPy2.4.4/Polars1.39.3을 확인했다. 원격 실험 경로는 Git checkout이 아니어서 source manifest 연결이 필요하다. 첫 조회에서 Git revision 확인이 실패한 기록도 [보존](5080_runtime_readonly.json)했다. 조회 중 source sync·학습·GPU 계산은 하지 않았다.

현재 CPU 계약을 그대로 CUDA 계약이라고 부르거나, engine의 CPU guard를 우회하지 않는다. 실제 5080 실행에는 source snapshot 연결, imported torch/CUDA/cuDNN 정보 및 결정적 연산 설정, 합성 데이터의 연속/재개 일치 검증과 별도의 GPU 계약이 필요하다.

**남은 작업 순서**

**5080에서 재현 계약을 검증할 제한된 실행 범위 확정 — 다음 작업, GPU 실행 승인 필요**

- 대상은 `paper_research`의 5080 runtime이다. 새로운 GPU 작업의 범위 안에서 고정 source와 runtime을 연결하고, 합성 데이터의 J/Q/T 연속·재개 동등성을 먼저 검증한다. 이 검증 전에는 연구 데이터의 본 학습을 시작하지 않는다.
- 현재 CPU 계약·checkpoint 증적을 보존하고 별도 GPU 계약을 작성한다. 정확한 device/runtime/source와 누적 실행 시간 제한이 연결돼야 GPU 실행 준비 완료다.

**J/Q/T 동일 budget 학습과 대응 selector 비교 — 재현 검증 후 다음 작업, 새 학습 승인 필요**

- 위 연구 설계·9개 arm·120 epoch·실행 한도에 따라 동일 표본 노출을 비교한다. Q↔J 수량, T↔J 시간 및 같은 step history를 각각 감사한다. 기존 scheduler를 자동 재개하지 않는다.
- 학습 실행과 완료가 확인되기 전에는 성능 개선·열화 또는 분리 구조의 우월성을 보고하지 않는다.

**분리 구조와 benchmark 비교 확정 — 진단 결과 후 다음 작업**

- 이득이 반복될 때만 Q+T의 전체 capacity와 학습·추론 compute를 통제하고, RMTPP·THP에도 같은 분리 기법을 적용한다.
- 공통 분리 기법의 이득과 TitanTPP 고유 기여를 구분한다. 단일 seed validation이나 이번 gradient snapshot을 논문의 우월성 증거로 승격하지 않는다.
