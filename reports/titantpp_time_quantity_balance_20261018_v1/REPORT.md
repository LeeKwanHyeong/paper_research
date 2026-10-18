# 시간 손실과 수량 손실의 균형 진단

**외부27 실험을 일시정지하고, MLP16의 제한된 Train 기울기 진단을 완료했다. 시간 손실이 세 데이터셋에서 일관되게 더 강하다는 설명은 지지되지 않는다.** 선택 checkpoint에서는 Intermittent의 시간 기울기가 컸고, RAF에서는 9배치 중앙값 기준 수량 기울기가 더 컸다. 손실 형태를 바꾸면 크기와 방향도 함께 달라지므로 같은 λ=1만으로 균형을 맞춘 비교라고 할 수 없다.

## 현재 외부 실험 — 일시정지

2026-10-18 10:09:33/34 KST에 사용자 요청으로 두 서버의 외부27 실행 큐를 멈췄다.

| 서버 | 학습 완료 | 메모리 상태를 유지하며 정지 | 미시작 대기 |
|---|---:|---:|---:|
| 5080 | 2 | 3 | 7 |
| 5090 | 9 | 3 | 3 |

dispatcher를 먼저 멈추고 신원을 확인한 소유 worker에만 SIGSTOP했다. 추가 읽기 전용 검사와 진단 전후 검사에서 두 dispatcher 및 전체 live 자식6개의 신원·T 상태가 유지됐다. 완료11개는 학습 단계 수이며 Test 완료 수가 아니다.

기존 source·checkpoint·원본 로그·실패 기록은 변경하지 않았다. 정지한 프로세스의 optimizer/RNG 등 메모리 상태와 CUDA context/VRAM이 유지된다. 이는 학습 종료나 새 복구 checkpoint 저장을 뜻하지 않는다. 기존 wall-clock 제한은 계속 흐르며 자동 재개·기한 연장은 하지 않았다.

## 비교한 모델과 데이터 — 완료

- 모델: Cap 기준 이력 MLP16. Taxi·RAF·Intermittent × seed42·52·62, 초기 상태9개와 기존 Validation-selected 상태9개를 비교했다.
- 순수 Train에서 seed별 첫 shuffled3배치, batch128과 기존 최대 이력 범위를 유지했다. 두 상태에서 target·truth·batch tensor가 정확히 같다. 같은 forward의 dropout을 네 손실 gradient가 함께 사용한다.
- 선택 checkpoint epoch는 Taxi10/5/5, RAF3/3/1, Intermittent3/14/5다. 기존 fullValidation rawRMSE 최초 strict finite 최소 선택을 그대로 사용했으며 재선택하지 않았다.
- 시간 손실은 기존 recorded positive-integer duration NLL이다. 수량은 기존 log-MSE, `((p-q)/C)^2`, `abs(p-q)/C`를 비교했다. C는 Train Q3 상한65/10/20, 수량 가중치 λ=1이다. 예측을 cap으로 자르지 않았다.
- 5080에서8조건, 5090에서10조건을 각각 최대3 CPU 프로세스로 실행했다. Python3.12.13/PyTorch2.11.0+cu130/4threads, device=CPU, CUDA_VISIBLE_DEVICES 빈 값이다. GPU 학습과 수치 동일성을 주장하지 않는다.
- 총18조건·54forward·216개 autograd 계산이다. optimizer update·새 fit·Validation/Test 추론·기존 CNN 추론은0이다. 학습된 MLP의 고정 상태를 읽어 기울기만 계산했다.

## 선택 상태에서의 공유 백본 기울기

아래는 각 데이터셋의 3seed × 3Train 배치, 총9배치에서 **수량 gradient norm / 시간 gradient norm**의 중앙값이다. 1보다 작으면 시간 쪽 norm이 더 크다. 배치 평균 손실의 기울기이므로 배치 안에서의 방향 상쇄도 포함하며, 개별 샘플 기울기 크기의 평균이 아니다. 서로 다른 seed·배치의 국소 기술통계이며 독립 표본의 성능 추정치가 아니다.

| 데이터셋 | 기존 log-MSE | 실수량 MSE / C² | 실수량 MAE / C |
|---|---:|---:|---:|
| Taxi | 1.326541 | 0.327738 | 1.773245 |
| RAF | 3.224139 | 0.518727 | 1.105808 |
| Intermittent | 0.054194 | 0.008899 | 0.174350 |

기존 log-MSE에서 Intermittent는 이 표의 중앙값 기준 시간 norm이 약18.45배다. 반면 RAF에서는 수량 norm이 약3.22배다. Taxi는 seed별 배치 중앙값이0.599291/1.378991/2.432128로 방향이 갈린다. 따라서 시간 손실만 공통 원인으로 지목하거나 하나의 가중치를 세 데이터셋에 일괄 적용할 근거는 없다.

cap 정규화 raw-MSE로 바꾸면 세 데이터셋 모두 수량의 상대 norm이 낮아졌다. 특히 Intermittent에서는 중앙값0.008899, 시간 쪽이 약112.38배다. **실수량 오차를 직접 겨냥한다는 이유만으로 시간과의 균형도 좋아지는 것은 아니다.** 이 결과는 현재 log-MSE로 학습된 고정 가중치에서의 대안 손실 민감도이며 raw-MSE로 재학습한 뒤의 균형 예측은 아니다.

현재 log-MSE의 seed별 배치 중앙값도 보존한다.

| 데이터셋 | seed42 | seed52 | seed62 |
|---|---:|---:|---:|
| Taxi | 0.599291 | 1.378991 | 2.432128 |
| RAF | 3.602154 | 3.224139 | 2.831727 |
| Intermittent | 0.054194 | 0.035738 | 0.073898 |

## 기울기 방향과 clipping 조건

공유 백본 gradient cosine이 음수였던 배치 수는 아래와 같다. 음수는 해당 배치에서 두 기울기의 내적이 음수였다는 뜻이다. 지속적인 충돌이나 성능 저하의 원인으로 확정하지 않는다.

| 데이터셋 | log-MSE | raw-MSE / C² | raw-MAE / C |
|---|---:|---:|---:|
| Taxi | 5/9 | 4/9 | 4/9 |
| RAF | 3/9 | 4/9 | 4/9 |
| Intermittent | 4/9 | 5/9 | 5/9 |

각 조건의 cosine 중앙값은 −0.027~0.104 범위다. 방향과 부호가 배치·손실에 따라 달라져 항상 서로 반대로 학습한다는 설명도 맞지 않는다.

전체 파라미터의 λ=1 결합 norm이 기존 clip1 기준을 넘은 배치 수는 Taxi4/1/4, RAF7/2/5, Intermittent9/9/9(log/rawMSE/rawMAE 순)다. 공유 백본 외 전용 head도 포함한 값이며 실제 clipping이나 optimizer update를 수행한 횟수가 아니다. 큰 결합 norm만으로 어느 손실이 성능을 해쳤다고 판단하지 않는다.

## 초기 상태와 검증

초기 수량 head.weight는 정확히0이다. 초기9조건의27배치·세 수량 손실81개 비교에서 수량→공유 백본 gradient norm은 모두0이었다. cosine·norm을 맞추는 λ는 미정/null로 기록했다. 선택 상태의81개 비교에서는 모두 양수였다. 초기 결과를 학습 전반의 시간 손실 지배로 해석하지 않는다.

공유 파라미터46개 tensor, 시간 전용4개, 수량 전용2개를 그래프 참여로 구분하고 정확한0 여부를 따로 기록했다. parameter/buffer/context·기존 checkpoint·입력·source SHA가 유지됐고, autograd는 .grad를 쓰지 않았으며 호출 종료 시 RNG를 복원했다.

합성18tests 및 독립 산술444개를 통과한 소스 commit은 paper_research 저장소 `codex/contextual-retrieval-pilot-20261007` 브랜치의 `9f1bd1fe`다. 과학 source941개를 동일 바이트로 상속하고 신규 진단 worker만 더한942개를 별도 경로에 설치했다. Train/Validation 입력3개와 선택 checkpoint9개만 묶었으며 full Test 입력은 포함하지 않았다. 두 서버의 진단 worker18개와 controller transport는 실제 exit0이며 신규 child PID 부재를 확인했다. supervisor 자신의 별도 OS exit를 관측했다고 주장하지 않는다.

원본 결과·로그72파일을 두 archive로 회수해 SHA를 확인했다. 독립 검산은 1,053개 파일 SHA, 원결과17,500검사·집계12,998필드에서 불일치0이었다. 상세 검산 및 저장된 group dot을 사용한 방향 대수 검산의 한계는 NATIVE_RESULTS_QA.json/.md에 기록했다. 개별 gradient vector를 저장하지 않았으므로 원벡터에서 dot을 독립 재구성한 검산은 아니다.

## 남은 판단 — 다음 작업

**손실 형태와 시간/수량 가중치의 효과를 분리하는 비교 조건을 확정해야 한다.** 같은 λ=1을 유지한 raw-loss 학습은 목적함수 모양과 상대 영향력을 동시에 바꾸므로 손실 형태만의 실험으로 설명할 수 없다.

공유 norm을 같게 만드는 λ를 배치별로 계산해 보존했지만, 이를 최적 λ로 채택하지 않았다. 선택 상태·표본 배치의 gradient는 AdamW momentum·실제 update·학습 전 과정·Validation/Test 성능을 대신하지 않는다. 한 가지 원인이나 개선 성공을 확정하지 않는다.

기존 외부27은 정지 상태다. 이번 진단에 이어 자동 재개하거나 새 손실18개 본 학습을 시작하지 않았다. 자세한162개 손실 비교와 seed별 범위는 BALANCE_PER_BATCH.csv, BALANCE_PER_SEED.csv, BALANCE_SUMMARY.json에 보존한다.
