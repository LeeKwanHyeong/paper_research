# TitanTPP 후속 검증: 데이터 계보와 36조건 실행 준비

2026-10-01 사용자 승인: “진행하자. 학습까지 승인할게.” 기존 MLP·84조건 결과·Core/효율 감사는 현재 기준선으로 보존했다. 본 작업은 새 구조 비교24조건, Deep Renewal12조건, 비학습 기준8행의 validation 검증이다.

## 1. 과거 test와 현재 평가 모집단 — 조사 완료, 일부 계보 미확인

성능표를 다시 비교하지 않고 manifest, 분할 코드, `(oper_part_no, seq, demand_dt)` 사건 키를 대조했다. 상세 해시와 경로는 `lineage.json`에 있다.

| 데이터 | 확인한 관계 | 최종 평가에서의 설명 |
|---|---|---|
| Taxi | 현재 보존된 과거 test와 현재 test 사건8,327개가 전부 일치. 과거 test 열람 기록이 있음. | 이전에 접근한 기존 holdout의 재평가. 미접근 독립 test로 표현하지 않음. |
| Instacart | 현재 보존된 과거 test와 현재 test 사건578,387개가 전부 일치. 과거 test 열람 기록이 있음. | 이전에 접근한 기존 holdout의 재평가. |
| Intermittent | 기존41,344개와 현재88,019개 사건의 문자 키 중복은0. 원천자료·엔티티명·사건 병합 규칙이 달라 키를 대응시키는 계보를 확정하지 못함. | 중복0을 독립성 증거로 쓰지 않음. 과거 노출 여부가 미확정인 기존 holdout. |
| RAF | 확인된 세 과거 test와 별도 데이터. 고정 test는2001-12~2002-12의5,226사건. 제한된 이력 검색에서 test 열람 확증은 찾지 못함. | 고정 시간 holdout. 완전한 비접근 이력을 입증했다고 쓰지 않음. |

Taxi·Instacart의 과거 실행 manifest에는 원본 데이터 SHA가 없어 **실행 당시 모든 바이트의 동일성**까지 복원한 것은 아니다. 현재 남아 있는 파일의 사건 동일성, 동일한 경로·분할 설정, 변하지 않은 분할 코드, 과거 열람 기록을 함께 근거로 삼았다.

현재 네 고정 test는 모델/선택 규칙 확정 후 별도 평가 계약으로 사용할 수 있으나, 모두를 새 독립 확인 자료라고 주장하지 않는다. 이번 학습 승인에 test 평가를 포함하지 않는다.

새 독립 확인의 범위는 다음과 같다. Taxi는 현재 사건의 마지막 날짜 이후 겹치지 않는 원자료 기간과 동일 집계 규칙, Intermittent는 현재 사업장별 종료일 이후 추출 또는 출처가 확인된 새 사업장/부품, Instacart는 별도로 수집된 주문 이력, RAF는 기존84개월을 벗어난 별도 부품/기간이다. 실제 확보 가능한 데이터·행 수·접근 이력은 아직 확정되지 않았으므로 새 독립 확인의 실행 준비가 완료된 것은 아니다.

접근 기록: 최초 manifest 점검 중 기존 요약 필드의 heldout 수량 분포 집계가 출력됐다. 성능값·예측값을 읽거나 heldout 수량/시간 열을 새로 materialize하지는 않았다. 이 사실을 `lineage.json.access_note`에 남겼으며 이후 읽기는 사건 키와 메타데이터로 제한했다.

## 2. 구현과 검증 — CPU 검증 완료

- `titantpp_current_only_param_matched`: 현재 표현만 받는8개64→6→64분기. 기존 가용성 임계값·고정/8 유지. 추가파라미터6,144개. 현재 encoder 표현 자체에는 인과적 이력이 있으므로 ‘이력을 전혀 쓰지 않는 모델’로 해석하지 않는다.
- `titantpp_all_available_history_mlp`: 현재+실제 직전 표현을 받는8개128→4→64분기. 직전 사건이 있으면 모두 가용, 고정/8 유지. 원래0으로 마스킹된 인덱스를 단순 해제하지 않고 실제 직전 인덱스를8분기에 사용한다. 활성 용량과 잔차 크기 효과를 함께 바꾸는 대조군이다.
- `deep_renewal_event_native_nb`: 원시 간격·수량을 받는2층64 LSTM, native shifted-negative-binomial 평균/전역분산, 시간·수량 NLL. Instacart 기록값30은 `P(NB>=29)`로 처리한다. 수량 예측은모든지표에서동일한조건부평균 `1+mu`다.
- Deep Renewal의 공식 GluonTS/MXNet 소스에서 gate 순서, Xavier 초기화, LSTM zero bias, dense bias5, alpha bias2를 확인했다. 독립적인 LSTM 식 계산과 분포 계산을 대조했다. 실제 MXNet Runtime에서 원 모델을 실행한 수치 대조는 수행하지 않았다. 공통 window의 마지막 target만 학습하는 next-event adapter이며 원 논문의 calendar-horizon 전체실험 재현과 구분한다.
- Deep Renewal의 층 사이 dropout을 명시적으로 적용하여 저장된 PyTorch RNG로 복구한다. 내부 dropout 상태가 state_dict에 없는 fused multi-layer cuDNN 경로를 사용하지 않는다. 두 LSTM층과 dropout0.1이라는 과학적 설계는 유지한다.
- 기존 scientific trainer를 이용한 네 데이터 의미의 합성3epoch 학습, selected/last replay, 모델명 재표기 거부, 인과성·target 배제·양쪽 padding, 초기 base tensor/RNG/예측 동일성, 모델/optimizer/RNG/shuffle의 실제 epoch 저장·복구 동등성을 검증했다.
- CPU65검사 통과(`final_cpu_tests.xml`). 첫 검사에서 발견한 test fixture의step수 누락, native telemetry 누락, RAF month 허용 누락은 본학습 전에 수정했다. 최초 실패 XML은 보존했다.
- 원래 공통 lognormal head 및 loss는 대조군에서 유지한다. Deep Renewal의 native NLL은 별도 모델의 목적함수이며 서로 다른 train loss의 합계를 성능 순위로 쓰지 않는다.

## 3. 승인된 실행 계약 — 동결 완료, native 검증 결과는 별도 receipt

실행 root: `search_artifacts/titantpp_pakdd_extension_20261001_v1`.

| 서버 | 대상 | 새 fit |
|---|---|---:|
| 5080 | Taxi·Intermittent·RAF ×3모델 ×seed42/52/62 |27|
| 5090 | Instacart ×3모델 ×seed42/52/62 |9|

batch128, max300, min40, patience40, 최초 strict validation raw수량 RMSE선택을 유지한다. 조건별36시간, 배포부터전체168시간. 첫2epoch는본학습에포함하며 이후298epoch×최대관측epoch시간×1.2+600초가남은조건/전체시간을넘으면 중단한다. 이는 최대학습 예산의 실행 가능성 검사이며 완료 ETA가 아니다. 준비 native 검증은90분 상한이다.

서버당 단독worker, tmux, 독립 timeout, 서버lease, 원자적checkpoint와terminalSHA전환을 사용한다. Mac ACK나 scheduler에 의존하지 않는다. 실패한 조건은원본/claim을보존하고해당서버큐를중단하며자동retry하지않는다. 다른서버의성공한큐는계속한다. 클라우드추가임대비0,전기료미측정이다.

준비 단계와 native GPU 통과, 실제 scientific fit 시작은 `preparation_receipt.json`, `native/<host>/receipt.json`, `launch_confirmation.json`으로 구분한다. 검사명이나 배포만으로 학습 시작을 판단하지 않는다.

## 남은 작업 순서

1. **원격 검증과 학습 시작 — 완료 / 학습 진행 중**: 양 GPU 검증과 기존 MLP validation 재평가가 통과했다. 2026-10-01 23:30 KST 소유 PID/GPU와 실제 학습 진입을 확인했다. 5080 Taxi seed42 현재 표현 전용 대조군은1epoch/300updates 저장,5090 Instacart seed42 같은 대조군은첫epoch 진행 중이다.36조건전체완료는아니다.
2. **완료 결과 원본 회수와 비교 — 다음 작업**: checkpoint와source, 노출/선택/replay를검증하고모든seed의결과·실패·시간/메모리를함께집계한다. 추가학습으로기존완료결과를대체하지않는다.
3. **최종 평가 대상과 주장 동결 — 다음 작업**: 대조군결과에따라방법의기여범위를확정하고기존holdout평가계약을완성한다. 새 독립 데이터의 확보 가능성은별도작업으로남긴다.
4. **평가 실행과 원고 반영 — 승인 필요**: 이번승인은validation학습까지다. 최종test/새자료평가의범위와고정모델을검토한뒤실행한다.

## 배포 전 검증 보완 기록

최초5080 native검증은통과했다. 최초5090은길이64인모델에준비용길이256을넣어positional interpolation의비결정적CUDA backward가거부됐다. 실제loader는64이내이므로검증부하를32/64로정정했다.모델·데이터·Runtime·determinism·학습·선택은불변이다. 원본claim과실패로그를보존하고새`prelaunch_v2`사본에동결했으며최초시작/마감은유지한다.두서버v2의native통과receipt를받은후에만학습허가서를쓴다.

## 현재 실행 증거

활성계약SHA `08f7e9ed402ba74fde099321f039b78de3a11b0bd93cba27492b1a84454f6a6c`,109소스closure `989153f9eea2dac6a790835ab1b7c502fc9c0461829800915a6c602166b6ea92`. `prelaunch_v2/launch_confirmation.json` 및 `monitor/20261001T143010729329Z`를참조한다. 전체마감2026-10-08 23:23:50.973276 KST. 첫2epoch비용gate는위최초관측시점에아직확인되지않았다.

[Notion 실행 기록](https://app.notion.com/p/3ecbbe405613815d8f36e19cd7f73601)

## 비학습 기준 비교 — 완료

동일 validation target에서 마지막 수량·이력 평균 기준8행을 회수하고 SHA를 확인했다. 기존 MLP의 세 seed 평균과 비교했다. **Intermittent의 마지막 수량 기준은 MAE0.564733으로 MLP0.700523보다 낮다. RMSE는 MLP1.673034가 기준1.707803보다2.04%낮다.** Taxi·RAF에서는 MLP의 MAE/RMSE가 두 기준보다 모두 낮다. Instacart의 이력 평균 대비 개선은MAE1.46%/RMSE0.34%다. 이 결과는 성능 주장의 범위를 검토할 근거이며, 단순 기준을 불리하다는 이유로 제외하지 않는다. [전체 비교](nonlearned_comparison.md)와 원본SHA/계산JSON을 보존했다. 비학습 기준은seed없는결과이며통계적유의성으로해석하지않는다.

### 23:34 KST 시작 검증 보완

두 서버에서 저장된 실제 epoch checkpoint를 확인했다. 5080은 Taxi seed42 현재 표현 전용 대조군23epoch/6,900updates,5090은 Instacart seed42 같은 대조군1epoch/15,557updates다. 5080의 첫2epoch13.167/12.749초와 비용gate통과를 확인했다.5090의 첫epoch는214.387초이며두번째완료후비용gate검사가남았다. 진행기록을최종성능으로해석하지않는다. `prelaunch_v2/launch_checkpoint_confirmation.json`참조.
