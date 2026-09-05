# Hard-LMM 경과시간 표현 후보와 검증 계약

대상은 `paper_research/master`의 별도 backbone `titantpp_elapsed_age_static_memory`이다. 사용자는 계약 확정, 로컬 구현·검증·독립 커밋과 5090 CUDA·Taxi/Instacart 전체 e1을 승인했다. 이후 성능 screening, 추가 seed·dataset, held-out test와 push는 이번 실행에 포함하지 않는다.

기계 판독 계약은 [hard_lmm_elapsed_age_v1.json](hard_lmm_elapsed_age_v1.json), 실행 계약은 [hard_lmm_elapsed_age_cuda_e1_5090_v1.json](hard_lmm_elapsed_age_cuda_e1_5090_v1.json)이다. 아래 성능 기준은 새 후보 결과를 보기 전에 고정하며 **e1에서 채택 판정에 사용하지 않는다.**

## 단일 가설과 변경 위치

Instacart의 비교 가능한 짧은 이력 중심 body 집단에서 실제 시간 배치와 사건 번호상의 배치가 크게 다른 경우 오류도 높았다. 이 시간 배치의 차이를 encoder attention에 직접 전달하면 상태 표현을 개선할 수 있는지 시험한다. 근거는 [미사용 series의 사전 균형 진단](../results/hard_lmm_instacart_balanced_20260905/README.md)에 한정되며, 인과적 encoder 결함이나 개선 가능 폭이 입증된 것은 아니다.

각 query 사건 i에서 보이는 과거 key j에 대해 다음 값을 계산한다.

```text
B(i,j) = (i와 j 사이의 경과시간 / 첫 관측부터 i까지의 경과시간)
       - (i와 j 사이의 사건 수 / 첫 관측부터 i까지의 사건 수)

attention_score(layer, head, i, j) += beta(layer, head) * B(i,j)
```

현재 관측 사건 i의 gap까지 포함하며, 분모는 반드시 **i 시점까지의 prefix**에서 계산한다. 마지막 관측 행에서 B의 RMS는 진단에 사용한 J와 같다. 두 layer·네 head의 signed 계수 `encoder.elapsed_age_beta[2,4]`, 총 **8개**만 추가한다. 계수의 부호를 제한하지 않으며 별도 gate·projection·loss를 추가하지 않는다.

기존 log-gap·log-quantity 입력, 사건 위치 embedding, persistent token 16개, separate-key K/V, top-4, temperature 1, `h+r` 결합, quantity/time head, loss, optimizer와 checkpoint 선택을 유지한다. 절대적인 gap 크기는 기존 입력 경로로 계속 전달된다.

## 시간·이력·누출 처리

- 첫 관측의 gap은 window 바깥 사건과 연결될 수 있으므로 B의 누적 전에 제외한다. 기존 log-gap 입력 경로는 그대로 둔다.
- 관측 수는 padding을 제외해 센다. 관측 mask가 유효 위치의 부분집합이 아니면 입력 오류로 거부한다. Target·padding 값은 누적 계산 전에 0으로 만든다. Query마다 자기 prefix의 누적시간과 사건 수를 사용한다.
- 미래·padding·target·persistent key pair의 B는 0이다. 첫 query, 관측 수 H≤2, 경과시간 합이 0인 prefix도 B=0이다. 같은 간격에서는 수학적으로 B=0이다.
- Taxi의 시간 단위는 시간, Instacart는 일이지만 비율에서 소거된다. 새 전역 통계나 validation 기반 정규화를 만들지 않는다.
- 유한한 float32 범위의 gap을 대상으로 음수는 geometry에서 0으로 제한한다. FP64 누적·비율 계산 후 FP32 geometry를 만들고, 더하기 직전에는 기존 attention 점수의 dtype으로 변환한다. 기존 AMP 경로를 FP32로 일괄 승격하지 않는다.
- 첫 gap 변경과 공통 시간 배율에 대한 불변성은 **B에만** 요구한다. 기존 log-gap 경로를 사용하는 전체 모델의 출력 불변성을 요구하지 않는다.

## 초기 출력과 학습·복원 계약

- Beta는 RNG를 사용하지 않는 0 초기화다. 같은 seed의 separate-key와 공통 파라미터 및 생성 후 RNG 상태가 같아야 한다.
- Beta=0에서 encoder·검색·최종 출력은 CPU에서 정확히 일치해야 한다. CUDA float32는 사전 허용오차 `rtol=atol=1e-6`으로 확인한다.
- Beta가 0이라고 계산을 건너뛰지 않는다. `beta * B`를 autograd에 남겨 불규칙한 H≥3 이력에서 유한한 비영 gradient와 optimizer 갱신이 발생해야 한다.
- 실제 e1의 갱신은 beta 텐서의 norm과 해당 AdamW의 `exp_avg`, `exp_avg_sq` norm 및 step이 양수인지 확인한다. 8개 계수 중 적어도 하나가 움직여야 하며 전부 비영일 필요는 없다.
- Beta의 변화가 실제 attention·검색·예측으로 전달되는지 합성 조건에서 확인한다. 동일 간격의 B=0이 학습 후 Taxi 보존을 보장하지는 않으므로 아래 성능 기준도 별도로 둔다.
- 새 backbone·role·contract metadata를 사용하며, 기존 separate-key checkpoint와 상호 오인하거나 자동 변환하지 않는다. Fresh 학습을 사용하고 기존 출력 경로의 자동 resume를 차단한다.
- 모델·optimizer 저장/복원 후 같은 RNG의 다음 학습 step을 비교한다. CPU는 exact, CUDA는 `rtol=atol=1e-6`이다.

## 향후 성능 평가의 고정 기준

기존 원본 기준과 separate-key 기준을 **동시에** 만족해야 한다. 기준선은 hash가 고정된 기존 seed42 validation artifact의 전체 정밀도 값이다. 향후 별도 승인된 e300/min40/patience40, batch128, lr0.001, 같은 data/context/loss/selector의 fresh run에서만 적용한다.

| 대상·비교 | Body MAE | 전체 RMSE | >p99 MAE | Time NLL |
|---|---|---|---|---|
| 두 dataset, 원본 Hard-LMM 대비 | 5% 이상 개선 | 악화 2% 이하 | 악화 2% 이하 | 증가 0.01 이하 |
| Instacart, separate-key 대비 | **5% 이상 개선** | 악화 1% 이하 | 악화 1% 이하 | 증가 0.01 이하 |
| Taxi, separate-key 대비 | **악화 1% 이하** | 악화 1% 이하 | 악화 1% 이하 | 증가 0.01 이하 |

동시 적용으로 Time NLL은 원본·separate-key 중 낮은 값 +0.01 이하여야 한다. 모든 지표는 유한해야 한다. 예를 들어 Instacart body 상한은 약 **3.251659**, Taxi body 상한은 약 **17.135879**, Taxi RMSE 상한은 **120.584501**, Taxi >p99 MAE 상한은 **703.368908**이다. 반올림한 수치를 실제 비교 경계로 사용하지 않는다.

기존 원본은 일반 기준선으로 유지한다. 성능 기준 통과가 시간 가설의 기전 증명과 동일하지는 않으며, 향후 개선이 있으면 high-J 오류가 함께 감소하는지도 별도 검토해야 한다.

## 승인된 실행과 완료 조건

**로컬 — 모델·계약 검증 후 독립 커밋**
- 독립 공식과 J 일치, 초기 동등성, causal prefix·target·padding·persistent 처리, gradient와 실제 예측 변화, finite 계산, 엄격한 checkpoint route와 optimizer 복원을 확인한다.
- 기존 원본·separate-key의 계약도 확인한다. 모델 등록과 train/validation 전용 데이터 경로가 실제 CLI에 연결돼야 한다.
- 완료된 소스와 기록을 `paper_research/master`에 커밋한다. 기존 미추적 `scripts/`는 포함하지 않는다.

**5090 — CUDA 테스트와 전체 e1 실행**
- GPU 점유, 최소 12,000MiB 여유, 기존 Runtime, 데이터 checksum을 확인한다. 커밋된 소스를 새 snapshot에 전송하고 패키지 및 파일별 hash를 대조한다. 기존 checkout·서비스·Runtime을 변경하지 않는다.
- CUDA 테스트는 h64, 길이64/256의 실제 학습 형태를 포함하고 실패·skip이 없어야 한다. 이어서 Taxi, Instacart 전체 e1을 순서대로 실행한다.
- Train target은 Taxi **38,393**, Instacart **1,991,192**개, validation target은 각각 **8,268**, **503,733**개여야 한다.
- Beta와 optimizer 상태의 실제 갱신, peak CUDA 메모리, 저장·복원, 전체 처리 건수, 유한성, 소스·데이터 무결성과 held-out 부재를 감사한다. 결과를 로컬에 보존한다.
- **완료는 정상 실행과 계약 준수다.** e1 수치로 모델 채택이나 다음 성능 실험을 자동 결정하지 않는다.
