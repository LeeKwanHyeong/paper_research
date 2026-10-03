# Encoder 1 CNN + GRU latent history correction 재검토

2026-10-03. 상태: **설계 검토 완료 / 구현·성능·GPU 실험 미실행**.
사용자가 제안한 결합 구조를 검토한 문서다. 앞선 단층 CNN 보정 후보와 별도 제안이며 실행 계약을 동결하거나 과거 계약을 변경하지 않는다.

## 권고

Encoder 1의 **event Q/K/V**에 causal depthwise kernel-3 residual을 추가하고, encoder 사이의 기존 8개 MLP 보정을 **64→24→단방향 GRU(24)→64 residual**로 교체하는 결합 후보를 권한다. 전체 주요 경로는 64차원으로 유지한다. 24차원은 기존 보정의 파라미터 수에 가깝게 두기 위한 제안이며 최적 폭이라는 결과는 없다. 16차원은 더 작은 후보다.

    log1p inputs + embedding
      → Encoder 1 [event Q/K/V + causal depthwise CNN; persistent bank unchanged]
      → h_i (64)
          ├─ identity residual ──────────────────────────┐
          └─ P_down (64→24) → GRU(s_prev) → P_up (24→64) ├─ add
                                                       ↓
      → Encoder 2 → static top-4 retrieval → existing quantity/duration heads

기존 Attention은 유지한다. CNN은 event Q/K/V 각각의 현재·직전 두 관측 사건을 필터링하고 persistent-bank 토큰에는 적용하지 않는다. GRU는 encoder 1의 contextual hidden state를 관측 순서대로 누적한다. 이 누적 상태가 기존 pair MLP와의 핵심적인 설계 차이다. 다만 encoder 1도 이미 과거를 참조하므로 GRU가 새로운 원자료를 얻는 것은 아니다.

전체 정보를 24차원에 강제로 통과시키지 않는다. 다음 residual 경로를 유지한다.

    z_i = P_down h_i
    s_i = GRU(z_i, s_previous)
    corrected_i = h_i + observed_i * 1[H_i >= 2] * alpha * P_up s_i

H_i는 현재를 포함한 correction segment의 관측 사건 수다. 첫 관측은 GRU 상태에 반영하되 residual은 이전 관측 사건이 있을 때 켠다. 이는 기존 8개 문턱 전체를 보존하는 것이 아니다.

## 초기화와 residual 계수

- CNN은 `q_new = q + causal_conv(q)` 형태이므로 **추가 kernel 전체 0**으로 초기화한다. current 계수를 1로 놓으면 2q가 되어 항등 초기화가 아니다. K/V도 동일하다.
- GRU의 입력 projection과 recurrent weight는 일반적인 비영 초기화, 출력 projection P_up만 0으로 초기화한다.
- 첫 후보는 고정 alpha=1을 제안한다. 기존 `/8` 제거는 후보 정의의 일부이며 순수 recurrence 효과와 구분한다. 여덟 분기의 합산 자체가 없어지더라도 residual 계수는 학습에 영향을 주므로 “제거해도 아무 차이가 없다”고 볼 수 없다.
- alpha와 P_up을 동시에 0으로 시작하면 양쪽 gradient가 막힐 수 있다. 이중 zero gate를 추가하지 않는다.
- P_up=0인 첫 backward에서는 GRU 쪽 보정 경로 gradient가 0이다. 이후 P_up이 갱신된 뒤 gradient를 확인해야 한다. 새 경로의 항등 초기화는 같은 공유 가중치의 무보정 모델 함수에서 시작한다는 뜻이며, 이미 학습된 MLP checkpoint 전체와 같다는 뜻은 아니다.

## 코드에서 확인한 연결 지점과 재사용 한계

- `models/TPPs/CountAwareTitanMultiLagDetail.py::_encode_base`: features→input projection→position→encoder.layers[0]→보정 residual→encoder.layers[1]. `encode_task_states`에서 이후 static memory가 적용된다.
- `models/TPPs/CountAwareTitanCoreAblation.py::HistoryCorrection`: 기존 8개 bias-free 128→4→64 GELU branch, output zero init, sum/8.
- `models/TPPs/CountAwareTitanCausalQKV.py::CausalQKVMemoryAttention`: event-only Q/K/V convolution, +576개, persistent K/V 연결 이전 위치, 기존 attention parameter 재사용을 참고할 수 있다.
- 기존 `CountAwareTitanCausalQKVTPP` 전체 모델·checkpoint validator는 legacy time head 및 별도 route에 묶여 있다. 이를 현재 observed-time 모델의 대체 클래스로 그대로 사용하지 않는다. 새 결합에는 현재 head/손실과 새로운 checkpoint identity가 필요하다.
- 기존 CNN helper는 **물리적인 tensor lag1/lag2**다. padding을 건너뛰는 관측 사건 순서 및 withheld-valid 경계 의미를 자동으로 만족하지 않는다. 새 설계에 재사용하려면 mask/segment 연결을 별도로 구현·검증해야 한다.
- 기존 correction의 segment reset과 encoder attention의 참조 범위는 다르다. GRU 상태를 reset한다고 encoder 1에 이미 포함된 모든 과거 정보까지 제거되는 것은 아니다. 전체 attention mask를 임의로 바꾸지 않는다.

## GRU 상태의 범위

첫 후보는 **각 sample/prefix 시작에서 상태를 0으로 초기화**한다. 같은 prefix 안에서만 이전 상태를 전달하고 batch/sample/개체 사이에는 넘기지 않는다. 겹치는 sliding window에 상태를 이어 붙이면 같은 사건을 반복 반영하거나 비교군보다 긴 이력을 보게 될 수 있다.

padding에서는 상태를 갱신하지 않는다. GRU에는 bias가 있으므로 입력을 0으로 만드는 것만으로 갱신을 막을 수 없다. withheld valid 위치에서는 correction용 상태와 이력 수를 reset하고, 현재 baseline의 관측·구간 의미를 따른다. 미래 값과 target은 입력에 넣지 않는다. train과 validation에 같은 규칙을 적용한다. 이는 지속적인 streaming memory 실험이 아니다.

## 파라미터 검산

로컬 PyTorch의 bias-free Linear 두 개와 1층·단방향·bias=True GRU를 실제 생성하여 numel 합산했다. 학습·forward·자료 평가는 실행하지 않았다. GRU 보정 수는 `128r + 6r² + 6r = 6r² + 134r`이다.

| 구성 | 비교되는 보정·CNN 파라미터 | 기존 대비 |
|---|---:|---:|
| 기존 History MLP | 6,144 | 기준 |
| CNN + 기존 MLP | 6,720 | +576 |
| GRU r=24만 | 6,672 | +528 |
| CNN + GRU r=24 | 7,248 | +1,104 |
| CNN + GRU r=16 | 4,256 | -1,888 |

공통 backbone 파라미터는 이 표에서 제외했다. r=24는 근접한 용량이며 정확한 parameter matching은 아니다. 파라미터 수와 실행 속도는 별개다. GRU에는 사건 순서에 따른 의존성이 있으므로 GPU 처리량·추론 지연을 실측해야 한다.

## 검증할 가설과 비교 설계

가설은 “국소 QKV 필터링과 관측창 내부의 recurrent residual이 함께 유용한가”다. 기존 입력 log1p, quantity loss, observed-time likelihood, 두 번째 encoder, 각 bank, heads, split, 선택 규칙을 유지한다. 기존 MLP baseline은 재사용하되 모든 새로운 조건은 같은 초기화·학습 규칙으로 별도 학습한다. baseline의 학습된 checkpoint에서 새 후보만 이어 학습하여 from-scratch 대조처럼 표현하지 않는다.

CNN 유무 × MLP/GRU 보정의 2×2:

1. 기존 encoder + MLP: 기존 결과 재사용.
2. CNN encoder + MLP: CNN 추가 효과.
3. 기존 encoder + GRU: 보정 모듈 교체 효과.
4. CNN encoder + GRU: 결합 후보.

이 분리는 모듈 수준의 비교다. GRU 교체에는 활성화/용량/계수 변경도 포함되므로 recurrence만의 인과 효과를 곧바로 입증하지 않는다. 결합에서 유망한 결과가 확인되면 같은 입력·residual 계수·활성 조건·유사 용량의 비순환 대조와 비교하고 여러 seed에서 재확인한다. validation에서 후보를 개발하며 이미 접근한 test를 미접근 평가로 표현하지 않는다.

## 논문 주장 범위

CNN과 Attention을 결합한 선행 구조가 있고 GRU도 잘 알려진 recurrent unit이다. 이 세 모듈의 조합만으로 새로움이나 여러 데이터에서의 우위를 입증하지는 않는다. 불규칙 사건의 시간 간격은 현재 입력 표현에 들어 있지만 일반 GRU가 명시적 시간 경과 기반 decay를 자동으로 구현하는 것은 아니다. 초기 후보에 새로운 decay·손실 변경·전체 latent 압축까지 동시에 추가하지 않는다.

- [Conformer 원 논문](https://arxiv.org/abs/2005.08100): CNN과 Transformer를 결합한 선행 예시. 이번 후보의 성능 근거는 아님.
- [GRU 비교 원 논문](https://arxiv.org/abs/1412.3555): 음악·음성 과제에서 gated recurrent unit을 비교. 수요/TPP 성능 보장은 아님.
- [PyTorch GRU 공식 문서](https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html): gate 식, recurrent weight와 bias의 형태.

## 남은 순서

1. **현재 관측 계약에 CNN·GRU 명세 연결 — 다음 작업**: 관측 사건의 CNN lag, GRU reset/gating, r=24, alpha=1, checkpoint identity를 구현 가능한 계약으로 정리한다.
2. **구현과 합성 검증 — 위 연결 이후**: 미래 값·padding·sample 순서·segment 경계·초기 항등과 gradient를 확인한다. 코드 호환성 조사와 parameter count를 전체 구현 검증으로 표시하지 않는다.
3. **2×2 validation 실험 준비 — 검증 이후**: 데이터·서버·시간·비용을 구체화한다. 이번 재검토에서 새 학습·원격 실행은 하지 않았다.
