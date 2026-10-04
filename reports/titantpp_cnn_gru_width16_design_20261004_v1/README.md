# 폭16 기준 CNN·GRU 모듈 대조 설계

2026-10-04. **설계 명세·파라미터 검산 완료 / 후보 구현·학습 미실행.**
이 문장은 설계 당시 상태다. 이후 세 후보의 구현·CPU 검증은
[별도 구현 기록](../titantpp_cnn_gru_implementation_20261004_v1/README.md)에 연결하며 아래 설계 JSON은 원래대로 보존한다.
범위는 Taxi·Intermittent·RAF이며 Instacart는 후속이다. 현재 진행할 폭8·12 대조와 별도 단계다. 기존 폭16을 후속 기준선으로 사용하고 폭4 결과를 비교 기록으로 보존한다.

## 문제와 권고

해결할 구조 문제는 **국소 사건 패턴을 Q/K/V에 반영하는 것과 관측창 안에서 누적 잠재 상태를 사용하는 것이 기존 pair MLP 보정보다 유용한가**이다. 폭 확대 자체의 효과는 먼저 폭4·8·12·16 대조로 판단한다. CNN·GRU가 큰 수량 오차 또는 시간 NLL 악화의 원인을 해결한다는 사실은 아직 확인되지 않았다. 이번 산출물은 모델 구조·실험 설계이며 서비스/UI/인프라 변경이 아니다.

주 대조에는 **Encoder 1의 event-only depthwise causal Q/K/V CNN(k=3)**과 **64→54→GRU(54)→64 잔차 보정**을 사용한다. GRU의 입력·상태 폭54는 보정 파라미터를 폭16 MLP에 가깝게 맞추기 위한 값이다. 최적 폭이라는 증거는 없다. 주요 표현은 64차원이며 identity 경로가 유지된다. 기존 제안의 GRU24는 폭4 MLP에 근접한 예산이었다. 기준선을 폭16으로 바꾸었으므로 이 문서는 그 예산 선택만 새로 정한다.

| 비교 가능한 선택지 | 구현 비용 | 운영·계산 비용 | 공동 작업 복잡도 | 이후 변경 비용·해석 |
|---|---|---|---|---|
| CNN + MLP16 | mask-aware event lag 연결 필요 | CNN +576개, 사건 축 병렬 처리 가능 | 기존 보정·checkpoint 형식을 대부분 유지 | CNN 추가 효과를 가장 직접적으로 구분 |
| GRU16 / CNN+GRU16 | GRU54와 같은 상태·마스크 구현 필요 | 보정3,680개로 작지만 순차 연산 존재 | 새 상태·checkpoint 계약 필요 | 용량 축소와 순환 방식 변경이 섞임; 경량 후속 후보 |
| **GRU54 / CNN+GRU54** | 상태·마스크·새 checkpoint identity 구현 | 보정24,732개; MLP16보다156개 많음, 지연은 실측 필요 | 두 GRU 조건이 동일 구현을 공유 | **선택안**: 보정 예산 차이0.635%로 줄이고 모듈 교체를 비교 |

작은 잠재 상태 자체를 검증하려면 GRU16을 별도 대조해야 한다. GRU54를 ‘16차원 latent 실험’으로 부르지 않는다. 근접한 파라미터 수는 동일한 함수 용량·FLOPs·속도 또는 순수 recurrence 대조를 뜻하지 않는다.

## 2×2 대조 계약

| 조건 | Encoder 1 | 중간 보정 | 보정+CNN 파라미터 | 전체 참조값(max_len256) | 상태 |
|---|---|---|---:|---:|---|
| MLP16 | 기존 attention+bank | 8×(128→16→64) | 24,576 | 114,435 | 기존 결과 재사용 |
| CNN+MLP16 | event Q/K/V에 CNN 추가 | 동일 MLP16 | 25,152 | 115,011 | 구현 필요 |
| GRU54 | 기존 attention+bank | 64→54→GRU54→64 | 24,732 | 114,591 | 구현 필요 |
| CNN+GRU54 | event Q/K/V에 CNN 추가 | 동일 GRU54 | 25,308 | 115,167 | 구현 필요 |

모든 조건은 Encoder 2→static top-4 prototype retrieval→기존 수량·시간 head를 공유한다. 전체 수는 기존 공통부의 참조값에 새 모듈 수를 더한 **계산값**이며 완성 모델을 생성해 확인한 값은 아니다. 위치 embedding 길이에 따라 달라진다. RAF max_len84 참조값은 순서대로103,427 /104,003 /103,583 /104,159개다. 실제 동결 설정으로 생성한 전체 numel과 trainable numel을 실행 전 기록해야 한다.

향후 이 2×2를3seed에 적용하면 기존9조건(Taxi·Intermittent·RAF×42·52·62)을 재사용하고 **3개 신규 구조×3개 데이터×3seed=27개 신규 조건**이다. 현재 폭8·12의18조건 학습 큐에 이27조건을 추가하지 않는다. 구현·원본 연결·자원 qualification이 끝난 후 별도 실행 계약으로 배치한다. 서버5080·5090의 독립 GPU 조건은 병렬 가능하나 공통 모델 구현·계약 검증은 먼저 완료해야 한다.

## 변경 경계

| 항목 | MLP16 | GRU54 대체 |
|---|---|---|
| 즉시 보정 입력 | 현재·직전 관측 hidden pair | 현재 hidden을 투영하고 이전 recurrent state 사용 |
| 누적 범위 | 같은 pair를8개 분기에 제공 | 같은 sample/prefix 안의 correction segment 전체 |
| 활성화 | 기존8개 관측 이력 문턱 | 첫 관측은 상태만 갱신, 두 번째부터 보정 |
| 크기 계수 | 분기 합/8 | 단일 보정×고정 alpha1 |
| 비선형·게이트 | GELU | GRU update/reset gate와 tanh |
| 파라미터 | 24,576 | 24,732 |

따라서 GRU 비교는 **문맥·활성화·계수까지 포함하는 보정 모듈 교체 효과**다. recurrence만의 인과적 이득이라고 쓰지 않는다. 그런 주장을 하려면 이후 같은 입력 범위·활성 조건·계수·근접 용량의 비순환 대조를 추가해야 한다.

## 구현 가능한 연산·마스크 명세

**CNN:** 첫 encoder attention의 event Q/K/V projection 직후, persistent K/V를 연결하기 전에 적용한다. Q/K/V마다64채널의 bias-free kernel(3,64)을 독립 사용한다. `q_new = q + Σ_l kernel[l] * q[observed_predecessor_l]`, l=0,1,2이며 K/V도 같다. 현재와 직전 두 **관측 사건**만 사용하고 누락 lag는0이다. padding은 사건 수를 늘리지 않는다. valid이지만 관측 허용이 아닌 위치는 CNN의 추가 경로에 대해 segment를 끊는다. persistent bank 토큰은 convolution 대상에서 제외한다.

기존 attention에 전달되는 observed mask만으로 padding과 withheld-valid를 구분할 수 없다. `_encode_base`에서 valid·observed로 segment/predecessor 정보를 만들고 첫 attention에 **명시적 인수**로 넘기거나 동등한 순수 함수 경로를 마련해야 한다. 모듈의 숨은 mutable cache로 batch별 마스크를 전달하지 않는다. 기존 attention mask와 참조 범위는 유지한다. CNN/GRU segment reset은 Encoder 1이 이미 contextualized hidden에 담은 더 이전 문맥까지 제거한다는 뜻이 아니다.

**GRU:** `z_i = Linear_no_bias(64,54)(h_i)`, `s_i = GRUCell_equivalent(z_i,s_previous)`, `Δh_i = observed_i * 1[segment_count_i≥2] * Linear_no_bias(54,64)(s_i)`, `h_corrected=h+Δh`. 구현은1층·단방향·bias=True GRU와 동등한 gate 식을 사용한다. GRU dropout0, 별도 LayerNorm·learned alpha·시간 decay를 추가하지 않는다. sample/prefix 시작마다 s=0; padding은 state/count를 유지하고 output correction0; withheld-valid는 state/count reset과 correction0; 첫 observed는 state update하되 correction0. batch/sample/개체/겹치는 window 사이에는 상태를 넘기지 않는다.

**초기화:** CNN의 추가 kernel 전부0; GRU down projection과 gate weight는 비영 기본 초기화, up projection만0; alpha1 고정. 신규 초기화는 공통 모델 RNG를 보존하는 fork 안에서 수행한다. 공유 가중치를 맞춘 초기 모델은 무보정 경로와 동일하며, 학습 완료 MLP checkpoint 전체와 동일하다는 뜻은 아니다. MLP16과 CNN+MLP16는 공통 가중치·MLP 초기값까지 같아야 한다. GRU-only와 CNN+GRU도 같은 GRU 초기값을 사용한다. CNN kernel0와 GRU up0의 상호작용 및 첫 optimizer step 이후 gradient 도달을 검증한다.

## 공통 과학 조건과 판단

- 입력 변환(log1p), 출력부, log 수량 손실, observed-time likelihood `positive_integer_round_clamp_v1`, Encoder2, 두 persistent bank, 마지막 prototype bank, optimizer·학습률·batch·loader·분할·seed를 데이터별 기존 width16 동결 계약과 동일하게 유지한다.
- seed42/52/62, min40/max300/patience40 및 **최초 최소 finite 전체 validation raw 수량 RMSE** checkpoint 선택을 유지한다. 큰 수량 지표나 Time NLL로 사후 epoch를 바꾸지 않는다.
- 전체RMSE·MAE·큰 수량RMSE·Time NLL을 같은 선택 epoch에서 읽는다. 큰 수량 경계는 train에서 정한 기존값을 재사용하며 validation으로 다시 정하지 않는다. train과 validation 진단을 분리해 보존한다.
- 3seed 평균·표본 표준편차, seed별 변화 방향과 범위를 모두 보고한다. 일부 지표의 개선과 시간 NLL 손해를 상쇄한 임의 종합점수로 승자를 정하지 않는다. 이번 설계로 새로운 수치 중단 기준을 만들지 않는다.
- 같은 seed별 CNN 효과는 `CNN+MLP16 − MLP16`과 `CNN+GRU54 − GRU54`를 각각 계산한다. GRU 모듈 교체는 CNN 유무별로 비교한다. metric별 interaction `combined − cnn_only − gru_only + baseline`도 기록하되 작은3seed만으로 통계적 유의성·강건성을 단정하지 않는다.
- validation으로 후보를 개발한다. 기존에 열람한 Test를 독립 미접근 평가라고 하지 않으며 이번 설계에서는 새 Test 평가나 Test 기반 후보 선택을 실행하지 않는다.

## 코드 근거·검증 상태

연결 지점은 `CountAwareTitanMultiLagDetail._encode_base`의 첫 encoder 뒤 보정, 그리고 첫 `MemoryAttention` 내부 event QKV다. `CountAwareTitanCausalQKV.CausalQKVMemoryAttention`의 파라미터 재사용·영 kernel 방식을 참고하되, 기존 helper는 물리적 tensor lag이고 전체 candidate 클래스는 legacy time head라서 현재 모델에 그대로 연결할 수 없다. 각 신규 모델에 고유 alias/metadata/identity 및 잘못된 체크포인트 relabel 거부가 필요하다.

`verify_design.py`가 실제 CPU PyTorch Linear/GRU/Parameter를 생성하여 수를 검산했다. `verification.json`에 결과와 읽은 코드 SHA를 보존했다. 수식은 MLP `1536w`, GRU `6r²+134r`, CNN `9×64=576`이다. 자료·checkpoint·forward·학습은 읽거나 실행하지 않았다.

프로젝트 `.codex/agents/architecture-reviewer.toml`은 역할 모델을 `gpt-6-astra`, effort `high`로 지정한다. 이는 확인한 설정값이며 실제 서비스 실행 모델을 별도로 검증한 기록은 아니다. 지정된 `.agents/skills/_shared/runtime/execution-protocols/codex.md`는 현재 경로에 없었다. `.agents/` 수정 금지와 소유 범위에 따라 결과를 이 reports 폴더에 저장했다.

**남은 검증 — 다음 작업**

- 새 모델을 구현한 뒤 미래 값·padding 삽입·withheld 경계·batch 순서·sample 독립성, CNN event-only 적용과 GRU 첫 관측 처리, 첫 초기 항등과 optimizer update 뒤 gradient를 검증한다. padding 삽입 시 위치 embedding 변화와 모듈 마스크 효과를 구분해 모듈 입력 수준 검증도 포함한다.
- 동일 frozen checkpoint의 공유 head/출력/metric 재현, 새 identity mismatch 거부, 실제 전체 numel을 검사한다. GRU에서 GPU 비결정적 kernel 문제가 있으면 묵시적으로 deterministic 조건을 낮추지 않고 qualification 실패로 기록한다.
- 데이터별 GPU 최대 메모리·epoch 속도·추론 지연을 측정하고 실행 계약을 동결한다. 파라미터 수로 속도나 비용을 추정하지 않는다.
- 폭8·12 대조와 시간 예측 진단을 마친 후 이27조건 실행 범위·배치를 정한다. 검증되지 않은 CNN·GRU 구조를 지금의 width 학습과 섞지 않는다.
