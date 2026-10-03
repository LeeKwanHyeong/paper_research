# TitanTPP HistoryCorrection의 Causal CNN 교체 제안

상태: **설계 검토 완료 / 구현·실험 전 제안**. 서브 세션에서 확정한 명세로 간주하지 않는다. 이 문서는 실행 계약이나 GPU 실행 승인이 아니다.

## 첫 변경 위치

두 causal encoder 사이의 HistoryCorrection을 교체한다. 현재 64차원 encoder, persistent bank, 최종 static top-4 retrieval, quantity/duration head를 유지한다. 이는 사용자가 제공한 기존 Figure 1의 초록색 블록에 해당한다.

| 항목 | 기존 History MLP | 첫 CNN 후보 |
|---|---|---|
| 삽입 위치 | encoder 1과 encoder 2 사이 | 동일 |
| 직접 참조하는 상태 | 현재 + 직전 관측 상태 | 현재 + 직전 2개 관측 상태 |
| 연산 | 서로 다른 8개 MLP, 각각 128→4→64, GELU | bias 없는 dense causal Conv1d 64→24, kernel 3 → GELU → pointwise Conv1d 24→64 |
| convolution 설정 | 해당 없음 | stride 1, dilation 1, groups 1; 출력 convolution kernel 1 |
| 중간 특징 | 분기당 4, 최대 8분기 | 하나의 24채널 표현 |
| 활성화 | 관측 이력 수가 1,2,4,8,16,32,64,128 문턱을 넘을 때 각 분기 활성화 | 현재 관측 위치에서 이전 관측 사건이 하나 이상 있을 때 활성화 |
| 불충분한 이력 | 이전 관측 상태가 없으면 0 보정 | 두 번째 이전 상태가 없으면 0 벡터; 이전 상태가 전혀 없으면 보정 전체 0 |
| residual 계수 | 8개 출력을 마스킹 후 합산하여 /8 | 단일 출력을 /8; 이제 분기 평균이 아니라 고정 계수 |
| 출력 초기화 | 8개 출력 행렬을 0으로 초기화 | pointwise 출력 행렬을 0으로 초기화 |
| 보정 파라미터 | 8×(128×4+4×64)=6,144 | 64×24×3+24×64=6,144 |
| 전체 모델 파라미터 | 데이터별 기존 모델의 수 | 같은 데이터의 기존 모델과 동일 |

여기서 “3개 상태”는 **보정이 직접 읽는 창**이다. 각 상태는 encoder 1에서 허용된 더 긴 과거를 이미 반영한다. 모델 전체의 과거 범위가 3개 사건으로 제한되는 것은 아니다. 창의 단위는 관측 사건이며 날짜/일수 간격이 아니다.

## 제안 연산과 마스킹

관측된 상태를 각 유효 구간 안에서 시간 순서로 모은다. 현재 상태를 h_i, 직전과 두 번째 이전 관측 상태를 h_prev, h_prev2라고 하면:

    u_i = GELU(W0 h_i + W1 h_prev + W2 h_prev2)       # 24차원
    correction_i = observed_i * 1[history_count_i >= 2] * V u_i / 8
    corrected_i = h_i + correction_i                # 64차원

W0, W1, W2는 각각 24×64, V는 64×24이며 bias는 없다. history_count는 현재를 포함한 구간 내 관측 사건 수다. 누락된 h_prev2는 0이다. 단순한 tensor 행 이동으로 padding을 사건으로 계산해서는 안 된다. 현재 기준선의 segment 규칙을 따라 withheld valid row에서는 참조 구간을 끊고, target 및 미래 위치는 보정의 입력으로 쓰지 않는다. batch padding 위치의 보정은 0이다. 정확한 gather/compact 구현은 다음 구현 단계에서 기준선과 대조한다.

## 공통 실험 기준에 연결

- 입력의 log1p, d_model=64, 두 encoder, 각 persistent bank, static retrieval, 예측 head, 시간 likelihood, quantity 목적함수는 기존 observed-time History-MLP 기준을 유지한다.
- 동일 데이터의 기존 train/validation 분할·전처리·seed·optimizer·batch 설정을 기준으로 삼는다. 첫 탐색 seed42, min40/max300/patience40, 최초 최소 raw quantity validation RMSE epoch 선택을 유지하는 것이 제안이다. 기존 baseline을 재학습하지 않는다.
- 새 후보 개발 판단은 validation에서 한다. 이미 접근한 test를 새로운 미접근 독립 평가로 표현하지 않는다. 이 설계 작업에서는 성능/예측 파일을 새로 열지 않았다.
- 6,144개 보정 파라미터가 같아도 연산량, 학습 속도, 메모리, 실제 residual/gradient 크기가 같다는 뜻은 아니다. 전체 파라미터 수는 데이터별 positional embedding 길이 등과 함께 확인한다.
- `/8`은 명시적으로 유지하지만 기존 여덟 출력 합과 후보 단일 출력의 규모까지 맞추는 통제는 아니다. 보정 RMS·gradient 크기 확인은 후속 구현 검증 항목이다.
- 현재 RunPod routing/placement 실험과 별도 후보다. 기존 계약·실행 소스·학습·예산은 변경하지 않는다.

## 무엇을 검증할 수 있는가

검증 가설: **직전 두 관측 상태까지 직접 읽는 단일 보정 모듈이 기존 8분기 보정보다 유용한가?**

이는 참조 창, 중간 특징 구성, 분기 활성화 규칙이 함께 바뀌는 모듈 대조다. CNN만의 효과, 4차원 bottleneck만의 효과, 단계적 활성화만의 효과를 분리한 실험으로 해석하면 안 된다. 기존 all-available control도 이 차이를 모두 통제하지 않는다.

특히 `kernel=2, 64→4→64`를 8분기로 구성하고 같은 가중치·gate·/8을 쓰면 기존 pair MLP와 대수적으로 같은 연산이다. 제안한 단일 kernel=3 convolution 역시 세 상태를 연결한 공유 MLP `192→24→64`로 표현할 수 있다. 따라서 **CNN이라는 이름만으로 새 구조의 참신함이나 성능 우위를 주장할 수 없다**. 신호가 확인되면 같은 참조 창·용량을 가진 단순 window MLP 등과 비교하여 유효한 설계 차이를 분리해야 한다. 대수적으로 동일한 구현을 별도 과학 비교군으로 중복 학습할 필요는 없다.

일반적인 causal convolution 배경: Bai et al., [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271). 이 논문은 이 후보의 성능을 보장하는 근거로 사용하지 않는다.

## 다른 변경 위치를 먼저 선택하지 않는 이유

| 위치 | 판단 |
|---|---|
| HistoryCorrection 교체 | 첫 후보. 기존 그림과 baseline에 바로 대응하며 주변 경로를 유지할 수 있음 |
| encoder 1의 Q/K/V convolution | 이미 별도의 legacy causal-QKV 실험·구현이 있음. 현재 보정 교체와 같은 연구로 합치지 않음 |
| attention encoder 자체를 CNN으로 교체 | backbone 전체 표현과 계산 특성이 함께 바뀜. 첫 모듈 검증 이후 별도 연구 질문으로 다룰 범위 |

## 재사용한 근거

- `models/TPPs/CountAwareTitanCoreAblation.py`: 기존 HistoryCorrection.
- `models/TPPs/CountAwareTitanMultiLagDetail.py`: 기존 관측 이력 참조와 downstream 연결.
- `paper/titantpp_pakdd_2027_draft/main.tex`: 현재 Figure 1과 모델 설명.
- `reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluator_sources/core/models/TPPs/`: 과거 평가에 묶인 source. 현재 관련 baseline 파일과의 연결은 직전 bottleneck 검토를 재사용.
- `reports/titantpp_logloss_bottleneck_review_20261003_v1/`: 기존 log 변환·국소 bottleneck 검토.
- `models/TPPs/CountAwareTitanCausalQKV.py`, `paper/contracts/hard_lmm_causal_qkv_v1.md`: 기존 QKV convolution은 별도 구조.

## 남은 작업 순서

1. **보정 모듈 구현과 인과성 검증 — 다음 작업**: 독립 후보로 구현하고 미래 상태 변경, padding 삽입, 구간 경계, 짧은 이력, zero-output 초기화 및 파라미터 수를 합성 자료로 검증한다. 기존 모델 소스는 유지한다.
2. **기존 실험 기준에 실행 계약 연결 — 위 검증 이후**: 대상 데이터·실행 자원·비용·비교 범위를 구체화한다. 현재 문서의 제안을 실행이 확정된 계약으로 취급하지 않는다.
3. **validation 탐색과 해석 — 준비 이후**: 이득과 악화를 모두 기록하고 보정 모듈 전체의 효과로 해석한다. 유망할 때 다중 seed와 원인 분리 검토를 진행한다.
