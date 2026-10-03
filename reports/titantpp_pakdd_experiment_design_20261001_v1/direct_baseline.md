# 다음 사건 수량 예측에 직접 대응하는 비교군

**Deep Renewal을 우선 비교군으로 선정한다 — 원문·공식 소스 검토 완료**

현재 여섯 외부 비교군은 공통 head로 연결한 encoder 비교다. 이를 보완하려면 간격과 수량을 원래부터 함께 다루는 수요 예측 방법이 필요하다. 우선 비교군은 **Deep Renewal의 interval–size 모델**로 정한다. [원 논문](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764)은 간헐 수요를 사건 간격과 양의 수량으로 표현하고 RNN으로 분포를 갱신한다. RAF도 실험에 포함한다. 원 논문의 달력 구간 예측과 우리의 다음 사건 예측은 평가 대상이 다르므로 여기서는 다음 사건에 맞춘 적용으로 표기한다.

공식 GluonTS의 [고정 revision](https://github.com/awslabs/gluonts/tree/889a3df86a89a365880b4bc1488bcf4c039f265e/src/gluonts/mx/model/renewal)을 기준으로 `_network.py`, `_estimator.py`, `neg_binomial.py`를 대조했다. [소스 목록](sources.json)에 로컬 사본·URL·SHA를 남겼다. MXNet 실행이나 정식 이식 검증은 아직 하지 않았다.

### 계산식과 출력

이하 식은 확인한 GluonTS 코드의 파라미터화를 사용한다. 마지막 관측까지의 interval–size를 인과 LSTM에 넣어 다음 두 평균을 얻는다.

\[
h_i=\operatorname{LSTM}((d_i,q_i),h_{i-1}),\quad
(\mu_{d,i+1},\mu_{q,i+1})=\operatorname{softplus}(Wh_i+b)+10^{-5}.
\]

두 dispersion은 각각 학습되는 **전역 scalar**다: \(\alpha_j=\operatorname{softplus}(a_j)+10^{-5}\). \(X\sim\operatorname{NB}(\mu,\alpha)\)이면 \(E[X]=\mu\), \(\operatorname{Var}(X)=\mu+\alpha\mu^2\), \(r=1/\alpha\)이고,

\[
\log p(k)=\log\Gamma(k+r)-\log\Gamma(k+1)-\log\Gamma(r)
+k\log\frac{\alpha\mu}{1+\alpha\mu}-r\log(1+\alpha\mu).
\]

간격과 수량은 각각 \(1+\operatorname{NB}\)이며, 수량 점예측은 \(1+\mu_q\)다. RMSE와 MAE에 같은 평균 예측을 사용한다. 미래 정답 간격은 수량 예측의 입력에 넣지 않는다.

### 현재 비교 조건으로 연결하는 범위

| 구성 | 결정 |
|---|---|
| 모델 이름 | `deep_renewal_event_native_nb` — native 분포를 유지한 다음 사건 adapter |
| recurrent 구조 | LSTM hidden64, 2층, dropout0.1. 기존 비교의 hidden 폭과 맞추되 전체 파라미터 동등성은 주장하지 않음 |
| 초기화 | GluonTS dense bias5·dispersion bias2·Xavier 경로를 기준으로 이식. MXNet LSTM gate/forget-bias 및 padding 의미는 구현 검증 완료 전 실행 금지 |
| 입력 정보 | 기존과 같은 과거 간격·수량, 같은 window·target ID. 공식 interval–size 경로처럼 raw pair 사용; TitanTPP의 log1p 변환과는 구별 |
| padding | 가짜 padding은 LSTM에 관측으로 넣지 않음. 인과 shift의 시작 zero token은 한 번만 처리. 공식 구현의 right-aligned 동작과 달라지는 부분을 adapter 변경으로 기록 |
| 손실 | 마지막 target의 간격 NLL + 수량 NLL. TitanTPP의 수량 log-MSE head로 대체하지 않음 |
| 학습 범위 | 두 구조 비교와 같은 전체 train target, batch128, max300/min40/patience40, seed42·52·62, AdamW lr0.001/weight_decay0.01/clip1.0 |
| 선택/평가 | 최초 최소 validation raw 수량 RMSE; 동일 checkpoint MAE·RMSE·시간 NLL. quantity NB NLL은 부가 지표 |
| 신규 조건 | 네 데이터 × 3seed = **12조건**, 특정 데이터의 성능을 보고 포함 여부를 바꾸지 않음 |

이 구성은 원 논문의 소규모 데이터 LSTM20×1·Adam lr0.1·달력 horizon 설정을 그대로 재현하는 실험이 아니다. 공유된 학습 조건에서 native 분포를 보존한 직접 비교다. 단일 설정이므로 원 방법의 튜닝된 최상 성능이라는 표현도 피한다. 초기 loss·gradient·학습률 적합성은 train/validation에서 먼저 확인하고 변경이 필요하면 조건을 다시 고정한 뒤 모든 데이터·seed에 동일하게 적용한다. 유리한 비교 결과를 보고 사후에 설정을 고르지 않는다.

### 시간 관측과 자료 지원 범위

[train·validation 지원 범위 검사](data_support.json)에서 네 데이터의 수량은 전부 양의 정수였다. 원본 간격 0은 모두 각 entity의 첫 train 행이며 target으로 사용되지 않는다. **기존 loader가 history의 0을 1로 바꾸는 규칙을 그대로 공유**한다. 이 비교를 위해 새 행을 제외하거나 수량을 반올림하지 않는다.

| 데이터 | train target | validation target | 시간 관측 |
|---|---:|---:|---|
| Taxi | 38,393 | 8,268 | 양의 정수 시간 간격 |
| Intermittent | 393,824 | 86,285 | 양의 정수 주 간격 |
| Instacart | 1,991,192 | 503,733 | 양의 정수 일 간격, 30 top-code |
| RAF | 25,779 | 6,690 | 양의 정수 월 간격 |

Instacart는 기록된 30을 정확히 30인 잠재 간격으로 취급하지 않는다. \(D=1+X\)일 때 기록값30의 질량은 \(P(D\ge30)=P(X\ge29)\)다. 작은 tail에서의 cancellation을 막는 미분 가능한 log-survival 구현과 합성 극단값·gradient 검증이 구현 완료 조건이다. 다른 데이터는 해당 정수의 NB 질량을 쓴다. 기존 lognormal의 정수 bin 질량과 **동일하게 기록된 사건에 대한 시간 NLL**을 비교한다. 서로 다른 quantity head의 총 학습 loss 수치를 성능 순위로 사용하지 않는다.

### 값싼 예측 기준과 FlexTPP의 위치

같은 관측 window에서 **마지막 수량**과 **관측 수량 산술평균**을 계산하는 두 비학습 기준도 추가한다. 데이터별 한 결과이며 seed 세 개의 독립 학습처럼 복제하지 않는다. 다음 사건 수량 MAE·RMSE를 보고하고, 분포가 없는 기준의 시간 NLL은 N/A다. 이는 공급망의 전체 달력 수요량 예측 실험과 구분한다.

[FlexTPP 원문](https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html)과 [연결 저장소](https://github.com/czi-ai/FlexTPP)는 mixed-type autoregressive 예측을 다룬다. 확인한 저장소는 스스로 unofficial implementation으로 표기한다. 속성별 flow·조건부 샘플링을 유지한 adapter에는 시간 관측 변환, 미래 시간을 적분한 수량 점예측, decoder 수치 검증이 추가로 필요하다. 현재의 간헐 수요 논점을 먼저 검증하기 위해 Deep Renewal을 선정했고 FlexTPP는 후순위로 둔다. 성능을 확인하고 제외한 것이 아니다. 원고가 일반적인 mixed-type TPP 우월성을 주장한다면 FlexTPP 비교 범위를 다시 검토해야 한다.

**실제 비교를 실행할 조건 — 다음 작업**

PyTorch 이식·native 식/초기화 대조·인과성·padding·top-code·유한 gradient·target pairing·checkpoint 재개 검증 후 source를 동결한다. 그 뒤 현재 5080/5090 가용 상태와 실측 비용을 확인해 호스트·조건별 상한·전체 마감을 정한다. 기존 GPU 캠페인이나 2026-09-27의 광범위 계획을 이번 12조건의 실행 승인으로 재사용하지 않는다.
