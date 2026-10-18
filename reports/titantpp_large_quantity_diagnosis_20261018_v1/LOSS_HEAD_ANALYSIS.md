# 큰 수량 오류의 손실·출력 경로 검토

코드·수식 검토 완료. Cap27 실행, 기존 모델·결과를 변경하지 않았고 새 학습·추론·원격 호출 및 Test 결과 열람은 하지 않았다. 이 문서는 행별 오류 분석 결과가 아니라 그 해석에 필요한 계산 경로를 정리한다.

## 확인한 실행 경로

Cap36 계약의 source revision은 `e33f540c02b5de69fcd75e84375ec0f5c6a89bd3`이다. 관련 소스 10개를 현재 파일과 SHA 대조했으며 모두 계약 바이트와 일치했다. 목록과 해시는 JSON에 기록했다.

| 위치 | 확인 내용 |
|---|---|
| `paper/scripts/count_aware_tpp_backbone/core.py:29` | demand_qty를 Float64 scale_residual로 전달한다. 이 전처리에서 log 변환이나 예측 clipping을 하지 않는다. |
| `core.py:62–105` | 마지막 target 수량을 가리고 관측 이력의 상태로 예측한다. 공동 손실은 시간 NLL + λ·수량 손실이며 현재 λ=1이다. |
| `models/TPPs/CountAwareTPP.py:434–448` | 공통 입력은 log1p(dt), log1p(q)이다. 이 함수에 Train μ·σ 표준화는 없다. |
| `models/TPPs/CountAwareTitanMultiLagDetail.py:225–238` | 두 입력을 학습된 투영으로 표현하고 Encoder·이력 보정·메모리를 거친다. MLP 폭 후보는 이 클래스를 상속한다. |
| `models/TPPs/CountAwareFactory.py:447–462` | 해당 MLP 후보·구조 대조군은 공통 log 회귀 head와 시간 head를 유지한다. |
| `models/TPPs/CountAwareTPP.py:390–409` | 수량 head는 Linear(64,1). train_log_mean은 초기 bias를 정한다. train_log_std는 활성 log-MSE의 분모가 아니며 선택하지 않은 lognormal variant의 scale 초기화에 쓰인다. |
| `models/TPPs/CountAwareTPP.py:770–821` | ẑ=softplus(a), p=expm1(ẑ), 수량 손실=(ẑ−log1p(q))². scale 출력은0이고 분포/꼬리 보조 손실도0이다. |
| `paper/scripts/count_aware_tpp_backbone/training.py:829–865,885–892` | log 손실로 학습하되 전체 Validation raw RMSE가 처음 엄격히 최소가 되는 epoch를 선택한다. 선택 기준과 최적화 손실은 다르다. |
| `paper/scripts/run_titantpp_mlp_cap36.py:182–216` | q>cap은 빈 꼬리로 N/A, q==cap은 별도 그룹. 예측값 자체는 cap으로 자르지 않는다. |

따라서 **현재 손실을 ‘정규화된 log 수량 MSE’라고 설명하면 틀린다.** 활성 경로는 표준화하지 않은 log1p 수량 MSE다. 실수의 정확한 대수에서는 expm1(softplus(a))=exp(a)이므로 출력은 양수이며 유한한 상한이 없다. 실제 구현의 부동소수점 오차와 softplus 선형 근사 구간은 이 대수적 설명과 구분한다. 비활성 tail variant에 있는 clamp를 현재 출력의 clipping으로 읽으면 안 된다.

## log 손실은 실수량 RMSE와 다른 오차를 줄인다

q≥0, 예측 p>0이면 L=[log((1+p)/(1+q))]²이다. 직접 미분하면

- ∂L/∂p = 2 log((1+p)/(1+q))/(1+p)
- ∂L/∂a = 2[softplus(a)−log1p(q)] sigmoid(a)
- p=q+δ이고 |δ|≪1+q이면 L≈δ²/(1+q)²
- p=(1+r)q, r>−1이면 q가 클 때 L→[log(1+r)]²

같은 작은 절대 오차는 큰 q에서 더 작게 벌점을 받는다. 같은 비율 오차는 큰 q에서 비슷한 벌점을 받는다. 다만 이것은 **raw 예측 p에 대한 손실 기하**다. 전체 backbone gradient에는 ∂a/∂θ와 공유 시간 손실도 관여하므로 “큰 수량 행의 모든 파라미터 gradient가 항상 더 작다”로 일반화할 수 없다. log 손실은 큰 절대 규모의 영향을 줄이지만 유계 손실이나 모든 이상치에 강건한 보장은 아니다.

다음은 실제 데이터와 무관한 수식 예시다.

| q | p | 절대 오차 | raw 제곱 오차 | log-MSE |
|---:|---:|---:|---:|---:|
| 10 | 9 | 1 | 1 | 0.00908403037 |
| 1000 | 999 | 1 | 1 | 9.99000916e-07 |
| 10 | 5 | 5 | 25 | 0.367400612 |
| 1000 | 500 | 500 | 250000 | 0.479069793 |

## 평균과 중앙값을 구분한다

관측 정보 X를 고정하고 충분한 표현력과 완전한 최적화를 가정한 **수량 손실 단독의 모집단 최적값**은 z*(X)=E[log(1+Q)|X], p*(X)=exp(z*)−1이다. 이는 Q+1의 조건부 기하평균에서1을 뺀 값이며 일반적으로 raw 평균도 raw 중앙값도 아니다.

Jensen 부등식으로 p*(X)≤E[Q|X]다. raw 제곱오차의 최적값은 조건부 평균이고 절대오차는 조건부 중앙값이므로, 학습 목표와 보고 지표가 일치하지 않을 수 있다. log(1+Q)|X가 대칭이라는 추가 가정이 있을 때 log 평균과 log 중앙값이 같아져 역변환값을 raw 중앙값으로 설명할 수 있다. 현재 코드는 수량의 조건부 Gaussian 분포나 분산 보정을 구현하지 않으므로 이를 무조건 중앙값이라고 부르거나 exp(μ+σ²/2)로 보정해야 한다고 단정하지 않는다. [scikit-learn 공식 지표 설명](https://scikit-learn.org/stable/modules/model_evaluation.html), [저자 공개 교재의 역변환·편향 보정 설명](https://otexts.com/fpp3/toolbox.html).

이 모집단 결과는 유한한 모델의 개별 예측이나 실제 과소예측 비율을 증명하지 않는다. 공동 시간 손실, 용량, 최적화, 규제, 표본 수, 조기 종료가 모두 영향을 줄 수 있다. Validation raw RMSE로 checkpoint를 선택해도 학습 gradient 자체가 raw MSE로 바뀌지는 않는다. log1p 입력은 수학적으로 단조·가역이므로 값을 단순히 ‘잃는다’고 할 수 없다. 입력 압축이 표현 학습에 미친 효과는 target log 손실의 효과와 별개다.

## Q3 Cap은 예측 목표와 관측 이력을 함께 바꾼다

`benchmark_data/Cap/CAP_POLICY.json`은 상한 적용 전 Train 전체 event 행의 Q75를 기준으로 Qc=min(Q,C)를 모든 split에 적용한다. C는 Intermittent20, RAF10, Taxi65다. 개체 첫 event도 threshold 산출에 포함된다. 모델에는 원래 수량을 별도로 주지 않는다.

P(Qc=C|X)=P(Q≥C|X)이므로 cap 위치의 질량에는 원래 C였던 행과 C보다 컸던 행이 함께 들어간다. 이 비율이 정확히25%라는 뜻은 아니다. 이산값 동률과 split 분포 차이를 고려해야 한다. q>cap 그룹은 정의상 비어 있고 q==cap 그룹이 실제로 분석 가능한 별도 집단이다. 상한을 넘은 원래 값과 이력의 차이는 capped 수량만으로 구분할 수 없다.

현재 손실에는 top-code를 위한 censored likelihood가 없다. 잘린 Qc 자체를 정답으로 학습·평가하는 **별도 수량 과제**다. 출력 p를 C로 자르지 않으므로 p>C도 가능하다. Cap에서 오류가 작아져도 상한 적용 전 큰 수량을 잘 예측한다는 결론으로 옮길 수 없다. Cap에서도 조건부 log 평균의 역변환과 조건부 raw 평균은 일반적으로 다르다.

## 다음 해석에 필요한 진단과 선택지

**저장 Train·Validation 행 분석에 적용할 해석 기준.** 추가 계산량과 구현 비용이 작고 기존 실험을 건드리지 않는다. 고정 Train 수량 구간별 support·signed error·AE·SSE·log error를 seed와 개체 구성에 맞춰 비교한다. 큰 실제값으로 나눈 집단의 과소예측만으로 calibration 실패를 주장하지 않는다. 그 집단은 이미 큰 정답을 조건으로 선택했기 때문이다. Train에서 정한 예측값 또는 관측 이력 기준 집단의 raw/log 평균 잔차도 함께 확인한다. Cap에서는 cap-hit/비cap, 상한 적용 전 자료에서는 반복되는 큰 수준/갑작스런 상승, 드문 과대예측이 SSE를 지배하는 경우를 나눈다. 저장 epoch 곡선이 있으면 Train부터 지속되는 미적합과 Validation에서만 생기는 차이를 구분한다. 모델 규모가 크다는 사실만으로 과적합이나 데이터의 단순함을 확정하지 않는다.

**목표 손실·head 대조 학습 — 현재 미실행.** 직접 raw-MSE나 분포 기반 평균을 비교하면 estimand 차이를 검증할 수 있지만, backbone만 바꾸는 실험과는 연구 질문이 달라지고 GPU·설계·해석 비용이 더 크다. 현재 Cap27을 바꾸지 않는다. 이번 코드 분석만으로 새 학습이나 편향 보정을 정당화하지 않으며 행 분석 결과로 필요성을 판단한다. 기존 결과를 새로운 head로 재평가하지 않는다.

참고로 `search_artifacts/model_enhancement_direct_raw_qty_q012_inter_seed42_e50_0713/experiment_manifest.json`에는 구형 marked TitanTPP small_lmm의 Intermittent seed42·50epoch, direct_raw_qty와 global/causal RevIN/shrinkage RevIN 조합이 등록돼 있다. 이번에는 명세만 확인했고 성능은 읽지 않았다. 입력·decoder·학습 조건이 함께 다른 구형 실험이므로 현재 MLP4/16에서 입력 log1p만 제거한 대조로 사용할 수 없다.

## 판정

현재 코드에서 **손실과 raw RMSE/MAE가 겨냥하는 통계량의 차이**는 확인됐다. 그러나 이를 큰 수량 오류의 단일 원인으로 확정할 근거는 아직 없다. 특정 데이터셋의 큰 수량이 본질적으로 예측 불가능하다거나 복잡한 backbone이 불필요하다는 결론도 나오지 않는다. Cap27은 그대로 유지하고 저장 Train·Validation 행에서 어떤 오차가 집중되는지 먼저 판단한다. 문헌상 신규성이나 방법론 개선을 주장하지 않는다.
