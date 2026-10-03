# 마지막 Backbone 후보: 두 시간척도의 관측 전이 메모리

## 승인 범위와 종료 조건

사용자는 기존 benchmark 4개를 5080에서, 새 Backbone의 학습·검증을 5090에서
수행하도록 승인했다. 이 계약은 추가 탐색을 한 후보로 제한한다. 성능 기준 미달이면
TitanTPP(B)를 유지한다. 기존 Instacart baseline 패배와 모든 실패 기록은 보존한다.
새 seed42 결과를 본 뒤 구조·loss·selector·기준을 수정하지 않는다.

## 해결하려는 문제와 근거의 한계

B의 static prototype bank는 학습 시 갱신되고 개별 관측 이력에 맞춰 갱신되지 않는다.
선택한 네 행의 산술평균은 검색 순위에 미분 가능한 학습 경로를 주지 않는다.
사용량 정규화나 frozen-B의 한 단계 gradient 수정은 공통 개선 조건을 통과하지 않았다.
이번 가설은 관측된 상태 전이를 표본별로 축적하고, 현재 상태와 비슷한 과거 상태에서
나타난 변화를 검색하면 static prior에 없는 예측 상태를 구성할 수 있다는 것이다.
이는 검증할 가설이며, 기존 진단이 효과를 입증했다는 뜻은 아니다.

## 모델 계약

두 encoder 층 사이에서 첫 층의 관측 상태 `h_i`를 읽는다. 각 관측 전이의 key는
이전 상태, value는 현재와 이전 상태의 차이다. key/query는 8차원 양수 특징으로,
value는 bounded tanh 투영으로 구성한다. 관측 변화로 계산한 학습 가능한 sigmoid
신뢰도를 write weight로 사용한다.

각 시점에서 최근 유효 전이 8개는 local memory, 그보다 오래된 전이는 global memory다.
두 집합은 겹치지 않는다. global은 해당 표본의 입력 창 내 오래된 prefix를 뜻하며,
다른 고객·series나 train 전체를 inference 시 공유한다는 뜻이 아니다.
정적 final Hard-LMM bank는 전역 학습 prior로 유지된다.

각 경로는 `S=sum(weight*key*value^T)`, `z=sum(weight*key)`를 축적하고
현재 query로 `query*S/(query*z)`를 읽는다. query와의 가중 support mass가
`1e-8` 이하인 경로는 정확히 0이며 fusion에서 제외한다. sigmoid underflow로
관측 횟수는 양수이지만 실제 write mass가 0인 경우도 여기에 포함된다.
각 normalized read에 `-expm1(-rank*mass)`를 곱해 절대 support가 적은 경로를 억제한다.
전이 하나에서도 write confidence가 read 크기에서 상쇄되지 않는다. 데이터셋별 계수 없이
동일한 rank 8과 같은 함수를 사용한다. 그 뒤 두 경로의 support와 현재 관측 변화에
기반한 같은 confidence 함수가 경로를 결합한다.
`h1 + tanh(alpha)*bounded_read`를 두 번째 encoder 층에 입력한다.
time/quantity가 같은 최종 상태를 사용하며 head와 loss는 기존 B와 같다.

현재 관측 사건까지는 write할 수 있고 다음 target은 write할 수 없다. padding,
미관측 query token, 비연속 mask를 검증한다. 상태는 forward마다 각 표본의 prefix에서
다시 만들며 batch·series·train/validation 사이에 mutable state를 공유하지 않는다.
훈련의 충분통계는 누적합으로 계산한다. 최근 전이 경계의 `searchsorted`까지 포함한
추가 비용은 `O(T*d² + T*r*d + T*log(T))`이며 rank 8은
검색 key/query의 차원이다. 전체 encoder의
attention까지 선형이라고 주장하지 않는다. 내부 gradient optimizer는 없다.

`alpha=0`에서 같은 seed의 B와 공유 parameter·RNG·출력·공유 gradient가 일치한다.
이는 이미 학습된 B를 이어 학습하는 것이 아니라 같은 초기 상태에서 공정하게 비교하는
계약이다. alpha가 활성화된 뒤 query/key/value/write/fusion으로 수량 gradient가 실제
전달되는지 별도로 검사한다. 0 초기값에서 모든 새 parameter가 즉시 학습된다고 주장하지 않는다.

## 기존 후보 및 문헌과의 관계

기존 inter-layer 후보는 동일 static bank의 read 한 번을 추가했고, FiLM은 readout을
변경했다. 이번 후보는 관측된 전이를 기록하는 동적 메모리를 encoder 내부에서 구성한다.
MAC와 달리 내부 gradient optimizer를 실행하지 않는다. 기존 SurpriseGatedMemory도
관측 이력을 쓰는 fast-weight update와 학습 가능한 Q/K/V가 있으므로 부분적으로 겹친다.
이번 후보의 구체적인 차이는 상태 차이를 value로 쓰는 전이 저장, 현재 관측까지의 write,
최근 8개와 오래된 prefix의 겹치지 않는 분리, event loop를 대신한 누적합,
encoder 층 사이의 배치다. 새 Backbone 실험 경로라는
사실과 학술적으로 독창적인 방법이라는 주장은 구분한다.

양수 kernel의 누적 충분통계는 [Linear Transformers](https://proceedings.mlr.press/v119/katharopoulos20a.html)
및 [fast-weight 해석](https://proceedings.mlr.press/v139/schlag21a.html)과 관련된다.
[Titans](https://arxiv.org/abs/2501.00663)의 다중 시간척도 memory 관점에서 동기를 얻지만
Titans의 test-time gradient update를 그대로 구현하거나 발명했다고 주장하지 않는다.

## 실행과 판정

1. 로컬 인과성·초기 일치·gradient·상태·저장복원 계약 검사.
2. 5090 CUDA 계약 검사와 B 대비 비용 측정. batch 128, 길이 64/256에서
   step 중앙값 2배 이하, peak allocated memory 3배 이하, parameter 1.25배 이하.
3. 세 데이터셋 전체 e1. 실제 처리 건수, 유한 계산, checkpoint 복원만 판정한다.
4. Instacart → Taxi → Intermittent seed42, 최대 300/minimum 40/patience 40.
   validation raw RMSE의 가장 이른 엄격한 최솟값으로 선택·조기 종료한다.

모든 데이터셋에서 raw RMSE는 B보다 낮아야 한다. 전체 MAE 악화 1% 이하,
body(p95 이하)·>p99 MAE 악화 각각 2% 이하, 기존 clamped time loss 증가 0.01 이하다.
기존 시간 점수는 정상화된 Time NLL이라고 부르지 않는다. 원본 대비 body MAE 5%라는
별도 주장은 별도로 입증해야 한다. 3-seed·held-out 없이 채택·통계적 우월성을 확정하지 않는다.
정확한 입력 SHA, B 수치, 실행 한도는 같은 이름의 JSON 계약에 고정했다.
