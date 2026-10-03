# TitanTPP의 로그 변환·수량 손실·보정 폭 검토와 후속 설계

작성일: 2026-10-03. 상태: **코드·기록 검토 완료, 후속 실험 설계 제안**.

사용자 요청은 `log1p` 사용과 MLP의 작은 잠재 차원이 현재 결과를 제한했는지 확인하고 검토 방향을 다시 설계하는 것이다. 이번 작업은 로컬 소스·기존 연구 기록·수학 검토에 한정한다. 새 모델 구현, 학습, 추론, 원격 접속, 임대, 현재 캠페인 변경, Test 재집계 또는 원고 편집은 실행하지 않았다. 아래 조건은 실행 승인을 받은 계약이 아니다.

## 1. 코드에서 확인한 사실 — 완료

검토의 기준은 최종 평가에 묶인 동결 core 소스다. 현재본과 HistoryCorrection/수행 경로 두 파일은 전체 SHA가 일치한다. 공통 모델·학습 코드의 전체 파일은 다를 수 있으므로, 필요한 네 함수의 AST 일치를 따로 확인했다. 결과: [algebra_checks.json](algebra_checks.json), 재확인 코드: [check_algebra.py](check_algebra.py). 자료·checkpoint를 열거나 학습하지 않는 검사다.

| 위치 | 현재 동작 | 해석 |
|---|---|---|
| 입력 | `log1p(dt)`와 `log1p(history_quantity)`의 2개 채널 → 64차원 투영 | 로그 변환은 비음수 영역에서 가역적이다. 큰 값의 거리와 학습 조건은 바꾸지만 원래 수량을 수학적으로 삭제하지 않는다. 유한 정밀도와 후속 신경망의 학습은 별개다. |
| 수량 학습 | 예측 log 위치와 `log1p(target_quantity)` 사이 MSE | 원래 수량 MSE를 직접 최소화하는 목적함수와 다르다. |
| 수량 출력 | `log_prediction=softplus(a)`, `prediction=expm1(log_prediction)` | 단순 역변환이며 조건부 산술평균 보정은 없다. 실수 연산에서 `expm1(softplus(a))=exp(a)`이나, 동결 구현의 연산을 바꾸지는 않는다. |
| 공동 목적함수 | 현재 관측시간 likelihood의 음의 로그값 + 수량 손실 | 수량 손실만 바꿔도 공유 encoder의 시간/수량 학습 균형이 바뀔 수 있다. 현재 시간 likelihood를 과거 legacy clamped RMTPP와 혼동하지 않는다. |
| 보정 분기 | 8개 각각 `128 → 4 → 64`, GELU, output projection 0 초기화, 합계 `/8` | 내부 폭 4는 실제로 좁다. 하지만 전체 backbone의 hidden이 4차원인 것은 아니다. |
| 주 경로 | attention block 1의 64차원 상태에 보정을 더하고 block 2와 static memory를 통과 | 64차원 residual 경로가 유지된다. 이전 상태도 attention을 거친 맥락 상태다. |

동결 소스:

- [HistoryCorrection](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluator_sources/core/models/TPPs/CountAwareTitanCoreAblation.py:43)
- [보정의 residual 경로](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluator_sources/core/models/TPPs/CountAwareTitanMultiLagDetail.py:211)
- [입력 변환](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluator_sources/core/models/TPPs/CountAwareTPP.py:435)
- [수량 출력과 손실](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluator_sources/core/models/TPPs/CountAwareTPP.py:770)
- [공동 목적함수와 target 마스킹](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluator_sources/core/paper/scripts/count_aware_tpp_backbone/core.py:63)

### 로그 입력과 로그 손실을 구분해야 하는 이유

로그 손실에서 같은 100단위 과소예측이라도 `(정답, 예측)=(200,100)`의 손실은 약 0.4736, `(10100,10000)`은 약 0.0000990이다. raw MSE는 둘 다 10,000이다. 이는 **합성 산술 예시**이며 실제 모델 성능 수치가 아니다. 큰 수량의 절대 오차를 상대적으로 덜 벌주는 학습 기하가 확인된다. 이것만으로 실제 성능 저하의 원인을 확정할 수는 없다.

수량 단독 모집단 최적화, 충분한 모델 표현력과 필요한 조건부 모멘트를 가정하면:

\[
\hat q_{\log}(H)=\exp(E[\log(1+Q)\mid H])-1
\le E[Q\mid H]=\hat q_{\mathrm{raw}}(H).
\]

Jensen 부등식이다. 왼쪽을 일반적으로 중앙값이라고 부르면 안 된다. 실제 유한 모델의 공동 학습·정규화·최적화에서는 이 식이 예측 순서를 보장하지 않는다. 역변환된 위치와 원래 단위의 평균이 다르다는 일반적인 설명은 [Forecasting: Principles and Practice 3e §5.6](https://otexts.com/fpp3/ftransformations.html)도 참고한다. 해당 자료의 중앙값 해석에는 변환 공간의 대칭성 가정이 있다.

현재 비교표의 공통 수량 인터페이스에도 로그 목적함수가 사용된다. 그러므로 이 불일치만으로 TitanTPP와 S2P2의 순위 차이를 설명할 수 없다. 입력·구조·학습과의 상호작용을 확인해야 한다.

### 작은 MLP 폭이 의미하는 범위

대표 `local` 모드의 8개 분기는 서로 다른 실제 lag를 읽는 것이 아니라 **같은 직전 관측 상태**를 읽는다. `[1,2,4,8,16,32,64,128]`은 이 모델에서는 활성화 문턱이다. 최종 예측에 사용되는 관측 이력 수를 H라 하면:

\[
K(H)=\sum_{\ell\in\{1,2,4,8,16,32,64,128\}}1[H>\ell].
\]

H=2일 때 K=1, H=3~4이면 K=2, H=5~8이면 K=3, H=9~16이면 K=4다. 활성 분기가 K개이고 각 폭이 r이면, 고정된 활성 집합에서 **보정 출력의 열공간 차원 상한**은 `min(64,K*r)`이다. r=4에서는 K=2/3일 때 최대 8/12, 8개 모두 켜져도 최대 32다. 실제 학습된 rank 측정값이 아니며, residual과 이후 block을 포함한 전체 네트워크의 표현력 상한도 아니다.

| 분기 내부 폭 r | 보정 파라미터 | 길이256 설정의 전체 파라미터¹ | 목적 |
|---:|---:|---:|---|
| 4 | 6,144 | 96,003 | 기존 기준선 |
| 16 | 24,576 | 114,435 | 첫 용량 대조 |
| 32 | 49,152 | 139,011 | 필요할 때 추가 용량 대조 |

¹ 최대 길이256인 Taxi·Intermittent 구성에서 `96,003−6,144+8(128r+64r)`로 계산한 예상 수다. 아직 구현한 새 모델의 실측 수가 아니다. 학습 가능한 위치 임베딩이 `max_seq_len×64`이므로 Instacart·RAF 전체 파라미터 수에는 그대로 적용하지 않는다. 일반식은 `해당 데이터의 r4 기준 전체 수−6,144+1,536r`이다.

현재 소스와 161개 로컬 계약을 확인한 범위에서는 **동일 History-MLP의 폭만 바꾼 독립 비교를 찾지 못했다**. current-only의 폭 6은 입력 64이며 직전 상태를 제거했으므로 폭만의 대조가 아니다. all-available, active normalization, recent4/attention/post-block 후보도 폭 4를 유지한다. 이 비교들은 독립적인 width sweep을 대신하지 않는다.

## 2. 기존 연구를 재사용하는 범위 — 완료

과거 B에서 로그/raw 목적함수와 출력 link를 비교하는 설계가 이미 있었다. 관련 계약은 [quantity_objective_output_comparison_v1.json](/Users/igwanhyeong/PycharmProjects/paper_research/paper/contracts/quantity_objective_output_comparison_v1.json)이다. 실제 실행과 validation 결과의 확인은 [PRIOR_STUDIES.md](PRIOR_STUDIES.md)에 별도로 기록한다.

이전 B의 legacy 시간 손실, 120epoch, 당시 선택 규칙은 현재 History-MLP의 관측시간 likelihood와 동일하지 않다. 과거 결과는 운영 위험·가설의 근거로 재사용하고, 현재 구조에서의 직접 대조 결과로 합치지 않는다. 단순 raw 손실 교체, 혼합 손실, output link 교체를 새로운 방법론이라고 주장하지 않는다.

## 3. 폭과 목적함수를 분리해서 비교한다 — 다음 작업 / 설계 제안

대상은 **Taxi·Intermittent·RAF·Instacart 전체**, 첫 단계는 seed42다. 짧은 이력 문제를 주장하면서 Instacart를 제외하지 않는다. 기존 split, 이력, 현재 시간 head/likelihood, min40/max300/patience40, 최초 최소 validation raw 수량 RMSE checkpoint 선택, `/8`, 분기 가용성, zero output 초기화와 수량 출력 head는 고정한다.

| 조건 | 분기 폭 | 수량 손실 | 입력 수량 | 확인할 점 |
|---|---:|---|---|---|
| A | 4 | 기존 log-MSE | 기존 log1p | 기준선 |
| B | 16 | 기존 log-MSE | 기존 log1p | 폭 확대 효과 |
| C | 4 | raw-MSE / s² | 기존 log1p | 목적함수 교체 효과 |
| D | 16 | raw-MSE / s² | 기존 log1p | 폭과 목적함수의 상호작용 |

실행 순서는 A의 재사용 적격성 확인 → B의 폭 대조 → C/D의 교차 확인이다. 독립 조건의 학습은 실행 자원이 정해진 후 병렬화할 수 있다. r=32는 기본 전체 실행에 포함하지 않고, r=16의 학습 적합 개선과 validation 양상을 본 후 별도 확장 여부를 결정한다. 선택적으로 추가하면 탐색 단계임을 기록한다.

**s의 제안 정의:** 데이터셋별 canonical next-event TRAIN target에서 `max(1,sqrt(mean(q²)))`를 계산해 동결한다. target별·batch별로 변하지 않으며 validation/Test 통계를 쓰지 않는다. 고정 상수이므로 수량 단독 raw-MSE 최적해를 바꾸지 않지만 공동 시간/수량 손실의 비중에는 영향을 준다. 이전 연구의 train identity가 일치하면 해당 통계를 검증 후 재사용한다.

**비교 가능성:** 기존 A checkpoint를 우선 재사용한다. 동일 데이터·target 모집단·소스 의미·선택 기준·수치 환경을 확인한다. GPU나 초기화/RNG/batch 순서가 맞지 않으면 이를 새 조건과 엄밀히 짝지어진 기준선이라고 표현하지 않는다. A의 재학습을 자동으로 요구하거나 실행하지 않는다. 새 기준선이 필요한지는 실행 계약에서 별도로 확정한다.

범위는 4조건×4데이터=16개 seed42 비교 셀이다. A의 적격 재사용이 가능하면 새 학습은 12개다. 이는 비용·시간 예측이나 실행 승인이 아니다. 세 seed 확장은 기제와 후보가 정리된 다음 단계다.

### 해석과 실패를 가르는 기록

- 같은 선택 epoch의 raw RMSE·MAE·시간 NLL을 함께 보고한다. 학습 중 최솟값을 지표마다 따로 골라 합치지 않는다.
- 고정 train/validation 표본에서 log/raw 수량 손실의 학습곡선, exact K별 보정 크기/기본 상태 크기, 활성값 공분산 특이값을 확인한다. 작은 유효 rank만으로 원인을 확정하지 않는다.
- 시간/수량의 공유 encoder gradient norm·방향, clipping 빈도, 비유한 값 여부를 확인한다. quantity head가 0 weight로 시작하므로 처음 한 step의 encoder 수량 gradient만으로 균형을 보정하지 않는다.
- 같은 output head에서 raw 손실은 `dL/da=2(qhat−q)*qhat/s²`이다. 큰 출력의 gradient 증폭을 고정 s가 제거하지 않는다. 발산을 임의 target clipping이나 숨은 손실 변경으로 해결하지 말고 실패로 남긴다.
- 폭 확대가 train만 개선하고 validation을 악화시키면 일반화 이득으로 보지 않는다. 시간 NLL/MAE 악화와 데이터별 상반된 결과도 남긴다.
- B 개선은 폭 변경 효과의 근거, C 개선은 손실 변경의 근거, D만 개선하면 상호작용 가설의 근거다. 폭 변경은 파라미터 수뿐 아니라 출력/gradient 규모와 최적화에도 영향을 줄 수 있어 개선을 순수 rank 효과로 단정하지 않는다. seed42만으로 인과 확정이나 여러 데이터 우월성을 선언하지 않는다.

## 4. 입력 log1p 자체의 영향을 별도로 확인한다 — 다음 작업 / 3번 이후

위 2×2는 **입력 log1p의 영향을 검증하지 않는다**. 이를 빠뜨리지 않도록 r=4/log-MSE라는 사전 지정 조건에서 입력 수량 채널만 바꾸는 대조를 제안한다. 시간 입력의 log1p, 수량 target의 log1p와 출력 head는 유지한다.

단순 raw 입력은 크기 차이가 너무 크므로, 고유 TRAIN 사건의 동일 모집단에서 구한 평균·population SD(ddof=0)를 사용한다:

\[
T_{raw}(q)=\mu_{log}+\frac{\sigma_{log}}{\sigma_{raw}}(q-\mu_{raw}).
\]

이는 원래 log 입력과 평균·분산을 맞춘 affine raw 입력이다. 입력 통계 모집단은 중복 window로 가중한 target 모집단이 아닌 고유 train 사건으로 명시하며, 손실의 s 계산 모집단과 구분한다. 입력 변환은 음수가 될 수 있고 정상이다. 변환 뒤 padding/target 등 비관측 위치를 0으로 마스킹하며, 추가 clamp나 log를 적용하지 않는다. 비유한 값/0분산은 규칙 위반으로 표시한다.

왜도·꼬리·사건 간 거리는 여전히 달라지므로, 결과는 비선형 압축을 제거한 표현/최적화 효과다. 가역적인 입력에서 정보가 사라졌음을 증명하는 실험은 아니다. 데이터별 하나의 추가 조건, 총 4개 seed42 fit 제안이며 3요인 전체 factorial은 아니다. A~D와 합치면 총20개 비교 셀, 기존 A 재사용 적격 시 새 fit16개다.

## 5. 원인을 확인한 뒤 새 backbone 기여를 정한다 — 다음 작업 / 진단 이후

| 확인되는 결과 | 이어갈 방향 | 그 결과만으로 할 수 없는 주장 |
|---|---|---|
| 폭 확대가 여러 데이터에서 안정적으로 개선 | 필요한 보정 용량과 분기 가용성 관계를 확인하고, 같은 파라미터의 단순 대조와 비교 | 폭을 키운 것만으로 새로운 backbone 기여 또는 효율 우위 |
| raw 손실만 개선 | 같은 손실을 S2P2와 구조 대조군에도 적용해 구조 효과를 분리 | TitanTPP 구조 자체의 개선 |
| affine raw 입력만 개선 | 수량 level을 보존하는 입력/보정 경로 가설 검토 | log1p가 정보를 삭제했다는 주장 |
| 폭×손실 상호작용이 있음 | 필요한 수량 표현 용량과 목적함수를 함께 설명하는 구조 설계 | 하나의 원인만으로 모든 데이터 차이를 설명 |
| 모두 불안정하거나 이득 없음 | 더 큰 MLP를 중단하고 입력 정보, 시간/수량 공유 상태, 다른 실패 원인을 재검토 | 새 attention/MoE를 붙이면 해결된다는 보장 |

이전의 level/change 분리나 적응형 결합 제안은 **조건부 후속 가설**로 내린다. 이미 존재하는 elapsed-time state transport, 단순 폭 확대, raw 손실을 새로 발명한 것처럼 제안하지 않는다. 새로운 기여는 실패 원인과 설계의 연결, 단순 대조군보다 나은 결과, 여러 seed의 재현으로 입증해야 한다.

현재 공개한 Test는 개발에서 이미 접근한 자료다. 후속 후보 선택은 train/validation에서 하고, 동일 Test의 추가 결과를 독립 평가로 재명명하거나 유리한 재분할로 교체하지 않는다. 이번 설계 검토에서는 새 Test 파일을 열거나 수치를 계산하지 않았다.

## 이전 이력 구간 설명의 정정

기존 [Test 구간 보고서](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_test_stratified_analysis_20261003_v1/REPORT.md)의 보조 경계 `[1,3,7,15,31,63,127]`은 분기 문턱을 참고해 고정한 **설명용 이력 구간**이며, 같은 활성 분기 수를 보장하는 구간은 아니다. 예를 들어 `(1,3]`에는 K=1인 H=2와 K=2인 H=3이 섞인다. 기존 수치·경계·계약은 보존하며, 이를 K별 성능 증거로 해석하지 않는다고 정정한다. 이번 폭 진단에서는 정확한 K를 직접 계산한다.

## 남은 작업 순서

1. **실험 계약을 구체화한다 — 다음 작업:** `paper_research`에서 A 재사용 가능성, r16 구현 명세, train 통계·초기화·선택·기록 기준을 확정한다. GPU 실행 대상과 비용은 아직 정하지 않았다.
2. **구현과 합성·계약 검증을 한다 — 다음 작업:** 폭만 변경한 경로의 shape/파라미터, r4 동일성, 마스킹과 raw 손실 수치 안정성을 검증한다. 입력 변환 대조의 구현 검증은 공통 계약 확정 후 분리 가능하다.
3. **실제 학습 자원·범위를 확정해 실행한다 — 승인 필요:** 구체적 서버·동시 실행 수·시간·비용·회수 계획을 갖춘 별도 실행 계약으로 진행한다. 현재 A100 캠페인은 변경하지 않는다.
4. **후보와 주장 범위를 정한다 — 외부 작업 대기:** 위 실험과 독립 검증 후 단순 대조군 및 여러 seed로 확장할 후보를 결정한다.
