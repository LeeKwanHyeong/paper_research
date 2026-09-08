# Q/K/V 경로 진단 결과

**완료: V를 제거하는 QK-only 후속 후보는 이번 진단으로 지지되지 않는다.**
추가 V 경로를 끄면 Taxi와 Intermittent의 수량 오차가 크게 증가했고 시간 loss도
개선하지 못했다. 동시에 Intermittent의 기존 시간 loss 악화를 Backbone 표현
악화로 해석한 근거에 문제가 있음을 확인했다. 서로 구분해야 하는 두 결과다.

새 모델을 학습하거나 채택하지 않았다. 기존 Causal-QKV의 seed42 validation
기준 미달 기록은 보존하며, 이 train 진단을 성능 재평가로 사용하지 않는다.

## 실행과 재현

- 대상: `paper_research / codex/hard-lmm-causal-qkv`, 로컬 CPU.
- 사전 계약: `2a7f3ba`, JSON SHA-256
  `a4ae310d254ed124fc8be4c340a10d22c462a33fb08d669afd38504cf978ddea`.
- 추출기 source: `01f6c3a`. 실제 모델·loader는 동결 source
  `84668e207d5121f211a6a93af12ca2f96068a25e`의 405개 Python 파일을 검증해 사용했다.
- 분석기 source: `d73534e`. 독립 재계산과 분석기의 모든 데이터셋·조합·fold
  MAE/RMSE/시간 loss가 오차 1e-13 이내에서 일치했다.
- Taxi·Intermittent·Instacart 각각 train target 4,096개, series가 겹치지 않는
  두 fold에 각각 2,048개. B와 8개 경로 조합에서 총 110,592개 예측을 계산했다.
- 최종 실행: 2026-09-08 10:40:00–10:41:16 KST. GPU·optimizer update·head 재학습·
  validation/held-out 행 접근 없이 완료했다.
- 실제 6개 checkpoint를 포함한 추출기 테스트 9개와 분석기 테스트 6개,
  통합 실행 **15 passed**. 실제 고정 batch의
  B/FULL 출력 재현, 미래 target 수량·시간 및 padding 차단, 모든 조합의 finite 계산,
  모델·head·checkpoint 복원과 gradient 정리를 통과했다.

초기 실행은 감사 코드 16줄이 Git source 기록보다 앞서 실행 파일에 추가된 상태여서
최종 증적에서 제외했다. 추가 코드까지 커밋한 뒤 같은 identity로 재실행했다.
두 실행의 모든 NPZ 배열은 bitwise하게 같았다. 이 기록은
`independent_local_audit.json`에 남겼다. 표본·판정 기준을 다시 정하지 않았다.

## 1. 추가 V를 제거하면 필요한 수량 정보도 손실된다

아래는 **같은 train 표본**에서 계산한 값이다. FULL은 학습된 Q/K/V를 모두 유지하고,
QK는 그 checkpoint의 추가 V kernel만 0으로 만든 결과다.

| 데이터셋 | FULL MAE | QK MAE | FULL raw RMSE | QK raw RMSE | RMSE 변화 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Taxi | 24.0568 | 32.4270 | 76.7014 | 103.8435 | +35.39% |
| Intermittent | 0.5777 | 1.2989 | 1.4047 | 3.0313 | +115.79% |
| Instacart | 3.8941 | 3.9425 | 5.7714 | 5.7969 | +0.44% |

Taxi 두 fold의 RMSE 증가는 각각 20.79%, 45.01%다. Intermittent는 각각
96.50%, 135.39%로 두 fold에서 같은 방향이다. Instacart는 +2.61%, −1.31%로
방향이 엇갈린다. 전체 MAE도 1.25% 증가하여 사전 FULL 보존 기준을 충족하지 못했다.

시간 loss도 FULL→QK에서 Taxi 1.026189→1.659710, Intermittent
−1.602675→−1.600497, Instacart 3.148501→3.149237로 pooled 평균이 악화했다.
따라서 V를 시간 악화의 단일 원인으로 보고 제거할 근거가 없다.

QK 변경으로 top-4의 선택 집합이 바뀐 비율은 Taxi 59.84%, Intermittent 27.17%,
Instacart 50.93%였다. 검색 변화는 확인됐지만 이것이 수량 개선을 뜻하지는 않았다.
이 비교는 같은 checkpoint 내부의 prototype ID에만 적용했다.

ZERO는 같은 Causal-QKV checkpoint의 추가 Q/K/V를 모두 끈 결과이며 별도로 학습된
B가 아니다. FULL·QK·ZERO 비교는 함께 학습된 경로의 민감도와 상호 의존성을
보여준다. **QK-only를 처음부터 재학습하면 실패한다는 증거는 아니다.**

Taxi tail은 fold0에서 7 targets/2 series, fold1에서 30 targets/2 series에 불과했다.
사전에 정한 tail 최소 10 targets/5 series를 충족하지 않아 tail 보호는 판정 유보다.
전체·body 수량 기준 실패는 이 tail 한계와 별도로 확인됐다.

## 2. Intermittent 시간 점수의 해석을 바로잡아야 한다

기존 legacy 시간 계산은 `c=min(w×duration,10)`을 사용한다. Intermittent의 target
duration은 loader 계약상 모두 1 이상이며, 저장된 w는 B 89.5077, FULL 13.5038이다.
따라서 두 모델 모두 모든 표본에서 c=10이다. 새 train 표본에서도 100% 포화를 확인했다.

이때 intercept를 I라고 하면 시간 loss는

\[
L(I)=-I-10+\frac{e^I}{w}\operatorname{expm1}(10),\qquad
\min_I L=1-\log w+\log(1-e^{-10}).
\]

최솟값을 만드는 intercept는 두 모델 모두 허용 범위에 있다. 같은 duration에서
출력한 시간분포가 얼마나 정확한지와 별개로 w가 커지면 점수의 최솟값이 낮아진다.

아래 표의 실제 시간 loss는 **이미 존재하는 선택 checkpoint의 validation summary**다.
이번에 validation을 다시 실행한 값이 아니다.

| 항목 | B | Causal-QKV |
| --- | ---: | ---: |
| 이론적 legacy loss 최솟값 | −3.494370 | −1.603019 |
| 기존 validation legacy loss | −3.483794 | −1.602208 |
| 최솟값 초과분 | 0.010576 | 0.000811 |

관측된 차이 +1.881585는 w에 따른 최솟값 차이 +1.891351과 나머지 −0.009766으로
분해된다. **시간 점수가 수치상 증가한 사실은 맞지만, 이를 시간 예측 품질이나
Backbone 표현 품질의 악화로 해석할 수 없다.** 반대로 더 좋은 시간 예측을 입증한
것도 아니다. `legacy_time_score_audit.json`에 checkpoint·summary hash와 독립적인
수식 검증을 기록했다.

사전 gate의 원래 수치 판정은 변경하지 않는다. 다만 이렇게 전 구간이 포화되고
w가 다른 B/FULL의 시간 격차를 회복하도록 요구하는 조건은 표현 품질 판정에
적합하지 않다. 이 제한을 별도 기록한다. 같은 checkpoint의 FULL/QK/ZERO는 w가
같으므로 그 사이의 시간 변화는 여전히 경로 개입의 민감도로 비교할 수 있다.
이번 QK 보류 결론은 이 부적절한 cross-B 시간 조건을 제외해도 수량 기준에서 유지된다.
원래 수치 기준의 실패 27개 중 이 문제가 있는 시간 조건 4개를 제외해도 23개가 실패했다.

또한 이 clamp 식은 충분히 큰 duration에서 log score가 상수로 남으므로 전체
적분이 1인 density가 아니다. 이번 시간 loss 표는 기존 학습 점수의 재현값으로
해석하며, 정상화된 Time NLL의 성능 비교에는 별도 공통 likelihood 계약이 필요하다.

## 3. 모든 데이터셋에 공통인 V 또는 gradient 병목은 확인되지 않았다

현재 FULL의 Taxi Q kernel에서는 8개 진단 batch 모두 시간·수량 gradient 내적이
음수였고 cosine 중앙값은 −0.3569였다. 반면 V는 중앙값 +0.0877, 음수 3/8이었다.
Intermittent와 Instacart는 경로별 부호가 섞여 있었다. 과거 T1 tail-shared 감사값을
재사용하지 않고 현재 checkpoint에서 계산한 결과다.

이 작은 eval-mode gradient 표본은 학습 과정 전체의 충돌이나 task별 encoder 분리의
효과를 입증하지 않는다. V가 세 데이터셋에서 공통으로 시간 예측을 손상한다는
단일 가설과도 맞지 않는다.

Taxi는 이번 train 표본에서 FULL의 시간 loss가 B보다 낮지만, 기존 validation에서는
훨씬 높다. 남아 있는 문제는 이 일반화 격차가 어떤 간격·표본과 계산 항에서 생기는지다.
현재 train 진단만으로 그 원인을 확정할 수 없다.

Intermittent FULL의 기존 validation raw RMSE는 B의 1.499555에서 1.548392로
3.26% 악화했다. 이 수량 실패는 시간 점수의 해석 문제와 독립적으로 남는다.
Instacart의 기존 raw RMSE 기준 미달도 유지된다. 시간 평가를 바로잡는 것만으로
세 데이터셋의 공통 Backbone 개선이 해결되지는 않는다.

## 남은 작업 순서

**다음 작업 / 로컬 — 데이터셋별 일반화 격차와 시간 평가의 의미를 확인한다.**
- 기존 공통 정상화 likelihood 계약과 평가 증적을 재사용할 수 있는지 확인하고,
  Backbone 비교에서 legacy 점수 변화와 시간 예측 품질을 구분한다.
- Taxi의 기존 validation 증적에서 시간 간격·intercept·지수 누적 항과 큰 오류의
  관계를 검토할 범위를 고정한다. 새 validation 행 진단은 이번 train 계약과 분리한다.
- Intermittent의 validation 수량 RMSE 악화와 Instacart의 개선 부재도 다음
  Backbone 후보가 해결할 대상으로 유지한다. 기존 수량 구간별 증적을 먼저 재사용한다.
- 완료 조건은 다음 Backbone 변경이 해결할 실패 유형을 하나로 좁히는 것이다.

**후속 작업 / 로컬 — 그 근거로 공통 Backbone 후보 하나의 계약을 작성한다.**
- V 제거 구현은 보류하고 수량에 필요한 경로를 보존할 변경 지점을 정한다.
- 세 데이터셋에 같은 구조·loss·selector를 적용하고 Backbone 기여와 시간 평가
  정렬을 각각 판정한다. Gradient 부호만으로 경로 분리를 선택하지 않는다.
- 이번 결과만으로 새 GPU 학습, 추가 seed 또는 master 병합을 진행하지 않는다.
