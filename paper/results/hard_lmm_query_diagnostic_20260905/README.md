# Hard-LMM 이력 통계 query 진단 결과

상태: **진단 완료 · query 후보 구현 보류**. 로컬 `paper_research/master`,
실행 기준선 `1003033`. 분석 종료: **2026-09-05 07:39:49 KST**.

과거 평균 log 수량과 최근 편차를 query에 직접 넣는 가설은 이번에 고정한
진단에서 지지되지 않았다. 사용자 승인에 포함된 후속 구현·5090 CUDA/e1은
진단 통과를 전제로 하므로 실행하지 않았다. 원본 Hard-LMM을 일반 기준선으로
유지한다. 이 결과는 후보를 실제 학습한 성능 결과가 아니다.

## 1. 범위와 판정 방법

- Taxi와 Instacart 각각 **동일 train target 8,192개**를 원본·separate-key에
  입력했다. Seed42 무작위 비복원 표본을 정렬하여 사용했다.
- 원본 train population은 Taxi 38,393 targets/131 series,
  Instacart 1,991,192 targets/206,209 series다. 표본에는 각각
  **131 / 7,848 series**가 포함되었다.
- 두 내부 분할은 series 단위로 분리했다. Taxi의 target 수는 4,761/3,431,
  Instacart는 4,008/4,184다. 양쪽 방향을 모두 평가했다.
- 각 표본은 반대 분할에서 normalized local state의 cosine이 가장 가까운
  64개 표본을 찾았다. 이웃 평균 잔차와, 동일 이웃에 두 이력 통계를 더한
  local ridge 보정을 비교했다. Ridge penalty는 1이며 탐색하지 않았다.
- 잔차는 `log1p(실제 다음 수량) - softplus(동결 모델 수량 logit)`이다.
  실제 다음 수량은 **진단 label에만** 사용했으며 검색·이력 통계에는 들어가지
  않았다. 별도의 반대 분할 평균 잔차를 상수 대조군으로 사용했다.
- Instacart separate-key에서 이웃 대조군 대비 pooled MSE 1% 이상 개선,
  양쪽 분할 개선, 상수 대조군 개선, series bootstrap 개선량 하한 양수를 모두
  요구했다. Taxi pooled MSE 악화는 1% 이하여야 했다.

위 표본·방법·기준은 추론과 결과 열람 전에
`paper/contracts/hard_lmm_query_diagnostic_v1.json`에 고정했다.
진단 중 threshold·이웃 수·ridge를 변경하지 않았다.

## 2. 두 통계를 추가해도 잔차 설명력이 개선되지 않았다

아래는 **동결 separate-key의 잔차를 설명하는 보조 진단 MSE**다.
새 query 모델의 quantity MAE/RMSE와 혼용하면 안 된다.

| Dataset | 상수 대조군 MSE | 이웃 대조군 MSE | 두 통계 추가 MSE | 이웃 대비 변화 |
| --- | ---: | ---: | ---: | ---: |
| Taxi | 0.141261 | 0.138724 | 0.139975 | **0.902% 악화** |
| Instacart | 0.245778 | 0.248975 | 0.254749 | **2.319% 악화** |

- Instacart는 분할 0에서 **2.999%**, 분할 1에서 **1.690% 악화**했다.
  상수 대조군보다도 **3.650% 악화**했다. 이웃 대조군 자체도 상수보다 나빴다.
- Instacart의 `(이웃 squared error - 두 통계 squared error)` 평균은
  **−0.005774**였다. Series 단위 500회 bootstrap의 5%/95% 값은
  **−0.007253 / −0.004529**였다. 이는 train 내부의 조건부 안정성 범위이며
  독립 일반화나 통계적 유의성 주장이 아니다.
- Taxi도 두 방향 모두 악화했다(0.770% / 1.073%). Pooled 악화가 1% 이내라
  보호 조건만 통과했으며, 이력 통계의 이익이 확인된 것은 아니다.
- 원본 checkpoint에서도 Instacart는 **1.636% 악화**했다. 원본 Taxi는
  0.951% 개선했지만 bootstrap 하한은 음수였다. 원본 결과를 이용해
  separate-key에 고정한 판정을 대체하지 않았다.

**판정: Instacart의 네 필수 조건이 모두 실패했다. Query 후보를 정의하거나
구현하는 단계로 진행할 근거가 부족하다.**

## 3. 검색은 바뀌었지만 수요 상황 구분의 이익은 확인되지 않았다

모두 이번 동일 train 표본·seed42에서 계산한 값이다. 과거 validation/다중 seed
진단의 약 90% 수치와 모집단이 다르므로 직접 이어 붙이지 않는다.

| Dataset | 모델 | 유효 prototype 수¹ | 상위 4개 선택 비중 | 수량 head에 투영한 memory 기여의 표준편차 |
| --- | --- | ---: | ---: | ---: |
| Taxi | 원본 | 17.611 | 38.208% | 0.042534 |
| Taxi | Separate-key | 8.731 | 62.259% | 0.019323 |
| Instacart | 원본 | 6.792 | 77.899% | 0.007618 |
| Instacart | Separate-key | 4.195 | **99.222%** | **0.003228** |

¹ Prototype 선택 빈도 분포의 entropy를 지수화한 값이다.

- Separate-key의 평균 최대 검색 가중치는 Taxi **0.29513**, Instacart
  **0.29335**였다. 원본의 실제 가중치는 항상 0.25다.
- Instacart separate-key에서 과거 수량 수준의 1~4사분위별 평균 memory 기여는
  **0.263223 / 0.262564 / 0.262491 / 0.262309**로 거의 일정했다.
  선택된 네 value의 수량 방향 기여 차이도 평균 **0.018814**로 작았다.
- 같은 slot ID의 top-4 집합이 원본과 완전히 일치한 표본은 두 데이터셋 모두
  0%였다. 그러나 서로 학습된 모델의 slot 의미와 latent 좌표가 달라질 수
  있으므로, 이것만으로 검색의 질이나 key 분리만의 인과효과를 주장하지 않는다.
- Separate-key에서 이웃들의 평균 cosine은 Taxi **0.98676**, Instacart
  **0.97716**였다. 가까운 방향의 이력 안에서도 제안한 두 통계의 추가 설명력이
  안정적으로 나타나지 않았다.

**해석:** Instacart의 memory 출력이 수량 예측 방향에서 거의 일정해진 것은
확인된 현상이다. 하지만 과거 평균·최근 편차를 query에 추가하면 해결된다는
근거는 나오지 않았다. 또한 Taxi도 선택 집중도가 높아졌는데 앞선 validation
성능은 개선되었으므로, prototype 사용 다양성 자체를 성공 목표로 삼지 않는다.

Memory를 동결 모델에서 제거한 차이는 사후 개입 결과다. Encoder·memory·head가
함께 학습된 모델에서 memory를 제거한 수치를 fresh memory-free 모델의 성능이나
학습 중 memory의 순기여로 해석하지 않는다.

## 4. 검증과 한계

- **22개 테스트 통과:** 원본/KV 실제 검색 수식, 공식 수량 예측 일치,
  target/padding 차단, series 분리, 영분산/공선성, 동률 이웃, 재현성,
  반대 분할 통계만 사용, 판정 실패 조건을 검증했다.
- 원본 registry·data·split·checkpoint 파일/state digest와 separate-key audit를
  확인했다. 실행 전후 모델 state와 입력·소스 hash가 일치했다.
- 모든 cache와 분석 수치가 finite이며, 저장된 보정값에서 pooled/fold MSE를
  NumPy로 별도 재계산해 일치함을 확인했다.
- 독립 감사에서도 **169개 hash 일치**, paired identity와 series 분리,
  네 bootstrap 범위의 재현을 확인했다. MSE 재계산의 최대 차이는
  **5.55e−17**였다. `independent_audit.json`에 결과와 해석의 한계를 기록했다.
- Runtime: macOS ARM CPU, Python 3.12.10, PyTorch 2.14.0,
  Polars 1.31.0, CPU threads4. 두 모델을 같은 runtime에서 새로 추론했다.
  과거 CUDA 학습과 bitwise 동일한 실행 환경이라고 주장하지 않는다.
- Backbone은 이미 전체 train을 학습했고 checkpoint 선택에는 validation이
  사용되었다. 내부 분할은 보조 진단의 안정성을 확인할 뿐, 새 표본에 대한
  일반화를 검증하지 않는다. 이번에는 validation/test 행을 읽거나 추론하지 않았다.
- Instacart weighted-only e300 대조군은 없으므로 key/value 분리만의 효과를
  확정할 수 없다. 고정 표본·선형 local probe의 실패가 모든 비선형 query
  개선 가능성을 부정하지도 않는다.
- 그림은 생성하지 않았다. 표와 기계 판독 결과를 검산했다.

## 5. 산출물과 남은 작업

- `execution_manifest.json`: 데이터·소스·cache hash, 표본, 실제 검색 지표.
- `analysis.json`, `crossfit_metrics.csv`: 네 조합의 pooled/분할별 진단 결과.
- `evidence_decision.json`: 고정 기준의 자동 판정.
- `analysis_verification.json`, `independent_audit.json`, `local_pytest.xml`:
  계산 검산·독립 감사·테스트 증적.
- 원시 cache와 행별 보정은 git-ignored
  `search_artifacts/hard_lmm_query_diagnostic_20260905`에 보존했다.

현재 순서:

1. **완료:** 동일 train 표본 검색 진단과 후보 진행 여부 판단.
2. **차단됨 — 진단 조건 미충족:** 두 통계 query의 모델 계약·구현·5090 CUDA/e1.
   승인이 없어서 멈춘 것이 아니며, 사용자가 정한 조건부 실행 경계를 적용했다.
3. **다음 작업:** 거의 일정한 수량 방향 memory 기여가 무엇을 뜻하는지 바탕으로
   Hard-LMM 내부의 value 표현 또는 결합 방식에 대한 새 가설을 검토한다.
   이번에 다른 후보를 자동 구현하지 않았고, 상대시간 변경을 지지할 별도 근거도 없다.

원본 Hard-LMM·separate-key 모델, loss, head, 기존 실험 결과와 5090은 변경하지 않았다.
