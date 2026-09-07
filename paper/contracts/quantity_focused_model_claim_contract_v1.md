# 수량 중심 모델·주장 계약 v1

- 작성일: 2026-09-07 KST
- 대상: `paper_research`, `codex/aligned-k1-duration-head`
- 상태: 모델·평가·주장 범위 동결 / 최종 논문 증거 미완료
- 근거: [수량 비교 감사](../results/quantity_focused_feasibility_audit_20260907/README.md)
- 이번 범위: 로컬 감사·문서화. 새 GPU 학습, 추가 seed, held-out 평가는 실행하지 않는다.

## 1. 모델과 연구 질문

수량 중심 경로의 고정 주 모델은 **B: original mark-free Hard-LMM + validation
raw-quantity-RMSE checkpoint selector**다. A와 B는 같은 Hard-LMM 구조이며,
선택과 early stopping 기준의 변경을 새로운 Backbone 구조 기여로 서술하지 않는다.
기존 B seed42의 세 checkpoint identity는
[동일 selector 실행 계약](raw_rmse_baseline_alignment_seed42_v1.json)의 dataset별
파일 SHA-256과 state SHA-256을 그대로 사용한다.

공통 구조는 hidden 64, static Hard-LMM, persistent tokens 16, prototypes 64,
top-k 4다. 입력은 관측 이력의 `log1p(delta_t)`와 `log1p(quantity)`이며
quantity-derived mark/residual을 사용하지 않는다. Quantity objective는 unweighted
log1p-MSE이고 raw 예측은 기존 구현의 expm1 경로를 유지한다. Quantile-adaptive loss,
separate-key, 경과시간 후보, causal time adapter, CDF calibration, K=2는 주 모델에
포함하지 않는다. 데이터셋별 모델 분기는 허용하지 않는다.

연구 질문은 **같은 수량 인터페이스와 checkpoint 선택에서 각 encoder가 다음 사건의
수량을 얼마나 정확히 예측하는가**다. 고정 기간 누적 수요 예측이나 전체 TPP
시간·수량 결합분포의 우월성을 주장하지 않는다.

## 2. 비교·데이터 계약

주 비교 행은 동일 입력·head·loss·selector의 **count-aware RMTPP, count-aware THP,
Hard-LMM B**다. A는 selector 민감도 보조 비교다. 기존 baseline을 수량 예측용으로
공통 확장했다는 점을 명시하며 원본 논문 모델의 무수정 결과로 표기하지 않는다.

Intermittent-5000, Taxi, Instacart 모두 주 비교 데이터셋으로 유지한다. 각 역할은
간헐 수량, 시간 단위 집계 수량, 일 단위 basket 수량의 서로 다른 관측 조건 비교다.
Instacart의 불리한 결과를 보고 대상에서 제외하거나 부록으로 숨기지 않는다.
관측된 성능만으로 long-history/heavy-tail의 인과적 효과를 주장하지 않는다.

데이터·split SHA-256, lookback 520/168/52, max sequence length 256/256/64,
target identity·quantity hash, train/validation 처리 건수는 기존 동일 selector
계약을 참조한다. 이 계약은 기존 split을 바꾸거나 test를 열지 않는다.

학습은 공통 log1p-MSE와 기존 legacy time objective, AdamW, lr 0.001, batch 128,
gradient clip 1, max/min epochs 300/40, patience 40을 유지한다. Time head는
실행 당시 `legacy_clamped_rmtpp`, intercept cap 300, wd cap 10을 명시한다.
현재 코드 기본값으로 대체하지 않는다. 기존 source가 다른 B와 baseline의
재사용은 해당 실행 계약·target identity·출력 경로 감사에 의존한다.

모든 모델에서 **validation raw quantity RMSE의 가장 이른 엄격한 유한 최솟값**을
선택하고 early stopping도 같은 지표를 사용한다. MAE용 checkpoint를 따로 고르지
않는다. e300은 최대 예산이며 계약에 맞는 조기종료를 완료로 인정한다.

## 3. 지표와 주장

- 주 표: raw quantity RMSE(주 지표), raw quantity MAE(필수 동반 지표).
- 보조: train-only 경계로 나눈 body/`>p99` MAE와 수량 구간별 오차. 구간별로
  checkpoint나 모델을 다시 고르지 않는다.
- 최종 목표: 각 데이터셋·모델의 seeds 42/52/62 평균과 sample SD, paired-seed
  차이와 승수. 데이터셋 간 raw error를 pooling하지 않는다.
- 시간: 현재 알려진 열화와 실패한 개선 실험을 limitation에 명시한다.
  `time_nll`이라는 legacy artifact 필드는 본문/부록에서 **clamped time loss**로
  부른다. 정상화된 duration NLL과 수치 비교하지 않는다. Proper time likelihood의
  공정 비교는 별도 연구로 둘 수 있지만, 이미 관측한 실패를 미평가로 바꾸지 않는다.

현재 허용되는 수량 주장은 다음 두 가지다.

1. 세 데이터셋 seed42 validation에서 B는 A보다 전체 quantity MAE와 raw RMSE가
   낮았다. 이는 checkpoint/early-stopping 정책 변화의 관측 결과이며 새로운
   Backbone 구조의 효과나 held-out 일반화 증거가 아니다.
2. 동일 raw-RMSE selector의 Instacart seed42에서 B는 RMTPP·THP 모두보다 전체
   MAE와 RMSE가 소폭 높았다. 추가 seed가 없어 유의한 열등성이나 동등성도 확정하지
   않는다. B의 body MAE가 조금 낮다는 사실을 전체 수량 성능 우위로 대체하지 않는다.

현재 금지되는 주장은 다음과 같다.

- 세 데이터셋에서 B가 RMTPP·THP보다 수량 예측이 우월하다.
- Quantity-aware 입력 또는 direct regression 자체가 다른 인터페이스보다 우월하다.
  이 주장은 동일 조건의 해당 ablation이 없으므로 기여에서 제외한다.
- B의 개선은 새로운 memory/Backbone 구조의 결과다.
- Time NLL을 주 표에서 제외하면 전체 TPP 우위가 입증된다.
- seed42 validation 차이를 통계적 유의성, 동등성 또는 held-out 성능으로 표현한다.

기여 후보는 공통 수량 인터페이스에서의 정적 memory 모델 평가와 selector가 수량
오차에 미치는 영향 분석이다. **논문의 출판 가능성이나 새 기법의 독창성이
확정됐다는 의미는 아니다.** 완성된 동일 조건 결과와 기존 문헌 대비 차별성이
필요하다. 수량 우위가 확보됐다는 전제로 기여를 확정하지 않는다.

## 4. 기존 결과 재사용과 잠금

재사용: B seed42 3개, baseline CUDA·e1 6개, Instacart RMTPP/THP seed42 full fit
2개. Old joint-selector 3-seed 표는 별도 역사적 증적으로만 유지한다.
기존 summary의 raw RMSE 최솟값만 읽어 checkpoint를 복원했다고 간주하지 않는다.
재선택하려면 해당 epoch의 실제 weights와 동일 early-stopping 학습 경로가 있어야 한다.

현재 frozen campaign은 Instacart gate 실패로 종료됐다. 남은 seed42 4개 및
추가 seed 18개는 자동 실행 대상이 아니다. 혼합 결과를 포함한 완결 평가로 범위를
변경할 경우, 실패 결과를 유지한 새 실행 계약을 먼저 작성해야 한다. 모든
데이터셋 승리를 요구했던 과거 gate를 사후 통과로 변경하지 않는다.

최종 held-out은 모든 모델·selector·seed checkpoint와 metric 구현을 validation에서
고정한 후 잠금 해제한다. 완전한 3모델×3데이터셋×3seed 비교라면 27 checkpoint
평가이며 추가 학습 27개라는 뜻이 아니다. Test 결과로 loss·selector를 다시
조정하거나 유리한 데이터셋/seed만 고르지 않는다.

## 5. K=2 단계 경계

이번 감사로 수량 중심 **범위**는 정의했으나, 수량 우위와 최종 논문 증거의 확정
조건은 충족되지 않았다. 따라서 사용자 요청의 선후 조건에 따라 **K=2 구현과
로컬 학습 가능성 검증은 보류**한다. 이 보류는 K=2의 수학적 불가능성을 뜻하지 않는다.

후속으로 K=2를 진행하면 shared-median scale mixture의 latent continuous median
보존, 초기 density/CDF/survival 근접성, component symmetry 해소와 실제 gradient,
integer/censored likelihood의 안정성부터 검증한다. 정수 관측 median은 관측 모델의
quantile 정의까지 별도 검증한다. 수량 경로는 고정하며, 성공해도 우선 시간분포
head 개선으로 서술한다. Quantity identity를 보존하는 K=2는 현재의 quantity
RMTPP/THP 순위를 바꿀 수 없다.
