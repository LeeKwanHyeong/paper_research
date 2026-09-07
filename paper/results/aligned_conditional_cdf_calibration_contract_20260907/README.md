# 공통 conditional CDF calibration 계약 검토

## 판정

후보는 **로컬 수학·수치 계약을 충족해 구현 가능한 단일 공통 후보**로 판정했다. 이번 단계에서는 학습 runner, CUDA e1, full fit과 held-out 평가를 실행하지 않았다.

기존 aligned-B가 제공하는 행별 log-normal CDF를 `u=F_B(t|h)`라 두고 다음 변환을 적용한다.

\[
F_C(t\mid h)=1-\left(1-u^{a(h)}\right)^{b(h)}.
\]

`a(h),b(h)`는 aligned-B의 detached 64차원 `time_hidden`을 입력으로 받는 하나의 `Linear(64,2)`에서 계산한다. 가중치와 bias를 0으로 초기화하면 모든 행에서 `a=b=1`이므로 density, CDF, survival, 양의 정수 구간확률과 base quantile level이 aligned-B와 정확히 일치한다. 전체 trainable parameter는 세 데이터셋에서 동일하게 130개다.

이 변환은 Taxi의 `D=1`이나 Instacart의 `D=30`을 모델 입력 또는 구조에 넣지 않는다. Intermittent, Taxi와 Instacart가 공유하는 것은 동일한 latent CDF와 calibration layer다. 연속 density, 정수 구간확률과 top-code survival의 차이는 데이터 수집 방식에 따른 observation operator이며 calibrator의 구조나 파라미터 분기가 아니다.

## 로컬 수치 검증

`simple_lab_test/search/tests/test_conditional_cdf_calibrator.py`의 8개 테스트로 다음을 확인했다.

- `z=-40`부터 `z=40`까지 초기 density, CDF와 survival이 base tensor와 bitwise하게 일치한다.
- 비초기 shape에서 변환 CDF가 단조이며 `logaddexp(log CDF, log survival)` 오차가 `1e-10`보다 작다.
- 변환 density의 수치 적분값이 `1`과 absolute tolerance `1e-4` 안에서 일치한다.
- 첫 구간, 일반 정수 구간과 마지막 survival을 합친 전체 확률질량이 `1`과 absolute tolerance `1e-10` 안에서 일치한다.
- 연속 likelihood와 정수·censor likelihood 모두 epoch 0에서 두 shape head의 gradient가 finite하고 0이 아니다.
- 상보적이지 않은 base CDF/survival과 float32 likelihood 입력은 fail-closed로 거부한다.

초기 구현은 `torch.where`의 선택되지 않은 극단-tail 분기에서도 autograd가 NaN을 계산하는 문제가 있었다. 이를 인덱스로 선택된 분기만 평가하는 방식으로 수정했다. 오른쪽 극단에서는 `log(1-F_B^a)`를 `log(a)+log(S_B)`로, 왼쪽 극단에서는 `log F_C`를 `log(b)+a log(F_B)`로 계산하고, 수치 identity 변환과의 차이만 base log-probability에 더한다.

Kumaraswamy 분포는 닫힌형 CDF와 quantile을 갖는 두 shape 분포이며, 일반적인 base CDF에 적용해 더 유연한 generated family를 구성할 수 있다. 기본 식과 tractability는 [Jones (2009)](https://doi.org/10.1016/j.stamet.2008.04.001)에 근거한다. 극단 shape와 tail에서 직접 확률 계산이 불안정할 수 있다는 점은 [Wasserman and Mateos (2024)](https://arxiv.org/abs/2410.00660)의 안정화 분석과 일치한다. Time-to-event 문제에서 행별 conditional calibration을 별도로 평가해야 한다는 근거는 [Qi, Yu, and Greiner (2024)](https://arxiv.org/abs/2410.20579)에 정리되어 있다.

## 고정된 후속 판정 기준

향후 학습은 세 데이터셋 모두 동일한 optimizer, batch size, epoch budget, shape bound와 selector를 사용한다. Candidate는 각 데이터셋에서 다음 조건을 모두 만족해야 한다.

1. aligned-B보다 primary observation NLL을 `0.005` 이상 개선한다.
2. aligned-A의 primary NLL보다 `0.01`을 초과해 높지 않다.
3. 두 개의 global shape만 학습한 control보다 `0.005` 이상 낮다.
4. 동일 용량의 permuted-hidden control보다 `0.005` 이상 낮다.
5. aligned-B 대비 continuous reference NLL 악화가 `0.01` 이하다.
6. time median MAE와 RMSE 악화가 각각 `2%` 이하다.
7. quantity prediction과 source model state가 정확히 보존된다.

세 데이터셋 모두 통과해야 공통 개선으로 해석한다. Taxi seed42가 실패하면 나머지 full fit을 중단하고, 결과를 본 뒤 shape bound, loss, selector 또는 데이터셋별 설정을 바꾸지 않는다.

## 주장 경계

이 후보는 aligned-B hidden state를 다시 인코딩하거나 Hard-LMM memory를 변경하지 않는다. 따라서 성공하더라도 **Backbone 자체의 개선**이 아니라, Hard-LMM의 수량 경로를 보존하는 **공통 conditional time-distribution interface** 기여다. Backbone 개선이라고 주장하려면 이후 별도 실험에서 encoder 또는 memory 표현의 변화가 세 데이터셋의 quantity와 time 성능을 함께 높인다는 증거가 필요하다.
