# 신규 후보의 이득·손해와 후속 연구 판단

선택 epoch는 Validation에서 동결됐다. Test 순위는 사후 탐색 결과이며 이후 모델 선택에 사용하면 해당 Test의 독립 평가 자격을 유지할 수 없다.

## 데이터별 관측

| 데이터 | 신규 중 Validation RMSE 최저 | 전체 Test RMSE 관측1위 | 신규 중 Test RMSE 최저 | Validation 4목적 Pareto의 신규 후보 |
|---|---|---|---|---|
| Taxi | CNN+MLP16 | MLP16 | CNN+GRU54 | CNN+MLP16 |
| Intermittent | CNN+MLP16 | 모든 가용 분기 MLP | GRU54 | GRU54, CNN+GRU54 |
| RAF | CNN+MLP16 | GRU54 | GRU54 | CNN+MLP16, GRU54 |

## 동일 seed42에서 MLP16·S2P2·지표별 최선 외부 모델과 비교

NLL은 절대 차이만 표시한다. 큰 수량 기준은 원래 TRAIN의 엄격 초과이며 Test에서 다시 정하지 않았다.

| 데이터 | split | 범위 | 신규 | 지표 | 기준 | 절대 Δ | 수량 변화율 |
|---|---|---|---|---|---|---:|---:|
| Intermittent | test | overall | CNN+GRU54 | qty_rmse | MLP16 | 0.180075 | 6.212597% |
| Intermittent | test | overall | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 | 0.179715 | 6.199409% |
| Intermittent | test | overall | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 0.179715 | 6.199409% |
| Intermittent | test | overall | CNN+MLP16 | qty_rmse | MLP16 | -0.091327 | -3.150794% |
| Intermittent | test | overall | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 | -0.091687 | -3.162819% |
| Intermittent | test | overall | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -0.091687 | -3.162819% |
| Intermittent | test | overall | GRU54 | qty_rmse | MLP16 | -0.247828 | -8.550080% |
| Intermittent | test | overall | GRU54 | qty_rmse | S2P2 · 공통 출력부 | -0.248188 | -8.561435% |
| Intermittent | test | overall | GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -0.248188 | -8.561435% |
| Intermittent | test | overall | CNN+GRU54 | qty_mae | MLP16 | 0.092014 | 8.054826% |
| Intermittent | test | overall | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 | -0.055187 | -4.279598% |
| Intermittent | test | overall | CNN+GRU54 | qty_mae | THP · 지표최선외부 | -0.048568 | -3.785725% |
| Intermittent | test | overall | CNN+MLP16 | qty_mae | MLP16 | 0.030435 | 2.664225% |
| Intermittent | test | overall | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 | -0.116766 | -9.054864% |
| Intermittent | test | overall | CNN+MLP16 | qty_mae | THP · 지표최선외부 | -0.110147 | -8.585629% |
| Intermittent | test | overall | GRU54 | qty_mae | MLP16 | -0.078694 | -6.888825% |
| Intermittent | test | overall | GRU54 | qty_mae | S2P2 · 공통 출력부 | -0.225895 | -17.517436% |
| Intermittent | test | overall | GRU54 | qty_mae | THP · 지표최선외부 | -0.219275 | -17.091864% |
| Intermittent | test | overall | CNN+GRU54 | time_nll | MLP16 | -11.262760 | — |
| Intermittent | test | overall | CNN+GRU54 | time_nll | S2P2 · 공통 출력부 | 1.089820 | — |
| Intermittent | test | overall | CNN+GRU54 | time_nll | THP · 지표최선외부 | 11.065566 | — |
| Intermittent | test | overall | CNN+MLP16 | time_nll | MLP16 | -15.631058 | — |
| Intermittent | test | overall | CNN+MLP16 | time_nll | S2P2 · 공통 출력부 | -3.278478 | — |
| Intermittent | test | overall | CNN+MLP16 | time_nll | THP · 지표최선외부 | 6.697268 | — |
| Intermittent | test | overall | GRU54 | time_nll | MLP16 | -13.185823 | — |
| Intermittent | test | overall | GRU54 | time_nll | S2P2 · 공통 출력부 | -0.833243 | — |
| Intermittent | test | overall | GRU54 | time_nll | THP · 지표최선외부 | 9.142503 | — |
| Intermittent | test | tail | CNN+GRU54 | qty_rmse | MLP16 | -1.687070 | -10.624964% |
| Intermittent | test | tail | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 | -2.176204 | -13.295893% |
| Intermittent | test | tail | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -2.176204 | -13.295893% |
| Intermittent | test | tail | CNN+MLP16 | qty_rmse | MLP16 | -4.672803 | -29.428756% |
| Intermittent | test | tail | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 | -5.161937 | -31.537744% |
| Intermittent | test | tail | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -5.161937 | -31.537744% |
| Intermittent | test | tail | GRU54 | qty_rmse | MLP16 | -2.344395 | -14.764722% |
| Intermittent | test | tail | GRU54 | qty_rmse | S2P2 · 공통 출력부 | -2.833530 | -17.311937% |
| Intermittent | test | tail | GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -2.833530 | -17.311937% |
| Intermittent | test | tail | CNN+GRU54 | qty_mae | MLP16 | -1.497098 | -10.043037% |
| Intermittent | test | tail | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 | -1.818438 | -11.941280% |
| Intermittent | test | tail | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 · 지표최선외부 | -1.818438 | -11.941280% |
| Intermittent | test | tail | CNN+MLP16 | qty_mae | MLP16 | -5.528990 | -37.090329% |
| Intermittent | test | tail | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 | -5.850329 | -38.417828% |
| Intermittent | test | tail | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 · 지표최선외부 | -5.850329 | -38.417828% |
| Intermittent | test | tail | GRU54 | qty_mae | MLP16 | -3.465248 | -23.246055% |
| Intermittent | test | tail | GRU54 | qty_mae | S2P2 · 공통 출력부 | -3.786588 | -24.865692% |
| Intermittent | test | tail | GRU54 | qty_mae | S2P2 · 공통 출력부 · 지표최선외부 | -3.786588 | -24.865692% |
| Intermittent | test | tail | CNN+GRU54 | time_nll | MLP16 | 4.1291e-16 | — |
| Intermittent | test | tail | CNN+GRU54 | time_nll | S2P2 · 공통 출력부 | -0.027105 | — |
| Intermittent | test | tail | CNN+GRU54 | time_nll | NHP · 지표최선외부 | -2.3272e-09 | — |
| Intermittent | test | tail | CNN+MLP16 | time_nll | MLP16 | 2.4789e-12 | — |
| Intermittent | test | tail | CNN+MLP16 | time_nll | S2P2 · 공통 출력부 | -0.027105 | — |
| Intermittent | test | tail | CNN+MLP16 | time_nll | NHP · 지표최선외부 | -2.3247e-09 | — |
| Intermittent | test | tail | GRU54 | time_nll | MLP16 | 1.3929e-28 | — |
| Intermittent | test | tail | GRU54 | time_nll | S2P2 · 공통 출력부 | -0.027105 | — |
| Intermittent | test | tail | GRU54 | time_nll | NHP · 지표최선외부 | -2.3272e-09 | — |
| Intermittent | validation | overall | CNN+GRU54 | qty_rmse | MLP16 | 0.135899 | 8.324758% |
| Intermittent | validation | overall | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 | -0.719114 | -28.909292% |
| Intermittent | validation | overall | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -0.719114 | -28.909292% |
| Intermittent | validation | overall | CNN+MLP16 | qty_rmse | MLP16 | 0.087644 | 5.368781% |
| Intermittent | validation | overall | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 | -0.767369 | -30.849222% |
| Intermittent | validation | overall | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -0.767369 | -30.849222% |
| Intermittent | validation | overall | GRU54 | qty_rmse | MLP16 | 0.140669 | 8.616937% |
| Intermittent | validation | overall | GRU54 | qty_rmse | S2P2 · 공통 출력부 | -0.714344 | -28.717542% |
| Intermittent | validation | overall | GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -0.714344 | -28.717542% |
| Intermittent | validation | overall | CNN+GRU54 | qty_mae | MLP16 | -0.008676 | -1.215269% |
| Intermittent | validation | overall | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 | -0.235423 | -25.026955% |
| Intermittent | validation | overall | CNN+GRU54 | qty_mae | RMTPP · 지표최선외부 | -0.204004 | -22.436297% |
| Intermittent | validation | overall | CNN+MLP16 | qty_mae | MLP16 | -5.2700e-05 | -0.007382% |
| Intermittent | validation | overall | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 | -0.226799 | -24.110225% |
| Intermittent | validation | overall | CNN+MLP16 | qty_mae | RMTPP · 지표최선외부 | -0.195380 | -21.487889% |
| Intermittent | validation | overall | GRU54 | qty_mae | MLP16 | -0.023367 | -3.272963% |
| Intermittent | validation | overall | GRU54 | qty_mae | S2P2 · 공통 출력부 | -0.250113 | -26.588650% |
| Intermittent | validation | overall | GRU54 | qty_mae | RMTPP · 지표최선외부 | -0.218694 | -24.051956% |
| Intermittent | validation | overall | CNN+GRU54 | time_nll | MLP16 | -0.054087 | — |
| Intermittent | validation | overall | CNN+GRU54 | time_nll | S2P2 · 공통 출력부 | 0.043255 | — |
| Intermittent | validation | overall | CNN+GRU54 | time_nll | S2P2 · 공통 출력부 · 지표최선외부 | 0.043255 | — |
| Intermittent | validation | overall | CNN+MLP16 | time_nll | MLP16 | 0.050584 | — |
| Intermittent | validation | overall | CNN+MLP16 | time_nll | S2P2 · 공통 출력부 | 0.147926 | — |
| Intermittent | validation | overall | CNN+MLP16 | time_nll | S2P2 · 공통 출력부 · 지표최선외부 | 0.147926 | — |
| Intermittent | validation | overall | GRU54 | time_nll | MLP16 | -0.070155 | — |
| Intermittent | validation | overall | GRU54 | time_nll | S2P2 · 공통 출력부 | 0.027187 | — |
| Intermittent | validation | overall | GRU54 | time_nll | S2P2 · 공통 출력부 · 지표최선외부 | 0.027187 | — |
| Intermittent | validation | tail | CNN+GRU54 | qty_rmse | MLP16 | 1.302136 | 17.572086% |
| Intermittent | validation | tail | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 | -7.566521 | -46.480512% |
| Intermittent | validation | tail | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -7.566521 | -46.480512% |
| Intermittent | validation | tail | CNN+MLP16 | qty_rmse | MLP16 | 1.364259 | 18.410420% |
| Intermittent | validation | tail | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 | -7.504399 | -46.098898% |
| Intermittent | validation | tail | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -7.504399 | -46.098898% |
| Intermittent | validation | tail | GRU54 | qty_rmse | MLP16 | 2.101905 | 28.364814% |
| Intermittent | validation | tail | GRU54 | qty_rmse | S2P2 · 공통 출력부 | -6.766753 | -41.567601% |
| Intermittent | validation | tail | GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -6.766753 | -41.567601% |
| Intermittent | validation | tail | CNN+GRU54 | qty_mae | MLP16 | 1.211741 | 21.927026% |
| Intermittent | validation | tail | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 | -6.288838 | -48.276065% |
| Intermittent | validation | tail | CNN+GRU54 | qty_mae | RMTPP · 지표최선외부 | -5.147761 | -43.310370% |
| Intermittent | validation | tail | CNN+MLP16 | qty_mae | MLP16 | 1.616101 | 29.244102% |
| Intermittent | validation | tail | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 | -5.884478 | -45.172012% |
| Intermittent | validation | tail | CNN+MLP16 | qty_mae | RMTPP · 지표최선외부 | -4.743401 | -39.908316% |
| Intermittent | validation | tail | GRU54 | qty_mae | MLP16 | 1.883272 | 34.078696% |
| Intermittent | validation | tail | GRU54 | qty_mae | S2P2 · 공통 출력부 | -5.617307 | -43.121078% |
| Intermittent | validation | tail | GRU54 | qty_mae | RMTPP · 지표최선외부 | -4.476230 | -37.660484% |
| Intermittent | validation | tail | CNN+GRU54 | time_nll | MLP16 | 2.4398e-14 | — |
| Intermittent | validation | tail | CNN+GRU54 | time_nll | S2P2 · 공통 출력부 | -0.096620 | — |
| Intermittent | validation | tail | CNN+GRU54 | time_nll | NHP · 지표최선외부 | -1.4172e-08 | — |
| Intermittent | validation | tail | CNN+MLP16 | time_nll | MLP16 | 0.011586 | — |
| Intermittent | validation | tail | CNN+MLP16 | time_nll | S2P2 · 공통 출력부 | -0.085034 | — |
| Intermittent | validation | tail | CNN+MLP16 | time_nll | NHP · 지표최선외부 | 0.011586 | — |
| Intermittent | validation | tail | GRU54 | time_nll | MLP16 | 1.1172e-13 | — |
| Intermittent | validation | tail | GRU54 | time_nll | S2P2 · 공통 출력부 | -0.096620 | — |
| Intermittent | validation | tail | GRU54 | time_nll | NHP · 지표최선외부 | -1.4172e-08 | — |
| RAF | test | overall | CNN+GRU54 | qty_rmse | MLP16 | -0.751399 | -1.859588% |
| RAF | test | overall | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 | 0.017963 | 0.045318% |
| RAF | test | overall | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 0.017963 | 0.045318% |
| RAF | test | overall | CNN+MLP16 | qty_rmse | MLP16 | -0.624353 | -1.545170% |
| RAF | test | overall | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 | 0.145009 | 0.365838% |
| RAF | test | overall | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 0.145009 | 0.365838% |
| RAF | test | overall | GRU54 | qty_rmse | MLP16 | -0.803028 | -1.987361% |
| RAF | test | overall | GRU54 | qty_rmse | S2P2 · 공통 출력부 | -0.033666 | -0.084935% |
| RAF | test | overall | GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -0.033666 | -0.084935% |
| RAF | test | overall | CNN+GRU54 | qty_mae | MLP16 | -0.103071 | -1.005932% |
| RAF | test | overall | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 | -0.115392 | -1.124823% |
| RAF | test | overall | CNN+GRU54 | qty_mae | RMTPP · 지표최선외부 | 0.276477 | 2.802092% |
| RAF | test | overall | CNN+MLP16 | qty_mae | MLP16 | -0.111975 | -1.092830% |
| RAF | test | overall | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 | -0.124296 | -1.211616% |
| RAF | test | overall | CNN+MLP16 | qty_mae | RMTPP · 지표최선외부 | 0.267573 | 2.711852% |
| RAF | test | overall | GRU54 | qty_mae | MLP16 | -0.211807 | -2.067148% |
| RAF | test | overall | GRU54 | qty_mae | S2P2 · 공통 출력부 | -0.224128 | -2.184764% |
| RAF | test | overall | GRU54 | qty_mae | RMTPP · 지표최선외부 | 0.167741 | 1.700054% |
| RAF | test | overall | CNN+GRU54 | time_nll | MLP16 | -0.040085 | — |
| RAF | test | overall | CNN+GRU54 | time_nll | S2P2 · 공통 출력부 | 0.310007 | — |
| RAF | test | overall | CNN+GRU54 | time_nll | NHP · 지표최선외부 | 0.528247 | — |
| RAF | test | overall | CNN+MLP16 | time_nll | MLP16 | -0.106697 | — |
| RAF | test | overall | CNN+MLP16 | time_nll | S2P2 · 공통 출력부 | 0.243396 | — |
| RAF | test | overall | CNN+MLP16 | time_nll | NHP · 지표최선외부 | 0.461636 | — |
| RAF | test | overall | GRU54 | time_nll | MLP16 | -0.308370 | — |
| RAF | test | overall | GRU54 | time_nll | S2P2 · 공통 출력부 | 0.041723 | — |
| RAF | test | overall | GRU54 | time_nll | NHP · 지표최선외부 | 0.259963 | — |
| RAF | test | tail | CNN+GRU54 | qty_rmse | MLP16 | -11.328622 | -2.725580% |
| RAF | test | tail | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 | 4.676090 | 1.170087% |
| RAF | test | tail | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 4.676090 | 1.170087% |
| RAF | test | tail | CNN+MLP16 | qty_rmse | MLP16 | -13.211735 | -3.178643% |
| RAF | test | tail | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 | 2.792976 | 0.698880% |
| RAF | test | tail | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 2.792976 | 0.698880% |
| RAF | test | tail | GRU54 | qty_rmse | MLP16 | -13.271386 | -3.192994% |
| RAF | test | tail | GRU54 | qty_rmse | S2P2 · 공통 출력부 | 2.733325 | 0.683954% |
| RAF | test | tail | GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 2.733325 | 0.683954% |
| RAF | test | tail | CNN+GRU54 | qty_mae | MLP16 | -17.921186 | -4.810162% |
| RAF | test | tail | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 | 0.845250 | 0.238904% |
| RAF | test | tail | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 · 지표최선외부 | 0.845250 | 0.238904% |
| RAF | test | tail | CNN+MLP16 | qty_mae | MLP16 | -17.950772 | -4.818103% |
| RAF | test | tail | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 | 0.815664 | 0.230542% |
| RAF | test | tail | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 · 지표최선외부 | 0.815664 | 0.230542% |
| RAF | test | tail | GRU54 | qty_mae | MLP16 | -21.406556 | -5.745657% |
| RAF | test | tail | GRU54 | qty_mae | S2P2 · 공통 출력부 | -2.640119 | -0.746212% |
| RAF | test | tail | GRU54 | qty_mae | S2P2 · 공통 출력부 · 지표최선외부 | -2.640119 | -0.746212% |
| RAF | test | tail | CNN+GRU54 | time_nll | MLP16 | -0.492835 | — |
| RAF | test | tail | CNN+GRU54 | time_nll | S2P2 · 공통 출력부 | 0.192212 | — |
| RAF | test | tail | CNN+GRU54 | time_nll | NHP · 지표최선외부 | 0.580524 | — |
| RAF | test | tail | CNN+MLP16 | time_nll | MLP16 | -0.309164 | — |
| RAF | test | tail | CNN+MLP16 | time_nll | S2P2 · 공통 출력부 | 0.375883 | — |
| RAF | test | tail | CNN+MLP16 | time_nll | NHP · 지표최선외부 | 0.764194 | — |
| RAF | test | tail | GRU54 | time_nll | MLP16 | -0.758709 | — |
| RAF | test | tail | GRU54 | time_nll | S2P2 · 공통 출력부 | -0.073662 | — |
| RAF | test | tail | GRU54 | time_nll | NHP · 지표최선외부 | 0.314650 | — |
| RAF | validation | overall | CNN+GRU54 | qty_rmse | MLP16 | 0.145013 | 0.426713% |
| RAF | validation | overall | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 | 0.166047 | 0.488908% |
| RAF | validation | overall | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 0.166047 | 0.488908% |
| RAF | validation | overall | CNN+MLP16 | qty_rmse | MLP16 | -0.127540 | -0.375297% |
| RAF | validation | overall | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 | -0.106507 | -0.313598% |
| RAF | validation | overall | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -0.106507 | -0.313598% |
| RAF | validation | overall | GRU54 | qty_rmse | MLP16 | 0.168130 | 0.494736% |
| RAF | validation | overall | GRU54 | qty_rmse | S2P2 · 공통 출력부 | 0.189164 | 0.556973% |
| RAF | validation | overall | GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 0.189164 | 0.556973% |
| RAF | validation | overall | CNN+GRU54 | qty_mae | MLP16 | -0.045613 | -0.488899% |
| RAF | validation | overall | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 | -0.027622 | -0.296640% |
| RAF | validation | overall | CNN+GRU54 | qty_mae | RMTPP · 지표최선외부 | 0.209090 | 2.304001% |
| RAF | validation | overall | CNN+MLP16 | qty_mae | MLP16 | -0.135063 | -1.447653% |
| RAF | validation | overall | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 | -0.117072 | -1.257246% |
| RAF | validation | overall | CNN+MLP16 | qty_mae | RMTPP · 지표최선외부 | 0.119640 | 1.318339% |
| RAF | validation | overall | GRU54 | qty_mae | MLP16 | -0.212250 | -2.274974% |
| RAF | validation | overall | GRU54 | qty_mae | S2P2 · 공통 출력부 | -0.194259 | -2.086166% |
| RAF | validation | overall | GRU54 | qty_mae | RMTPP · 지표최선외부 | 0.042453 | 0.467798% |
| RAF | validation | overall | CNN+GRU54 | time_nll | MLP16 | -0.007853 | — |
| RAF | validation | overall | CNN+GRU54 | time_nll | S2P2 · 공통 출력부 | 0.161438 | — |
| RAF | validation | overall | CNN+GRU54 | time_nll | NHP · 지표최선외부 | 0.198958 | — |
| RAF | validation | overall | CNN+MLP16 | time_nll | MLP16 | -0.099926 | — |
| RAF | validation | overall | CNN+MLP16 | time_nll | S2P2 · 공통 출력부 | 0.069366 | — |
| RAF | validation | overall | CNN+MLP16 | time_nll | NHP · 지표최선외부 | 0.106885 | — |
| RAF | validation | overall | GRU54 | time_nll | MLP16 | -0.126860 | — |
| RAF | validation | overall | GRU54 | time_nll | S2P2 · 공통 출력부 | 0.042432 | — |
| RAF | validation | overall | GRU54 | time_nll | NHP · 지표최선외부 | 0.079951 | — |
| RAF | validation | tail | CNN+GRU54 | qty_rmse | MLP16 | -1.301323 | -0.395988% |
| RAF | validation | tail | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 | 6.414862 | 1.998957% |
| RAF | validation | tail | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 6.414862 | 1.998957% |
| RAF | validation | tail | CNN+MLP16 | qty_rmse | MLP16 | -7.666151 | -2.332784% |
| RAF | validation | tail | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 | 0.050034 | 0.015591% |
| RAF | validation | tail | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 0.050034 | 0.015591% |
| RAF | validation | tail | GRU54 | qty_rmse | MLP16 | -2.663618 | -0.810530% |
| RAF | validation | tail | GRU54 | qty_rmse | S2P2 · 공통 출력부 | 5.052566 | 1.574447% |
| RAF | validation | tail | GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 5.052566 | 1.574447% |
| RAF | validation | tail | CNN+GRU54 | qty_mae | MLP16 | -5.971757 | -2.090704% |
| RAF | validation | tail | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 | 6.281672 | 2.297778% |
| RAF | validation | tail | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 · 지표최선외부 | 6.281672 | 2.297778% |
| RAF | validation | tail | CNN+MLP16 | qty_mae | MLP16 | -7.571822 | -2.650885% |
| RAF | validation | tail | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 | 4.681607 | 1.712489% |
| RAF | validation | tail | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 · 지표최선외부 | 4.681607 | 1.712489% |
| RAF | validation | tail | GRU54 | qty_mae | MLP16 | -6.710376 | -2.349294% |
| RAF | validation | tail | GRU54 | qty_mae | S2P2 · 공통 출력부 | 5.543053 | 2.027597% |
| RAF | validation | tail | GRU54 | qty_mae | S2P2 · 공통 출력부 · 지표최선외부 | 5.543053 | 2.027597% |
| RAF | validation | tail | CNN+GRU54 | time_nll | MLP16 | -0.170302 | — |
| RAF | validation | tail | CNN+GRU54 | time_nll | S2P2 · 공통 출력부 | 0.220314 | — |
| RAF | validation | tail | CNN+GRU54 | time_nll | THP · 지표최선외부 | 0.340065 | — |
| RAF | validation | tail | CNN+MLP16 | time_nll | MLP16 | -0.272333 | — |
| RAF | validation | tail | CNN+MLP16 | time_nll | S2P2 · 공통 출력부 | 0.118283 | — |
| RAF | validation | tail | CNN+MLP16 | time_nll | THP · 지표최선외부 | 0.238034 | — |
| RAF | validation | tail | GRU54 | time_nll | MLP16 | -0.398787 | — |
| RAF | validation | tail | GRU54 | time_nll | S2P2 · 공통 출력부 | -0.008171 | — |
| RAF | validation | tail | GRU54 | time_nll | THP · 지표최선외부 | 0.111580 | — |
| Taxi | test | overall | CNN+GRU54 | qty_rmse | MLP16 | 10.008278 | 10.432729% |
| Taxi | test | overall | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 | -14.377585 | -11.949713% |
| Taxi | test | overall | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | -14.377585 | -11.949713% |
| Taxi | test | overall | CNN+MLP16 | qty_rmse | MLP16 | 32.457782 | 33.834317% |
| Taxi | test | overall | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 | 8.071920 | 6.708854% |
| Taxi | test | overall | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 8.071920 | 6.708854% |
| Taxi | test | overall | GRU54 | qty_rmse | MLP16 | 79.387548 | 82.754374% |
| Taxi | test | overall | GRU54 | qty_rmse | S2P2 · 공통 출력부 | 55.001685 | 45.713822% |
| Taxi | test | overall | GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 55.001685 | 45.713822% |
| Taxi | test | overall | CNN+GRU54 | qty_mae | MLP16 | 1.459348 | 4.527050% |
| Taxi | test | overall | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 | -5.822297 | -14.733343% |
| Taxi | test | overall | CNN+GRU54 | qty_mae | AttNHP · 공통 출력부 · 지표최선외부 | -3.903409 | -10.381700% |
| Taxi | test | overall | CNN+MLP16 | qty_mae | MLP16 | 5.659766 | 17.557185% |
| Taxi | test | overall | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 | -1.621879 | -4.104171% |
| Taxi | test | overall | CNN+MLP16 | qty_mae | AttNHP · 공통 출력부 · 지표최선외부 | 0.297009 | 0.789939% |
| Taxi | test | overall | GRU54 | qty_mae | MLP16 | 18.986020 | 58.896617% |
| Taxi | test | overall | GRU54 | qty_mae | S2P2 · 공통 출력부 | 11.704375 | 29.617963% |
| Taxi | test | overall | GRU54 | qty_mae | AttNHP · 공통 출력부 · 지표최선외부 | 13.623262 | 36.233105% |
| Taxi | test | overall | CNN+GRU54 | time_nll | MLP16 | 0.561348 | — |
| Taxi | test | overall | CNN+GRU54 | time_nll | S2P2 · 공통 출력부 | 0.574135 | — |
| Taxi | test | overall | CNN+GRU54 | time_nll | THP · 지표최선외부 | 0.595312 | — |
| Taxi | test | overall | CNN+MLP16 | time_nll | MLP16 | 0.862849 | — |
| Taxi | test | overall | CNN+MLP16 | time_nll | S2P2 · 공통 출력부 | 0.875636 | — |
| Taxi | test | overall | CNN+MLP16 | time_nll | THP · 지표최선외부 | 0.896812 | — |
| Taxi | test | overall | GRU54 | time_nll | MLP16 | 0.418469 | — |
| Taxi | test | overall | GRU54 | time_nll | S2P2 · 공통 출력부 | 0.431256 | — |
| Taxi | test | overall | GRU54 | time_nll | THP · 지표최선외부 | 0.452433 | — |
| Taxi | test | tail | CNN+GRU54 | qty_rmse | MLP16 | 131.979476 | 28.875643% |
| Taxi | test | tail | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 | 63.426918 | 12.067201% |
| Taxi | test | tail | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 63.426918 | 12.067201% |
| Taxi | test | tail | CNN+MLP16 | qty_rmse | MLP16 | 380.048121 | 83.150305% |
| Taxi | test | tail | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 | 311.495563 | 59.263160% |
| Taxi | test | tail | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 311.495563 | 59.263160% |
| Taxi | test | tail | GRU54 | qty_rmse | MLP16 | 667.173195 | 145.970080% |
| Taxi | test | tail | GRU54 | qty_rmse | S2P2 · 공통 출력부 | 598.620636 | 113.889746% |
| Taxi | test | tail | GRU54 | qty_rmse | S2P2 · 공통 출력부 · 지표최선외부 | 598.620636 | 113.889746% |
| Taxi | test | tail | CNN+GRU54 | qty_mae | MLP16 | 91.666613 | 25.578723% |
| Taxi | test | tail | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 | 25.581099 | 6.026795% |
| Taxi | test | tail | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 · 지표최선외부 | 25.581099 | 6.026795% |
| Taxi | test | tail | CNN+MLP16 | qty_mae | MLP16 | 343.505145 | 95.851944% |
| Taxi | test | tail | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 | 277.419631 | 65.358856% |
| Taxi | test | tail | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 · 지표최선외부 | 277.419631 | 65.358856% |
| Taxi | test | tail | GRU54 | qty_mae | MLP16 | 624.663892 | 174.306701% |
| Taxi | test | tail | GRU54 | qty_mae | S2P2 · 공통 출력부 | 558.578378 | 131.598631% |
| Taxi | test | tail | GRU54 | qty_mae | S2P2 · 공통 출력부 · 지표최선외부 | 558.578378 | 131.598631% |
| Taxi | test | tail | CNN+GRU54 | time_nll | MLP16 | -2.9286e-28 | — |
| Taxi | test | tail | CNN+GRU54 | time_nll | S2P2 · 공통 출력부 | -2.1423e-15 | — |
| Taxi | test | tail | CNN+GRU54 | time_nll | RMTPP · 지표최선외부 | -1.0058e-64 | — |
| Taxi | test | tail | CNN+MLP16 | time_nll | MLP16 | -2.9286e-28 | — |
| Taxi | test | tail | CNN+MLP16 | time_nll | S2P2 · 공통 출력부 | -2.1423e-15 | — |
| Taxi | test | tail | CNN+MLP16 | time_nll | RMTPP · 지표최선외부 | -1.0058e-64 | — |
| Taxi | test | tail | GRU54 | time_nll | MLP16 | -2.9286e-28 | — |
| Taxi | test | tail | GRU54 | time_nll | S2P2 · 공통 출력부 | -2.1423e-15 | — |
| Taxi | test | tail | GRU54 | time_nll | RMTPP · 지표최선외부 | 4.0043e-61 | — |
| Taxi | validation | overall | CNN+GRU54 | qty_rmse | MLP16 | -4.179352 | -5.065933% |
| Taxi | validation | overall | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 | -26.092422 | -24.989814% |
| Taxi | validation | overall | CNN+GRU54 | qty_rmse | RMTPP · 지표최선외부 | -15.899203 | -16.874729% |
| Taxi | validation | overall | CNN+MLP16 | qty_rmse | MLP16 | -6.011562 | -7.286816% |
| Taxi | validation | overall | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 | -27.924632 | -26.744598% |
| Taxi | validation | overall | CNN+MLP16 | qty_rmse | RMTPP · 지표최선외부 | -17.731413 | -18.819357% |
| Taxi | validation | overall | GRU54 | qty_rmse | MLP16 | 2.618534 | 3.174013% |
| Taxi | validation | overall | GRU54 | qty_rmse | S2P2 · 공통 출력부 | -19.294536 | -18.479192% |
| Taxi | validation | overall | GRU54 | qty_rmse | RMTPP · 지표최선외부 | -9.101317 | -9.659746% |
| Taxi | validation | overall | CNN+GRU54 | qty_mae | MLP16 | -1.520404 | -5.683340% |
| Taxi | validation | overall | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 | -7.414158 | -22.710978% |
| Taxi | validation | overall | CNN+GRU54 | qty_mae | RMTPP · 지표최선외부 | -2.999254 | -10.624049% |
| Taxi | validation | overall | CNN+MLP16 | qty_mae | MLP16 | -1.935437 | -7.234751% |
| Taxi | validation | overall | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 | -7.829190 | -23.982302% |
| Taxi | validation | overall | CNN+MLP16 | qty_mae | RMTPP · 지표최선외부 | -3.414286 | -12.094190% |
| Taxi | validation | overall | GRU54 | qty_mae | MLP16 | 0.221448 | 0.827783% |
| Taxi | validation | overall | GRU54 | qty_mae | S2P2 · 공통 출력부 | -5.672305 | -17.375353% |
| Taxi | validation | overall | GRU54 | qty_mae | RMTPP · 지표최선외부 | -1.257401 | -4.454006% |
| Taxi | validation | overall | CNN+GRU54 | time_nll | MLP16 | 0.686828 | — |
| Taxi | validation | overall | CNN+GRU54 | time_nll | S2P2 · 공통 출력부 | 0.673131 | — |
| Taxi | validation | overall | CNN+GRU54 | time_nll | THP · 지표최선외부 | 0.712697 | — |
| Taxi | validation | overall | CNN+MLP16 | time_nll | MLP16 | 0.846194 | — |
| Taxi | validation | overall | CNN+MLP16 | time_nll | S2P2 · 공통 출력부 | 0.832497 | — |
| Taxi | validation | overall | CNN+MLP16 | time_nll | THP · 지표최선외부 | 0.872064 | — |
| Taxi | validation | overall | GRU54 | time_nll | MLP16 | 0.411733 | — |
| Taxi | validation | overall | GRU54 | time_nll | S2P2 · 공통 출력부 | 0.398036 | — |
| Taxi | validation | overall | GRU54 | time_nll | THP · 지표최선외부 | 0.437603 | — |
| Taxi | validation | tail | CNN+GRU54 | qty_rmse | MLP16 | -36.112357 | -8.890468% |
| Taxi | validation | tail | CNN+GRU54 | qty_rmse | S2P2 · 공통 출력부 | -153.763643 | -29.352992% |
| Taxi | validation | tail | CNN+GRU54 | qty_rmse | RMTPP · 지표최선외부 | -96.930055 | -20.755475% |
| Taxi | validation | tail | CNN+MLP16 | qty_rmse | MLP16 | -55.923033 | -13.767639% |
| Taxi | validation | tail | CNN+MLP16 | qty_rmse | S2P2 · 공통 출력부 | -173.574319 | -33.134788% |
| Taxi | validation | tail | CNN+MLP16 | qty_rmse | RMTPP · 지표최선외부 | -116.740731 | -24.997503% |
| Taxi | validation | tail | GRU54 | qty_rmse | MLP16 | 9.631512 | 2.371173% |
| Taxi | validation | tail | GRU54 | qty_rmse | S2P2 · 공통 출력부 | -108.019774 | -20.620633% |
| Taxi | validation | tail | GRU54 | qty_rmse | RMTPP · 지표최선외부 | -51.186186 | -10.960415% |
| Taxi | validation | tail | CNN+GRU54 | qty_mae | MLP16 | -50.907038 | -15.529011% |
| Taxi | validation | tail | CNN+GRU54 | qty_mae | S2P2 · 공통 출력부 | -124.658098 | -31.042682% |
| Taxi | validation | tail | CNN+GRU54 | qty_mae | RMTPP · 지표최선외부 | -79.838957 | -22.379472% |
| Taxi | validation | tail | CNN+MLP16 | qty_mae | MLP16 | -55.679823 | -16.984932% |
| Taxi | validation | tail | CNN+MLP16 | qty_mae | S2P2 · 공통 출력부 | -129.430883 | -32.231214% |
| Taxi | validation | tail | CNN+MLP16 | qty_mae | RMTPP · 지표최선외부 | -84.611742 | -23.717320% |
| Taxi | validation | tail | GRU54 | qty_mae | MLP16 | -8.904575 | -2.716309% |
| Taxi | validation | tail | GRU54 | qty_mae | S2P2 · 공통 출력부 | -82.655635 | -20.583120% |
| Taxi | validation | tail | GRU54 | qty_mae | RMTPP · 지표최선외부 | -37.836494 | -10.605859% |
| Taxi | validation | tail | CNN+GRU54 | time_nll | MLP16 | -3.8977e-37 | — |
| Taxi | validation | tail | CNN+GRU54 | time_nll | S2P2 · 공통 출력부 | -4.8681e-15 | — |
| Taxi | validation | tail | CNN+GRU54 | time_nll | RMTPP · 지표최선외부 | 9.0563e-123 | — |
| Taxi | validation | tail | CNN+MLP16 | time_nll | MLP16 | -3.8977e-37 | — |
| Taxi | validation | tail | CNN+MLP16 | time_nll | S2P2 · 공통 출력부 | -4.8681e-15 | — |
| Taxi | validation | tail | CNN+MLP16 | time_nll | RMTPP · 지표최선외부 | 3.7120e-93 | — |
| Taxi | validation | tail | GRU54 | time_nll | MLP16 | -3.8977e-37 | — |
| Taxi | validation | tail | GRU54 | time_nll | S2P2 · 공통 출력부 | -4.8681e-15 | — |
| Taxi | validation | tail | GRU54 | time_nll | RMTPP · 지표최선외부 | 2.0620e-62 | — |

## 해석 범위

- 신규3seed 안정성과 통계적 우월성은 미확인이다. 모든 불리한 지표와 비교군을 보존했다.
- CNN은 Encoder1 Q/K/V 국소 경로 추가이며 GRU는 중간 보정 구조 전체 변경이다. 연산 하나의 독립 기여로 단정하지 않는다.
- 수량·tail·시간의 목적이 다르므로 지표를 임의 가중 합산하거나 데이터 간 NLL을 합산하지 않는다.
- Test 관측1위를 후속 탐색 후보로 고를 수 있으나, 이를 최종 모델 우월성이나 독립 Test 검증으로 주장할 수 없다.
- 후속 구조의 기여는 고정 용량/출력부/손실/선택 규칙의 CNN만·GRU만·결합 대조와 추가 seed에서 검증해야 한다.
- 원본 binary 회수·CPU 감사·속도 효율 측정은 이 로컬 보고서 생성과 별도다.
