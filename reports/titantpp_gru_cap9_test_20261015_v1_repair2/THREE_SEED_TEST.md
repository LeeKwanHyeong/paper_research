# CAP9 고정 checkpoint Test 비교

Validation 전체 raw 수량 RMSE 최초 strict finite 최소 epoch를 고정했다. 기존 CNN/FULL18은 재실행하지 않았다.

| 데이터셋 | 구조 | RMSE 평균±표본SD | MAE 평균±표본SD |
|---|---|---:|---:|
| Taxi | CNN | 135.246858±31.744940 | 40.615289±7.196753 |
| Taxi | CNN+GRU | 158.706128±35.292013 | 46.880951±10.657791 |
| Taxi | CAP(ρ=0.25) | 145.873812±13.833816 | 44.273003±2.727981 |
| RAF | CNN | 39.600663±0.023654 | 9.972138±0.036428 |
| RAF | CNN+GRU | 39.895919±0.481128 | 10.131450±0.146749 |
| RAF | CAP(ρ=0.25) | 39.766269±0.280259 | 10.027510±0.053397 |
| Intermittent | CNN | 2.616125±0.078209 | 1.061696±0.036434 |
| Intermittent | CNN+GRU | 2.914264±0.238046 | 1.160457±0.093891 |
| Intermittent | CAP(ρ=0.25) | 2.877853±0.181393 | 1.119554±0.061276 |

CAP−기존 모델: 음수는 개선, 양수는 악화다.

| 데이터셋 | seed | 비교 | ΔRMSE | ΔMAE |
|---|---:|---|---:|---:|
| Taxi | 42 | CAP−CNN | -13.715677 | -2.876396 |
| Taxi | 42 | CAP−CNN+GRU | -35.124775 | -9.861315 |
| Taxi | 52 | CAP−CNN | +23.334167 | +7.171040 |
| Taxi | 52 | CAP−CNN+GRU | -34.255696 | -9.337152 |
| Taxi | 62 | CAP−CNN | +22.262372 | +6.678498 |
| Taxi | 62 | CAP−CNN+GRU | +30.883525 | +11.374624 |
| RAF | 42 | CAP−CNN | -0.072561 | +0.062440 |
| RAF | 42 | CAP−CNN+GRU | +0.057078 | -0.203985 |
| RAF | 52 | CAP−CNN | +0.500820 | +0.089756 |
| RAF | 52 | CAP−CNN+GRU | -0.343526 | +0.049114 |
| RAF | 62 | CAP−CNN | +0.068558 | +0.013918 |
| RAF | 62 | CAP−CNN+GRU | -0.102500 | -0.156950 |
| Intermittent | 42 | CAP−CNN | +0.556827 | +0.138066 |
| Intermittent | 42 | CAP−CNN+GRU | +0.141835 | -0.078078 |
| Intermittent | 52 | CAP−CNN | +0.069171 | +0.066793 |
| Intermittent | 52 | CAP−CNN+GRU | +0.083101 | +0.096697 |
| Intermittent | 62 | CAP−CNN | +0.159186 | -0.031286 |
| Intermittent | 62 | CAP−CNN+GRU | -0.334169 | -0.141326 |

전체·큰수량 수량 RMSE/MAE와 recorded positive-integer TimeNLL의 여섯 지표 및 개별 seed 원문은 JSON에 모두 보존했다.
이번 후보는 기존 Test 순위 반전을 본 뒤 설계한 탐색적 후속 실험이다. 새 독립 평가나 원인 확정으로 해석하지 않는다.
Intermittent42/52는 5090/cuDNN92000, seed62는 5080/cuDNN91900이다. 시간은 host별로 분리한다.
