# 고정 checkpoint의 행별 오류 진단

## Material Passport

- Workflow: academic-research-suite / experiment-agent / validate
- Verification Status: ANALYZED
- 대상: 선택 모델72개, Train·Validation·Test216개 모델/분할 결과
- 고유 진단 조건162개(현재 계약의 실제 추론148개, SHA 승인된 Validation 재사용14개)와 기존 외부 Test54개를 읽었습니다. 분석 코드 자체의 학습·추론·원격 호출은0입니다.
- 기준: 최초 strict finite Validation raw 수량 RMSE 최소 epoch를 그대로 유지했습니다.
- 통계: 3seed 평균과 표본SD, 행별 정확한 쌍 비교. 통계적 유의성·원인 확정은 주장하지 않습니다.

## 이번 행 분석으로 새로 확인한 내용

Train raw 오류는 고정 checkpoint를 모든 예측 가능한 학습 대상에 적용한 결과입니다. 학습 중 온라인 손실과 다르며, 선택 epoch나 모델을 바꾸지 않았습니다. 아래는 3seed 평균±표본SD입니다.

| 데이터셋 | 모델 | 고정 Train raw RMSE | 고정 Train raw MAE |
|---|---|---:|---:|
| Taxi | CNN+GRU | 79.755270±3.957326 | 24.486922±1.190226 |
| Taxi | CNN | 85.135354±1.262633 | 26.356645±0.173790 |
| RAF | CNN+GRU | 45.943471±0.275217 | 10.540761±0.040464 |
| RAF | CNN | 45.952279±0.070475 | 10.619992±0.046016 |
| Intermittent | CNN+GRU | 1.437767±0.056090 | 0.623813±0.013239 |
| Intermittent | CNN | 1.539286±0.077459 | 0.665434±0.045546 |

Taxi와Intermittent는 CNN+GRU의 Train raw RMSE·MAE·log 손실이 세 seed 모두 CNN보다 낮았습니다. 따라서 이 두 데이터셋에서는 학습 데이터부터 raw 수량을 잘 못 맞춘다는 설명을 지지하지 않습니다. RAF Train raw RMSE 평균은 거의 같고 CNN+GRU가 seed42·52에서는 더 높았습니다. RAF는 절대 Train RMSE가 Test보다 높아 Train<Test라는 전제로 과적합을 주장할 수도 없습니다.

### Test에서 log 손실과 raw 오류가 같이 나빠졌는지 확인

차이는 CNN+GRU−CNN입니다. 양수는 CNN+GRU의 손해입니다. 모든 값은 같은 고정 모델의 직접 저장된 original log_qty_mse와 raw 오류로 계산했습니다.

| 데이터셋 | 평균 Test raw RMSE 차이 | 평균 Test raw MAE 차이 | 평균 Test model log MSE 차이 |
|---|---:|---:|---:|
| Taxi | +23.459270 | +6.265662 | +0.033451 |
| RAF | +0.295255 | +0.159312 | +0.007659 |
| Intermittent | +0.298139 | +0.098760 | +0.001028 |

세 데이터셋 모두 Test 평균에서는 CNN+GRU의 raw RMSE·MAE뿐 아니라 model log MSE도 더 높았습니다. 따라서 전체 평균 반전을 Test에서 log 손실만 좋아지고 raw RMSE만 나빠지는 목적 차이 하나로 설명할 수 없습니다. log 목적과 raw 평가의 차이는 여전히 설계상의 차이지만, 이번 평균 반전의 충분한 원인으로 확인된 것은 아닙니다.

| 데이터셋 | seed | Test raw RMSE 차이 | Test raw MAE 차이 | Test model log MSE 차이 |
|---|---:|---:|---:|---:|
| Taxi | 42 | +21.409098 | +6.984919 | +0.035967 |
| Taxi | 52 | +57.589863 | +16.508192 | +0.051699 |
| Taxi | 62 | -8.621153 | -4.696126 | +0.012686 |
| RAF | 42 | -0.129639 | +0.266425 | +0.001176 |
| RAF | 52 | +0.844346 | +0.040643 | +0.003510 |
| RAF | 62 | +0.171058 | +0.170868 | +0.018291 |
| Intermittent | 42 | +0.414992 | +0.216144 | +0.003651 |
| Intermittent | 52 | -0.013931 | -0.029904 | +0.005060 |
| Intermittent | 62 | +0.493355 | +0.110040 | -0.005627 |

Intermittent seed62는 raw RMSE가 나빠져도 log MSE는 더 낮아 두 목적의 충돌이 실제로 관측되는 예외입니다. Taxi seed62는 반대로 raw RMSE·MAE가 낮아도 log MSE가 높습니다. RAF seed42는 raw RMSE만 낮고 MAE·log MSE는 높습니다. Intermittent seed52는 CNN+GRU의 raw RMSE·MAE가 조금 낮지만 log MSE는 높습니다. 이 예외를 평균 결과로 덮지 않습니다.

### CNN+GRU 대 CNN의 반전 구간

- Taxi: Test q>3449 대상83개에서 CNN+GRU 평균 편향은−925.568, CNN은+68.109였습니다. 이 구간의 평균 SSE 차이를 전체 Test 대상8327개로 나눈 MSE 기여5929.334는 전체 평균 MSE 차이7054.446의84.05%입니다. RMSE 차이의84.05%라는 뜻은 아닙니다. 목표 수량이 직전 수량보다 큰 조건의 MSE 기여는+11445.667이고 직전보다 작은 조건은−4393.689로 상쇄됩니다. 따라서 내부 비교의 손해는 주로 수량 상승에 대한 과소 예측 양상으로 좁혀집니다. q>3449의 목표 평균은Validation4296.253→Test4309.988로 비슷하지만 관측 이력 평균은2705.333→2272.895로 낮아졌습니다. 이력 수준과 목표 수량의 상대 조건 변화는 확인됐으나 GRU 자체가 그 원인이라는 증명은 아닙니다.

- Intermittent: 목표가 관측 이력 평균보다 작은 행은 Train26.93%→Validation42.29%→Test65.26%로 늘었습니다. Test에서 이 조건의 평균 MSE 기여+1.736112는 전체 격차+1.682523보다 커 다른 조건의 이득이 일부 상쇄합니다. 직전 수량보다 작은 조건도Test대상의36.16%이며 그 기여는+1.862831입니다. Test46<q≤187 대상1923개의 CNN+GRU/CNN 편향은+6.770/+3.705이고 이 구간의 MSE 기여+0.971132는 전체 MSE 격차의57.72%입니다. 이 데이터셋의 주요 차이는 극단적 큰 수량만의 문제가 아니라 수량 하락 조건에서의 중간 수량 과대 예측입니다. seed52는 다른 구간의 이득이 이를 상쇄해 전체 raw 오류가 조금 좋습니다.

- RAF: q>200 Test대상39개의 MSE 기여+18.407326는 전체 MSE 차이+23.625727의77.91%입니다. Validation→Test에서 같은2428개 개체만 남겨도 평균 MSE 차이가−24.445100→+22.889615로 반전됩니다. 전체 격차 변화+40.681922 중 공통 개체 내부 오류 항은+52.481220이고 개체 구성·가중치 항은 합쳐−11.799299로 악화를 일부 상쇄합니다. 따라서 다른 품목이 들어와서만 발생한 반전은 아닙니다. seed52의 큰 수량 손해가 크고 seed42는 전체 RMSE 우세를 유지하므로 모든 seed의 같은 실패로 표현하지 않습니다.

길이129 이상 이력에 Taxi 손해가 집중됐지만 이 대상은 raw 수량도 큽니다. Intermittent는65..128 이력 구간의 MSE 기여가+1.692263이고129 이상은−0.044005로 일부 상쇄됩니다. RAF의 실제 모든 대상 이력은64 이하입니다. 이 차이는 길어진 이력이나 GRU의 독립적인 효과를 증명하지 않습니다.

### CNN 대 외부 비교군은 별도로 해석

- Taxi: CNN의 Train/Validation raw RMSE는 S2P2·AttNHP보다 낮았지만 Test에서 각각+19.705345/+3.927050 높아졌습니다. S2P2 대비 Test MSE 차이+5586.372884에서 q>1562 두 구간의 기여는+5024.780623(89.95%)입니다. 이는 Test 격차의 분모이며, 이전 Validation→Test 격차 변화에서의89.86%와 다릅니다. 목표가 직전 수량보다 작은 조건의 기여는+5853.942296이며, CNN/S2P2 편향은+35.476/+15.092입니다. AttNHP 대비 같은 조건 기여도+5478.899509입니다. 즉 외부 비교의 주요 양상은 하락 조건의 CNN 과대 예측이며, 내부 비교의 상승 조건 CNN+GRU 과소 예측과 하나의 원인으로 합치면 안 됩니다. CNN이 나머지 네 외부 모델보다 평균적으로 좋은 Test 결과도 보존합니다.

- RAF: CNN−S2P2 전체 평균 MSE 차이는 Validation+29.874965→Test−13.690730입니다. 그러나 양쪽에 등장하는2428개 개체만 보면 이미Validation−3.013596→Test−14.942190로 CNN이 좋았습니다. 전체 변화−43.565695의 공통 개체 내부 항은−1.335674이고 개체 구성·가중치 항 합계는−42.230021입니다. 전체 순위 개선의 상당 부분은 구성 차이와 연결되므로 동일한 품목 전반에서 일반화가 좋아졌다고 확대하지 않습니다. Test MAE는 RMTPP·NHP가 CNN보다 낮은 예외도 남습니다.

- Intermittent: CNN은 Train·Validation·Test 평균 raw RMSE·MAE에서 외부 여섯 모델보다 낮았습니다. S2P2와의 Test MSE 차이−3.540460 중 직전보다 작은 수량 조건 기여는−3.880494이며 다른 조건 일부가 상쇄합니다. 내부 CNN+GRU 비교의 반전을 외부 경쟁력 상실로 해석할 근거는 없습니다.

같은 개체 내부의 오류 변화에는 서로 다른 시점·목표 수량·관측 이력이 함께 포함됩니다. 위의 구간과 날짜·개체 분석은 실패 조건을 좁히는 관측 근거이며 과적합, GRU, 모델 용량, 손실 함수 중 하나의 인과효과를 분리한 결과는 아닙니다. 목표 대 last/mean 조건은 실제 목표를 보고 나눈 사후 구간이므로 예측 시점의 모델 선택 규칙으로 사용할 수 없습니다.

### 실행과 검증 증적

- 실행 조건162개는 현재 native 추론148개와 SHA승인된 Validation 캐시14개를 합친 고유 조건입니다. 기존 외부 Test54개는 재추론하지 않았습니다.
- 원본216개 모델/분할 결과와189개 쌍 비교,63개 개체 분해를 분석했습니다. 모든 참조 파일2687개의 SHA를 분석 후 다시 확인했습니다.
- 분석 전용 집중 테스트14개가 실제 종료0으로 통과했습니다. 명령은 `PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m unittest paper.tests.test_analyze_titantpp_frozen_rows -v`입니다.
- Python의 resolved 경로가 worker receipt에 직접 저장되지 않은 경우 실제 worker argv와 봉인된 평가기의 전후 resolve 검사 증적으로 연결했습니다. 없는 경로 필드를 직접 관측한 값으로 만들지 않았습니다.

## 고정 모델의 Train·Validation·Test raw 오류

Train은 각 개체의 첫 행을 제외한 모든 예측 가능 next-event 대상입니다. 학습 중 온라인 log 손실과 고정 모델의 Train raw 오류를 혼용하지 않습니다.

| 데이터셋 | 모델 | Train RMSE | Validation RMSE | Test RMSE | Train MAE | Validation MAE | Test MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| Taxi | CNN+GRU | 79.75527±3.95733 | 81.92273±5.02739 | 158.70613±35.29201 | 24.48692±1.19023 | 25.81563±1.13509 | 46.88095±10.65779 |
| Taxi | CNN | 85.13535±1.26263 | 86.65005±2.09874 | 135.24686±31.74494 | 26.35665±0.17379 | 27.79279±0.49999 | 40.61529±7.19675 |
| Taxi | RMTPP | 90.59281±4.17829 | 94.72835±3.40060 | 246.64728±82.50643 | 26.99642±0.85279 | 28.51236±0.63443 | 65.27698±16.10348 |
| Taxi | THP | 114.70804±2.59195 | 107.59766±4.50405 | 202.17400±36.21825 | 35.18120±0.54513 | 32.40951±0.81440 | 58.34864±11.07589 |
| Taxi | NHP | 327.85835±3.60041 | 318.81478±5.30111 | 381.07624±26.23487 | 102.10198±1.85988 | 94.26935±1.87518 | 116.33238±9.71280 |
| Taxi | SAHP | 113.77068±10.19509 | 120.52734±1.99963 | 148.24948±30.21568 | 31.89091±2.98861 | 34.22603±0.45395 | 40.90079±4.92398 |
| Taxi | S2P2 | 104.20576±3.50445 | 106.35405±1.91353 | 115.54151±6.40227 | 33.34114±0.47090 | 33.02315±0.55687 | 36.83585±2.38147 |
| Taxi | AttNHP | 126.79724±10.23122 | 127.12903±8.99192 | 131.31981±2.42794 | 36.72907±2.41847 | 35.98109±1.60877 | 38.75277±1.05645 |
| RAF | CNN+GRU | 45.94347±0.27522 | 34.18235±0.23401 | 39.89592±0.48113 | 10.54076±0.04046 | 9.18115±0.03900 | 10.13145±0.14675 |
| RAF | CNN | 45.95228±0.07047 | 34.43141±0.07320 | 39.60066±0.02365 | 10.61999±0.04602 | 9.21827±0.02800 | 9.97214±0.03643 |
| RAF | RMTPP | 45.77571±0.36302 | 34.55033±0.25063 | 40.46317±0.24158 | 10.57617±0.08114 | 9.13620±0.05568 | 9.90635±0.03702 |
| RAF | THP | 45.58688±0.24830 | 34.74452±0.32041 | 40.55509±0.29526 | 10.64070±0.11987 | 9.30136±0.18190 | 10.12295±0.23048 |
| RAF | NHP | 47.61972±0.42195 | 36.27308±0.31864 | 42.17633±0.30048 | 10.70437±0.11517 | 9.17925±0.08816 | 9.96915±0.09318 |
| RAF | SAHP | 46.19033±0.20993 | 35.41927±0.21305 | 41.04934±0.15611 | 10.45590±0.15238 | 9.23317±0.02088 | 10.26164±0.10324 |
| RAF | S2P2 | 45.01513±0.17289 | 33.99470±0.12612 | 39.77294±0.16091 | 10.70502±0.10401 | 9.27299±0.05168 | 10.09581±0.14549 |
| RAF | AttNHP | 45.26794±0.27412 | 34.87664±0.06757 | 40.20362±0.20440 | 10.48694±0.05418 | 9.19166±0.04340 | 9.98393±0.02844 |
| Intermittent | CNN+GRU | 1.43777±0.05609 | 1.70054±0.05055 | 2.91426±0.23805 | 0.62381±0.01324 | 0.71102±0.02025 | 1.16046±0.09389 |
| Intermittent | CNN | 1.53929±0.07746 | 1.72297±0.07300 | 2.61613±0.07821 | 0.66543±0.04555 | 0.73564±0.06111 | 1.06170±0.03643 |
| Intermittent | RMTPP | 2.31320±0.31469 | 2.54993±0.16040 | 4.21147±0.21262 | 0.80527±0.08086 | 0.89488±0.04940 | 1.69232±0.18077 |
| Intermittent | THP | 2.53237±0.50361 | 2.75335±0.17843 | 3.48394±0.12320 | 0.87992±0.17629 | 0.89933±0.06610 | 1.31228±0.02705 |
| Intermittent | NHP | 11.89559±0.47345 | 12.87546±1.04050 | 5.90646±0.85074 | 3.96134±0.06548 | 4.51877±0.17059 | 2.32775±0.15934 |
| Intermittent | SAHP | 4.14511±0.81205 | 7.02983±0.76727 | 12.25244±1.51818 | 1.22954±0.15131 | 1.86195±0.08993 | 3.67698±0.68916 |
| Intermittent | S2P2 | 2.40108±0.38468 | 2.57584±0.07656 | 3.21535±0.27425 | 0.88045±0.02590 | 0.93589±0.00857 | 1.31969±0.02635 |
| Intermittent | AttNHP | 2.59768±0.25144 | 3.07994±0.29817 | 12.28910±2.68907 | 0.91741±0.07838 | 1.04126±0.09376 | 4.34928±0.70113 |

## 비교를 분리한 순위 변화

차이는 왼쪽 모델−오른쪽 모델입니다. 양수면 오른쪽 모델의 오류가 작습니다. MSE 차이는 정확히 합산할 수 있고 RMSE 차이는 합산하지 않습니다.

| 데이터셋 | 비교 | Train RMSE 차이 | Validation RMSE 차이 | Test RMSE 차이 | Train log MSE 차이 | Validation log MSE 차이 |
|---|---|---:|---:|---:|---:|---:|
| Taxi | CNN+GRU−CNN | -5.38008 | -4.72732 | +23.45927 | -0.015533 | -0.033152 |
| Taxi | CNN−RMTPP | -5.45745 | -8.07830 | -111.40042 | +0.000911 | +0.047442 |
| Taxi | CNN−THP | -29.57268 | -20.94761 | -66.92714 | -0.034554 | +0.051737 |
| Taxi | CNN−NHP | -242.72300 | -232.16474 | -245.82939 | -0.124519 | +0.002484 |
| Taxi | CNN−SAHP | -28.63533 | -33.87729 | -13.00262 | +0.000814 | -0.032552 |
| Taxi | CNN−S2P2 | -19.07041 | -19.70400 | +19.70535 | -0.054106 | +0.065872 |
| Taxi | CNN−AttNHP | -41.66188 | -40.47899 | +3.92705 | -0.023534 | +0.006015 |
| RAF | CNN+GRU−CNN | -0.00881 | -0.24906 | +0.29526 | -0.015063 | -0.003291 |
| RAF | CNN−RMTPP | +0.17657 | -0.11892 | -0.86250 | +0.004793 | +0.005684 |
| RAF | CNN−THP | +0.36540 | -0.31311 | -0.95443 | -0.005902 | -0.010892 |
| RAF | CNN−NHP | -1.66744 | -1.84167 | -2.57567 | +0.004254 | +0.018201 |
| RAF | CNN−SAHP | -0.23805 | -0.98786 | -1.44868 | +0.015376 | -0.011341 |
| RAF | CNN−S2P2 | +0.93715 | +0.43670 | -0.17227 | -0.003378 | -0.002646 |
| RAF | CNN−AttNHP | +0.68434 | -0.44524 | -0.60296 | +0.011510 | +0.003933 |
| Intermittent | CNN+GRU−CNN | -0.10152 | -0.02243 | +0.29814 | -0.001311 | -0.002781 |
| Intermittent | CNN−RMTPP | -0.77392 | -0.82696 | -1.59535 | +0.001270 | +0.002967 |
| Intermittent | CNN−THP | -0.99308 | -1.03039 | -0.86782 | -0.000661 | +0.001770 |
| Intermittent | CNN−NHP | -10.35630 | -11.15249 | -3.29033 | -0.027344 | -0.026400 |
| Intermittent | CNN−SAHP | -2.60583 | -5.30687 | -9.63632 | -0.001121 | -0.003570 |
| Intermittent | CNN−S2P2 | -0.86180 | -0.85288 | -0.59923 | -0.000845 | +0.001289 |
| Intermittent | CNN−AttNHP | -1.05839 | -1.35697 | -9.67297 | -0.000789 | +0.001113 |

## 같은 개체의 오류 변화와 개체 구성 변화

Validation→Test MSE 차이 변화를 네 항으로 합산했습니다. 한쪽에만 있는 개체의 구성, 공통 개체의 대상 수 가중치, 공통 개체 내부 오류 변화가 모두 포함됩니다. 이 수학적 분해는 인과효과를 식별하지 않습니다.

| 데이터셋 | 비교 | 전체 MSE 격차 변화 | Test 개체구성 항 | Validation 개체구성 항 | 공통 개체 가중치 항 | 공통 개체 내 오류 항 |
|---|---|---:|---:|---:|---:|---:|
| Intermittent | CNN+GRU−CNN | +1.761160 | +0.000000 | +0.000000 | -0.020743 | +1.781903 |
| Intermittent | CNN−AttNHP | -142.421348 | +0.000000 | +0.000000 | +1.316888 | -143.738236 |
| Intermittent | CNN−NHP | +135.006392 | +0.000000 | +0.000000 | +1.598084 | +133.408308 |
| Intermittent | CNN−RMTPP | -7.371316 | +0.000000 | +0.000000 | +0.122911 | -7.494227 |
| Intermittent | CNN−S2P2 | +0.126248 | +0.000000 | +0.000000 | +0.058054 | +0.068193 |
| Intermittent | CNN−SAHP | -97.971853 | +0.000000 | +0.000000 | +1.687548 | -99.659401 |
| Intermittent | CNN−THP | -0.669791 | +0.000000 | +0.000000 | +0.089890 | -0.759681 |
| RAF | CNN+GRU−CNN | +40.681922 | +0.736112 | -7.388906 | -5.146505 | +52.481220 |
| RAF | CNN−AttNHP | -17.288479 | -13.745280 | +13.897897 | -13.941980 | -3.499115 |
| RAF | CNN−NHP | -80.411641 | +15.476326 | +34.753588 | -27.141013 | -103.500541 |
| RAF | CNN−RMTPP | -60.852235 | +2.084052 | +5.705119 | -5.335841 | -63.305565 |
| RAF | CNN−S2P2 | -43.565695 | +1.251460 | -32.888561 | -10.592919 | -1.335674 |
| RAF | CNN−SAHP | -47.822538 | +9.874811 | +1.577128 | -44.220381 | -15.054096 |
| RAF | CNN−THP | -54.836148 | +8.298031 | -3.664136 | -15.802024 | -43.668020 |
| Taxi | CNN+GRU−CNN | +7837.429922 | +0.000000 | +0.000000 | +8.807665 | +7828.622257 |
| Taxi | CNN−AttNHP | +10419.245168 | +0.000000 | +0.000000 | -3.927719 | +10423.172887 |
| Taxi | CNN−NHP | -32563.975639 | +0.000000 | +0.000000 | -29.458410 | -32534.517229 |
| Taxi | CNN−RMTPP | -44939.544931 | +0.000000 | +0.000000 | -21.862114 | -44917.682817 |
| Taxi | CNN−S2P2 | +9388.829728 | +0.000000 | +0.000000 | -2.265422 | +9391.095150 |
| Taxi | CNN−SAHP | +3395.310303 | +0.000000 | +0.000000 | -13.500164 | +3408.810467 |
| Taxi | CNN−THP | -18705.681495 | +0.000000 | +0.000000 | -7.631805 | -18698.049690 |

## 구간별 행 진단

ROW_ANALYSIS.json의 pair_splits에는 각 seed·분할의 고정 수량5구간, history64/128, 실제 recorded gap, 월, target 대 last/mean 수량 방향, 수량과 history/방향의 교차 구간이 있습니다. 모든 구간은 대상 수·SSE·절대오차·signed bias를 보존합니다.
개체별 집계는 모든 개체로 계산했습니다. 보고서의 상위10개 목록은 사후 오류 기여 설명용이며 새 하위집단 결론이나 선택 규칙으로 사용하지 않습니다.

## 해석의 한계

- Fixed-model Train errors are in-sample diagnostic errors, not an independent generalization estimate.
- No new checkpoint selection, thresholds, model fitting, correction or Test-driven architecture choice is performed.
- Quantity/target-to-history strata use the observed target and cannot be inference-time model-selection rules.
- Raw Train/Validation/Test populations differ in quantity severity, date and eligible target composition; raw gaps alone do not prove overfitting.
- A lower fixed-model log loss and higher raw RMSE reveals an objective/metric tradeoff but does not prove that changing loss would repair it.
- CNN+GRU and CNN have different active parameter counts; this experiment cannot separate GRU design from model capacity or optimization effects.
- External comparisons share frozen targets and quantity heads but differ in historical source/campaign/runtime; this is not a fresh same-runtime architecture experiment.
- Intermittent seeds42/52 use native5090/cuDNN92000 and seed62 native5080/cuDNN91900; paired component arms match within seed but cross-seed runtimes differ.
- Taxi and Intermittent chronological splits are entity-relative and have overlapping global calendar ranges, so common-calendar future deployment is not established.
- This Test has prior access history; exploratory diagnoses and top-entity ranks are not independent confirmatory evidence.
- Arithmetic membership/weight/within-entity attribution includes different targets and histories inside each entity and is not a causal decomposition.
- Existing external Test rows do not contain original log-space model losses; those fields remain N/A rather than being replaced by a loss derived from raw predictions.

## 통계적 오류 점검

11/11 항목을 검토했습니다. 표본은 seed3개이고 개별 사건은 독립 실험 반복으로 취급하지 않습니다.

- Simpson's paradox: CHECKED — Micro, equal-entity and exact shared-entity decomposition are separately reported.
- Ecological fallacy: CHECKED — Per-example paired errors do not identify personal/entity causal effects.
- Non-collapsibility: NOT_APPLICABLE — No odds ratios or nonlinear regression coefficients are interpreted.
- Confounding: CAUTION — Component active capacity differs; external campaign/source/runtime differs. Histories and regimes are descriptive controls, not causal adjustment.
- Base rate neglect: CHECKED — Every group reports n and population contribution; conditional error is not substituted for the global contribution.
- Regression to the mean: CAUTION — Worst-entity ranks are posthoc descriptions with no improvement or independent replication claim.
- Survivorship bias: CHECKED — Only complete, approved 72 fixed fits with exact populations enter; all 216 model/split rows are required.
- Look-elsewhere effect: CAUTION — All seven pairs and declared partitions are retained; no formal significance claim or favorable slice-only conclusion.
- Garden of forking paths: CAUTION — Exploratory diagnostics after Test exposure; no new thresholds, checkpoint selection, fitting or exclusions.
- Correlation versus causation: CAUTION — Arithmetic attribution and observed associations do not prove loss/GRU/distribution causal mechanisms.
- Reverse causality: CHECKED — History fields precede targets; target-relative strata are outcome-conditioned diagnostics, not predictors available before the target.
