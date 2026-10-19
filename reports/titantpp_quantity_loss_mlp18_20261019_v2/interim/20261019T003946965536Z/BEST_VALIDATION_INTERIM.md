# 손실 비교의 저장 best Validation — 중간 결과

관측: 2026-10-19 09:39:47 KST. Cap·MLP16·3seed·시간 가중치0. 시간·수량 입력은 모두 log1p인 현재 실험이다.
신규18개 학습 종료9·진행6·대기3·실패0·미확정0. 종료 후 native Validation 재현은0/18이다.
기존 log 손실9개는 완료·회수·선택·SHA를 확인한 저장 결과다. 신규 두 손실은 모두 중간 비교로 구분한다.

| 데이터 | 손실 | RMSE 평균±표본SD | MAE 평균±표본SD | 상태 |
|---|---|---:|---:|---|
|Taxi|log_mse|5.995211±0.018141|3.114605±0.139224|저장 Validation 완료|
|Taxi|raw_mse|5.999896±0.096508|3.072916±0.095521|학습종료2/3, best보유3/3; 재현대기|
|Taxi|raw_mae|5.981532±0.075574|2.912077±0.085317|학습종료1/3, best보유3/3; 재현대기|
|RAF|log_mse|2.607249±0.008908|1.854419±0.017944|저장 Validation 완료|
|RAF|raw_mse|2.579694±0.002455|1.904680±0.031603|학습종료3/3, best보유3/3; 재현대기|
|RAF|raw_mae|2.691066±0.019286|1.704316±0.027036|학습종료3/3, best보유3/3; 재현대기|
|Intermittent|log_mse|0.326140±0.016926|0.171692±0.013809|저장 Validation 완료|
|Intermittent|raw_mse|3seed 미구성|3seed 미구성|학습종료0/3, best보유2/3; 재현대기|
|Intermittent|raw_mae|3seed 미구성|3seed 미구성|학습종료0/3, best보유1/3; 재현대기|

| 데이터 | seed | 손실 | 학습상태 | 저장epoch | bestepoch | RMSE | MAE |
|---|---:|---|---|---:|---:|---:|---:|
|Taxi|42|log_mse|stored_selected_validation_complete|None|5|5.993084|2.995577|
|Taxi|52|log_mse|stored_selected_validation_complete|None|5|5.978229|3.267701|
|Taxi|62|log_mse|stored_selected_validation_complete|None|3|6.014322|3.080537|
|Intermittent|42|log_mse|stored_selected_validation_complete|None|24|0.314770|0.159482|
|RAF|42|log_mse|stored_selected_validation_complete|None|24|2.617028|1.834036|
|RAF|52|log_mse|stored_selected_validation_complete|None|3|2.599595|1.867833|
|RAF|62|log_mse|stored_selected_validation_complete|None|12|2.605126|1.861388|
|Intermittent|52|log_mse|stored_selected_validation_complete|None|21|0.318060|0.168917|
|Intermittent|62|log_mse|stored_selected_validation_complete|None|17|0.345592|0.186678|
|Taxi|42|raw_mse|train_exited0|46|6|5.972663|3.022493|
|Taxi|42|raw_mae|train_exited0|56|16|5.920069|2.870609|
|Taxi|52|raw_mse|train_exited0|44|4|5.919931|3.183083|
|Taxi|52|raw_mae|train_active_verified|43|12|6.065912|3.010201|
|Taxi|62|raw_mse|train_active_verified|41|4|6.107094|3.013171|
|Taxi|62|raw_mae|train_active_verified|31|11|5.958613|2.855420|
|Intermittent|42|raw_mse|waiting_train|None|None|미생성|미생성|
|Intermittent|42|raw_mae|waiting_train|None|None|미생성|미생성|
|RAF|42|raw_mse|train_exited0|60|20|2.582411|1.934365|
|RAF|42|raw_mae|train_exited0|51|11|2.670737|1.694931|
|RAF|52|raw_mse|train_exited0|56|16|2.579034|1.871456|
|RAF|52|raw_mae|train_exited0|57|17|2.709105|1.683222|
|RAF|62|raw_mse|train_exited0|43|3|2.577636|1.908217|
|RAF|62|raw_mae|train_exited0|45|5|2.693354|1.734794|
|Intermittent|52|raw_mse|train_active_verified|10|10|0.336472|0.189115|
|Intermittent|52|raw_mae|train_active_verified|10|8|0.374637|0.189130|
|Intermittent|62|raw_mse|train_active_verified|9|7|0.346726|0.176557|
|Intermittent|62|raw_mae|waiting_train|None|None|미생성|미생성|

RMSE의 최초 최소epoch를 선택하고 그 epoch의 MAE를 함께 비교했다. MAE 최소epoch를 따로 골라 섞지 않았다.
기존 log 대비 모든 seed의 절대·상대차이·악화는 JSON과 CSV에 보존한다.
Taxi 평균은 rawMAE가 현재 두 지표 모두 낮으나 seed52 RMSE는 악화했다. rawMSE는 평균 RMSE가 소폭 악화하고 MAE만 개선했다.
RAF는 rawMSE의 RMSE 개선·MAE 악화, rawMAE의 RMSE 악화·MAE 개선이 각각3seed 모두 나타났다.
Intermittent rawMSE는52·62, rawMAE는52만 best를 보유하므로3seed 평균을 만들지 않았다. 모두 초기 진행 중이다.
현 실험의 학습·회수·후속 대기 작업은 변경하지 않았고 Test는 읽거나 실행하지 않았다.
