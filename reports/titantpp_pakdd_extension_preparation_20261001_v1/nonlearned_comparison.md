# 비학습 기준선과 기존 TitanTPP 비교

같은 frozen validation target/window를 사용한다. 비학습 기준은seed없는한번의결과이고 TitanTPP는기존3seed의평균이다. 시간분포를제공하지않는기준에시간NLL을임의부여하지않았다.

| 데이터 | 모델 | MAE | RMSE |
|---|---|---:|---:|
| yellow_trip_hourly | TitanTPP MLP (3seed mean) | 25.623661 | 79.711120 |
| yellow_trip_hourly | last_observed_quantity | 41.195090 | 134.393100 |
| yellow_trip_hourly | mean_quantity_in_same_observed_window | 120.377075 | 368.784314 |
| intermittent_frozen_5000 | TitanTPP MLP (3seed mean) | 0.700523 | 1.673034 |
| intermittent_frozen_5000 | last_observed_quantity | 0.564733 | 1.707803 |
| intermittent_frozen_5000 | mean_quantity_in_same_observed_window | 4.553695 | 13.782119 |
| insta_market_basket | TitanTPP MLP (3seed mean) | 3.991574 | 5.882742 |
| insta_market_basket | last_observed_quantity | 4.967427 | 7.347325 |
| insta_market_basket | mean_quantity_in_same_observed_window | 4.050744 | 5.903032 |
| raf_spare_parts | TitanTPP MLP (3seed mean) | 9.250602 | 33.950679 |
| raf_spare_parts | last_observed_quantity | 11.313602 | 44.574172 |
| raf_spare_parts | mean_quantity_in_same_observed_window | 10.856359 | 36.849906 |

Intermittent의 마지막 수량 기준은MAE에서기존MLP평균보다낮다. RMSE에서는MLP가약2%낮다. 따라서 기존외부TPP대비큰수량개선과단순기준을포함한비교를구분해야한다. 불리한기준을제외하지않고후속Deep Renewal 결과와함께성능주장범위를검토한다.

세부 source SHA 및계산은nonlearned_comparison.json을따른다.
