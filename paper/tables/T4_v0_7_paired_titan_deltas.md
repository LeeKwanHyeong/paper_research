# T4 v0.7. Paired TitanTPP comparisons

> Candidate minus reference. A negative delta and a larger better-seed count favor TitanTPP.

| Dataset | Reference | MAE delta | MAE better seeds | RMSE delta | RMSE better seeds | Clamped-time delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Intermittent-5000 | Count-aware RMTPP | -2.155606 | 3/3 | -8.659224 | 3/3 | 0.006417 |
| Intermittent-5000 | Count-aware THP | 0.080538 | 0/3 | -0.231219 | 3/3 | 0.006407 |
| Taxi | Count-aware RMTPP | 2.291208 | 1/3 | -0.821700 | 2/3 | 0.006034 |
| Taxi | Count-aware THP | 0.992132 | 2/3 | -3.503206 | 2/3 | 0.003401 |
| Instacart | Count-aware RMTPP | 0.018033 | 0/3 | 0.025818 | 1/3 | 0.001227 |
| Instacart | Count-aware THP | 0.008236 | 1/3 | 0.014733 | 1/3 | 0.000302 |
