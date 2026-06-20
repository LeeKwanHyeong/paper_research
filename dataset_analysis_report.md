# Dataset Analysis Report

Generated: 2026-06-09 23:37:41

## Executive Summary

이 문서는 `simple_lab_test` 실험에서 사용하는 데이터 파일의 구조와 sequence 특성을 정리한 모델 설계용 데이터 프로파일입니다. 특히 `max_seq_len`, recurrent hidden size, Titan candidate depth, batch size를 정할 때 참고할 수 있도록 series length, `delta_t`, `demand_qty`, mark/log-scale 분포를 함께 제공합니다.

- `head_office/marked_target_df.parquet`, `insta_market_basket/instacart_marked_target_df.parquet`, `new_york_taxi/yellow_trip_hourly.parquet`은 학습에 직접 들어가는 event table입니다.
- `new_york_taxi/yellow_trip.parquet`은 raw taxi log이며, `simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb`를 통해 `yellow_trip_hourly.parquet`으로 변환한 뒤 학습에 사용합니다.
- 모델 capacity는 row 수보다 series별 event 길이와 mark/quantity tail에 더 민감합니다. 아래 sequence length 분포를 먼저 보고 `max_seq_len`과 hidden dimension을 잡는 것이 안전합니다.

## Dataset Inventory

| dataset | CLI/train name | path | file size | rows | series | sequence length | min time | max time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Head Office Marked Target | intermittent | sample_data/head_office/marked_target_df.parquet | 1.06 MB | 242,888 | 23,387 | p50=6, p95=35, max=110 | 201712 | 202702 |
| Instacart Market Basket | insta_market_basket | sample_data/insta_market_basket/instacart_marked_target_df.parquet | 18.42 MB | 3,279,521 | 206,209 | p50=10, p95=50, max=100 | 0 | 365 |
| Yellow Trip Raw | source only | sample_data/new_york_taxi/yellow_trip.parquet | 294.33 MB | 12,748,986 | raw trips | NA | 2015-01-01 00:00:00 | 2015-01-31 23:59:59 |
| Yellow Trip Hourly | yellow_trip_hourly | sample_data/new_york_taxi/yellow_trip_hourly.parquet | 0.16 MB | 55,119 | 131 | p50=405, p95=743, max=744 | 20150101000000 | 20150131230000 |

## Head Office Marked Target

- Role: 본사 간헐 수요 episode-level event table
- Path: `sample_data/head_office/marked_target_df.parquet`
- Training dataset name: `intermittent`
- File size: 1.06 MB
- Rows: 242,888
- Series count: 23,387
- demand_dt range: 201712 to 202702

### Schema

| column | dtype |
| --- | --- |
| oper_part_no | String |
| demand_dt | Int64 |
| seq | UInt32 |
| delta_t | Int32 |
| demand_qty | Float64 |
| z | Float64 |
| mark | Int32 |

### Sequence Length Summary

| metric | mean | std | min | p50 | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| events per series | 10.39 | 13.59 | 2 | 6 | 24 | 35 | 73 | 110 |
| seq span per series | 292.04 | 120.90 | 10 | 314 | 426 | 467 | 471 | 512 |

Top 10 longest/most active series:

| series | events | seq span | total qty |
| --- | --- | --- | --- |
| 04810-00220 | 110 | 472 | 41,795.00 |
| E6301-13151 | 110 | 472 | 25,300.00 |
| E6820-60011 | 110 | 472 | 21,026.00 |
| T4145-82052 | 110 | 472 | 13,893.00 |
| T4835-82551 | 110 | 472 | 11,660.00 |
| U3215-50323 | 110 | 472 | 11,089.00 |
| T4835-82501 | 110 | 472 | 9,437.00 |
| 04810-00800 | 110 | 472 | 7,599.00 |
| T5710-69941 | 110 | 472 | 6,494.00 |
| T4115-63603 | 110 | 472 | 5,940.00 |

### Quantity and Time Distribution

| metric | count | sum | mean | std | min | p50 | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demand_qty | 242,888 | 1,643,589.00 | 6.77 | 54.42 | 1.00 | 2.00 | 10.00 | 16.00 | 63.00 | 5,000.00 |
| delta_t | 242,888 | 6,806,567.00 | 28.02 | 43.23 | 0.00 | 3.00 | 97.00 | 152.00 | 157.00 | 313.00 |
| series_total_qty | 23,387 | 1,643,589.00 | 70.28 | 827.44 | 2.00 | 13.00 | 79.00 | 140.00 | 548.00 | 58,500.00 |

### Existing Mark Distribution

파일에 저장된 `mark` 컬럼 기준 분포입니다. 학습 파이프라인은 선택한 `scale_base`에 따라 `demand_qty`에서 mark/residual을 다시 만들 수 있습니다.

| mark | count | share |
| --- | --- | --- |
| 0 | 98,888 | 0.4071 |
| 1 | 51,776 | 0.2132 |
| 2 | 19,837 | 0.0817 |
| 3 | 34,028 | 0.1401 |
| 4 | 35,944 | 0.1480 |
| 6 | 2,172 | 0.0089 |
| 7 | 243 | 0.0010 |

### Log-Base Order Distribution from demand_qty

| base | raw order | count | share | cum share |
| --- | --- | --- | --- | --- |
| log10 | 0 | 215,127 | 0.8857 | 0.8857 |
| log10 | 1 | 26,117 | 0.1075 | 0.9932 |
| log10 | 2 | 1,540 | 0.0063 | 0.9996 |
| log10 | 3 | 104 | 0.0004 | 1.0000 |
| log4 | 0 | 170,501 | 0.7020 | 0.7020 |
| log4 | 1 | 60,163 | 0.2477 | 0.9497 |
| log4 | 2 | 9,809 | 0.0404 | 0.9901 |
| log4 | 3 | 1,813 | 0.0075 | 0.9975 |
| log4 | 4 | 500 | 0.0021 | 0.9996 |
| log4 | 5 | 100 | 0.0004 | 1.0000 |
| log4 | 6 | 2 | 0.0000 | 1.0000 |
| log2 | 0 | 98,888 | 0.4071 | 0.4071 |
| log2 | 1 | 71,613 | 0.2948 | 0.7020 |
| log2 | 2 | 41,133 | 0.1693 | 0.8713 |
| log2 | 3 | 19,030 | 0.0783 | 0.9497 |
| log2 | 4 | 7,001 | 0.0288 | 0.9785 |
| log2 | 5 | 2,808 | 0.0116 | 0.9901 |
| log2 | 6 | 1,215 | 0.0050 | 0.9951 |
| log2 | 7 | 598 | 0.0025 | 0.9975 |
| log2 | 8 | 323 | 0.0013 | 0.9989 |
| log2 | 9 | 177 | 0.0007 | 0.9996 |
| log2 | 10 | 86 | 0.0004 | 0.9999 |
| log2 | 11 | 14 | 0.0001 | 1.0000 |
| log2 | 12 | 2 | 0.0000 | 1.0000 |

### Model Configuration Implications

- Sequence length reference: p95=35, p99=73, max=110. `max_seq_len` should cover at least p95 for stable validation, while p99/max coverage is useful for stress tests.
- Quantity tail reference: demand_qty p99=63.00, max=5,000.00. Large gaps between p99 and max indicate that scale-wise MAE should be reported in addition to global MAE.
- Capacity hint: medium-length histories can justify hidden sizes around 64-128 and Titan candidates with moderate depth.

## Instacart Market Basket

- Role: Instacart basket-count demand event table
- Path: `sample_data/insta_market_basket/instacart_marked_target_df.parquet`
- Training dataset name: `insta_market_basket`
- File size: 18.42 MB
- Rows: 3,279,521
- Series count: 206,209
- demand_dt range: 0 to 365

### Schema

| column | dtype |
| --- | --- |
| oper_part_no | String |
| demand_dt | Int32 |
| seq | Int32 |
| delta_t | Int16 |
| demand_qty | Float64 |
| z | Float64 |
| mark | Int16 |

### Sequence Length Summary

| metric | mean | std | min | p50 | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| events per series | 15.90 | 16.04 | 1 | 10 | 37 | 50 | 83 | 100 |
| seq span per series | 168.07 | 102.13 | 1 | 146 | 333 | 355 | 365 | 366 |

Top 10 longest/most active series:

| series | events | seq span | total qty |
| --- | --- | --- | --- |
| user_175294 | 100 | 330 | 2,307.00 |
| user_178736 | 100 | 323 | 2,121.00 |
| user_7744 | 100 | 359 | 1,641.00 |
| user_41356 | 100 | 332 | 1,571.00 |
| user_118221 | 100 | 343 | 1,546.00 |
| user_66500 | 100 | 327 | 1,389.00 |
| user_75664 | 100 | 269 | 1,363.00 |
| user_146147 | 100 | 229 | 1,329.00 |
| user_138113 | 100 | 305 | 1,295.00 |
| user_173431 | 100 | 299 | 1,294.00 |

### Quantity and Time Distribution

| metric | count | sum | mean | std | min | p50 | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demand_qty | 3,279,521 | 33,819,106.00 | 10.31 | 7.72 | 1.00 | 8.00 | 20.00 | 25.00 | 36.00 | 177.00 |
| delta_t | 3,279,521 | 34,451,600.00 | 10.51 | 9.20 | 0.00 | 7.00 | 30.00 | 30.00 | 30.00 | 30.00 |
| series_total_qty | 206,209 | 33,819,106.00 | 164.00 | 206.01 | 3.00 | 90.00 | 395.00 | 579.00 | 1,017.00 | 3,725.00 |

### Existing Mark Distribution

파일에 저장된 `mark` 컬럼 기준 분포입니다. 학습 파이프라인은 선택한 `scale_base`에 따라 `demand_qty`에서 mark/residual을 다시 만들 수 있습니다.

| mark | count | share |
| --- | --- | --- |
| 0 | 275,851 | 0.0841 |
| 1 | 3,233 | 0.0010 |
| 2 | 80,457 | 0.0245 |
| 3 | 1,613,419 | 0.4920 |
| 4 | 29,640 | 0.0090 |
| 5 | 7,998 | 0.0024 |
| 6 | 298,250 | 0.0909 |
| 7 | 8,910 | 0.0027 |
| 8 | 27,930 | 0.0085 |
| 9 | 745 | 0.0002 |
| 10 | 33,705 | 0.0103 |
| 11 | 16,718 | 0.0051 |
| 12 | 99,642 | 0.0304 |
| 13 | 23,184 | 0.0071 |
| 14 | 32,665 | 0.0100 |
| 15 | 442,986 | 0.1351 |
| 16 | 50,043 | 0.0153 |
| 17 | 27,351 | 0.0083 |
| 18 | 187,303 | 0.0571 |
| 19 | 18,693 | 0.0057 |
| 20 | 798 | 0.0002 |

### Log-Base Order Distribution from demand_qty

| base | raw order | count | share | cum share |
| --- | --- | --- | --- | --- |
| log10 | 0 | 1,836,551 | 0.5600 | 0.5600 |
| log10 | 1 | 1,442,874 | 0.4400 | 1.0000 |
| log10 | 2 | 96 | 0.0000 | 1.0000 |
| log4 | 0 | 538,331 | 0.1641 | 0.1641 |
| log4 | 1 | 2,081,814 | 0.6348 | 0.7989 |
| log4 | 2 | 658,200 | 0.2007 | 0.9996 |
| log4 | 3 | 1,176 | 0.0004 | 1.0000 |
| log2 | 0 | 150,810 | 0.0460 | 0.0460 |
| log2 | 1 | 387,521 | 0.1182 | 0.1641 |
| log2 | 2 | 902,367 | 0.2752 | 0.4393 |
| log2 | 3 | 1,179,447 | 0.3596 | 0.7989 |
| log2 | 4 | 595,740 | 0.1817 | 0.9806 |
| log2 | 5 | 62,460 | 0.0190 | 0.9996 |
| log2 | 6 | 1,158 | 0.0004 | 1.0000 |
| log2 | 7 | 18 | 0.0000 | 1.0000 |

### Model Configuration Implications

- Sequence length reference: p95=50, p99=83, max=100. `max_seq_len` should cover at least p95 for stable validation, while p99/max coverage is useful for stress tests.
- Quantity tail reference: demand_qty p99=36.00, max=177.00. Large gaps between p99 and max indicate that scale-wise MAE should be reported in addition to global MAE.
- Capacity hint: medium-length histories can justify hidden sizes around 64-128 and Titan candidates with moderate depth.

## Yellow Trip Raw

- Role: NYC taxi raw pickup log; preprocessing source only
- Path: `sample_data/new_york_taxi/yellow_trip.parquet`
- Training dataset name: `source only`
- File size: 294.33 MB
- Rows: 12,748,986
- Pickup time range: 2015-01-01 00:00:00 to 2015-01-31 23:59:59
- Required preprocessing columns missing: none

### Schema

| column | dtype |
| --- | --- |
| VendorID | Int64 |
| tpep_pickup_datetime | String |
| tpep_dropoff_datetime | String |
| passenger_count | Int64 |
| trip_distance | Float64 |
| pickup_longitude | Float64 |
| pickup_latitude | Float64 |
| RateCodeID | Int64 |
| store_and_fwd_flag | String |
| dropoff_longitude | Float64 |
| dropoff_latitude | Float64 |
| payment_type | Int64 |
| fare_amount | Float64 |
| extra | Float64 |
| mta_tax | Float64 |
| tip_amount | Float64 |
| tolls_amount | Float64 |
| improvement_surcharge | Float64 |
| total_amount | Float64 |

### Coordinate Quality

| metric | value |
| --- | --- |
| null pickup longitude | 0 |
| null pickup latitude | 0 |
| longitude range | -121.92581 to 78.66265 |
| latitude range | 0.00000 to 404.70001 |
| rows inside loose NYC bounds | 12,505,244 |
| inside-bounds share | 0.9809 |

### VendorID Distribution

| VendorID | count | share |
| --- | --- | --- |
| 2 | 6,647,797 | 0.5214 |
| 1 | 6,101,189 | 0.4786 |

### payment_type Distribution

| payment_type | count | share |
| --- | --- | --- |
| 1 | 7,881,388 | 0.6182 |
| 2 | 4,816,992 | 0.3778 |
| 3 | 38,632 | 0.0030 |
| 4 | 11,972 | 0.0009 |
| 5 | 2 | 0.0000 |

### Modeling Note

이 raw 파일은 현재 TPP 모델에 직접 들어가지 않습니다. `simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb`에서 hourly grid-cell pickup count event table로 변환한 뒤 `yellow_trip_hourly` dataset으로 학습합니다.

## Yellow Trip Hourly

- Role: hourly grid-cell pickup-count event table
- Path: `sample_data/new_york_taxi/yellow_trip_hourly.parquet`
- Training dataset name: `yellow_trip_hourly`
- File size: 0.16 MB
- Rows: 55,119
- Series count: 131
- demand_dt range: 20150101000000 to 20150131230000

### Schema

| column | dtype |
| --- | --- |
| oper_part_no | String |
| demand_dt | Int64 |
| seq | UInt32 |
| delta_t | Int32 |
| demand_qty | Float64 |
| time_bucket | Datetime(time_unit='us', time_zone=None) |
| source_resolution | String |
| grid_size_deg | Float64 |
| min_active_buckets | Int32 |

### Sequence Length Summary

| metric | mean | std | min | p50 | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| events per series | 420.76 | 261.73 | 72 | 405 | 741 | 743 | 744 | 744 |
| seq span per series | 740.55 | 9.30 | 654 | 744 | 744 | 744 | 744 | 744 |

Top 10 longest/most active series:

| series | events | seq span | total qty |
| --- | --- | --- | --- |
| -3699_2038 | 744 | 744 | 1,321,805.00 |
| -3698_2038 | 744 | 744 | 878,650.00 |
| -3700_2037 | 743 | 744 | 2,187,035.00 |
| -3700_2036 | 743 | 744 | 1,610,040.00 |
| -3699_2037 | 743 | 744 | 1,373,247.00 |
| -3700_2038 | 743 | 744 | 1,113,536.00 |
| -3698_2039 | 743 | 744 | 380,354.00 |
| -3697_2037 | 743 | 744 | 32,994.00 |
| -3701_2036 | 742 | 744 | 649,816.00 |
| -3699_2039 | 742 | 744 | 569,196.00 |

### Quantity and Time Distribution

| metric | count | sum | mean | std | min | p50 | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demand_qty | 55,119 | 12,498,482.00 | 226.75 | 646.20 | 1.00 | 7.00 | 669.00 | 1,547.00 | 3,445.00 | 6,489.00 |
| delta_t | 55,119 | 96,881.00 | 1.76 | 2.91 | 0.00 | 1.00 | 3.00 | 5.00 | 15.00 | 115.00 |
| series_total_qty | 131 | 12,498,482.00 | 95,408.26 | 320,044.06 | 75.00 | 825.00 | 177,526.00 | 649,816.00 | 1,610,040.00 | 2,187,035.00 |

### Log-Base Order Distribution from demand_qty

| base | raw order | count | share | cum share |
| --- | --- | --- | --- | --- |
| log10 | 0 | 29,684 | 0.5385 | 0.5385 |
| log10 | 1 | 13,617 | 0.2470 | 0.7856 |
| log10 | 2 | 7,769 | 0.1409 | 0.9265 |
| log10 | 3 | 4,049 | 0.0735 | 1.0000 |
| log4 | 0 | 22,315 | 0.4049 | 0.4049 |
| log4 | 1 | 10,292 | 0.1867 | 0.5916 |
| log4 | 2 | 8,834 | 0.1603 | 0.7518 |
| log4 | 3 | 5,475 | 0.0993 | 0.8512 |
| log4 | 4 | 4,223 | 0.0766 | 0.9278 |
| log4 | 5 | 3,739 | 0.0678 | 0.9956 |
| log4 | 6 | 241 | 0.0044 | 1.0000 |
| log2 | 0 | 13,030 | 0.2364 | 0.2364 |
| log2 | 1 | 9,285 | 0.1685 | 0.4049 |
| log2 | 2 | 5,844 | 0.1060 | 0.5109 |
| log2 | 3 | 4,448 | 0.0807 | 0.5916 |
| log2 | 4 | 4,903 | 0.0890 | 0.6805 |
| log2 | 5 | 3,931 | 0.0713 | 0.7518 |
| log2 | 6 | 2,861 | 0.0519 | 0.8038 |
| log2 | 7 | 2,614 | 0.0474 | 0.8512 |
| log2 | 8 | 1,859 | 0.0337 | 0.8849 |
| log2 | 9 | 2,364 | 0.0429 | 0.9278 |
| log2 | 10 | 2,101 | 0.0381 | 0.9659 |
| log2 | 11 | 1,638 | 0.0297 | 0.9956 |
| log2 | 12 | 241 | 0.0044 | 1.0000 |

### Model Configuration Implications

- Sequence length reference: p95=743, p99=744, max=744. `max_seq_len` should cover at least p95 for stable validation, while p99/max coverage is useful for stress tests.
- Quantity tail reference: demand_qty p99=3,445.00, max=6,489.00. Large gaps between p99 and max indicate that scale-wise MAE should be reported in addition to global MAE.
- Capacity hint: long histories justify larger `max_seq_len`, deeper Titan/THP candidates, and checkpointing because training cost and overfitting behavior become more sensitive.

## Cross-Dataset Configuration Guidance

| dataset | series | len p50 | len p95 | len p99 | len max | suggested max_seq_len | initial hidden_dim range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| intermittent | 23,387 | 6 | 35 | 73 | 110 | 64 for p95, 128 for p99/max | 64-128 |
| insta_market_basket | 206,209 | 10 | 50 | 83 | 100 | 64 for p95, 128 for p99/max | 64-128 |
| yellow_trip_hourly | 131 | 405 | 743 | 744 | 744 | 256 truncated, 768 full-context | 128+ |

### Practical Setup Notes

- `intermittent`: series 수는 많지만 event 길이는 짧습니다. `max_seq_len=64`면 p95를 덮고, `128`이면 거의 전체 tail까지 확인할 수 있습니다. 모델은 RMTPP hidden 64 또는 Titan `small_lmm`부터 시작하는 것이 안전합니다.
- `insta_market_basket`: row 수와 series 수가 가장 커서 throughput이 중요합니다. sequence 길이는 `intermittent`보다 조금 길지만 p99가 83이므로 `max_seq_len=128`이면 충분한 full-history stress test가 됩니다.
- `yellow_trip_hourly`: series 수는 131개로 적지만 각 series가 최대 744 hourly events입니다. `max_seq_len=256`은 최근 10.7일 정도를 보는 truncated setup이고, 전체 31일 context를 보려면 `max_seq_len=768`이 필요합니다. Titan/THP 계열의 장점은 이 데이터셋에서 더 잘 드러날 가능성이 큽니다.

## Reproducibility Notes

- Statistics were computed directly from the four parquet files listed in the inventory table.
- `yellow_trip_hourly.parquet` is expected to be generated by `simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb` from the raw taxi file.
- For event tables, mark/log-order distributions are computed from positive `demand_qty`; training code may rebuild `mark` and `scale_residual` for the selected `scale_base`.
- Sequence length means number of positive-demand events per `oper_part_no`, not dense calendar length.
