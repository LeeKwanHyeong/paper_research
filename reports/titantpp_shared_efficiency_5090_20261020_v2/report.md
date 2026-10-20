# Shared15 / Titans-MAC 원본 Train 입력 비용 측정

상태: 완료. native5090의 동일 GPU에서 fresh seed42 모델을 새 프로세스로 만들어 36개 비용 측정을 실행했다. 원본 Train 표본만 사용했으며 Validation/Test 예측·성능은 평가하지 않았다. 이 문서의 수치는 회수한 원본 JSON과 CSV에서 계산했다.

## 지연시간

각 반복의 예열 5개를 제외한 32개 batch에서 계산 시간 중앙값을 구한 뒤, 반복 3회의 평균 ± 표본표준편차를 표시했다. batch 크기는 128이다. 속도비는 같은 반복 번호의 MAC 중앙값 / Shared15 중앙값 3개의 평균 ± 표본표준편차다. 1보다 크면 이번 입력에서 Shared15의 시간이 짧다. 반복 번호는 시간 측정 반복이며 학습 seed 3개를 뜻하지 않는다.

| 데이터 | 경로 | Shared15 ms/batch | MAC ms/batch | MAC/Shared15 |
| --- | --- | --- | --- | --- |
| RAF | 학습 step | 8.542 ± 0.162 | 67.137 ± 0.465 | 7.862 ± 0.192 |
| RAF | 평가 batch | 3.605 ± 0.030 | 25.177 ± 0.339 | 6.984 ± 0.118 |
| Taxi | 학습 step | 18.712 ± 0.108 | 175.800 ± 3.043 | 9.395 ± 0.142 |
| Taxi | 평가 batch | 5.215 ± 0.029 | 63.186 ± 0.810 | 12.116 ± 0.220 |
| Intermittent | 학습 step | 19.260 ± 0.076 | 178.008 ± 1.475 | 9.243 ± 0.104 |
| Intermittent | 평가 batch | 5.472 ± 0.048 | 63.623 ± 0.419 | 11.627 ± 0.115 |

계산 시간은 CPU→GPU 전송을 제외하고 CUDA 동기화로 양쪽 경계를 구분한 host wall time이다. 학습 경로는 forward·공통 head·target loss·zero_grad·backward·gradient clip·AdamW step을 포함한다. 평가 경로는 eval/no-grad forward·head·target loss와 MAC의 온라인 메모리 갱신을 포함하므로 순수 inference 시간으로 해석하지 않는다. 반복별 산술평균·사분위수·첫 batch·예열·전체 프로세스 시간은 [repetitions.csv](repetitions.csv)에 있다.

## 처리량

각 반복에서 측정 target 4,096개를 해당 32개 batch 시간의 합으로 나눈 후 반복 평균 ± 표본표준편차를 계산했다. 연산 처리량은 compute만, 전송 포함 처리량은 CPU→GPU 전송+compute를 분모로 쓴다. 두 값 모두 loader·직렬화·로그·유한성 확인을 제외하므로 전체 데이터 파이프라인 처리량은 아니다.

| 데이터 | 경로 | Shared15 연산 targets/s | MAC 연산 targets/s | Shared15 전송 포함 targets/s | MAC 전송 포함 targets/s |
| --- | --- | --- | --- | --- | --- |
| RAF | 학습 step | 14957.3 ± 317.4 | 1694.9 ± 14.5 | 14850.2 ± 314.8 | 1693.3 ± 14.5 |
| RAF | 평가 batch | 35404.1 ± 281.9 | 5073.6 ± 79.2 | 34846.0 ± 269.5 | 5061.2 ± 78.9 |
| Taxi | 학습 step | 6834.6 ± 21.2 | 710.0 ± 11.4 | 6802.9 ± 21.1 | 709.6 ± 11.4 |
| Taxi | 평가 batch | 24544.4 ± 123.0 | 2008.6 ± 29.2 | 24164.5 ± 121.6 | 2005.7 ± 29.1 |
| Intermittent | 학습 step | 6645.3 ± 31.2 | 701.1 ± 5.2 | 6615.9 ± 31.2 | 700.8 ± 5.2 |
| Intermittent | 평가 batch | 23367.7 ± 210.4 | 1997.8 ± 12.6 | 23024.5 ± 208.1 | 1994.9 ± 12.6 |

## 메모리와 parameter

예열 5개 뒤 peak를 reset하고 측정 32개 동안 얻은 allocated/reserved peak의 반복 평균 ± 표본표준편차다. allocated는 tensor에 할당된 메모리이며 reserved는 allocator cache를 포함한다. 상대 변화는 (Shared15 평균 / MAC 평균 − 1) × 100이며 양수는 Shared15가 더 큰 경우다. 예열 직후 baseline·측정 중 최소 baseline·peak 증분도 CSV에 남겼다.

| 데이터 | 경로 | Shared15 allocated MiB | MAC allocated MiB | allocated 상대 변화 | Shared15 reserved MiB | MAC reserved MiB | reserved 상대 변화 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAF | 학습 step | 540.76 ± 0.00 | 2873.66 ± 0.00 | -81.18% | 562.00 ± 0.00 | 2890.00 ± 0.00 | -80.55% |
| RAF | 평가 batch | 94.48 ± 0.00 | 86.24 ± 0.00 | +9.55% | 116.00 ± 0.00 | 106.00 ± 0.00 | +9.43% |
| Taxi | 학습 step | 1877.71 ± 0.00 | 7631.00 ± 0.00 | -75.39% | 2042.00 ± 0.00 | 7646.00 ± 0.00 | -73.29% |
| Taxi | 평가 batch | 389.24 ± 0.00 | 97.83 ± 0.00 | +297.87% | 416.00 ± 0.00 | 130.00 ± 0.00 | +220.00% |
| Intermittent | 학습 step | 1877.71 ± 0.00 | 7631.00 ± 0.00 | -75.39% | 2042.00 ± 0.00 | 7646.00 ± 0.00 | -73.29% |
| Intermittent | 평가 batch | 389.24 ± 0.00 | 97.83 ± 0.00 | +297.87% | 416.00 ± 0.00 | 130.00 ± 0.00 | +220.00% |

Shared15의 peak가 MAC보다 큰 반복을 모두 아래에 기록했다.

| 데이터 | 경로 | 반복 | 메모리 지표 | Shared15 MiB | MAC MiB | 상대 변화 |
| --- | --- | --- | --- | --- | --- | --- |
| RAF | eval | 1 | peak_allocated_mib | 94.48 | 86.24 | +9.55% |
| RAF | eval | 2 | peak_allocated_mib | 94.48 | 86.24 | +9.55% |
| RAF | eval | 3 | peak_allocated_mib | 94.48 | 86.24 | +9.55% |
| RAF | eval | 1 | peak_reserved_mib | 116.00 | 106.00 | +9.43% |
| RAF | eval | 2 | peak_reserved_mib | 116.00 | 106.00 | +9.43% |
| RAF | eval | 3 | peak_reserved_mib | 116.00 | 106.00 | +9.43% |
| Taxi | eval | 1 | peak_allocated_mib | 389.24 | 97.83 | +297.87% |
| Taxi | eval | 2 | peak_allocated_mib | 389.24 | 97.83 | +297.87% |
| Taxi | eval | 3 | peak_allocated_mib | 389.24 | 97.83 | +297.87% |
| Taxi | eval | 1 | peak_reserved_mib | 416.00 | 130.00 | +220.00% |
| Taxi | eval | 2 | peak_reserved_mib | 416.00 | 130.00 | +220.00% |
| Taxi | eval | 3 | peak_reserved_mib | 416.00 | 130.00 | +220.00% |
| Intermittent | eval | 1 | peak_allocated_mib | 389.24 | 97.83 | +297.87% |
| Intermittent | eval | 2 | peak_allocated_mib | 389.24 | 97.83 | +297.87% |
| Intermittent | eval | 3 | peak_allocated_mib | 389.24 | 97.83 | +297.87% |
| Intermittent | eval | 1 | peak_reserved_mib | 416.00 | 130.00 | +220.00% |
| Intermittent | eval | 2 | peak_reserved_mib | 416.00 | 130.00 | +220.00% |
| Intermittent | eval | 3 | peak_reserved_mib | 416.00 | 130.00 | +220.00% |

Parameter 수는 이번 native 모델 초기화 증적에서 읽었다. parameter 상대 변화와 실제 peak 메모리는 다른 지표다.

| 데이터 | Shared15 parameter | MAC parameter | 상대 변화 |
| --- | --- | --- | --- |
| RAF | 88,451 | 111,302 | -20.53% |
| Taxi | 99,459 | 122,310 | -18.68% |
| Intermittent | 99,459 | 122,310 | -18.68% |

## 입력 범위

데이터마다 전체 Train target에서 seed42 순열의 서로 다른 4,736개를 뽑았다. 예열 640개와 측정 4,096개를 모든 모델·모드·반복이 공유한다. 유효 길이는 target을 포함한 mask 길이이며 아래 통계는 측정 32개 batch 기준이다.

| 데이터 | 전체 Train target | padded 길이 | 유효 길이 평균 | 최소 | 최대 | padding 비율 |
| --- | --- | --- | --- | --- | --- | --- |
| RAF | 25779 | 84 | 4.56 | 2 | 14 | 94.57% |
| Taxi | 38393 | 256 | 111.16 | 2 | 169 | 56.58% |
| Intermittent | 393824 | 256 | 52.93 | 2 | 193 | 79.33% |

모델별 초기 state SHA와 batch tensor SHA를 모든 반복에서 확인했고, 준비 과정의 공통 head 초기 tensor 동일성 검사를 확인했다. 서로 다른 encoder의 parameter 수나 encoder tensor 동일성은 요구하지 않았다. 실제 Train 표본 optimizer step은 666개, 평가 batch는 666개이며 예열을 포함한다. 별도 합성 자격 검증은 CPU/CUDA × 두 모델 4개 프로세스에서 합계 12개 합성 optimizer step을 수행했다.

## 증적과 해석 범위

- 원본 bundle: [titantpp_shared_efficiency_5090_20261020_v2](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_shared_efficiency_5090_20261020_v2)
- [terminal manifest](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_shared_efficiency_5090_20261020_v2/retrieved/terminal_manifest.json)의 230개 파일 SHA와 local source 985개·운영 소스 2개 SHA를 확인했다.
- 비용 worker 36개, 합성 검증 4개, 준비 1개의 실제 자식 OS exit0와 controller OS exit0·PID 부재를 확인했다. 종료 시 GPU compute process가 없었다.
- 실행 시간은 준비를 포함해 407.181초다. 측정 32개 안의 추가 compiled graph는 전체 0개다. 느린 batch와 첫 batch를 제거하지 않았다. 첫 batch에는 compilation·CUDA·allocator 초기화 비용이 함께 들어갈 수 있다.
- contract canonical SHA: `e61b7c4b9dcded33f103ffaa116b3a53f8f304650ff2b3c79bbb332ac22eb81c`
- source closure SHA: `7905cb02aa6b51c3be5754260e48709f9d4f486276fc85c3ff460166b7032d7f`
- [기계판독 집계](analysis.json), [검증 결과](verification.json), [반복별 집계](repetitions.csv), [batch별 원시 시간](per_batch.csv), [입력 요약](inputs.csv)

이 결과는 fresh seed42로 초기화한 짧은 scratch 학습 step과 Train 입력 평가 경로의 비용이다. scientific fit, scientific checkpoint의 정확도, 수렴까지 총비용, 동일 정확도 효율, 실측 초/epoch 또는 모든 길이의 복잡도를 입증하지 않는다. 세 번의 시간 반복으로 학습 seed 변동성을 추정하지 않는다. 평가 경로 비용을 inference-only 성능으로 바꾸어 표현하지 않는다. 5080 과거 결과나 기존 scientific checkpoint 성능과 합치지 않는다.
