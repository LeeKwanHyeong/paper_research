# 5090 native qualification — 검증 영수증 기반 보고

5090의 **합성 입력 native 동작 검증은 통과**했고, 다섯 검색 방식 모두 T64·T256에서 사전 비용 한도에 적격이다. 이 결과는 예측 정확도나 pilot 승리 판정이 아니다. 본 보고서는 qualification 종료 증적까지만 다루며, 이후 본학습의 진행 상태는 포함하지 않는다.

- 검증 완료: 2026-10-07T08:36:58+09:00 / 종료·프로세스 확인: 2026-10-07T08:38:45+09:00.
- `qualification_shell_exit.returncode=0`, qualification 실패 기록 없음, 해당 확인 시점의 소유 프로세스와 GPU compute 프로세스 없음.
- 대상: Taxi(`yellow_trip_hourly`) seed42, NVIDIA GeForce RTX 5090, `cuda:0`, batch128. T64·T256은 합성 프로파일의 길이이며 실제 데이터 split이 아니다.
- 해당 qualification의 실제 Train optimizer updates는 **0**, 실제 scientific training launch는 **False**, held-out Test 평가는 **False**이며 Test 행을 materialize하지 않았다.

## 합성 native 동작 검증 — 완료

다섯 방식의 native 반복 실행은 조건별 3회 합성 optimizer update를 두 번 수행했고, 출력·경사·최종 상태 fingerprint가 bitwise 일치했다. 미래 관측이 이전 prefix를 바꾸지 않는지, withheld target이 차단되는지, padding을 건너뛰는지, withheld valid 행에서 검색 요약 구간이 초기화되는지도 통과했다. 별도의 점수 학습 검사는 조건별 4회 합성 update에서 fixed 512개, adaptive·generic 각각 521개 추가 파라미터 원소 모두의 비영 경사를 확인했고 두 학습 gate가 활성화됐다. 합성 optimizer 검증과 실제 Train 데이터 학습을 구분한다.

검증은 strict deterministic algorithms 및 cuDNN deterministic, TF32 비활성 조건에서 수행됐다. Runtime은 Python 3.12.13, PyTorch 2.11.0+cu130, CUDA 13.0이다. GPU UUID는 `GPU-c9adc246-ef66-7906-bc07-6e152fa9952f`이다.

## 후보 비용 적격성 — 다섯 방식 모두 적격

각 길이에서 별도로 만든 모델로 warmup5·실측20회를 수행했다. 추론과 학습 step은 실측 표본의 중앙값, 메모리는 `peak_allocated_bytes`를 MiB로 환산했다. 각 칸의 괄호는 **동일 길이의 top4 평균 대비 증가율** `(candidate/baseline−1)×100`이며 기준선은 0%다. 비용 한도는 추론 +20%, 학습 step +30%, peak allocation +20%이고 두 길이 모두 충족해야 적격이다. 비용 적격성은 technical qualification과 별개이며 학습 launch 중단 gate로 쓰지 않았다.

### T64 — batch128 합성 native 측정

| 방법 | 총 파라미터 | 추론 중앙값 ms (증가율) | 학습 step 중앙값 ms (증가율) | Peak allocation MiB (증가율) | 비용 적격 |
|---|---:|---:|---:|---:|---|
| top4 평균 | 102,879 | 14.066 (+0.00%) | 41.357 (+0.00%) | 447.244 (+0.00%) | 적격 |
| cosine softmax top4 | 102,879 | 14.105 (+0.28%) | 41.762 (+0.98%) | 453.494 (+1.40%) | 적격 |
| 시간·수량 50:50 | 103,391 | 15.216 (+8.17%) | 43.136 (+4.30%) | 457.875 (+2.38%) | 적격 |
| 이력 가변 비중 | 103,400 | 15.251 (+8.43%) | 43.352 (+4.82%) | 461.377 (+3.16%) | 적격 |
| 일반 학습 검색 | 103,400 | 15.236 (+8.32%) | 43.085 (+4.18%) | 453.846 (+1.48%) | 적격 |

### T256 — batch128 합성 native 측정

| 방법 | 총 파라미터 | 추론 중앙값 ms (증가율) | 학습 step 중앙값 ms (증가율) | Peak allocation MiB (증가율) | 비용 적격 |
|---|---:|---:|---:|---:|---|
| top4 평균 | 115,167 | 46.338 (+0.00%) | 138.129 (+0.00%) | 2020.667 (+0.00%) | 적격 |
| cosine softmax top4 | 115,167 | 46.343 (+0.01%) | 138.764 (+0.46%) | 2039.479 (+0.93%) | 적격 |
| 시간·수량 50:50 | 115,679 | 47.301 (+2.08%) | 140.211 (+1.51%) | 2056.985 (+1.80%) | 적격 |
| 이력 가변 비중 | 115,688 | 47.463 (+2.43%) | 140.532 (+1.74%) | 2072.988 (+2.59%) | 적격 |
| 일반 학습 검색 | 115,688 | 47.274 (+2.02%) | 140.059 (+1.40%) | 2040.363 (+0.97%) | 적격 |

길이별 총 파라미터 차이는 모든 방식에서 **12,288개**다. 프로파일러가 `max_seq_len`을 64와 256으로 각각 바꾸어 별도 모델을 생성하고, 부모의 학습 가능한 위치 임베딩이 `[1,max_seq_len,64]`이므로 `(256−64)×64=12,288`개 차이가 난다. 추가 검색 예산은 길이와 무관하게 mean·weighted 0개, fixed 512개, adaptive·generic 521개다. 실제 Taxi fit의 `max_seq_len=256`에서 사용할 총 수는 T256 표의 115,167 / 115,167 / 115,679 / 115,688 / 115,688개다. T64 측정을 T256 본학습 모델의 파라미터 수로 보고하지 않는다.

각 프로파일에는 warmup과 실측을 합친 **25회 합성 optimizer update**가 포함되며 실제 Train update0과 혼동하지 않는다. 위 실측은 전체 fit의 완료 시각·epoch 시간·전체 ETA로 환산하지 않는다. 데이터 로딩·전체 Train/Validation 순회·early stopping·endpoint 검증·원본 회수는 이 작은 합성 측정으로 추정하지 않았다.

## 원본 근거와 자체 확인 — 완료

- [Native receipt](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1/qualification/5090/receipt.json): RAW SHA256 `ef2e19f81f5578bdd03333c04f92665ad804c113d7d46dab5e9236082559c3f8`.
- [Latest readback](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1/qualification/5090/1791329924967488000.json): RAW SHA256 `1748028294ad62c7d2e7fa764af5ce3ad7735b0fb86c5ac3ea1ba0eacbc0f0cc`.
- [Latest pointer](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1/qualification/5090/latest.json): RAW SHA256 `d7a8efee2af0e914234181456d503bfc269085b9fd374fc3f8743255ec2d633a`. pointer가 가리키는 readback SHA와 실제 bytes를 대조했다.
- [Execution contract](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1/execution_contract.json): canonical SHA256 `5372f2ec93c98e6b45ecf37d2b19ae1375a838fdfd9eb8a9c4f3a908456e7953`.
- Source closure SHA256 `918f7cb883b62cbdc6f3f178cc53b6e20aac2e71a49b6e036769aa2fea76155a` / operation closure SHA256 `e4c25f8b21443f264f70df27fa5f33f28a43df1716ec84b04357c80d88380597`.
- [길이별 별도 프로파일 생성](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1/operation/qualify.py:161), [학습 가능한 위치 임베딩](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1/source/models/Titan/backbone.py:148).
- receipt의 raw bytes가 readback의 `receipt_json_text` 및 `receipt_sha256`와 일치하고 receipt payload도 동일함을 확인했다. 계약·source/operation closure, 실제 shell exit0, 반복 fingerprint, 모든 중앙값·기준선 비율·적격 판정을 원본 표본에서 다시 계산했다. 보고서 작성 과정에서는 원격 호출·모델 실행·Test 결과 열람을 하지 않았다.

## 남은 작업 — 외부 실행 결과 대기

본학습의 완료 여부와 예측 성능은 별도 종료 증적으로 확인해야 한다. 모든 실행이 끝난 후 같은 RMSE 선택 epoch의 전체·꼬리 RMSE/MAE/TimeNLL과 GPU별 비용으로 사전 pilot 규칙을 판정한다. 5080 측정과 비용은 해당 GPU의 영수증으로 별도 보고하며 두 GPU 결과를 independent seed 또는 독립 Test 검증으로 해석하지 않는다.
