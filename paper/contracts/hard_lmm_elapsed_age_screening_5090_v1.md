# Hard-LMM 경과시간 후보: 5090 seed42 e300 판정 계약

**현재 기준선 · 완료 / `paper_research/master`**

- 경과시간 후보는 separate-key encoder의 attention 점수에 사건 순서 대비 상대 경과시간 차이를 더합니다. 계수는 layer 2개 × head 4개의 8개이며, 모두 0에서 시작합니다. 동일한 초기화의 separate-key와 초기 출력이 일치합니다.
- 구현은 `77e5f4a`, AdamW scalar step의 장치 간 값 비교 테스트 수정은 `ad81a17`입니다. 검증 증적은 `e97a12a`에 기록했습니다. 최종 로컬 검증 122개와 CUDA 56개가 통과했고, Taxi·Instacart 전체 e1의 학습·저장·복원 및 독립 감사를 완료했습니다.
- e1은 정상 실행 확인이며 성능 채택 근거로 사용하지 않습니다. 현재 기본 모델은 original Hard-LMM입니다.

**실행과 판정 계약 · 확정 / 로컬**

- 사용자는 Taxi·Instacart seed42 e300 실행과 기존 기준에 따른 평가를 승인했습니다. 이 문서의 [JSON 계약](hard_lmm_elapsed_age_screening_5090_v1.json)은 이번 승인을 별도로 기록합니다. 기존 [후보 계약](hard_lmm_elapsed_age_v1.json)과 [e1 계약](hard_lmm_elapsed_age_cuda_e1_5090_v1.json)은 수정하지 않습니다.
- 원본 대비 개선, separate-key 대비 추가 효과, Taxi 개선 보존 기준은 새 결과를 보기 전에 확정된 값을 그대로 사용합니다. JSON의 `performance_gates`는 기존 후보 계약의 해당 객체와 완전히 같습니다. 그 안의 e1 시점 승인 제한 문장은 과거 기록이며, 이번 실행 권한은 새 계약의 `authorization`에 따릅니다.
- CUDA/e1 완료 증적, 당시 소스 manifest, 원본 registry와 separate-key validation 비교 자료의 SHA256을 고정합니다. 실행 소스는 별도 커밋으로 확정하고, 검증된 모델·학습 소스와 바이트 단위로 일치해야 합니다.

**승인된 실행 / 5090**

- 새 `paper_research/master` 소스 snapshot에서 Taxi 이후 Instacart를 순서대로 실행합니다. 데이터는 기존 서버 프로젝트에서 읽고, 결과는 별도의 새 디렉터리에 저장합니다. 기존 checkout과 미추적 `scripts/`는 실행 소스에 섞지 않습니다.
- Runtime은 Python 3.12.13, PyTorch 2.11.0+cu130, CUDA 13.0, Polars 1.39.3이며 Python은 `/opt/miniconda3/envs/ai_env/bin/python`입니다. 장시간 실행은 `/opt/miniconda3/envs/ai_env/bin/tmux`를 사용합니다.
- 실행 전 GPU compute process 0개, 여유 메모리 12,000 MiB 이상, GDM inactive, 최근 Xid/OOM 없음과 소스·데이터·split checksum 일치를 확인합니다. 확인 실패 시 기록을 남기고 멈춥니다. 서비스나 Runtime 설정을 변경하지 않습니다.
- 각 데이터셋은 새 seed42 파라미터로 시작합니다. e1이나 기존 reference checkpoint에서 학습을 이어가지 않습니다. 기존 loader·precision·AdamW 경로를 유지하며, 새로운 deterministic 실행 모드를 도입하지 않습니다.

| 조건 | Taxi (`yellow_trip_hourly`) | Instacart (`insta_market_basket`) |
| --- | ---: | ---: |
| Train target 수 | 38,393 | 1,991,192 |
| Validation target 수 | 8,268 | 503,733 |
| Lookback / target 포함 최대 길이 | 168 / 256 | 52 / 64 |
| 시간 단위 | hour | day |
| Train p95 / p99 수량 | 1,562 / 3,449 | 25 / 35 |
| 후보 파라미터 수 | 93,899 | 81,611 |

- 최대 300 epoch, 최소 40 epoch, patience 40, batch 128, learning rate 0.001, gradient clip 1을 고정합니다. 300은 상한이며 기존 조기 종료 조건을 충족한 종료도 정상 완료입니다.
- 기존 `validation_joint_objective = time_nll + log1p_quantity_mse`가 **엄격히 감소**할 때만 best를 갱신합니다. 동률이면 앞선 epoch를 유지합니다. 최종 평가는 이 기준으로 복원한 checkpoint 하나에만 적용하며, body·tail 지표를 보고 다른 epoch를 고르지 않습니다.
- Encoder 구성, 입력, K/V·top-4·temperature·`h+r` 결합, head·loss·optimizer와 데이터 경계를 모두 기존 후보 계약대로 유지합니다.

**성능 판정 / validation만 사용**

| 비교 대상 | Body MAE | 전체 RMSE | p99 초과 MAE | Time NLL |
| --- | --- | --- | --- | --- |
| 두 데이터셋 vs original Hard-LMM | 5% 이상 개선 | 악화 2% 이하 | 악화 2% 이하 | 증가 0.01 이하 |
| Instacart vs separate-key | 5% 이상 개선 | 악화 1% 이하 | 악화 1% 이하 | 증가 0.01 이하 |
| Taxi vs separate-key | 악화 1% 이하 | 악화 1% 이하 | 악화 1% 이하 | 증가 0.01 이하 |

- 모든 조건을 동시에 적용하고 모든 판정 지표가 finite여야 합니다. 두 reference 중 더 엄격한 상한이 적용됩니다. 두 데이터셋 모두 통과해야 전체 screening 통과입니다.
- Body는 target 수량이 train p95 이하인 validation target 전체의 수량 MAE입니다. 구간별 MAE를 단순 평균하지 않고 target 수로 가중합니다. Tail은 train p99를 **엄격히 초과**하는 target의 MAE입니다. RMSE와 MAE는 같은 point prediction을 사용합니다. Time NLL에 joint loss나 total NLL을 대신 넣지 않습니다.
- Reference는 [고정된 separate-key 비교 자료](../results/hard_lmm_key_value_screening_5090_20260904/validation_comparison.json)의 full-precision `baseline`과 `candidate` 값입니다. 새로운 JSON 계약에 두 reference의 지표와 계산된 상한을 함께 기록합니다. 아래 표시값은 읽기 위한 값이며, 실제 판정은 원본 정밀도에서 계산하고 허용 오차를 추가하지 않습니다.

| 후보가 동시에 만족해야 하는 상한 | Taxi | Instacart |
| --- | ---: | ---: |
| Body MAE | 17.135878570351 | 3.2516587761773206 |
| 전체 RMSE | 120.584501275849 | 6.056621032796696 |
| p99 초과 MAE | 703.3689079188094 | 23.461019329828012 |
| Time NLL | 1.375848076807801 | 3.2164400114488094 |

**완료 조건과 해석 범위**

- 전체 처리 건수, fresh seed42 history, 조기 종료와 best 선택, finite 지표, beta의 실제 학습 및 AdamW 상태, 최대 CUDA 메모리, checkpoint 저장·복원, source/data/reference 무결성을 검증합니다. Beta는 텐서 전체의 0 대비 L2 변화가 양수이면 됩니다. 8개 모두의 이동을 요구하지 않습니다.
- Last checkpoint optimizer의 step은 마지막 학습까지의 전체 batch 수와 일치해야 합니다. 선택된 best가 마지막 epoch보다 앞설 수 있으므로, best와 last 파라미터의 동일성을 요구하지 않습니다. Last에 저장된 best 복제본은 별도 best checkpoint와 일치해야 합니다.
- 성능 기준 실패는 해당 실패를 그대로 기록한 정상 screening 결과입니다. 실행·데이터·finite 감사가 통과했다면 나머지 승인 데이터셋도 실행합니다. 실행 감사 실패 시 자동 재시도·resume·후보 변경 없이 중단합니다.
- 로컬로 내려받은 결과의 무결성을 확인하고 독립 계산 감사 후 각 기준과 전체 판정을 보고합니다. 이번 작업은 held-out 평가, 추가 seed·데이터셋, reference 재학습, 자동 scheduler, Push를 포함하지 않습니다.
- 이력 길이가 짧은 train 집단에서 발견한 연관성이 개선을 보장하지 않습니다. 파라미터 8개 추가와 전체 학습 경로 변화가 포함되므로 파라미터 수를 맞춘 인과 증명도 아닙니다. 한 seed의 두 validation 결과는 screening 근거이며, 통과해도 일반적 우월성이나 최종 기본 모델 채택으로 확대하지 않습니다.
