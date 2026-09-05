# Hard-LMM 경과시간 encoder — seed42 e300 screening 결과

**완료: Taxi·Instacart 실행과 로컬 재감사는 통과했으며, 사전 성능 기준은 두 데이터셋 모두 미달했다.**
Taxi는 원본 Hard-LMM 대비 개선됐으나 separate-key가 확보한 개선을 보존하지 못했다. Instacart는 원본·separate-key 대비 요구한 body MAE 개선이 없었다. 이 단일 경과시간 bias 후보를 채택하지 않고 원본 Hard-LMM을 일반 기준선으로 유지한다.

## 실행과 검증

- 저장소/브랜치: `paper_research/master`. 실행 소스 `e6e59337a816d53e9d1d3afaf0f6cede48ef5e2f`, 시작 증적 `7769d84`.
- 서버: 5090. 2026-09-05 **10:19:55–13:16:19 KST**, 약 2시간 56분. Taxi 다음 Instacart를 직렬 실행했다.
- Fresh seed42, 최대300/min40/patience40, batch128, AdamW lr0.001, 기존 validation joint selector를 사용했다. 300은 상한이며 두 실행 모두 고정된 조기 종료 조건으로 정상 완료됐다.
- 모델·학습 의존 파일 167개가 CUDA/e1 검증 소스와 동일하며, 소스·참조 534개 및 Runtime·data·split·고정 참조 checksum을 검증했다. 실행 전 로컬 테스트31개와 기존 CUDA56개/full e1 증적을 확인했다.
- Python3.12.13, PyTorch2.11.0+cu130, CUDA13.0, Polars1.39.3. 공용 Runtime·서비스 변경 없이 별도 snapshot과 새 결과 경로를 사용했다.

| 데이터셋 | 완료 epoch | 선택 epoch | Train target / epoch | Validation target | 총 optimizer step | 실행·감사 시간 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Taxi | 42 | 2 | 38,393 | 8,268 | 12,600 | 6.33분 |
| Instacart | 66 | 26 | 1,991,192 | 503,733 | 1,026,762 | 170.05분 |

선택은 `time_nll + log1p_quantity_mse`가 엄격히 작아질 때만 갱신하며, 동률이면 앞선 epoch를 유지한다. Body·RMSE·tail 결과를 보고 checkpoint를 다시 고르지 않았다.

## 고정 기준에 따른 결과

Body는 train p95 이하 target의 수량 MAE, tail은 train p99를 엄격히 초과하는 target의 MAE다. Body는 구간별 target 수로 가중하며, 결과와 기준 비교는 반올림 전 수치로 계산한다.

| Body MAE | 원본 Hard-LMM | Separate-key | 경과시간 후보 | 원본 대비 | Separate-key 대비 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Taxi | 23.167407 | 16.966216 | 20.775113 | -10.3261% | +22.4499% |
| Instacart | 3.424546 | 3.422799 | 3.424271 | -0.0080% | +0.0430% |

변화율은 음수일수록 개선이다.

| 데이터셋 / 비교 대상 | Body MAE 변화 | RMSE 변화 | Tail MAE 변화 | Time NLL 차이 | 해당 비교 통과 |
| --- | ---: | ---: | ---: | ---: | --- |
| Taxi / 원본 | -10.3261% | -22.4607% | -30.4113% | +0.002615 | 통과 |
| Taxi / separate-key | +22.4499% | +17.9012% | +16.5415% | +0.003197 | 미달 |
| Instacart / 원본 | -0.0080% | +0.4192% | +1.1337% | -0.000064 | 미달 |
| Instacart / separate-key | +0.0430% | +0.0425% | -0.2173% | -0.000164 | 미달 |

- **Taxi:** 원본 대비 body 10.33%, RMSE22.46%, tail30.41% 개선으로 원본 기준은 모두 통과했다. 그러나 separate-key 대비 body22.45%, RMSE17.90%, tail16.54% 악화로 각각의 1% 보존 기준을 넘었다. Time NLL은 두 참조 모두 +0.01 이내다.
- **Instacart:** 원본 대비 body 개선은0.0080%이고 separate-key 대비0.0430% 악화했다. 두 참조 대비 요구한 body5% 개선 기준에 미달했다. RMSE·tail·Time NLL 기준은 모두 통과했다.
- **전체 판정: 0/2 통과.** 기존 기준을 완화하거나 모델·학습 조건을 바꿔 재실행하지 않았다.

## 감사와 해석 범위

- 모든 epoch의 전체 train target·batch 처리, 선택 checkpoint의 validation 구간별 건수와 가중 MAE/RMSE, 최초 허용 조기 종료, 최초 strict best 선택을 검사했다. 두 실행 모두 finite이며 traceback이나 실행 실패는 없다. Epoch별 validation 건수 필드는 별도로 저장되지 않으므로, 해당 건수의 직접 감사 범위는 선택 checkpoint다.
- 두 데이터셋 모두 beta8개가 초기0에서 갱신됐다. AdamW moments가 유한하며 step은 전체 학습 batch 합계와 같다. Last current state와 optimizer를 복원하고, 선택 checkpoint와 last 내부 best의 hash·복원 출력을 확인했다.
- 최대 할당/예약 GPU 메모리는 Taxi1.81/2.01GiB, Instacart0.36/0.43GiB였다. 종료 후 GPU compute process와 최근 kernel error가 없었다.
- 내려받은 raw checkpoint와 CSV에 [로컬 재감사](local_audit.json)를 실행해 통과했다. [독립 계산 감사](independent_final_audit.json)는 별도 계산으로 지표·조기 종료·hash·판정을 확인한다.
- 이번 결과는 한 seed의 고정 validation screening이다. 이 시간 표현 후보의 성능 기준 미달을 확인하며, 모든 시간 표현이 무효라고 일반화하거나 기존 train의 시간 불규칙성–오류 연관성을 부정하지 않는다.
- Held-out test, 추가 seed·데이터셋, 후보 변경, 자동 재시작, scheduler, Push는 실행하지 않았다.

## 증적

- [판정 원시값](validation_comparison.json), [실행 상태](status.json), [확정 실행 계약](execution_contract.json), [시작 명령](launch_record.json), [초기 학습 확인](initial_training_check.json), [소스 manifest](source_manifest.json).
- [Taxi summary](yellow_trip_hourly/summary.json), [Taxi audit](yellow_trip_hourly/audit.json), [Instacart summary](insta_market_basket/summary.json), [Instacart audit](insta_market_basket/audit.json).
- 서버 snapshot: `/home/leekwanhyeong/workspace/paper_research_elapsed_age_screening_e6e5933_5090`.
- 서버 raw: `/home/leekwanhyeong/workspace/paper_research/search_artifacts/hard_lmm_elapsed_age_screening_5090_20260905`.
- 로컬 raw: `search_artifacts/hard_lmm_elapsed_age_screening_5090_20260905/`. Checkpoint는 ignored raw 경로에 보존하고 증적·집계만 커밋한다.

## 남은 작업 순서

**현재 기준선 · 완료 — 실행·판정 / 로컬·5090**
- 승인된 e300 screening을 마쳤다. 후보는 두 데이터셋의 동시 성능 기준에 미달해 채택하지 않는다.

**다음 작업 / 로컬 — 실패 원인과 다음 가설 검토**
- 기존 학습 이력과 선택 checkpoint를 바탕으로, 시간 bias의 실제 작용과 수량·시간 손실의 관계를 검토한다. Train에서 발견한 연관성이 이번 수정 경로의 개선으로 이어지지 않은 이유를 좁힌다.
- 기존 readout·query·value 진단과 중복되는 후보를 피하고, 새로운 구현이나 GPU 실행을 제안하기 전에 근거와 단일 가설을 다시 확정한다.
