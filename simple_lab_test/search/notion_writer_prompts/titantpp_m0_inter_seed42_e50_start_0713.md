다음 실험 시작 내용을 Notion에 정리해주세요.

사후 범위 정정 (`2026-07-13`): 아래의 사전 gate에서 `RevIN track`은
`log2(qty)` direct decoder에 종속된 기존 log-domain M1-M4를 뜻한다. M0는
train-global normalization이며 RevIN 자체가 아니다. 이 사전 문구는 audit trail로
보존하되, M0 실패를 raw-quantity RevIN 실패로 해석하지 않는다.

위치:
- `5. Model Design Enhancement > Enhancement & Validation History`
- `2. Confirm and Refine Topic`에는 작성하지 않음
- 같은 제목의 페이지가 있으면 새 페이지를 만들지 말고 기존 페이지를 업데이트

페이지 제목:
- `TitanTPP M0 Intermittent Seed-42 e50 Validation Screening (2026-07-13)`

기준 시각:
- 실험 시작 시각: `2026-07-13 11:26:04 KST`
- 실행 서버: `5090` (`192.168.0.71`)
- tmux session: `titantpp_m0_inter_e50_0713`
- conda env: `ai_env`

실험 목적:
- M0 direct log2-magnitude decoder가 Intermittent의 confirmed V2 baseline보다 validation quantity MAE와 log2 quantity MAE를 개선하는지 확인
- direct regression 자체의 유효성을 먼저 검증해 M3/M4 shrinkage RevIN으로 진행할 근거가 있는지 판단
- marker/time likelihood와 mark accuracy가 V2 대비 안전 범위 안에 남는지 확인
- held-out test를 후보 선택에 사용하지 않고 validation-only gate로 승격 여부 결정

실험 순서:
1. V2 seed-42 e50 `best_val_nll` checkpoint를 현재 evaluator로 validation-only 재평가
2. 기존 V2에 없던 `log_qty_mae/rmse`를 포함한 reference를 고정
3. 같은 budget으로 M0 seed-42 e50 실행
4. M0 `best_val_nll` validation metric만 V2 reference와 비교
5. gate 통과 시 matched multi-seed 준비, 실패 시 M0에 종속된 기존 log-domain
   M1-M4 branch만 중단하고 raw-quantity RevIN은 미검증으로 유지

Matched budget:

| 항목 | V2 reference | M0 |
| --- | --- | --- |
| dataset/split | Intermittent fixed validation | 동일 |
| candidate | `small_lmm` | `small_lmm` |
| seed/epochs | `42 / 50` | `42 / 50` |
| lr/batch size | `1e-3 / 128` | 동일 |
| lookback/max sequence | `52 / 16` | 동일 |
| train loss scope | `target_only` | `target_only` |
| marker objective | plain CE | plain CE |
| checkpoint | `best_val_nll` | `best_val_nll` |
| quantity path | mark + residual V2 | direct `log2(qty)` M0 |
| continuous input | raw residual | global-normalized log2 quantity |

M0 옵션:
- `qty_decoder_mode=direct_log_qty`
- `magnitude_norm_mode=global`
- `magnitude_input_emb_dim=8`
- `lambda_magnitude=1.0`
- `lambda_qty=0.25`
- `magnitude_sigma_floor=0.0014535461338152059`
- `magnitude_exp_clamp_min/max=-2/15`
- `value_input_mode=none`
- `loss_mode=hybrid`
- `marker_loss_mode=ce`, `lambda_ordinal=0`
- `test_time_memory=none`

Frozen V2 validation reference:

| Metric | Value |
| --- | ---: |
| best epoch | `19` |
| total NLL | `5.666520` |
| marker NLL | `0.991274` |
| time NLL | `4.675246` |
| quantity MAE | `3.060182` |
| mark accuracy | `57.249%` |
| DT MAE | `42.064581` |
| validation targets | `41,901` |

- V2 log2 quantity MAE/RMSE는 첫 stage에서 현재 evaluator로 추가 고정
- validation reference script는 `held_out_test_read=false`를 기록하고 test sample을 구성하지 않음

Validation acceptance gate at `best_val_nll`:
- quantity MAE: V2 대비 최소 `3%` 개선
- log2 quantity MAE: V2 대비 최소 `3%` 개선
- marker NLL regression: 최대 `1%`
- total/time NLL regression: 각각 최대 `0.5%`
- mark accuracy gap: 최소 `-0.25%p`
- DT MAE regression: 최대 `2%`
- NaN/Inf/Traceback 및 artifact integrity 문제 없음
- M0가 실패하면 RevIN M3/M4를 시작하지 않고 V5b class-prior correction으로 복귀

실행 명령어:

```bash
ssh 5090 '/opt/miniconda3/envs/ai_env/bin/tmux new-session -d -s titantpp_m0_inter_e50_0713 "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_m0_inter_seed42_e50_0713.sh"'
```

Artifact 경로:
- V2 validation-only reference: `search_artifacts/model_enhancement_v2_inter_validation_reference_m0_0713`
- M0 e50: `search_artifacts/model_enhancement_m0_inter_short_e50_0713`

결과 작성란:
- 결과는 완료 후 업데이트 예정
- V2 reference부터 `held_out_test_read=false`, sample count, checkpoint epoch/hash 확인
- M0 artifact는 manifest, log, summary, histories, validation scale-wise, plots 순서로 확인
- M0 test artifact는 생성될 수 있으나 candidate/gate 결정 전 metric 내용을 읽지 않음
- best epoch와 quantity/log-quantity, marker/time safety gate를 표로 판정
- 통과 여부와 다음 작업을 사전 gate에 따라 기록

초기 해석 가설:
- M0가 quantity와 log-quantity를 함께 개선하면 mark argmax 우회 효과로 해석하고 multi-seed로 승격
- quantity만 좋아지고 log-quantity가 나빠지면 direct quantity loss가 일부 scale에 편향됐을 가능성을 확인
- marker/time safety가 깨지면 shared encoder task interference로 해석
- e50 seed 하나의 validation 결과만으로 최종 모델 우위를 주장하지 않음

실행 상태 업데이트:
- 최초 tmux 실행: `2026-07-13 11:26:04 KST`
- 첫 시도는 M0 학습 전 V2 reference JSON 직렬화 단계에서 중단
- 원인: legacy에 적용되지 않는 `val_magnitude_loss=NaN`을 strict JSON으로 저장하려 한 문제
- 조치: decoder 비적용 metric만 JSON `null`로 export하도록 수정; 계산식과 모델은 변경하지 않음
- 첫 실패 artifact는 `_attempt_1` 경로에 보존
- 재실행 시작: `2026-07-13 11:27:45 KST`
- V2 validation-only reference 완료: samples `41,901`, NLL `5.666520`, marker NLL `0.991274`, log2 quantity MAE `0.588742`
- M0 loader 진입: train/validation/test samples `136,256/41,901/41,344`
- M0 train-only stats: events `159,643`, mean/std `1.266239/1.453546`
- 초기 GPU process: PID `2930801`, memory `710 MiB`
- 현재 상태: `in progress`
- 초기 진입 확인 후 지속 polling하지 않음

문체:
- validation 사실과 모델 해석을 구분
- held-out test를 읽거나 인용하지 않음
- 통과하지 않은 항목을 긍정적으로 포장하지 않음
