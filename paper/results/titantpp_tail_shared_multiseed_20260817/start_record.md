# TitanTPP-T1 Three-seed Extension Start Record

- 상태: **실행 재개**
- 시작 기준 시각: `2026-08-17 19:05:17 KST`
- 실행 서버: `5080`
- tmux: `inter_tail_t1_multiseed_e300_5080_0817`
- Artifact: `search_artifacts/count_aware_tail_shared_multiseed_extension_e300_20260817`
- Source revision: `d7fb2da367fa7efbe2de232a3394c2af50c36bcf`
- Frozen `lambda_tail`: `0.09111380335463036`
- 실행 범위: TitanTPP-T1 seeds 52, 62, 최대 300 epoch, validation-only
- Held-out test: 미사용

## 실험 목적

기존 seed 42에서 통과한 T1-tail-shared를 seeds 52와 62에서 동일 조건으로 다시
학습한다. 완료 후 기존 RMTPP와 THP의 seeds 42, 52, 62 artifact와 결합하여 모델별
평균과 표준편차를 산출한다. 이 비교는 TitanTPP-T1 최종 방법의 반복 실행 안정성을
확인하기 위한 validation-only 비교이며, held-out test는 사용하지 않는다.

## 실험 계획

1. Intermittent fixed split과 train-only로 고정한 tail constants를 그대로 사용한다.
2. TitanTPP-T1의 seed 52와 seed 62를 checkpoint 재사용 없이 순차 학습한다.
3. 완료 artifact를 protocol 순서로 검증하고 기존 seed 42 결과와 결합한다.
4. Adapted RMTPP, Adapted THP, TitanTPP-T1의 3-seed 평균과 표준편차를 비교한다.

## 고정 조건

- Data SHA-256: `85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f`
- Split manifest SHA-256: `393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04`
- Epochs: `300`
- Batch size: `128`
- Learning rate: `1e-3`
- Lookback weeks: `520`
- Max sequence length: `256`
- Hidden dimension: `64`
- Tail threshold / normalization / clip cap / Huber delta: `46 / 46 / 187 / 1`
- Selection: best validation joint objective, min epochs `40`, patience `40`

## 실행 명령어

```bash
env \
  PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
  SOURCE_REVISION=d7fb2da367fa7efbe2de232a3394c2af50c36bcf \
  LAMBDA_TAIL=0.09111380335463036 \
  EXECUTION_ROLE=primary_5080_multiseed_extension \
  bash simple_lab_test/search/scripts/run_count_aware_tail_shared_multiseed_e300_20260817.sh
```

## Preflight

- RTX 5080 CUDA 인식: 통과
- Dataset 및 split checksum: 계약과 일치
- 로컬·5080 핵심 source checksum: 일치
- Shell syntax 및 focused contract test: `5 passed`
- 새 artifact 및 tmux 이름 충돌: 없음

## 결과

Seed 52 epoch 1은 train joint `1.493343`, validation joint `0.842131`, time NLL
`0.835752`, quantity MAE `0.972831`로 finite하게 완료됐다. 전체 결과는 seeds 52와
62가 모두 끝난 뒤 작성한다.

`2026-08-17` Windows 전환을 위한 서버 재부팅 요청으로 학습을 종료했다. Seed 52의
마지막 저장 지점은 epoch 26이며, `last_epoch_state.pt`의 epoch 값과 history 길이가
모두 26인 것을 확인했다. 학습 프로세스와 tmux는 종료됐고 2시간 모니터링 자동화도
일시중지했다. 동일 artifact로 runner를 다시 실행하면 epoch 27부터 재개된다.

`2026-08-17 20:39 KST`에 5080 재접속, GPU 유휴 상태, runner checksum, epoch 26
checkpoint를 재검증한 뒤 동일 tmux와 artifact로 학습을 재개했다. Seed 52 epoch 27은
validation joint `-2.943153`, time NLL `-2.948747`, quantity MAE `0.639393`으로
finite하게 완료됐고, 2시간 모니터링도 다시 활성화했다.

### 최종 완료

- 상태: **완료**
- 신규 run: seeds 52, 62 모두 성공
- Seed 52: best epoch `180`, completed epoch `220`, early stopped
- Seed 62: best epoch `259`, completed epoch `299`, early stopped
- NaN/Traceback: 없음
- Held-out test: 미사용
- 결과 결합: 기존 seed 42 및 RMTPP·THP·TitanTPP-T0 3-seed와 완료

| Model | Quantity MAE | Quantity RMSE | Time NLL |
| --- | ---: | ---: | ---: |
| Adapted RMTPP | `2.902523 +/- 0.170252` | `10.578742 +/- 0.604664` | `-3.599494 +/- 0.000000` |
| Adapted THP | `0.666380 +/- 0.081872` | `2.150737 +/- 0.555155` | `-3.599485 +/- 0.000004` |
| TitanTPP-T0 | `0.746917 +/- 0.068508` | `1.919518 +/- 0.293766` | `-3.593078 +/- 0.000661` |
| TitanTPP-T1 | `0.698884 +/- 0.059655` | `1.799715 +/- 0.182126` | `-3.593170 +/- 0.000919` |

TitanTPP-T1은 TitanTPP-T0 대비 MAE `6.43%`, RMSE `6.24%` 개선했고 time NLL도
`0.000092` 낮아졌다. Adapted THP 대비로는 RMSE가 `16.32%` 개선됐지만 MAE는
`4.88%` 높았고, seed별 MAE 우위도 `1/3`이었다. 따라서 T1은 Titan 내부 개선과
extreme tail 완화 근거는 제공하지만 THP에 대한 전 지표 우월 근거는 제공하지 않는다.

상세 결과는 `comparison.md`, 공통 조건과 재실행 판정은 `contract_audit.md`에 기록했다.

### Notion 반영

- 대상: `5. Model Design Enhancement` 하위 `TitanTPP Count-Aware Log-MSE + Tail-Aware Auxiliary Validation`
- 반영 내용: T1 3-seed 결과, T0 공통 비교 계약, 모델별 평균과 표준편차, tail 구간 해석
- 검증: 페이지 재조회 후 최종 수치와 완료 문구를 확인했고 기존 판정 보류 문구가 제거된 것을 확인함
