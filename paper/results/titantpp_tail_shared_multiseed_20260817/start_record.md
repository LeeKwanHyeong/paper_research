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
