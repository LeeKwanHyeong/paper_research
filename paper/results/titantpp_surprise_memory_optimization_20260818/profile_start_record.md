# TitanTPP Surprise-memory Sequential Optimization

## 상태

**완료** · 5080 CUDA pre/post profiling 및 speed gate 통과

- 시작 시각: `2026-08-18 12:52:17 KST`
- 실행 서버: `5080`
- 완료 시각: `2026-08-18 13:15:26 KST`
- tmux sessions: `titan_surprise_profile_0818`,
  `titan_surprise_profile_post_0818`, `titan_surprise_profile_compiled_0818`,
  `titan_surprise_profile_fullseq_0818`
- Source revision: `8c8a1b37286b2dc77d64831484097fea8fd995fd`
- Final implementation revision: `8bfaee168ea95046ee0de74f43a84a37bbb97e9f`
- CUDA regression revision: `573205f`
- Pre artifact: `search_artifacts/titantpp_surprise_memory_profile_pre_20260818`
- Final artifact:
  `search_artifacts/titantpp_surprise_memory_profile_post_fullseq_20260818`
- Held-out test: 미사용

## 목적

Surprise-memory의 순차 fast-weight 갱신에서 발생하는 CUDA 병목을 분해하고,
causal retrieval, padding 무시, chunk 단위 gradient detach 계약을 유지하면서 학습
속도를 개선한다. 성능 지표를 선택하거나 validation/test 데이터를 사용하는 실험이
아니며, 동일한 synthetic shape에서 구조별 forward와 backward 시간만 비교한다.

## 프로파일링 계약

| Shape | Batch | Sequence | 용도 |
| --- | ---: | ---: | --- |
| Intermittent | 128 | 16 | 현재 주 데이터의 짧은 sequence |
| Instacart | 128 | 64 | e1 smoke에서 확인된 주요 병목 재현 |
| Long | 32 | 256 | 긴 sequence에서 순차 비용 증가 확인 |

- 공통 hidden dimension: `64`
- 비교 backbone: hard-LMM, no-memory, gated soft-memory, surprise-memory
- 각 측정: warmup 5회, 본 측정 20회
- profiler: Instacart shape에서 surprise-memory 학습 step 3회
- memory residual gate: `0.5`로 열어 실제 gradient 경로 측정
- acceptance: 세 shape 모두 surprise/hard-LMM 학습 step 비율 `3.0x` 이하
- pre/post 비교는 동일 profiler와 synthetic seed를 사용

## 실행 계획

1. 현재 구현을 pre artifact에 측정한다.
2. projection과 gate를 sequence 단위로 벡터화한다.
3. 일반 forward에서는 diagnostics 생성을 생략한다.
4. 기존 process 결과와 gradient 계약을 focused test로 검증한다.
5. 같은 profiler를 post artifact에 실행하고 acceptance를 판정한다.

## 실행 명령어

```bash
ssh 5080 '/usr/bin/tmux new-session -d -s titan_surprise_profile_0818 \
  "env SOURCE_REVISION=8c8a1b37286b2dc77d64831484097fea8fd995fd \
  PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
  OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/titantpp_surprise_memory_profile_pre_20260818 \
  IMPLEMENTATION_LABEL=pre_vectorization \
  bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_titantpp_surprise_memory_profile_20260818.sh"'
```

## 결과

최적화 전에는 event마다 LayerNorm과 projection을 반복했고, 일반 학습 forward에서도
사용하지 않는 diagnostics를 계산했다. 1차로 event-local projection을 sequence 단위로
벡터화하고 query/key read를 하나의 batched operation으로 합쳤다. 2차로 CUDA에서
전체 sequence scan을 하나의 static graph로 compile했다. Gradient truncation은 graph
안에서 32 event마다 state를 detach하는 기존 계약을 유지한다.

| Shape | Pre Surprise train step | Final Surprise train step | Speedup | Final hard-LMM | Final ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| B128 x L16 | `16.725 ms` | `3.740 ms` | `4.47x` | `2.751 ms` | `1.36x` |
| B128 x L64 | `58.874 ms` | `4.779 ms` | `12.32x` | `2.912 ms` | `1.64x` |
| B32 x L256 | `236.052 ms` | `9.653 ms` | `24.45x` | `3.698 ms` | `2.61x` |

- Speed acceptance: **PASS**. 최대 Surprise/hard-LMM 학습 step 비율은 `2.6103x`로
  사전 기준 `3.0x` 이하다.
- Peak CUDA memory: L16 `50.1 MiB`, L64 `238.2 MiB`, L256 `366.7 MiB`로 같은
  shape의 hard-LMM보다 높지 않았다.
- CUDA operator profile: Instacart 3 step 기준 `aten::bmm` 호출이 벡터화 후
  `606`회에서 full-sequence compile 후 `36`회로 감소했다.
- Cold-run total: 세 shape compile, warmup, 측정, operator profile을 모두 포함해
  약 `4분 58초`가 걸렸다. 반복 학습 step timing에는 최초 compile 시간이 포함되지
  않으므로 이 경로는 e300 같은 장기 학습에 사용하고 짧은 smoke에서는 compile
  비용을 별도로 기록한다.
- Regression: 로컬 `61 passed, 1 skipped`, 5080 CUDA `24 passed`. CUDA에서
  compiled/eager output, input gradient, parameter gradient가 허용 오차 내에서
  일치했다.
- Held-out test와 validation 성능은 사용하지 않았다. 이번 결과는 runtime gate만
  통과한 것이며 모델 성능 우위는 다음 Intermittent seed-42 screening에서 판단한다.
- Notion: `5. Model Design Enhancement`의
  `2026-08-18 TitanTPP Surprise-memory Speed Optimization` 페이지에 결과를 직접
  반영하고 완료 상태, 최종 revision, artifact, speed gate와 CUDA test를 재조회했다.
