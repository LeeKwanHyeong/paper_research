# TitanTPP Surprise-memory Sequential Optimization

## 상태

**실행 준비** · 5080 CUDA pre/post profiling

- 시작 일자: `2026-08-18`
- 실행 서버: `5080`
- tmux session: `titan_surprise_profile_0818`
- Source revision: profiler 준비 커밋 후 고정
- Pre artifact: `search_artifacts/titantpp_surprise_memory_profile_pre_20260818`
- Post artifact: `search_artifacts/titantpp_surprise_memory_profile_post_20260818`
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
  "env SOURCE_REVISION=<checksum_synced_commit> \
  PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
  OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/titantpp_surprise_memory_profile_pre_20260818 \
  IMPLEMENTATION_LABEL=pre_vectorization \
  bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_titantpp_surprise_memory_profile_20260818.sh"'
```

## 결과

