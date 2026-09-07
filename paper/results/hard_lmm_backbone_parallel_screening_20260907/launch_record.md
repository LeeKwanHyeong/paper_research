# Hard-LMM backbone parallel screening launch record

## Frozen execution identity

- Launch observed: `2026-09-07 16:17:30 KST`
- Source revision: `ba9c297155fd87b8bdfe2e23bee5369c0199022b`
- Source archive SHA256: `554cc6319b77f2b055db18b906ddf62b4e3c82e32ea97494fdb2c500d4e27a6d`
- Screening contract SHA256: `7fc71597beba933bab4bfd5dc99ab2ffa7db54b2d5f8065e5b025baa16045f98`
- Evaluation scope: validation only; held-out test remains locked.

## 5090: inter-layer Hard-LMM

- SSH role: `5090` (`NVIDIA GeForce RTX 5090`)
- Source snapshot: `/home/leekwanhyeong/workspace/paper_research_interlayer_ba9c297_5090`
- Artifact root: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/hard_lmm_interlayer_ba9c297_5090_20260907`
- tmux session: `hardlmm_interlayer_ba9c297`
- Remote focused tests: `22 passed`
- Initial state: `running_e1`; CUDA process observed with 2530 MiB allocated.

## 5080: Memory-FiLM Hard-LMM

- SSH role: `5080` (`NVIDIA GeForce RTX 5080`)
- Source snapshot: `/home/leekwanhyeong/workspace/paper_research_memory_film_ba9c297_5080`
- Artifact root: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/hard_lmm_memory_film_ba9c297_5080_20260907`
- tmux session: `hardlmm_memory_film_ba9c297`
- Remote focused tests: `21 passed`
- Initial state: `running_e1`; CUDA process observed with 2256 MiB allocated.

## Campaign order and stop boundary

Each host runs full-data e1 for Intermittent, Taxi, and Instacart. A candidate that
passes all three execution audits continues with seed-42 screening in the order
Instacart, Taxi, Intermittent. The runner stops that candidate at the first frozen
gate failure. It does not run additional seeds or held-out test evaluation.

## Superseded startup attempt

The first `9b45929` startup exposed a contract-only role spelling mismatch for
Memory-FiLM before its training process started. The partial 5090 e1 process was
stopped so both candidates could restart from the same corrected revision and
contract digest. The superseded artifact roots ending in `9b45929` are not result
evidence and must not be aggregated with this campaign.

## Monitoring

The active hourly heartbeat is `hard-lmm-backbone-gpu-screening`. It reports only
phase changes, terminal results, failures, or required user action, and pauses
after both campaigns reach a terminal state.
