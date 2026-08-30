# Count-aware TitanTPP-MAC Primary Contract v1 Amendment 1

## Timing and information boundary

This amendment was frozen on 2026-08-30 at 20:12:40 KST, before any seed 52,
seed 62, Instacart TitanTPP-MAC, or held-out result was inspected. It attaches
to the byte-identical v1 JSON contract with SHA-256
`ff1216aedbb73343c08b6dc3e3e576bc9962bac25d35c6ed10a3ec7b3a65beb8`.
The original contract remains immutable.

All forward-facing text uses **Count-aware TitanTPP-MAC**, shortened to
**TitanTPP-MAC**. Historical role labels may appear only inside immutable
screening artifacts as provenance and are not model names.

## Compute amendment

The 3x cost ratio remains a visible efficiency target, but it is no longer a
hard model-selection gate or a blocker for three-seed validation. The
semantics-preserving adapter kept predictions and gradients within tolerance,
yet the RTX 5080 benchmark remained above the target. This failure must be
reported as a limitation together with same-device step or epoch time, peak
VRAM, and cold compile time when applicable.

No performance, statistical, causal, artifact-integrity, or held-out rule is
changed. TitanTPP-MAC can be selected only if the original primary performance
rule and every integrity guardrail pass. Compute is reported separately and
cannot be hidden or used to relax a performance threshold.

## Official validation execution

The missing validation grid uses the exact frozen training revision
`08e59880cd61cbd27cec40aa04636452b87bebfc` and backbone
`titantpp_titans_mac`. The later semantic-optimization adapter is not used, so
new runs remain directly comparable with the existing seed-42 checkpoints.
Existing seed-42 results may be reused only after contract and checkpoint
digest verification.

The nine missing runs are Intermittent, Taxi, and RAF seeds 52 and 62, plus
Instacart seeds 42, 52, and 62. Evaluation remains validation-only and held-out
test data stays locked.
