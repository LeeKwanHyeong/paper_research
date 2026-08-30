# Count-aware TitanTPP-MAC Semantic Optimization Contract v1

This pass may change execution only. The Titans-MAC neural-memory equations,
surprise/momentum/forgetting order, predict-before-write boundary, segment size,
parameters, heads, loss, checkpoint selection, and checkpoint keys are frozen.
The historical factory, B012 constants, and neural-memory implementation retain
their recorded SHA-256 digests. A separate execution adapter and primary runtime
apply the optimization, so prior screening artifacts remain reproducible.

The standard training forward may omit write diagnostics that no caller uses.
Its state-only recurrence is compiled separately. Final batches and final write
chunks are zero-padded to batch 128 and chunk 16 with write masks disabled for
padding, so the same graph can be reused. Real rows and token order are sliced
back unchanged after every chunk.

Acceptance requires strict loading of frozen B1 checkpoints, prediction/loss
agreement within `1e-5`, state agreement within `1e-6`, gradient agreement
within `1e-4`, causal/series tests, checkpoint replay, and finite extreme-input
forward/backward. On the RTX 5080, cold compile time, steady training-step time,
and peak VRAM are reported for batch 128 at sequence lengths 64, 84, and 256.
The B1/B0 steady-step ratio target is at most 3x and must be met before the
official three-seed expansion.
