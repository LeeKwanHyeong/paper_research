# TitanTPP-MAC Three-Seed Split Execution Contract v1

This contract changes only execution placement for the frozen nine-run
TitanTPP-MAC validation grid. It does not change the model, objective, data
split, checkpoint selection, or any acceptance gate in the parent contract.

The 5090 shard contains RAF, Taxi, Instacart, and Intermittent seed 52 plus the
previously missing Instacart seed 42. The 5080 shard contains seed 62 for all
four datasets. The two partitions are disjoint and their union must equal the
parent nine-run grid before canonical finalization can succeed.

Every run records its execution server. Runtime and GPU measurements from the
two servers are reported as operational evidence only and must not be treated
as a controlled model-compute comparison. A run can be reused only after the
full parent launch, summary, history, finite-metric, checkpoint-digest, and
held-out-lock checks pass.

The RAF seed-52 run that began under the parent 5090 launcher remains valid.
The parent launcher is held at the run boundary while that child finishes;
the split launcher then revalidates and reuses the completed artifact before
continuing the remaining 5090 shard.
