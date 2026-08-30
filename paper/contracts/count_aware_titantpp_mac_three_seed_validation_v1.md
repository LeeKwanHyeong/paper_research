# Count-aware TitanTPP-MAC Three-Seed Validation Contract v1

This contract executes the nine validation runs that were still unseen when
Primary Contract v1 Amendment 1 was frozen. All forward-facing records use the
name **TitanTPP-MAC**. Historical screening role labels may remain only inside
the frozen implementation metadata produced by revision
`08e59880cd61cbd27cec40aa04636452b87bebfc`.

Each run starts in an independent Python process on the 5090 server. The model,
loss, time head, data split, early stopping, and checkpoint rule are unchanged
from the existing seed-42 runs. The semantics-optimization adapter is excluded
to avoid mixing implementation revisions. Elevated Dynamo recompile limits are
an execution policy only and do not alter the model equations.

The run grid is Intermittent, Taxi, and RAF seeds 52 and 62, plus Instacart
seeds 42, 52, and 62. Runs are ordered from short to long so the launcher and
artifact validator are exercised before the expensive jobs. A completed run is
reused only after its launch contract, summary, history, finite metrics,
checkpoint digest, and held-out lock are revalidated.

The launcher records the frozen training revision and the separate
orchestration revision. Held-out test remains locked, and no test summary or
test prediction artifact may be created.
