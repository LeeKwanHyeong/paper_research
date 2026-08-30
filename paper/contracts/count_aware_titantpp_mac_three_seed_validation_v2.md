# Count-aware TitanTPP-MAC Three-Seed Validation Contract v2

This contract supersedes v1 execution after both Instacart shards exposed the
same legacy time-head overflow. It keeps the nine-run validation grid, memory
architecture, quantity loss, data splits, optimizer, early stopping, and
checkpoint selection unchanged. The only numerical correction is Primary
Contract Amendment 2: the declared `time_intercept_limit=30` is now enforced
before the legacy exponential calculation.

All nine runs use training revision
`b1d9e638c68bc7bab9ace6b5c8fa9d0af989c8f7`. Earlier RAF and Taxi runs remain
available for diagnosis but are not imported into this corrected grid. Each
run starts in an independent Python process and records the separate
orchestration revision. The Instacart run is placed first on each server so a
stability regression is surfaced before unrelated short runs consume time.

Held-out test remains locked. Canonical completion requires finite metrics,
matching launch contracts, checkpoint digests, source manifests, and the
absence of every test or held-out artifact.
