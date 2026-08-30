# TitanTPP-MAC Three-Seed Split Execution Contract v2

This split executes Validation Contract v2 on the 5080 and 5090. The shards
remain disjoint and their union is the same canonical nine-run grid. Every run
uses corrected training revision
`b1d9e638c68bc7bab9ace6b5c8fa9d0af989c8f7`; pre-amendment RAF and Taxi runs
are not reused.

Instacart is deliberately first on each server because it exposed the original
overflow after thousands of optimizer steps. This fail-fast order changes only
wall-clock scheduling, not model or validation semantics. RAF and Taxi follow,
and the longest Intermittent run remains last. Each run is isolated in its own
Python process, and held-out test remains locked.
