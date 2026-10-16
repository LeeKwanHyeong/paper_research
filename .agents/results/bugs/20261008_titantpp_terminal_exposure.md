# TitanTPP terminal exposure publication mismatch

Eight completed retrieval fits were falsely uncertain because the end-of-epoch receipt hashes exposure at N Train/N Validation passes, while normal trainer completion appends one selected Validation check. The immutable final manifest covers N Train/N+1 Validation. This also affected seven fresh fits; it is not unique to interrupted/resumed fixed.

A separate sealed observer proves the sole exposure transition by restoring the exact historical raw serialization hash, validates all other receipt hashes, CPU checkpoint metadata, terminal originals and actual process shutdown. The original science/current/control and all interruption claims/failures remain unchanged.

Regression: 47 tests and 60 subtests passed, covering fresh/resumed bridge, malformed histories/exposure/SHA/scope/owner/metadata and profile binding rejection. All10 terminal and three owner archives passed; no training or new evaluation was performed.

Results: /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contextual_retrieval_pilot_results_20261008_v1/README.md
