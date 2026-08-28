# Count-aware Titan B0/B1/B2 Screening Recovery Contract v1

## Purpose

This contract recovers the seed-42 B0/B1/B2 screening after an external GNOME
graphics session exhausted the 5080 VRAM while B1 was being initialized. It
does not change the model, loss, optimizer, data split, checkpoint selection,
or validation protocol frozen by `count_aware_titan_b012_screening_v1`.

The failed artifact remains immutable incident evidence. The successful
Intermittent B0 run is copied into a new `recovery1` artifact only after its
training revision, launch contract, finite metrics, checkpoint payload, and
canonical state digest all pass exact validation. Held-out test data remains
locked.

## Isolated Execution

The remaining eight dataset/backbone pairs run one at a time in separate
Python processes. Process exit is the CUDA-context cleanup boundary. Existing
successful shards are reused and a partial checkpoint may resume only under
the same frozen training source revision. Recovery never uses `--force-rerun`.

Single-backbone shards use the runner's experimental launch role solely to
permit process isolation. The recovery merger independently checks every
frozen T0 field and encoder contract before emitting the official
`titan_b012_screening` canonical launch contract. The training source revision
and the recovery orchestration revision are recorded separately.

## GPU Safety Boundary

Immediately before every isolated run, the 5080 must have at least 15000 MiB
of free VRAM, at most 512 MiB in use, no CUDA compute process, and no
`gnome-shell` or `Xwayland` graphics process in the NVIDIA process table. A
failed check stops before model construction and leaves existing artifacts
untouched.

The launcher uses an EXIT trap and atomic JSON writes. Any nonzero child exit
must produce a `failed` status containing the current dataset, backbone, and
exit code; a stale `running` status is not accepted.

## Canonical Completion

Canonical merge requires the reused Intermittent B0 and all eight isolated
shards. Each checkpoint state digest, summary, full validation breakdown,
source revision, and held-out lock is revalidated before copying. Only after
all nine runs are present does the original exact comparator produce the B2
acceptance decision.
