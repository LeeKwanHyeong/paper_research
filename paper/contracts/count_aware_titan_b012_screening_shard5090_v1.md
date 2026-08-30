# Count-aware Titan B0/B1/B2 5090 shard contract

This contract moves canonical screening runs 4-9 to the 5090 server without
changing model, loss, data, seed, or checkpoint-selection behavior. The 5080
server remains responsible for Intermittent B0/B1/B2 and the final canonical
merge.

## Run boundary

- Taxi B0, B1, and B2 are canonical runs 4-6.
- RAF B0, B1, and B2 are canonical runs 7-9.
- Each backbone runs in an independent Python process.
- Every fresh process applies the orchestration-only Dynamo policy before the
  frozen runner is imported: `recompile_limit=64` and
  `accumulated_recompile_limit=512`. The compiled scans remain
  `dynamic=False`; model equations and checkpoint selection do not change.
- The 5090 artifact contains isolated shards only. It does not run the final
  comparator or materialize canonical result directories.

## Safety boundary

- The frozen training revision is
  `08e59880cd61cbd27cec40aa04636452b87bebfc`.
- Frozen training files must match the SHA-256 values embedded in the JSON
  contract, including on a server without a Git checkout.
- Every run requires at least 30000 MiB free VRAM, at most 512 MiB used VRAM,
  no CUDA compute process, and no `gnome-shell`, `Xwayland`, or `Xorg` GPU
  process.
- A failed preflight stops before model allocation and preserves prior shards.
- RAF B1 and B2 must each pass a real-data e1 train-and-validation preflight
  under the same Dynamo policy before the e300 recovery is launched.
- Held-out test artifacts remain forbidden.

## Handoff boundary

All six shards must pass launch-contract, summary, history, checkpoint-digest,
finite-metric, and validation-only checks before transfer. The 5080 recovery
launcher may resume only after the transferred shards validate under its
existing `recovery1` contract.
