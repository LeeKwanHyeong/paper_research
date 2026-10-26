# Collector v2: frozen import and input-layout repair

## Preserved failure

`CPU_QUALIFICATION_01` failed before model inference. The original `collector.py`, specification, raw logs and failure receipts remain unchanged. The traceback identifies two operational issues: the inherited split-manifest relative path did not resolve because its sidecar had not yet been copied, and a dynamically imported quantity helper came from the active project rather than the frozen source.

The frozen `simple_lab_test/common/pathing.py` searches ancestors for the `models`, `utils` and `sample_data` directories. The copied source retained files but lacked the empty `sample_data` directory. The bootstrap therefore added the active project to `sys.path`; Python's namespace `paper` package expanded to include it. This was an import/layout failure, not an observed model or scientific-result failure.

## Bounded repair

- Root restored the two empty `source42/sample_data` and `source52/sample_data` directories and copied the original split manifest, recorded separately in `MATERIALS_SUPPLEMENT_02.json`. No model/helper source file was edited.
- `collector_v2.py` requires exact frozen-file inventory and SHA matches and rejects symlinks. Its import finder resolves the five research package roots only from that snapshot, checks each loaded file's bound SHA and fixes namespace search paths.
- The dynamically used quantity-data helpers are imported and verified before data preparation. The original `prepare_admitted_data` and its original split filtering, order, dtype and statistics checks remain in use.
- The working directory, input data root and original sidecar identity are explicit. The sidecar is verified before data preparation and after collection. Population identities, checkpoints, native forward, models, losses, tolerances and output semantics are unchanged.

## Verification

`COLLECTOR_V2_TEST_02.receipt.json` records 13 synthetic tests with actual exit 0. They include adversarial namespace-path injection, unbound helper refusal before execution, source-inventory changes, preloaded foreign packages and sidecar mutation. `COLLECTOR_V2_TEST_01` is preserved: its two failing test assertions expected `/var/...` while the implementation correctly returned macOS's resolved `/private/var/...`; only fixture expectations changed before TEST_02.

`COLLECTOR_V2_IMPORT_01.receipt.json` records separate seed-42 and seed-52 import-only children with actual exit 0. Both loaded all required frozen helpers and the appropriate builder, verified the bootstrap root and final file inventory, and performed no data/checkpoint read, model construction or forward. Its font-cache warning remains in stderr.

Frozen collector v2 SHA-256: `f15cb58666c9962a4d062cfda32027ef0881254a426672095f8c680d78165d36` (45,356 bytes).

Next: root binds the new version in separate v2 specifications, then runs the approved CPU qualification. Full inference has not been run by this worker.
