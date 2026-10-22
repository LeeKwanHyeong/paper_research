# Initial bootstrap recovery — prepared, no remote execution by this worker

The original 5080 SSH process returned **1**, with `timeout=false` and
`transport_error=null`. Its stderr was a `ModuleNotFoundError` at generated
line 79, after archive validation, extraction, and `sys.path.insert`.
This is a remote bootstrap command failure; SSH exit 255 was not observed.
All original attempt, stdout, stderr, receipt, and failure files remain intact.

A local fresh Python process with a nonexistent `PYTHONPATH` cached that path
with a `None` finder. Creating the directory/package and inserting the same
path did not repair the cache (actual exit 1). Calling
`importlib.invalidate_caches()` repaired the local import (actual exit 0).
The original remote cache was not queried; this is a locally reproduced cause
consistent with the native traceback. Raw local evidence is in
`../deployment_import_diagnostic_v1/`.

`ADMISSION.json` records recovery **within the existing explicit setup scope**.
It is not a new user response. The scientific contract, 991 scientific files,
9 inputs, 199 legacy files, deployment archive, original independent QA and
303-test receipt remain unchanged. Recovery has a separate independent review,
source pins and local state at `recovery_v1/initial_execution`.

## Required order

1. **Existing 5080 upload admission — next action after independent review.**
   One read-only SSH call, 45 seconds maximum. Recheck exact archive/contract,
   complete inventory and every SHA, no symbolic paths, no scientific phase
   attempts, native interpreter/environment/runtime/GPU/boot and paused-process
   guard. No upload, removal, overwrite or scientific launch occurs.
2. **5090 first deployment — depends on successful 5080 admission.**
   One 900-second maximum transfer of the unchanged archive. The original pure
   deployment generator is used with one explicit per-process importer-cache
   invalidation before imports. The 5090 root is created exclusively.
3. **Both-host native qualification — depends on both deployment admissions.**
   Launch only the original fixed manager qualification. Each launch is one
   45-second maximum SSH call. Each read-only check is one 45-second maximum
   call. Waiting cooldown is 900 seconds, with zero SSH before it expires.
   Preserve and verify 12 child actual exits, dispatcher actual exit, fixed
   identities, raw streams, source/runtime/boot/GPU and capacity receipts.
4. **Original initial fit/Validation — depends on both raw qualification gates.**
   Only the original 9 new single-MLP fits, 18 byte-preserving B/D reuse
   admissions and 27 Validation conditions may start. No scientific retry or
   additional Test evaluation is introduced. The returned launch evidence is
   compatible with the frozen continuation controller.

Every action requires explicit `--admission`, `--admission-sha`, `--self-sha`,
and `--recovery-qa-sha` pins. The independent review must be stored at
`recovery_v1/INDEPENDENT_QA.json` and bind this new driver and its tests.
Any unsuccessful transport or verification preserves original bytes and blocks
all further remote calls. An unfinished attempt also blocks remote calls.
The old controller state is not modified or relabelled.

The root agent owns final review, commit and actual execution. This worker
performed **zero remote calls**. Local verification: 17 focused tests passed,
full Ruff passed; all 15 files in the original source-bound QA remain unchanged.
