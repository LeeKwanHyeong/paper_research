# Collector v3: GPU UUID representation repair

The two v2 GPU qualifications stopped with `Unexpected GPU UUID` before data/checkpoint loading or model inference. Their original outputs and receipts remain in `failed_gpu_qualification_v2/`.

Root's read-only `5080/IDENTITY_DIAGNOSTIC.json` and `5090/IDENTITY_DIAGNOSTIC.json` show that PyTorch 2.11.0+cu130's `_CUuuid` string contains the same 128-bit identifier as the guard's `nvidia-smi` value, with only the `GPU-` prefix absent. The guards retained the same boot, GPU and protected-process identities. This evidence identifies a representation mismatch; it does not waive GPU identity verification.

`collector_v3.py` preserves v2 and accepts only a standard 8-4-4-4-12 hexadecimal UUID with an optional exact `GPU-` prefix. Hex case is normalized and the full 128-bit value must still match. Whitespace, braces, abbreviated values, MIG or other prefixes, missing UUIDs and changed identifiers are rejected. The successful runtime record retains raw, expected and canonical values; a failed comparison writes `GPU_UUID_FAILURE.json` before the original failure path exits.

No model, checkpoint, source helper, input, loss, tolerance, native inference or output metric was changed. The filename binding is versioned separately.

`COLLECTOR_V3_TEST_01.receipt.json` records 17 synthetic tests, actual exit 0, and unchanged v1/v2 SHA values. Tests cover prefix/case equivalence, changed identity, invalid syntax and preservation of failure values. This worker did not run GPU or model inference.

Frozen collector SHA-256: `4b965ba89cbe0130dbdb51a92dd3bd7723c55ad15f661b8a18519c92e3945c89` (46,617 bytes).

Next is root's independent QA and separately bound v3 qualification.
