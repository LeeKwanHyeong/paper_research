# A100 SXM candidate screen — launch report

Observed 2026-10-02 19:43:23 KST. Three active, three queued, zero completed. One A100-SXM4-80GB; owned Pod `751nbij4r4cwm2`.

## Scope and verification
- Two capacity-matched alternatives × Taxi, Intermittent, RAF = six fresh fits, seed42. Max300/min40/patience40; earliest strict finite validation raw-quantity RMSE selects MAE/RMSE/time NLL together. Common optimizer, loss, heads, target populations, original branch availability and fixed /8 retained.
- Cross-product candidate: GELU(A current) multiplied by B predecessor, then output projection. Recent-four candidate: existing MLP combines current with mean of up to four preceding observed states. Correction parameters6144 each, zero output initialization unchanged.
- Existing completed MLP is reused, not retrained. All three original MLP initial-state hashes matched exactly on pinned native runtime; all common candidate tensors and initialization RNG matched MLP within that runtime.
- Native model correctness and all three baseline validation replays passed (prespecified cross-GPU tolerance relative1e-6/absolute1e-8). No held-out performance/predictions accessed.
- Original CPU regression39 tests passed; nine candidate/trainer/replay/resume tests passed again after preparation correction.
- Source110 files, closure `11ad8517823e9a72c5c0aa50e507e0489f95039c0fd0d91775137e502a1c6034`; canonical contract `4b1ba84f73119fce0ef57ba24954af5b3739762e3c68732a57e3801a1fad2019`.
- PyTorch2.11.0+cu130, CUDA13.0, cuDNN91900, NumPy2.4.4, Polars1.39.3. Deterministic float32, TF32 off.

## Concurrency
Equal six-job synthetic work (180 measured updates per trial; process startup excluded):

|Concurrent jobs|Work seconds|
|---|---:|
|1|7.3774|
|2|6.6719|
|3|6.2100|
|6|6.0533|

Six simultaneous jobs were feasible, with summed allocated peak8.38GiB. Three were2.59% slower for equal synthetic work; the prespecified smallest-count-within5%-of-fastest rule therefore selected3. These scheduling probes are not manuscript efficiency results or epoch ETA.

Active: Intermittent cross-product, Intermittent recent-four, Taxi cross-product. Queued: Taxi recent-four and both RAF candidates. Three real GPU worker PIDs923/925/927, GPU97%, used7838MiB. Taxi stored2epochs (48.5963s,50.9378s) and passed its initial cost gate. Intermittent's first two full epochs and queued conditions' admission gates are pending. Full queue ETA remains unconfirmed.

## Cost and ownership
GPU USD1.59/h; conservative running storage-inclusive USD1.5969444444/h. Combined total capUSD100, work allowance80, cleanup reserve20; no automatic recharge. Account balance changes include other ongoing Pods and are not attributed to this Pod.

Native campaign deadline 2026-10-04 19:27:30 KST, per-fit36h cap. Provider `stopAfter` was included in the accepted creation request for 2026-10-04 20:27:30 KST. This fallback stops compute and preserves the persistent volume. Local manager retrieves originals with SHA verification, then stops/deletes this owned Pod earlier once terminal. A stopped volume still needs final deletion. First-two-epoch admission projects remaining300epoch compute with1.2 margin plus600seconds; a failed gate is retained without an automatic fit retry.

The first preparation attempt failed before any full fit because a Mac Torch2.14 initial-state hash was required on Linux Torch2.11. Original217 files/source110 hashes were verified, Pod rt7azpnede2hzx deleted, estimated chargeUSD0.128024 included in the same cap. Corrected verification binds the native MLP initial hash to original5080 checkpoint evidence; model and learning rules unchanged. No scientific fit was restarted/resumed.

## Remaining work
1. Running: finish the six approved fits, preserving failed and unstarted conditions.
2. Next: selected/last validation replay, original retrieval and SHA verification, owned-Pod cleanup and cost report.
3. Next: checkpoint CPU audit and same-seed comparison with completed MLP. This is single-seed exploration across GPUs, not a three-seed conclusion.

No candidate result is final at launch. Existing5080/5090/PRO4500 experiments and manuscript were not modified.
