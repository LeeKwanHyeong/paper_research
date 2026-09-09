# Local implementation evidence

- Source base: `1de2c31`, isolated branch `codex/hard-lmm-dual-timescale`.
- Existing training.py, core.py and CountAwareTPP.py are byte-identical to aligned baseline commit `9440609`; only the dedicated new route is registered.
- Candidate: observed transition memory between encoder layers, recent 8 vs older prefix, rank 8, per-sample state.
- Parameters: B 89,795; candidate 95,114; +5,319 (+5.92%).
- Combined local contracts: 31 passed / 14 CUDA skipped; model worker additionally reported 40 prior-route regression tests passed.
- Tests cover same-seed B identity/RNG/shared gradients, opened quantity gradients, causal prefix and target isolation, disjoint-memory reference replay, padding, finite extremes, save/restore and trainer interruption/resume, frozen gates.
- Alpha-zero identity does not imply all new weights receive nonzero gradients immediately. Quantity head and alpha must first learn. The normalized single-transition direction cancels q/k, but the absolute support gate retains query/key/write dependence in its magnitude; with no transitions the memory is exactly zero.
- No benchmark performance result has been generated locally. GPU qualification is pending.
- Local explicit global memory scope is the older part of each input window. This version has no unbounded stream API or cross-batch memory.
- Main dirty worktree, untracked scripts and existing checkpoint artifacts were untouched.

- Pre-GPU review added absolute support `-expm1(-rank*mass)` and strict actual-artifact model/AdamW restoration, including moment shapes and finite forward. Underflow and short-history monotonicity regressions passed.
