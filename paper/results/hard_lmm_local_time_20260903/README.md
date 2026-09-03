# Hard-LMM Quantity Memory / Local Time

## Implementation and Local Verification

2026-09-03, local `paper_research/master`. The subsequent user instruction
authorizes implementing the frozen route, local verification, then 5080 CUDA
contracts and full Taxi/RAF e1 smoke. It does not authorize e300 screening,
seed/dataset expansion, service changes, or push. The original contract's
authorization/status fields describe its earlier freeze phase and are preserved.

- Candidate: `titantpp_hard_memory_local_time`, role `t0_hard_memory_local_time`.
- Only route change: time receives masked local encoder output `h`, including
  persistent-token attention; quantity still receives original uniform `h+r`.
- No new parameters, loss, time head, optimizer, selector, or memory updates.
- Separate route metadata prevents silently relabeling a same-shaped checkpoint.
- Full train/validation data filtering occurs before materialization; test rows
  are excluded. The smoke wrapper rejects existing artifacts, runs each dataset
  in an isolated process, and records failures atomically without retry/resume.

Local runtime: Python 3.12.10, PyTorch 2.7.1; existing installation unchanged.
`local_tests.xml`: **128 passed, 1 skipped** across ten focused contract/regression
modules. The first regression pass found an exact supported-backbone list that
needed the new opt-in identifier; it was updated without changing legacy behavior.
`reference_route_parity.json`: pinned Taxi/RAF original checkpoint file and state
digests verified; synthetic-history quantity states/predictions match exactly.
No dataset rows, including held-out rows, were used for those checkpoint tests.

The tests cover parameter/RNG identity; single encoder execution; quantity output
and quantity-only gradient equality in train/eval; live quantity-to-prototype
gradients; absent time-to-prototype gradients; shared persistent gradients;
causal target/future/padding masking; series isolation; finite extreme-input
AdamW steps with global clipping; strict checkpoint inference replay; wrong-route
rejection; and fail-closed smoke orchestration/artifact audits. Optimizer state
serialization is checked, but exact stochastic training resumption is not claimed.

At identical weights and RNG, quantity is unchanged. After joint optimization,
time gradients alter the shared encoder and global gradient clipping, so unchanged
quantity predictions after training are **not** promised. e1 is a feasibility
check, not evidence that the original body/tail performance gates are passed.

## Next Boundary

5080 read-only preflight passed: GDM inactive, no CUDA compute processes, 15,801
MiB free VRAM, no recent accessible kernel Xid/OOM. Source transfer/CUDA/e1 are
pending at this implementation commit. The dedicated smoke entry point has no
screening option and cannot launch e300. Main Taxi/RAF seed-42 screening remains
a separate decision after completed CUDA/e1 artifact validation.
