# Causal QKV Hard-LMM v1

User approval: implement the selected causal convolution backbone and run training
on RTX 5090 only. Repository: `paper_research`, branch `codex/hard-lmm-causal-qkv`.

## Structural hypothesis

Learn short event-order patterns before attention by modifying only the first
Titan block's event Q/K/V. For each of Q, K and V and each hidden channel:

`x'_i = x_i + w0*x_i + w1*x_(i-1) + w2*x_(i-2)`.

Kernel size is 3; there are no convolution biases or extra gates. All weights
start at zero without consuming the base initialization RNG. With hidden64,
the added parameter count is 576. The implementation may use shifted tensor
products instead of grouped Conv1d, preserving the same linear operation.

Persistent memory tokens enter K/V after this operation and are not convolved.
Input and output masks exclude padded event projections, including QKV biases.
Left-of-history positions are zero; H1/H2 remain legal. Future targets are excluded
by the existing target_outputs contract. No cross-window/series state is added.

The second block, persistent tokens, static 64-vector top4 arithmetic-mean bank,
time and quantity heads are unchanged. Existing encoder/head parameters remain
trainable. This is an event-index local mixing hypothesis; it is not a new
physical elapsed-time encoding or a faithful Titans memory implementation.

## Local and CUDA contracts

- Identically seeded B and candidate have identical common parameter values,
  RNG state, initial outputs and common gradients at zero convolution weights.
- Each Q/K/V path including past-event coefficients receives finite nonzero
  gradients. Learning produces a measurable output change; it is not enough to
  report nonzero parameter count.
- Check prefix causality, target and padding perturbations, H1/H2, finite extreme
  inputs, checkpoint route isolation, optimizer and checkpoint roundtrip.
- CUDA tests must actually exercise CUDA. CPU tests in a CUDA environment alone
  do not qualify as CUDA evidence.
- On matched batch128/hidden64 L8,64,256 forward/backward/AdamW steps, all median
  candidate/B timing ratios must be ≤1.5 and peak allocated memory ratios ≤1.25.
  Five warmup and fifteen measured steps are repeated three times with alternating
  B/candidate order. Any failure stops the campaign before full-data learning.

## Frozen learning and adoption rules

The executable contract is `hard_lmm_causal_qkv_screening_v1.json`. Use fresh
seed42, log1p quantity MSE, lambda_qty1, tail0, AdamW lr0.001, batch128, clip1,
legacy time head cap300. Checkpoint and early stopping both use the earliest
strict minimum validation raw quantity RMSE. Screening max/min/patience are
300/40/40. The normal early-stop rule remains active.

Full-data e1 covers Intermittent, Taxi, Instacart. It checks execution, full target
counts, data/source hashes, learned lag weights, memory and checkpoint restore.
Only after all e1 audits pass, screen Instacart→Taxi→Intermittent. Stop on the first
dataset that misses any frozen B-relative condition:

- raw RMSE strictly improves;
- overall MAE regression ≤1%;
- body **≤train p95** MAE regression ≤2%;
- >train p99 MAE regression ≤2%;
- legacy clamped time loss increase ≤0.01.

Body explicitly comprises `le_p50`, `p50_p90`, `p90_p95`; `p95_p99` is excluded.
All validation target identity/quantity hashes, bin thresholds and counts must
match the pinned B evidence. The earlier original-Hard-LMM body5% objective is
a separate claim and is not a replacement for these B-relative screening rules.

Existing B is the performance control. This screening does not include a trained
pointwise-only control; any passing result is only a candidate-versus-B result.
A matched trained control is needed before claiming the improvement specifically
comes from past-event mixing rather than added capacity. Additional seeds,
held-out test evaluation, selector changes and data-specific tuning are outside
this campaign.

Reference: [Titans §4.4](https://arxiv.org/html/2501.00663v1) describes convolution
after Q/K/V projections. Applying that component does not by itself establish
novelty or an improvement on these datasets.
