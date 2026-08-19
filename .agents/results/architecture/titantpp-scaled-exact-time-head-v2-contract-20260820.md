# TitanTPP Scaled Exact Time Head v2 Contract

## Decision

Current H0 remains a negative control. H1 is the primary stable exact head. H2 is
opened only when H1 fails the frozen train-only stability gate.

The density remains the exact RMTPP/Gompertz density in scaled time with the
original-unit Jacobian. No `w * delta_t` clamp is introduced.

## Evidence

The 2026-08-19 matched memory screening produced finite validation metrics, but
the epoch-average train joint objective had medians between `1.013e7` and
`2.981e7`. M3a reached `1.301e15`. The current parameter contract permits
`w * tau=40`, an intercept range of `[-30, 30]`, and initializes the scaled
intercept at `log(time_scale)`. For Intermittent, this implies an original-time
initial hazard near 1 even though the train mean interval is approximately 3.

## Variants

| Variant | Exact mode | `w * tau` budget | Intercept | Initialization | Time-head LR |
| --- | --- | ---: | --- | --- | ---: |
| H0 | current scaled exact | 40 | hard clamp `[-30, 30]` | `log(time_scale)` | `1.0x` |
| H1 | stable scaled exact | 8 | smooth `6 * tanh(raw / 6)` | `log(time_scale / train_mean_dt)` | `1.0x` |
| H2 | stable scaled exact | 8 | same as H1 | same as H1 | `0.1x` |

The slope bound is derived only from train targets:

```text
tau_max = train_max_dt / time_scale
time_w_max = wd_safety_budget / tau_max
```

For Intermittent, `time_scale=3`, `train_max_dt=36`, and H1/H2 use
`time_w_max=8/12=2/3`. The initial scaled intercept is
`log(3 / 2.9969021695)`, approximately `0.001034`.

## Train-Only Selection

The stability runner reads train rows only and does not instantiate a validation
or test loader. It runs H0 and H1 first. H2 is executed only when H1 fails.

H1 or H2 passes when all conditions hold:

- every loss, parameter, and gradient telemetry value is finite;
- maximum epoch-average train joint objective is at most `100`;
- p99 batch-average train joint objective is at most `100`;
- maximum per-event time NLL is at most `10,000`;
- pre-clipping gradient clipping fraction is at most `25%`.

If H1 passes, H1 is selected without running H2. If H1 fails, H2 is evaluated
against the same gate. If H2 also fails, no validation screening is opened.

## Telemetry

Each train epoch records:

- train joint, time NLL, and quantity loss means;
- batch joint p50/p95/p99 and maximum;
- maximum per-event time NLL;
- pre-clipping gradient norm mean and maximum;
- gradient clipping count and fraction;
- positive time slope and configured optimizer multiplier.

## Validation Gate

Only a train-stability-selected head may enter seed-42 validation screening. The
selected head is compared with fresh H0 on Persistent-only Titan under matched
data, seed, batch, learning rate, context, quantity head, and checkpoint rule.

- selected-head Time NLL regression versus H0 must be at most `0.01`;
- quantity MAE and RMSE regression must each be at most `2%`;
- all validation and train telemetry values must be finite;
- the selected head must still satisfy the train stability gate.

Held-out test remains locked.

## Risks

- The train maximum interval can make the slope bound conservative. This is
  accepted because H1 is a stability control, not a capacity sweep.
- A smooth intercept bound changes gradients near the boundary relative to H0,
  but does not alter the exact conditional density for the bounded parameters.
- H2 changes optimization rather than architecture. It is therefore a fallback
  ablation and not the preferred model.

## Execution Order

1. Implement the stable exact mode, train-derived initialization, optimizer
   groups, and telemetry.
2. Run focused formula, density integration, finite-gradient, routing, and
   train-only leakage tests.
3. Commit before syncing to 5080.
4. Run 5080 CUDA model-test and Instacart top-20 e1 smoke.
5. Run Intermittent train-only H0/H1 calibration, conditionally H2.
6. Start seed-42 validation only when a stable head is selected.
