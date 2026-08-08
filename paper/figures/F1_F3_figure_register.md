# Figure register: F1-F3

> Status: manuscript-ready draft figures. F1 and F2 are implementation-backed schematics;
> F3 is generated from frozen fixed-split datasets and does not use model predictions.

## F1. Problem formulation

**Purpose.** Explain why zero-heavy demand is represented as a positive-demand event sequence
and why a categorical mark alone cannot recover continuous demand quantity.

**Caption draft.** Problem formulation for quantity-aware marked event prediction. Regular
zero-heavy observations are converted into positive-demand events with inter-event times.
Each positive quantity is factorized into a tail-clipped coarse magnitude mark
`m=min(floor(log_b q), M)` and a continuous residual `r=log_b q-m`, which permits exact
reconstruction through `q=b^(m+r)`. The next-event task jointly predicts the magnitude mark,
arrival time, and continuous quantity component.

**Files.** `F1_problem_formulation.{svg,pdf,png}`

## F2. TitanTPP architecture

**Purpose.** Show the controlled RMTPP-to-TitanTPP backbone redesign and the quantity-aware
prediction path used by the frozen comparison contract.

**Caption draft.** TitanTPP receives the observed magnitude mark, log inter-event time, and
quantity residual at each event. A causal Titan encoder and static local memory matching module
produce history representations for mark, time, and residual heads. The differentiable decoder
combines mark probabilities and residual predictions to reconstruct continuous quantity and
supports the hybrid residual-plus-quantity objective. RMTPP-matched retains the quantity input,
prediction tasks, decoder, and loss while using a GRU with shared heads. For Taxi, TitanTPP V3b
adds mark-conditioned residual experts and stops the quantity-loss gradient at the mark gate;
the TitanTPP V2 control isolates this specialization from the encoder change.

**Files.** `F2_titantpp_architecture.{svg,pdf,png}`

## F2-clean. TitanTPP architecture for manuscript

**Purpose.** Provide the main paper architecture figure without internal experiment-management
language or baseline-control notes.

**Caption draft.** TitanTPP architecture for quantity-aware event prediction. Each observed
event token combines a magnitude mark, an inter-event-time feature, and a quantity residual.
The Titan history encoder produces a causal history state, which feeds mark, time, and residual
prediction heads. The quantity decoder reconstructs continuous demand by combining the mark
probabilities and residual estimates, and the training objective combines mark, time, residual,
and reconstructed-quantity losses.

**Files.** `F2_titantpp_architecture_clean.{svg,pdf,png}`

## F3. Quantity and sequence distributions

**Analytical question.** How strongly do the three datasets differ in history length and
quantity tail behavior?

**Chart contract.** Two empirical survival plots with logarithmic horizontal and vertical
scales. Panel (a) uses one observation per sequence; panel (b) uses one observation per event.
Color and line style jointly identify each dataset.

**Caption draft.** Empirical survival distributions of sequence length and positive event
quantity in the frozen datasets. Taxi contains substantially longer event histories and a heavy
quantity tail, Intermittent combines mostly short histories with rare extreme quantities, and
Instacart contains many moderate-length user histories with a narrower quantity range. Curves
describe the complete fixed datasets and do not include model outputs or held-out performance.

**Files.** `F3_quantity_sequence_distributions.{svg,pdf,png}` and
`source_data/F3_quantity_sequence_distributions.csv`.

## Rebuild

```bash
python paper/scripts/build_f1_f3.py
```
