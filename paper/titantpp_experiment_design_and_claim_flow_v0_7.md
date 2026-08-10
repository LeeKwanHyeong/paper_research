# TitanTPP Experiment Design and Claim Structure

## 1. Purpose of This Document

This document defines the argument of the TitanTPP paper and the experiments required to support it. TitanTPP combines two proposed components:

1. a memory-augmented Titan backbone for representing longer event histories; and
2. an exponent-residual representation for reconstructing continuous, long-tailed demand quantities.

These components must be evaluated separately before their combined performance is reported. Otherwise, a performance difference cannot be attributed to either the history encoder or the quantity representation.

## 2. Problem Context

Intermittent demand consists of irregularly timed positive-demand events whose quantities can vary substantially. A useful model must therefore predict both the next event time and its associated count from the observed event history.

This setting creates two central challenges.

- **History representation:** Event sequences can contain both recent local changes and recurring patterns separated by many events. Compressing the entire prefix into a recurrent state may weaken access to older information.
- **Quantity representation:** Demand quantity is a non-negative count with a long-tailed distribution. A categorical event mark does not directly preserve continuous within-class variation, while a direct regression target may be sensitive to the target transformation and output constraint.

TitanTPP addresses these challenges with a memory-augmented causal encoder and an exponent-residual quantity decoder.

## 3. Contribution 1: Memory-Augmented History Encoding

### 3.1 Motivation

RMTPP processes events sequentially and summarizes the observed prefix in a fixed-dimensional recurrent state. Information from an older event must pass through multiple recurrent transitions before it contributes to a later prediction. This creates a long computational path and a single-state information bottleneck.

THP reduces this bottleneck through causal self-attention, which gives each position direct access to earlier event tokens within the observation window. THP is therefore an essential baseline: outperforming only RMTPP would not be sufficient evidence that the proposed memory-augmented encoder improves on attention-based history modeling.

The current TitanTPP `small_lmm` configuration combines causal event-token attention with learnable persistent memory and an LMM memory bank. Each encoded event token retrieves the top-k most similar learned memory vectors and adds their aggregate to the token representation. This provides a learned set of recurring prototypes in addition to direct causal interactions among observed events.

### 3.2 Bounded Architectural Claim

The current experiment uses `max_seq_len=96` and `memory_mode=static_lmm`. Its memory is learned with the model and is not an unbounded, series-specific record of prior events. The paper must therefore avoid claims of unlimited long-term memory or test-time memorization.

A defensible claim is:

> TitanTPP improves the representation of longer event histories within a fixed observation window through causal attention and learnable memory retrieval.

### 3.3 Controlled Backbone Experiment

The quantity representation, loss, split, seeds, checkpoint rule, and training budget remain fixed. Only the history encoder changes.

| Model | History encoder | Quantity representation |
|---|---|---|
| Adapted RMTPP | GRU | Exponent + residual |
| Adapted THP | Transformer | Exponent + residual |
| TitanTPP | Memory-augmented causal encoder | Exponent + residual |

The RMTPP and THP rows are variants of the original models. Their original history backbones are retained, but their output interface is adapted to the quantity-bearing demand task.

### 3.4 Evidence Required

Overall validation averages do not establish a long-history advantage. Performance must also be reported by available history length, for example:

- 1-16 observed events;
- 17-32 observed events;
- 33-64 observed events; and
- 65-96 observed events.

The exact boundaries may instead be derived from the training distribution, but they must be fixed before validation analysis. The long-history claim is supported only if TitanTPP's improvement is consistent across seeds and becomes meaningful in the longer-history strata. A comparison with THP is required because THP already provides direct attention over prior events.

### 3.5 Dataset Roles

- **Intermittent 5,000-series dataset:** Primary controlled backbone experiment because the three models use the same quantity interface and contract.
- **Taxi:** Evidence for the combined model, subject to careful interpretation. The current Taxi configuration changes residual-head and gradient-routing choices in addition to the backbone, so its result cannot be attributed solely to the Titan encoder.
- **Instacart:** Auxiliary evidence unless its average and tail-stratified results support a clear, reproducible claim.

## 4. Contribution 2: Exponent-Residual Quantity Modeling

### 4.1 Motivation

Standard marked TPP models usually predict a categorical event type. Demand quantity, however, is an ordered non-negative count. Mapping quantity to a categorical mark discards variation within each bin. Direct regression preserves continuity, but its performance depends on the target scale and output support.

TitanTPP decomposes quantity into a coarse exponent mark and a continuous within-scale residual. The exponent represents the approximate magnitude, while the residual reconstructs variation inside that magnitude range.

### 4.2 Why Adapted Baselines Are Necessary

Original RMTPP and THP do not natively produce the proposed continuous quantity reconstruction. To compare history backbones fairly, both baselines receive the same exponent-residual interface used by TitanTPP. These models must be described as **Adapted RMTPP** and **Adapted THP**, or explicitly as RMTPP and THP variants for the demand-quantity task.

This backbone comparison cannot establish the value of exponent-residual modeling because every row already uses it. A separate representation experiment is required.

### 4.3 Controlled Quantity-Representation Experiment

The RMTPP backbone is fixed and only the quantity target, output head, and reconstruction rule change.

| Quantity interface | Purpose |
|---|---|
| Uniform categorical binning | Tests equal-width discretization and tail sparsity |
| Quantile categorical binning | Tests balanced bins and within-bin information loss |
| Min-max scaling + sigmoid regression | Provides a bounded, non-negative regression baseline |
| Log-scale regression | Provides a strong conventional baseline for skewed quantities |
| Exponent + residual | Tests the proposed coarse-scale and within-scale decomposition |

The previous unconstrained raw-MSE experiment is not sufficient as a main baseline. Negative predictions indicate an avoidable output-design problem rather than an inherent limitation of regression.

### 4.4 Common Evaluation

Quantity thresholds must be computed from the training split only. Validation events should then be divided into common strata such as:

- below p50;
- p50-p90;
- p90-p95;
- p95-p99; and
- at or above p99.

All interfaces must use identical stratum membership and counts. The common metrics are quantity MAE, RMSE, and WAPE within each stratum. Event-time metrics should also be checked to ensure that a quantity interface does not degrade the temporal task.

The full training NLL should not be used to rank categorical and regression interfaces when their likelihood or objective definitions differ. Mark accuracy is likewise not a common metric for regression interfaces.

### 4.5 Interpretation Rules

- If exponent-residual modeling outperforms log-scale and bounded regression in the upper strata, the proposed quantity contribution is supported.
- If it outperforms uniform and quantile binning but not log-scale regression, the evidence supports an advantage over categorical discretization only.
- If overall errors are similar but exponent-residual modeling improves p95-p99 and p99+ errors, the contribution should be framed as a long-tail improvement rather than universal accuracy superiority.
- If bounded or log-scale regression performs better across the distribution, the necessity of exponent-residual modeling is not established in its current form.

## 5. Paper Argument

The paper should proceed in the following order.

1. Introduce intermittent demand as irregular event timing combined with non-negative, long-tailed event quantities.
2. Explain that neural TPPs learn event trajectories, but recurrent history compression and categorical marks do not directly address both challenges.
3. Introduce TitanTPP as the combination of a memory-augmented causal history encoder and an exponent-residual quantity decoder.
4. Evaluate the history encoder while holding the exponent-residual interface fixed across Adapted RMTPP, Adapted THP, and TitanTPP.
5. Evaluate the quantity representation while holding the RMTPP backbone fixed across categorical and regression alternatives.
6. Report the combined TitanTPP model in the main performance table after the two component-level effects have been separated.
7. Restrict the conclusion to claims directly supported by the controlled and stratified results.

## 6. Experiment-to-Claim Matrix

| Research question | Controlled factor | Changed factor | Required evidence |
|---|---|---|---|
| Does the Titan backbone improve longer-history modeling? | Exponent-residual interface and training contract | RMTPP, THP, or Titan encoder | Three-seed results by history-length stratum |
| Does exponent-residual modeling improve long-tail quantity prediction? | RMTPP encoder and training contract | Quantity representation and output head | Three-seed MAE/RMSE/WAPE by quantity stratum |
| Does the complete TitanTPP model improve the demand-event task? | Dataset split, seeds, and checkpoint rule | Complete model family | Main validation and final held-out test table |

## 7. Current Decision Boundary

The final manuscript should not state that TitanTPP captures long sequences better merely because its overall NLL is lower. It must show a controlled advantage in longer-history strata. Likewise, positivity alone is not sufficient evidence for exponent-residual modeling because bounded regression can enforce positive outputs. The quantity contribution requires a meaningful advantage over log-scale regression, particularly in the upper-quantity strata.

If either controlled experiment does not support its corresponding contribution, the motivation and conclusion must be narrowed rather than presenting the combined model as universally superior.
