# TitanTPP 4-page paper structure revision

Notion page: https://app.notion.com/p/3b6bbe405613813f9587dd319cfbcd93

## Purpose

This page reflects the professor feedback that the manuscript should be compressed into a 4-page short-paper structure. The goal is to reduce section count and keep contribution, baseline limitation, method, and experiment tightly connected.

## Revised paper structure

1. Introduction
2. Related Work
3. Method
   - 3.1 Problem Setup
   - 3.2 Limitations of RMTPP-style Modeling
   - 3.3 TitanTPP Architecture
4. Experiments
   - 4.1 Setup
   - 4.2 Results
   - 4.3 Ablation and Analysis
5. Conclusion

## Introduction

Use three paragraphs.

### Paragraph 1: Problem and Challenge

Introduce event-sequence-based demand prediction. The task is not only next-event-time prediction, but joint modeling of irregular event occurrence and continuous demand quantity.

Main challenges:

- long event history
- irregular event time
- sparse or intermittent demand
- continuous demand quantity

### Paragraph 2: Limitation and Contribution

RMTPP-style recurrent neural TPPs are useful baselines, but this demand setting exposes three limitations:

1. A recurrent hidden state may struggle to preserve old demand patterns and recent shifts in long event histories.
2. Categorical mark formulation is more natural for event types than continuous quantities.
3. A simple regression head can create conflict between time/event likelihood and quantity regression when both use the same representation.

TitanTPP is positioned as a long-history encoder plus transformed quantity modeling and quantity-specific heads/gradient separation.

### Paragraph 3: Experimental Summary

Use Intermittent, Taxi, and Instacart. Compare RMTPP, THP, and TitanTPP under fixed split, same seed, validation-only conditions. Avoid saying TitanTPP is always superior.

## Related Work

Keep it short and place it after Introduction.

Axes:

- Neural temporal point processes: Hawkes process, RMTPP, THP.
- Long-history sequence modeling and Titan architecture.

## Method

Problem Formulation moves inside Method as `3.1 Problem Setup`, because the current equations describe the proposed method more than a fully general task.

### 3.1 Problem Setup

Define event time, continuous demand quantity, transformed quantity, and inverse reconstruction.

### 3.2 Limitation of RMTPP-style Modeling

Explain recurrent history compression, categorical mark mismatch, and shared-representation conflict.

### 3.3 TitanTPP Architecture

Explain Titan encoder, time/event head, quantity head, transformed quantity prediction, inverse reconstruction, and gradient separation where applicable.

## Experiments

Use one combined section for setup, results, and ablation to save space.

### 4.1 Setup

- Datasets: Intermittent, Taxi, Instacart.
- Baselines: RMTPP, THP, TitanTPP.
- Training contract: fixed split, same seeds, e300 validation-only comparison.
- Final test: evaluate only once after validation-based selection.

### 4.2 Results

Use e300 validation results as the main table once all models are run under the same frozen contract.

### 4.3 Ablation and Analysis

Keep only ablation evidence that supports the paper's core contribution:

- power/quantity transform
- quantity head
- gradient separation
- long sequence sensitivity

## 2026-08-08 result update

RMTPP-matched and THP-matched e300 validation-only 18-run completed. This changes the experiment section and claim policy:

- RMTPP/THP e300 rows are final-ready validation baselines.
- Existing TitanTPP results remain draft-only.
- August 14 draft can use preliminary comparison.
- Final fair comparison table requires fresh TitanTPP e300 reruns.

Allowed claim:

> TitanTPP shows promising preliminary improvements for joint event-demand modeling. The improvement is strongest on Taxi, especially for quantity prediction. Intermittent shows lower validation NLL, but quantity improvement over RMTPP is modest. Instacart remains mixed under the current draft-only TitanTPP artifacts.

Avoid:

- TitanTPP consistently outperforms all baselines across all datasets.
- TitanTPP is superior on every metric.
- Instacart confirms the advantage of TitanTPP.
- Held-out test results confirm improvement.

## Updated figure/table plan

- T3 Main Validation Results: preliminary now, final-ready after fresh TitanTPP e300.
- F4 Validation NLL Comparison: `paper/results/e300_matched_20260808/figures/validation_nll_comparison.png`.
- F5 Quantity MAE Comparison: `paper/results/e300_matched_20260808/figures/quantity_mae_comparison.png`.
- F6 Applicability Matrix: internal planning figure, not manuscript figure.

## Updated next work

- Taxi TitanTPP V3b fresh strict e300 rerun.
- Intermittent TitanTPP V2 fresh strict e300 rerun.
- Instacart TitanTPP V2 fresh strict e300 rerun.
- Regenerate T3, F4, F5 after completion.
- Decide e800 continuation from validation convergence.
- Keep held-out test locked.
