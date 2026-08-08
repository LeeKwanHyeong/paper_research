# Reference List

Notion page: https://app.notion.com/p/3b4bbe4056138080a4aae22bda37498b

Local working file: `paper/references/related_work_reference_register.md`

## Use policy

This page links paper claims to supporting references. It separates what the literature directly supports from what our own experiments must establish.

- 핵심 문헌: 해당 주장을 직접 뒷받침한다.
- 보조 문헌: 연구 배경이나 설계 동기를 설명할 때 사용한다.
- 자체 근거: 데이터 통계, 비교 실험, ablation 결과로 확인해야 한다.
- 외부 문헌만으로 “TitanTPP가 더 우수하다”고 결론 내리지 않는다.

## Claim 1. RMTPP history representation can be limited for long event histories

Core references:

- Du et al., 2016. Recurrent Marked Temporal Point Processes: Embedding Event History to Vector.
- Mei and Eisner, 2017. The Neural Hawkes Process.
- Zhang et al., 2020. Self-Attentive Hawkes Process.
- Zuo et al., 2020. Transformer Hawkes Process.
- Behrouz et al., 2025. Titans: Learning to Memorize at Test Time.

Allowed wording:

> RMTPP summarizes observed event history using a fixed-dimensional recurrent state. Prior attention- and memory-based sequence models suggest that richer history representations can be useful for long-range dependencies. We examine this question under demand-event sequences with substantially different history lengths.

Avoid claiming that RMTPP necessarily fails on long sequences without our own breakdown evidence.

## Claim 2. Categorical marks alone cannot reconstruct continuous quantity

Core references:

- Shchur et al., 2021. Neural Temporal Point Processes: A Review.
- Wu et al., 2018. Decoupled Learning for Factorial Marked Temporal Point Processes.
- Draxler et al., 2025. Transformers for Mixed-type Event Sequences.

Allowed wording:

> A categorical mark can represent a demand scale or regime, but it cannot preserve within-class quantity variation without an additional continuous decoder.

Avoid saying categorical marks can never model continuous values. The precise point is that a categorical head alone cannot preserve within-class quantity variation.

## Claim 3. Intermittent demand can be represented as event time plus quantity

Core references:

- Croston, 1972. Forecasting and Stock Control for Intermittent Demands.
- Türkmen et al., 2021. Forecasting intermittent and sparse time series via deep renewal processes.

Allowed wording:

> Intermittent demand is characterized by irregular nonzero arrivals separated by zero-demand intervals. This structure naturally motivates an event-based formulation that predicts both the next demand time and its quantity.

## Claim 4. Joint objectives can create gradient conflict

Core references:

- Yu et al., 2020. Gradient Surgery for Multi-Task Learning.
- Bosser, 2024. Neural Marked Temporal Point Processes for Probabilistic Predictive Modeling of Continuous-Time Event Data.

Use these as general motivation only. TitanTPP-specific mark-quantity conflict must be supported by V2/V3b ablation evidence.

## Related Work mapping

| Related Work subsection | Core references | Role |
|---|---|---|
| Recurrent neural TPPs | RMTPP 2016, Neural Hawkes 2017, Neural TPP Review 2021 | Recurrent history representation and joint time/mark modeling |
| Attention and memory encoders | SAHP 2020, THP 2020, Titans 2025 | Attention-based TPP and memory architecture background |
| Marks and continuous quantity | Factorial MTPP 2018, Neural TPP Review 2021, Mixed-type Event Sequences 2025 | Discrete marker extensions and continuous attribute modeling |
| Intermittent-demand event modeling | Croston 1972, Deep Renewal Processes 2021 | Occurrence interval plus demand size formulation |
| Joint-objective optimization | PCGrad 2020, Bosser 2024 | V2/V3b gradient routing motivation |

## Local evidence files

- `paper/tables/T1_dataset_statistics.md`
- `paper/figures/F1_F3_figure_register.md`
- `paper/results/e300_matched_20260808/result_briefing.md`
- `paper/results/e300_matched_20260808/tables/preliminary_summary.md`
