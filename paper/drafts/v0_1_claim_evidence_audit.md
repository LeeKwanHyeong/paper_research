# Draft v0.1 claim-evidence audit

## Scope

This audit covers `introduction_v0_1.md`, `problem_formulation_v0_1.md`, and
`methodology_v0_1.md`. Result-dependent claims remain outside the prose until the frozen validation
comparison and one-time held-out evaluation are complete.

## Claim status

| Claim | Evidence | Status |
| :--- | :--- | :--- |
| Positive demand can be represented by event time and quantity | F1, T1b, Turkmen et al. (2021) | Supported |
| RMTPP summarizes history in a fixed-dimensional recurrent state | RMTPP source and Du et al. (2016) | Supported |
| Attention or memory may provide a less restrictive long-history representation | THP, Titans, F3 | Method motivation supported; empirical gain pending |
| Magnitude factorization reconstructs every positive quantity | F1 equation and frozen reconstruction audit | Supported by definition and audit |
| TitanTPP employs causal attention and static learned memory in the frozen contract | F2, T2, implementation source | Supported |
| V3b blocks the quantity-loss gradient at the mark-probability gate | TitanTPP implementation and T2 gradient contract | Supported |
| V3b improves optimization or predictive performance | Taxi V2/V3b strict ablation | Pending |
| TitanTPP improves upon RMTPP or THP | Qualified validation and held-out tables | Pending |

## Prose checks

- No final superiority, state-of-the-art, or all-metric claim appears in the draft.
- Dataset descriptions rely on frozen T1 and F3 statistics, not model predictions.
- The method distinguishes static learned memory from test-time memory updates.
- The V3b text limits detachment to the quantity gate; it does not claim full loss separation.
- Display and inline math delimiters are balanced.
- The three sections contain 2,204 words before references and final results are added.

## Items to resolve before the next draft

1. Convert author-year placeholders to the bibliography style required by the conference template.
2. Insert strict validation findings only after the artifact manifest marks all selected runs as
   `final_comparison_ready`.
3. Decide whether RMTPP-original remains a primary comparison. The current 39-run T2 contract
   includes nine RMTPP-original runs, but no qualifying artifact has been identified.
4. Run strict e300 TitanTPP for every retained dataset and the Taxi V2 control. Existing TitanTPP
   artifacts remain preliminary because their epoch budget or provenance differs from the frozen
   contract.

