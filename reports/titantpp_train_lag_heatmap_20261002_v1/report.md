# Train event-lag correlation heatmap and manuscript integration

## Scope and status — 완료

User request: “그리고 거기에 대해 인접 수량 상관에 대한 hitmap 같은걸 추가하는거는?” followed by “그러면 이거 먼저 진행해보자”. Implemented the proposed four-dataset × event-lag 1/2/4/8/16 heatmap, Appendix B interpretation, and Results 5.4 cross-reference in the existing LNCS main.tex. No training, inference, validation replay, remote command, or held-out performance read was performed.

The source remains `paper/titantpp_pakdd_2027_draft/main.tex`; the open document was edited in place. Authors, abstract, contributions, model equations, all existing result tables, Appendix A's audited 13-condition cutoff, Appendix C, efficiency results, and references are unchanged. Built-in LaTeX compilation succeeded on the final saved source. `main.diff` records the complete change.

## Method and source linkage

- Population: frozen **train** event rows only, filtered by `chronological_split == train` before materialization; sorted by entity and sequence index. File hashing checks full source bytes, while the analytical scan materializes only train rows.
- Identity: each dataset's source SHA, split-manifest SHA, exact canonical train-target identity and quantity SHA match the frozen Core/RAF contracts. See `sources.json`, `analysis.json`, and `verification.json`.
- Statistic: pooled Pearson correlation between log(1+q[j-lag]) and log(1+q[j]), after separate entity-specific means for the two members at each lag. Every pair remains within one series. Counts are sum(max(train-series length − lag, 0)). This is pair-weighted, not the mean of per-series correlations.
- Offsets: event order, not elapsed calendar time. The descriptive analysis includes all available train-sequence pairs, without imposing the model's input window. The plotted lags are not branch indices: all eight correction branches read the immediate predecessor.
- Support: each cell reports event-pair count P and series-with-pairs count S. Single-pair series contribute zero centered sums. A missing statistic is NA, never zero.
- Sensitivity: `analysis.json` additionally records pooled correlations restricted to series with at least four pairs and nonconstant members, plus a fixed cohort with at least 20 train events (at least four pairs even at lag16). These are descriptive checks, not significance tests or confidence intervals.

## Results

| Dataset | Lag 1 | Lag 2 | Lag 4 | Lag 8 | Lag 16 |
|---|---:|---:|---:|---:|---:|
| Taxi | 0.749 | 0.546 | 0.136 | −0.191 | −0.150 |
| Intermittent | 0.928 | 0.910 | 0.834 | 0.700 | 0.473 |
| RAF | −0.105 | −0.127 | −0.055 | −0.043 | NA |
| Instacart | 0.037 | 0.049 | 0.025 | −0.002 | −0.018 |

Lag1 agrees with all four existing Appendix B values to 1e−12. Taxi has strong immediate association that falls rapidly with event offset; Intermittent remains positively associated over the displayed range. RAF has no train pairs at lag16 and only 1,655 pairs from 846 series at lag8. Its short-sequence, mean-centered negative values do not establish a stable negative dependence law.

The fixed >=20-event cohort contains 131 Taxi, 4,968 Intermittent, and 31,020 Instacart series, with no RAF series. Instacart lag1/lag2 correlations increase to 0.074/0.092, while Taxi and Intermittent retain strong immediate association. Thus the contrast is not solely a difference in the availability of long series, although this restriction changes the population and does not establish causality.

## Claim–evidence map and prose revision

| Claim in revision | Evidence | Status |
|---|---|---|
| Taxi immediate association decays, Intermittent persists through lag16 | `analysis.json`, all-train rows | Observed descriptive result |
| Instacart remains weakly correlated even among longer series | `common_cohort_min20_train_events` | Observed sensitivity result |
| RAF lacks lag16 pairs; longer-lag support drops sharply | Explicit P/S counts | Observed coverage limitation |
| Recent within-series information may favor adjacent-state correction | Train profiles + existing audited Taxi parameter-matched control | Applicability hypothesis, not causal proof |
| Distribution alone does not determine MAE/RMSE trade-offs | Unchanged nonlearned baseline table and Appendix C | Existing validation evidence, preserved |

The earlier Appendix B closing paragraph was replaced by a method paragraph, the heatmap, descriptive findings, a support/sensitivity paragraph, and an applicability paragraph. Results 5.4 now links the lag profile to the existing error decomposition and audited structural control. No claim of universal superiority, optimal branch scheduling, or additional model validation was added. Exact before/after prose is preserved in `main_before.tex` and `main.diff`.

## Figure and validation

- `figure.tex`: vector TikZ embedded in main.tex, no new LaTeX dependency.
- `heatmap.png` / `heatmap.svg`: standalone previews using the same reviewed rows, palette, counts, and signed values. These are figure files, not a separate compiled manuscript PDF.
- Common signed range −1 to +1, orange/neutral/blue scale; gray NA is distinct from numerical zero. Every cell prints its value and support counts, preserving interpretation in grayscale.
- The PNG was visually checked for cell, tick, legend, and text overlap. Label colors were revised so foreground/background contrast is at least 4.5:1. Full native PDF-page layout was not separately screenshot-audited; final native source compilation was confirmed.
- Numeric checks: all 20 cells, lag1 agreement, all train identities, pair-count formula, no cross-series pairs, independent centered-sum/dot-product agreement, finite coefficient range, missing-cell handling, and all LaTeX references passed.
- Scope check reconstructs main.tex exactly from only the two intended block replacements and the generated figure. Existing tables and unrelated text are unchanged.

Reproduce analysis and figure locally with `/usr/local/bin/python3 reports/titantpp_train_lag_heatmap_20261002_v1/analyze.py` and `build_figure.py` in the same directory. `update_manuscript.py` intentionally refuses to overwrite the saved pre-edit source. `verify.py` checks the saved manuscript and current figure source.

## Remaining work in dependency order

1. **Follow-up results — 외부 작업 대기:** wait for the already authorized experiments; this task did not query or modify the servers.
2. **Newly completed results — 다음 작업:** collect and validate their originals, then establish complete three-seed groups. Reuse prior audits.
3. **Appendix A and final interpretation — 다음 작업:** update audited comparisons and reconcile the final claims with those results. The train heatmap does not require additional training.
