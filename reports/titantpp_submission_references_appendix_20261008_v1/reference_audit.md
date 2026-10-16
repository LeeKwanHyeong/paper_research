# Reference restoration audit — 2026-10-08

Status: completed (scholarly source audit and insertion fragment). Parent session owns manuscript integration.

Question: Which omitted references are necessary to restore the interdisciplinary argument without inflating novelty or claiming unperformed comparisons?

## Scope and evidence

Read only the old bibliography (main.tex lines 1833–1930), the new bibliography (lines 307–343), session protocol §0, and LITERATURE_COMPARISON.md. No local held-out, Test, or mixed Validation/Test performance was read. No GPU or experiment was run. The `.agents/` tree was not modified; this authorized report location supersedes the generic results path.

Route: web / primary academic sources. Docs/code plane: existing bibliography and prior scholarly memo only, no code audit. Community/market plane: not applicable and not searched. Academic plane: targeted restoration of 19 works, not exhaustive novelty search. Existing nine references were retained; their full methods were not independently re-audited here.

Trust: each finding below is **academic / primary**. Proceedings/publisher identity and author preprints establish provenance, not truth of every empirical claim. Strict oma-search registry has no numeric pattern for most publisher/arXiv domains, so their registry label is **unknown**, not a fabricated 0.95 score. The Washington author-hosted paper has **verified / 0.90** institutional domain trust. Confidence is high for the displayed metadata and narrowly worded methodological claims. No superiority or first-of-its-kind claim is inferred.

## Recommended restoration: 9 existing + 19 restored = 28

| Citation key | Supported use / limitation | Evidence |
|---|---|---|
| `croston1972forecasting` | Separate demand size and occurrence-frequency estimates motivate intermittent-demand modeling. | [Primary](https://link.springer.com/article/10.1057/jors.1972.50) |
| `syntetos2005accuracy` | SBA corrects the bias of Croston-style demand estimates; does not establish neural-method novelty. | [Primary](https://www.sciencedirect.com/science/article/pii/S0169207004000792) |
| `teunter2011obsolescence` | TSB updates occurrence probability each period and demand size after positive demand, addressing obsolescence. | [Primary](https://www.sciencedirect.com/science/article/pii/S0377221711004437) · [author text](https://pure.rug.nl/ws/portalfiles/portal/145394864/Intermittent_demand_Linking_forecasting_to_inventory_obsolescence.pdf) |
| `hawkes1971spectra` | Classical self-exciting point-process lineage. | [Primary](https://academic.oup.com/biomet/article-abstract/58/1/83/224809) |
| `mei2017nhp` | Continuous-time LSTM conditions event-type intensities on history. | [Primary](https://papers.nips.cc/paper_files/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html) |
| `zuo2020thp` | Self-attention captures temporal dependencies in neural TPPs. | [Primary](https://proceedings.mlr.press/v119/zuo20a.html) |
| `zhang2020sahp` | Self-attention summarizes event history and incorporates inter-event intervals. | [Primary](https://proceedings.mlr.press/v119/zhang20q.html) |
| `shchur2021review` | Neural TPP design overview; supports distinction among intensity, cumulative intensity, and density formulations. | [Primary](https://www.ijcai.org/proceedings/2021/623) |
| `shchur2020intensityfree` | Directly parameterizes conditional inter-event-time distributions; single lognormal head is not their full mixture model. | [Primary](https://openreview.net/forum?id=HygOjhEYDH) · [author text](https://arxiv.org/abs/1909.12127) |
| `omi2019fully` | Parameterizes integrated intensity and differentiates it; not the same as direct duration regression. | [Primary](https://proceedings.neurips.cc/paper/2019/hash/39e4973ba3321b80f37d9b55f63ed8b8-Abstract.html) |
| `salinas2020deepar` | Global autoregressive recurrent probabilistic forecasting on related time series; not an event-only model. | [Primary](https://www.sciencedirect.com/science/article/pii/S0169207019301888) |
| `behrouz2025titans` | Titans includes adaptive contextual memory at evaluation time as well as fixed persistent memory; current offline prototypes do not reproduce its adaptive mechanism. | [Primary](https://papers.nips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html) · [author text](https://arxiv.org/html/2501.00663v1) |
| `he2016residual` | Residual transformations are established architecture primitives. | [Primary](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) · [author text](https://arxiv.org/abs/1512.03385) |
| `houlsby2019adapters` | Adapters provide compact task-specific transformations with fixed original weights; current joint training does not reproduce this frozen-backbone setting. | [Primary](https://proceedings.mlr.press/v97/houlsby19a.html) |
| `bachlechner2021rezero` | Zero-initialized scalar residual gates are established; zero-initialized projection and ReZero scalar gate are distinct mechanisms. | [Primary](https://proceedings.mlr.press/v161/bachlechner21a.html) |
| `draxler2025flextpp` | Mixed-type MTPP with discrete and continuous prediction heads demonstrates numeric marks are compatible with TPPs. | [Primary](https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html) |
| `sukhbaatar2015memory` | End-to-end recurrent attention over external memory precedes current retrieval; does not establish identical storage semantics. | [Primary](https://proceedings.neurips.cc/paper/2015/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) |
| `gneiting2007proper` | Proper scores evaluate predictive distributions; a point-regression loss does not by itself validate distribution calibration. | [Primary](https://doi.org/10.1198/016214506000001437) · [author text](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) |
| `hyndman2006accuracy` | Forecast-error metrics have differing scaling and degeneracy properties; no claim that current protocol implements MASE. | [Primary](https://doi.org/10.1016/j.ijforecast.2006.03.001) · [author text](https://robjhyndman.com/papers/mase.pdf) |

The accompanying JSON retains complete bibitem metadata and DOI fields only where present and corroborated. `bibliography_additions.tex` preserves old keys and entries. The fragment has no bibliography wrapper. Change the manuscript wrapper from `{9}` to `{28}` and cite all included items substantively; do not use `nocite{*}` as a substitute for literature positioning.

## Suggested manuscript language

Intermittent demand involves both the occurrence of demand and its positive size. Croston separately estimates these components, SBA adjusts bias, and TSB updates occurrence probabilities in zero-demand periods [Croston; SBA; TSB]. DeepAR provides a related global probabilistic forecasting approach on time series, while deep renewal processes directly model demand intervals and sizes [DeepAR; deep renewal]. Consequently, jointly using interval and quantity history is established prior work, rather than a novelty claim of this study.

Temporal point processes model event arrivals; they are not intrinsically restricted to categorical marks [TPP review]. Common neural benchmarks and models use finite event-type vocabularies [NHP; THP; SAHP], whereas continuous numerical attributes can also serve as marks, as explicitly demonstrated by FlexTPP [FlexTPP]. Our setting omits a categorical event-type identifier and retains numerical quantity as an event attribute. Accordingly, “without categorical event-type labels” is preferable to an unqualified “mark-free.” Integer counts are numeric marks, not categorical types merely because their support is discrete. Continuous-time arrivals and continuous-valued marks are also distinct concepts.

Hawkes models established self-exciting event dynamics. Neural approaches include recurrent history encoders, continuous-time recurrent state, self-attention, and continuous-time state-space models [Hawkes; RMTPP; NHP; THP; SAHP; A-NHP; S2P2]. Intensity-free models directly parameterize conditional waiting-time distributions; fully neural TPPs parameterize integrated intensity [intensity-free; fully neural]. Our common output heads define an adaptation comparison and do not reproduce each original model’s native likelihood.

Residual mappings, compact adapters, and zero-initialized residual gates are established mechanisms [ResNet; adapters; ReZero]. The present question concerns the incremental value of specific local and recurrent correction paths under a common forecasting contract, rather than the invention of residual learning. A zero-initialized output projection should not be equated with ReZero’s scalar gate, and jointly trained correction paths should not be described as reproducing frozen-backbone adapter tuning.

Content-based memory retrieval is established in memory networks, persistent memory, and product-key memory [memory networks; persistent memory; product-key memory]. Titans combines fixed persistent parameters with contextual memory that continues adapting during inference [Titans, §§3.1–3.3, 4.1]. The present bank instead consists of offline-learned prototypes fixed during evaluation. This difference concerns the update mechanism; it does not justify calling all Titans memory adaptive or asserting that this bank stores actual historical events.

Point-forecast accuracy and predictive-distribution assessment answer distinct questions [Hyndman–Koehler; Gneiting–Raftery]. A quantity-regression objective should therefore not be presented as a validated quantity likelihood. Proper-score theory does not automatically certify clipped losses, a weighted mixed objective, calibration, or comparability across different targets. Report the exact quantity loss, duration score, transformation, aggregation and support separately.

These paragraphs are concise editorial suggestions, not a claim that the audited manuscript or code currently satisfies each contract.

## Excluded old references and reasons

- PyTorch, GELU, LayerNorm: component provenance rather than the central scientific positioning; restore only if their implementation choices are substantively discussed.
- HiPPO: no direct polynomial-projection contribution or model comparison in the current scope.
- PatchTST and iTransformer: regular-grid forecasting context is already represented by DeepAR; current event-only comparison does not evaluate these native models.
- EasyTPP: useful benchmarking infrastructure, but current audit establishes no use of its pipeline or benchmark contract; citing it as implementation evidence would be unsupported.
- Existing references to attention, AdamW, GRU, S2P2, RMTPP, A-NHP, deep renewal, persistent memory and product-key memory remain; do not duplicate the old A-NHP key `yang2022attnhp` alongside current `yang2022anhp`.

## Access and coverage limitations

All 19 proposed restored works have primary-source metadata/method evidence, but access depth varies. Direct page opening failed for SBA, TSB, DeepAR, the ResNet CVF page, the ICLR intensity-free page, and the two evaluation DOI links. SBA/TSB/DeepAR/Hyndman metadata were recovered from indexed publisher pages; TSB additionally from an institutional full-text copy; DeepAR, ResNet and intensity-free from author arXiv papers; Gneiting–Raftery from the author’s institutional PDF. Proceedings/abstract-level evidence was sufficient for broad positioning; it is not a full reproduction audit. Titans method text was checked directly. No paid sources were dispatched. No community claims were used. The closest time–quantity multi-view retrieval literature remains outside this targeted bibliography restoration; do not conclude novelty from absence in this set.

## Next integration checks

1. Parent: insert the 19 entries and connect citations to concrete claims.
2. Parent: preserve numeric-mark wording and native-model-versus-adaptation distinctions in the appendix.
3. Parent: compile and check duplicate keys, unresolved citations, bibliography count, and layout. This agent did not compile the manuscript.
