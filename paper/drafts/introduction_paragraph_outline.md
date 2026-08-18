# Introduction paragraph outline

> Status: first manuscript scaffold for the August 14 draft. This outline deliberately avoids
> universal performance claims until all validation runs and the one-time held-out evaluation
> are complete.

## Central narrative

Positive-demand forecasting requires two coupled answers: when the next event will occur and
how large it will be. Existing marked temporal point-process formulations naturally model event
times and categorical event types, but continuous, heavy-tailed demand quantity does not fit
cleanly into a categorical mark. TitanTPP addresses this mismatch by combining a long-context
causal memory encoder with a magnitude-factorized quantity interface and a differentiable
quantity reconstruction objective.

## Paragraph 1. Application problem and event-based view

**Topic sentence.** Intermittent demand, spatial mobility counts, and basket-size sequences all
contain irregular positive events separated by periods with no observed demand.

**Development.** Contrast fixed-grid regression with the event representation
`e_i=(t_i,q_i)`. Define the forecasting target as the next inter-event time and the quantity that
arrives at that event. Emphasize operational relevance without claiming a specific downstream
benefit that has not been measured.

**Evidence.** F1(a-b), T1b, and domain references for intermittent demand and marked event
forecasting `[citations needed]`.

**Closing transition.** This representation removes redundant zero observations, but it exposes
a second issue: demand quantity is continuous rather than a conventional categorical event type.

## Paragraph 2. Limits of the standard RMTPP formulation

**Topic sentence.** RMTPP supplies a principled joint model of the next event time and mark, yet
two aspects of the demand setting strain its standard formulation.

**Development.** First, a single recurrent state must summarize increasingly long histories,
which may restrict access to distant event interactions; phrase this as a modeling limitation,
not as proof of empirical failure. Second, assigning quantity to categorical bins discards
within-bin variation, while representative-quantity decoding makes predictions sensitive to bin
boundaries. Distinguish the backbone limitation from the quantity-interface limitation.

**Evidence.** F1(c), F3(a), the original RMTPP paper, recurrent long-context literature, and
marked TPP references `[citations needed]`.

**Closing transition.** A suitable demand TPP therefore requires both a stronger history encoder
and a mark definition that preserves continuous magnitude.

## Paragraph 3. Quantity-aware problem formulation

**Topic sentence.** We formulate positive demand as a magnitude-factorized marked temporal point
process.

**Development.** Introduce `m_i=min(floor(log_b q_i), M)`, `r_i=log_b q_i-m_i`, and
`q_i=b^(m_i+r_i)`. Explain that the logarithmic factorization compresses the quantity tail, the
mark retains a coarse and interpretable magnitude class, and residual regression restores
continuous variation, including values inside the merged tail class. Define the joint outputs for
mark, inter-event time, and quantity.

**Evidence.** F1(c-d), the exact reconstruction audit in `build_t1_t2.py`, and T1's dataset-specific
bases and mark cardinalities.

**Closing transition.** This interface can be shared across encoders, allowing the backbone
contribution to be evaluated under matched quantity supervision.

## Paragraph 4. TitanTPP method

**Topic sentence.** TitanTPP replaces recurrent history compression with a causal Titan memory
encoder while retaining a marked TPP output interface.

**Development.** Describe event-token construction, two causal memory-attention blocks, static
local memory matching, and the mark/time/residual heads. Introduce differentiable expected
quantity reconstruction and the hybrid objective. Mention the Taxi V3b specialization only after
the common V2 path: mark-conditioned residual experts model scale-dependent quantity behavior,
and a stopped mark-gate gradient reduces interference between quantity regression and mark
classification.

**Evidence.** F2, T2a, `models/TPPs/TitanTPP.py`, and the frozen model-source hashes.

**Closing transition.** The resulting design can be compared against RMTPP and THP without
changing the data split, quantity definition, or evaluation protocol.

## Paragraph 5. Evaluation scope and research questions

**Topic sentence.** We evaluate the formulation across three datasets selected to expose
different combinations of history length and quantity scale.

**Development.** Use F3 to characterize Intermittent, Taxi, and Instacart. State the controlled
comparisons: RMTPP-original tests the legacy quantity interface; RMTPP-matched supplies a
quantity-matched recurrent baseline; THP-matched provides an attention-based Hawkes baseline;
and TitanTPP tests the proposed encoder and quantity path. On Taxi, the TitanTPP V2 control
separates the common encoder effect from the V3b expert-head and gradient-routing changes.
Define the research questions around event likelihood, timing error, quantity error, long-history
behavior, and the V2/V3b ablation. Do not report final superiority in this paragraph until the
frozen evaluation table is available.

**Evidence.** F3, T1a, T2a-b, validation-only e300 results when complete, and the future one-time
held-out test table.

**Closing transition.** These comparisons separate gains from quantity reformulation, encoder
choice, and Taxi-specific gradient routing.

## Paragraph 6. Contributions

Draft as three precise contributions after the results stabilize:

1. A magnitude-factorized marked-event formulation that jointly represents event timing,
   categorical demand scale, and continuous within-scale quantity.
2. TitanTPP, a causal memory-augmented TPP architecture with differentiable quantity
   reconstruction and a scale-aware specialization for the Taxi regime.
3. A fixed-split, matched-comparison study across complementary demand regimes, including
   quantity, timing, event-likelihood, and architecture-ablation analyses.

Avoid contribution wording such as "consistently outperforms" or "state of the art" unless the
final held-out tests and statistical summaries directly support it.

## Figure placement

| Introduction location | Artifact | Narrative role |
| :--- | :--- | :--- |
| After Paragraph 3 | F1 | Converts the application problem into the formal prediction task |
| After Paragraph 4 or at the start of Methodology | F2 | Defines the proposed architecture, matched RMTPP baseline, and Taxi control |
| After Paragraph 5 or at the start of Experiments | F3 | Justifies the three datasets as complementary stress regimes |

## Claim-evidence map

| Planned claim | Evidence required | Current status |
| :--- | :--- | :--- |
| Categorical quantity marks lose within-class variation | F1 construction and formulation argument | Supported by definition; external citation desirable |
| RMTPP may struggle to retain distant interactions in long event histories | F3 sequence distributions, literature, and sequence-length breakdown results | Partially supported; empirical breakdown pending |
| Log-magnitude factorization preserves exact positive quantity | Reconstruction equation and frozen dataset audit | Supported |
| TitanTPP changes the encoder while matched baselines retain the quantity contract | F2, T2, source hashes | Supported |
| V3b reduces mark/quantity optimization interference | Gradient contract plus Taxi V2/V3b ablation | Mechanism supported; performance effect pending |
| TitanTPP improves final predictive performance | Three-seed validation and one-time held-out test tables | Pending; do not state yet |

## Citation queue

- Original RMTPP formulation and recurrent marked TPP likelihood.
- Transformer Hawkes Process as the attention-based comparison.
- Long-sequence limitations or information bottlenecks in recurrent models.
- Intermittent-demand event modeling and quantity transformation literature.
- Heavy-tailed count or demand regression where log/power-domain modeling is relevant.
