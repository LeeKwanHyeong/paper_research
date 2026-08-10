# v0.4 Introduction revision notes

Date: 2026-08-10 KST

## Revision target

- Source draft: `paper/titantpp_short_paper_draft_v0_3_manuscript.md`
- Revised draft: `paper/titantpp_short_paper_draft_v0_4_manuscript.md`
- Root manuscript copy: `README.md`
- Scope: Introduction only

## What changed

1. Opening paragraph was rewritten to start from the demand forecasting context.
   - Previous version began with a broad description of sparse demand.
   - v0.4 now opens with slow-moving items, spare parts, and long-tail retail products.
   - The paragraph states the two forecasting questions earlier: next event time and next positive quantity.

2. The transition from TPP to RMTPP was expanded.
   - Previous version moved directly from neural TPPs to RMTPP.
   - v0.4 first explains that neural TPPs improve classical TPPs by learning history representations from event trajectories.
   - RMTPP is then introduced as a representative recurrent neural TPP.

3. The third challenge was reframed.
   - Previous version could read as if joint learning itself were the limitation.
   - v0.4 identifies the sharper issue: rare large-demand events and gradient scale can interfere with time and mark likelihood learning.

4. TitanTPP is introduced before its contributions are discussed.
   - Previous version used TitanTPP immediately after listing limitations.
   - v0.4 explicitly defines TitanTPP as a quantity-aware TPP for intermittent demand events.
   - The contribution sentence now links Titan-style long-history encoding, transformed quantity reconstruction, and quantity-gradient separation to the three challenges.

## Remaining follow-up

- Method feedback should be reflected next, especially if the professor requests a clearer connection between problem formulation and architecture.
- Experiments should be updated after the final Instacart e300 run finishes.
