# Retrieval necessity independent QA — 2026-10-12

## Review Result: PASS

Status: completed read-only scientific-scope review; final parent plan follow-up pending.
Summary: New retrieval experiments are not necessary for the manuscript's current bounded claim about the combined CNN+GRU54 correction under a fixed prototype-equipped backbone. Zero new fits is the recommended present scope. Prototype contribution and retrieval-rule contribution are separate questions.
Files changed: none in project. This temporary review note only.

## CRITICAL
- None.
## HIGH
- None.
## MEDIUM
- None verified in the completed-evidence claims.
## LOW
- `paper/titantpp_pakdd2027_cnn_gru_submission_draft_20261008_v1/main.tex:51` — Present-tense “For retrieval, we compare” reads as a performed comparison although lines 59 and 423 correctly mark it pending. If kept as future work, proposed exact replacement: “A separate future study could compare uniform averaging, similarity weighting, separately normalized time and quantity queries, and a general learned query.” If retrieval is excluded from this submission's scope, remove this planned-comparison sentence and keep the actual mean readout description at lines 177–186. Do not edit this source during the planning-only request.

## Verified claim/evidence boundaries
- Current three contributions (main.tex:55–57) concern architecture, correction controls, and external/Instacart comparisons. Retrieval is not a claimed completed contribution. No new search fit is required to report them.
- The static prototype bank is a learned 64×64 residual, held fixed at inference (main.tex:177–186). Correction controls alter the local/recurrent paths (173–175), while the completed 27-condition design preserves the backbone (295–300). Thus the measured correction effect is conditional on this prototype-equipped architecture; it does not isolate prototype necessity or improvement.
- Historical MLP8 prototype comparisons (main.tex:682–720) differ from CNN+GRU54 and contain adverse prototype quantity effects (713). They motivate caution, not an extrapolated CNN+GRU prototype causal effect.
- Original Q2 requires fixed to improve overall RMSE over all three other rules for every seed on the same two or more core datasets (308) and meet fixed same-GPU cost limits at both T64 and T256 (310). A small pilot cannot satisfy missing comparisons or missing datasets.
- All evidence is development-validation; no new independently unexposed evaluation or CPU checkpoint re-inference exists (436). Additional retrieval tuning on the same validation cannot close this gap.

## Conditional minimum designs (total conditions, not automatically new-fit counts)
1. Present correction claim: zero new fits. Keep static-mean retrieval as implementation context; separate planned split-query methods and results placeholders from completed contributions.
2. Prototype presence effect conditional on full CNN+GRU54: on versus bypass, three seeds per predeclared dataset. One dataset = six conditions, two = twelve, all three core datasets = eighteen. This includes a changed active parameter count; it estimates the full intervention, not memory-specific superiority at equal capacity. Add a parameter-matched non-memory residual as a third arm only for a parameter-budget attribution claim: three core datasets × three seeds × three arms = twenty-seven conditions.
3. Whether correction benefit survives without prototypes: prototype on/off × correction full/bypass × three seeds × three core datasets = thirty-six conditions. Adding current-only to preserve the equal-active-parameter question gives fifty-four. These broader interaction questions are not necessary for the current bounded claim.
4. Retrieval rule comparison: mean/weighted/fixed/general512 × three seeds. One dataset = twelve-condition pilot; two predeclared datasets = twenty-four-condition complete comparison for a newly narrowed two-dataset scope; original three-core-dataset design = thirty-six conditions. Six conditions comparing fixed against mean at one dataset is only an engineering/development pilot. Neither six nor twelve establishes original Q2. A twenty-four-condition design can evaluate the logical two-dataset consistency rule only if its selection/scope and required cost measurements are frozen in a new contract; it is not completion of the original three-dataset study.
5. Actual additional fits depend on explicit source/input/initialization/model-selection/GPU/Runtime and cost-baseline reuse audit. No automatic reuse of A100 full as an RTX same-GPU mean reference.

## Post hoc selection and interpretation
- Choosing Taxi and Intermittent because the observed history-correction criterion succeeded there is outcome-informed selection. Selecting RAF because it failed is also outcome-informed. Both are permissible exploratory motivations when disclosed; neither becomes independent confirmation by registering only future fitting commands.
- A new seed on an already development-exposed dataset does not make validation independent. Never infer statistical significance from a three-seed direction rule or a low SD alone.
- Costs must be separately measured per physical GPU/Runtime and declared settings. Concurrent fit wall times do not establish architecture efficiency.
- Excluding retrieval results from the current submission must preserve earlier design and adverse exploratory findings as historical evidence; it must not select a favorable result after running more variants.

## Recommended manuscript-scope plan (no edits performed)
- main.tex:45,49,51,59,75: focus the motivating question on correction and mark retrieval-rule research as separate future work.
- main.tex:177–186: retain actual mean prototype readout for reproducibility.
- main.tex:188–233,302–310,422–423: move unexecuted learned-query derivations, cost contract, and empty result subsection out of the core completed-results narrative; retain an external future-study plan or clearly marked appendix if useful.
- main.tex:434,439: retain the limitation that no prototype/retrieval causal advantage is established, without presenting outstanding optional retrieval as required evidence for the completed correction claim.
- Preserve all numerical results, adverse conditions, hardware migration limitations, and independent-evaluation limits.

## Acceptance criteria
- [x] Protocol sections 0, 1, 9, and 12 read.
- [x] Open main.tex SHA agrees with integration validation receipt: 73adb89b64df1f821dff8fa1b2da7cb24beed5b9a558e3530414f0aab01ecbe9.
- [x] Seven source/receipt/contract SHA references in integration_evidence.json match local bytes.
- [x] Current claim, prototype presence effect, retrieval-rule effect, and correction×prototype interaction distinguished.
- [x] Pilot, narrowed complete comparison, and original full design fit counts distinguished.
- [x] Existing observed outcomes and post hoc dataset-selection limitations retained.
- [x] No Test/heldout or mixed result contents read; source/code/manuscript changes zero; remote/training/inference zero.
- [x] Automated local SHA checks used; npm/security/type tools are inapplicable to this no-code scientific review.

The referenced .agents/skills/_shared/runtime/execution-protocols/codex.md does not exist. No .agents files were created or changed, consistent with the explicit read-only rule and parent ownership.
