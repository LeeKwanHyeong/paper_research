# CNN+GRU54 Methods and claims audit — 2026-10-11

Read-only audit; no repository files changed, remote calls/training/checkpoint deserialization/held-out reading0. TEST_SESSION_PROTOCOL0/9/12 and Serena initial instructions read. Serena symbol reads returned no matches for ignored frozen-artifact paths, so focused file/line reads were used.

## Frozen evidence identity

- history canonical518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b recomputed by stdlib JSON SHA; source129 every file SHA matches contract. revision57da14a2a7a5687e1bdef7d3164f3e8c0051da59; sourceclosurec165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a.
- Instacart frozen science revision4274043717cf24640d9a18fb35231a1f7ba50c8e, operational revision57da14a...; source126 every file SHA matches its contract and all126 files exactly match their history129 counterparts. Instacart closure918f7cb883b62cbdc6f3f178cc53b6e20aac2e71a49b6e036769aa2fea76155a. This is frozen source equivalence, not equal fitted weights/trajectories.
- Source root H = search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/source/.

## Required prose fixes

1. Introduction says removal also removes parameters. Implementation retains the registered25308 parameters, freezes them, and bypasses both paths; active correction0. Proposed sentence: “Bypassing the correction removes its active trainable capacity, so that comparison alone cannot identify the benefit of explicit history input.” H/models/TPPs/CountAwareTitanCNNHistoryControls.py47–77,222–233; disabled total active89859 versus full/current115167.

2. Experimental Design must replace all-RTX/sameGPU claim. Proposed prose:
“The completed correction study comprises27 separately trained fits across three datasets and seeds, counted once by their original execution identities. Fifteen fits completed on A100: all nine Taxi fits and the six RAF fits for seeds42 and52. RAF seed62 used RTX5080 and all nine Intermittent fits used RTX5090, with the three arms for each RTX dataset–seed group sharing one GPU and runtime. RAF seed42/full and current-only resumed from epochs23 and46 on a different physical A100; their removal control did not undergo that migration. Thus that group is not a comparison of uninterrupted trajectories on one physical GPU. Software and hardware also differ across seeds, so training time is retained by physicalGPU segment and is not interpreted as structural efficiency.”
Actual provenance supports fresh initial fits but two resumed RAF42 originalID fits; avoid “27 fresh fits” if it suggests no resume/reuse. Native contracts remain warm_startfalse because initial fit is fresh, and operational recovery must be stated separately.
Evidence: RTXexecution_contract.json14 (assignments),454 (source revision),729+ training,746+ architecture,806+handoff; reports/titantpp_history_rtx_handoff_20261010_v1/README.md10,21–26,53; final observation39–40; reports/titantpp_history_recovery_20261010_v1/README.md8,36.

3. Retrieval remains pending; change the automatic ninefullmean reuse/27furtherfits/63newfits narrative. Proposed prose:
“Prototype retrieval is a pending follow-up comparison. Before execution, the mean, weighted, fixed and general-query arms require a common dataset–seed/GPU/runtime contract and the predeclared cost measurements. The A100 correction fits cannot be presented as matched RTX mean baselines. Whether existing fits can be reused, and consequently the number of additional fits, remains to be resolved under that contract.”
Keep the retrieval rules/cost criterion as planned methodology, explicitly label prospective; do not claim result support/automatic execution. PAPER_WORK_ORDER.md43–46 and FINAL_OBSERVATION controls support this boundary.

4. Instacart now completed9, supplementary architecture comparison only; not Q1 correction/retrieval isolation, external commonhead encoder comparison, matched efficiency, or untouched Test confirmation. Proposed provenance:
“The completed supplementary Instacart comparison contains MLP8, MLP12 and CNN+GRU54 at seeds42,52 and62. The three architectures share a GPU within each seed: seeds42 and62 ran on RTX5080, while seed52 ran on RTX5090. Cross-seed GPU/runtime differences remain, and the historical MLP16 panel is reported separately rather than included as a newly matched arm.”
Instacartexecution_contract.json14,171–183,227,498–507; approved9 originalscope untouched.

5. Update completed scopes throughout abstract/introduction/contributions/Results/Discussion/conclusion. CNN+GRU54 externalpanel9 is distinct from correctionstudyfull9; never replace externalpanel values with the latter or mix their checkpoints. Criterion is descriptive developmentValidation consistency; not statisticalsignificance, causalidentification of a singlelag, novelty/external superiority or independent confirmation.

## Mathematical and implementation match — no rewrite required

- Event inputlog1p(time/quantity), no training-momentstandardization: H/models/TPPs/CountAwareTPP.py435–448. Encoder width64/two layers/fourheads/feedforward128/dropout0.1/persistent16/learnedposition: samefile1039–1051; models/Titan/backbone.py147–148.
- LocalQKVafterLayerNorm andprojection beforepersistentconcat: CountAwareTitanCNNGRU.py131–156,184–198; current/lag1/lag2 observedsegment sourceindices85–110 and channelresidual113–128.576 kernels=3QKV×3taps×64. Padding skips; validwithheld resets eligibility while prefixattention visibility remains. Two attentionblocks264–284. Existing Eq(local) and Algorithm are consistent.
- RecurrentcorrectioninputprojectionA64→54, GRU54 withbias, B54→64 withoutbias, Bzeroinit: samefile201–238. State advances onlyobserved; resets withheldvalid; count>=2 gate. Residual AFTER currentevent.24,732=2×64×54+3×(54²+54²+2×54). Full+local25,308.
- current_only basesidentity/tanh/GELU withobservedcount1/2/3: CNNHistoryControls98–116. Two repeatedcurrent GRUsteps startingzero eachposition, >=2gate, no crosspositionstate:182–216. Allparameters/modules exactlyoriginalinitialization:222–233 and run_titantpp_cnn_history_controls.py123–142. Both encoder/currenthidden contain prefixhistory; functionalcapacity/depth/activations/cost differ despite equalactiveparamcount. Existing Methods control-limitparagraph correctly says this.
- Quantitysoftpluslinearlogprediction/expm1 raw, losslog1pMSE: CountAwareTPP770–789; no inverse trainstandardization. Heteroscedasticmu/sigmafloor1e-3+softplus:743–768. Recorded-timeprobability lawnormalizedfirstbinCDF1.5, regularbinCDFdifference, topcode30upper-tailInstacart: positive_integer_time.py39–58,101–128; CountAwareTPP450–460 observation dispatch. Current timeequations correct. For Instacart explicitly use D=min(30,max(1,round(T))) if compact clarificationdesired.
- Both approvedstudiesbatch128/min40/max300/patience40, AdamWlr1e-3/defaultwd0.01, clip1, lambda_log_qty1/lambda_tail0, fullValidationrawRMSEselector earlieststrictfinitemin; exacttiesearlier. H/paper/scripts/count_aware_tpp_backbone/training.py201–214,613–627,885–900; run_titantpp_cnn_history_controls.py148–155; contractdatasetmodel/optimizer fields. Sixmetrics fromchosenepoch, three-seedmean/sampleSD as drafted; no separatetimingselectedepoch.
- Instacartdiffs: nativeunitsday, scale7, topcode30, cap64,targetpopulation1991192train/503733Validation,52native-dayslookback, tailstrictquantity>35. CoreTaxi hour/scale1/cap256/lookback168, Intermittent week/scale3/cap256/lookback520, RAFmonth/scale6/cap84/lookback84. No head/training/selector formula discrepancy found.

## Claim wording suitable after results incorporation

“The full model satisfies the predeclared descriptive correction criterion on Taxi and Intermittent: it has strictly lower overallValidationquantityRMSE than each control for all three seeds on both datasets. RAF does not meet that criterion. Samecheckpoint timeNLL and largequantityerrors retain adverse conditions, and the current-only control still receives prefixencodedstates. These results support the implemented correction on these developmentdatasets, without establishing significance or independent confirmation.”
Parent result-audit owns exact numericaltable/harms verification; this methods audit did not read new percondition predictionfiles or reevaluate binaries.
