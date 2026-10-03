# B9 evidence retrieval QA

## Review Result: PASS

Status: completed retrieval implementation review and subsequent independent local original-evidence/result review. No new inference or bootstrap was run by this reviewer.

### CRITICAL
- None.

### HIGH
- None.

### MEDIUM
- None.

### LOW
- None.

## Summary
Reviewed `retrieve_complete.py` only; no sealed scientific source or prelaunch report was modified.

The remote destination is fixed to the approved 5080 campaign subdirectory. Local contract/root/host/registry and code seals must agree before network access. All four phases must declare exactly the nine approved conditions complete with zero failures and zero unstarted entries. Terminal/run manifests, gate evidence, launch and campaign identity bindings are checked before packaging. Existing remote/local retrieval directories and exposed runs/gate cause refusal instead of overwrite. Packaging preserves originals and includes generated logs/failure history, with explicit exclusions for unchanged sealed inputs, caches and locks.

The archive and manifest are SHA checked before exposure. Extraction accepts only unique regular files matching the manifest inventory, verifies safe normalized relative paths, file sizes, total bound and every SHA, and writes using exclusive creation. Links, traversal, extra/duplicate members and overwrites are rejected. Only verified runs and qualification gate are copied into the local campaign for the analyzer. Sealed local files are rechecked afterward.

## Automated local checks
- Parsed local script and compiled embedded remote Python without executing remote code.
- Seven unsafe path cases rejected.
- Exact nine-condition inventory accepted; duplicated/missing condition rejected.
- Synthetic archive success case passed.
- Nine rejection cases passed: traversal, symlink, hardlink, duplicate member, extra member, corrupt SHA, incorrect size, identity mismatch, and size limit exceeded.
- No network request, model import/inference, checkpoint deserialization, or training performed.

## Acceptance criteria
- [x] Approved remote ownership/path binding.
- [x] Nine-condition completion gate for all four phases.
- [x] SHA checks for archive, manifest, all retrieved/exposed files and frozen source bindings.
- [x] Safe extraction and exclusive local preservation.
- [x] Existing original evidence retained.
- [x] Completed archive retrieval and result verification were subsequently checked as documented below.

## Files changed
Only this `postrun_review.md` review artifact. No source or `.agents/` changes.

## Reviewed file SHA-256
`retrieve_complete.py`: `1bd2efa89b54181f12361180e6434e368c266b3a30e51e5cc8c7e2930f8f1dbb`

## Completed original-evidence audit
- Independently rehashed the 214,505,812-byte archive, manifest, all 625 original files (size + SHA), and all 616 exposed runs/gate files.
- Archive SHA: `840ea483e00feb9d14279ba87c5c58f9b91be803017b5527b1e246b8e9193f20`.
- Checked all sealed local source identities and every qualification gate evidence hash after retrieval.
- Checked 36 condition receipts across CPU, CUDA, full Validation and Test: each phase 9/9 complete, failures0, unstarted0. Contract/registry/checkpoint bytes/tensor/selected epoch/frozen bundle/split/immutability identities agree. Full Validation gate contains nine full replays.
- Existing legacy analysis and Test run manifest still match their frozen hashes.

## Completed scientific result/report checks
- Read stored B Test prediction parts for all nine fits and independently recomputed event-weighted MAE, RMSE and time NLL without inference. Recomputed three-seed means and sample SD (`ddof=1`) agree with analysis. Tolerance was `rtol=1e-10, atol=1e-12`; a 1.7e-14 reduction-order difference in Instacart SD is immaterial and correctly rounds identically.
- All existing seven-model means, SDs and individual seed metrics in every included panel match the legacy analysis exactly.
- All nine B/MLP pairs have identical receipt target counts, target identity, truth/history-summary digest and data identity. Test counts are Taxi8327, Intermittent88019, Instacart578387.
- Verified displayed aggregate strings for 24 dataset/model groups and 147 contrast/metric rows across seven primary/sensitivity panels against analysis, including displayed intervals and signs.
- Verified renderer output hashes and analysis linkage. Taxi168h/336h intervals remain unavailable; Taxi24h is descriptive. Intermittent site and Instacart user intervals are conditional on fixed trained seeds, unadjusted exploratory 95%, with no representative reselection.
- No independent bootstrap replay was run; reviewed stored bootstrap results, declared consistency evidence, interval scope and report transcription. History equality is verified through frozen loader contracts and recorded target/history summaries, not independent reconstruction of every history array.
- Parent completion report percentages and seed directions agree: Taxi MLP RMSE35.44% lower than B (all three quantity seeds improve); Intermittent B RMSE4.52% lower (all three quantity seeds improve, mean time NLL worsens); Instacart B RMSE0.096% lower with 2:1 seed direction. No causal overfitting conclusion or Test-based model replacement is made.

## Final reviewed result identities
- `analysis.json`: `e72c6b0dab8e998bbe4b086d3efd69818e9794728018db43442fa47ecf9f4a55`
- `review/summary.md`: `c4f84ffedfd478b42badfe23647ae9a095539b2b66d1516f82b22f546ca865b6`
- `review/tables.tex`: `961097682bcf6d9e8abaa80ce16c09e57b57caaaaa522a5056fd7f3cea5eacda`
- `review/render_receipt.json`: `74c66feaed1412827a8edfd47a17b0963134a4b3c08d93179d3ce68aa827b7e3`

Completion-report wording was corrected before final approval to state the exact history-summary verification scope. No remaining findings.
- `COMPLETION_REPORT.md`: `a2ad3fecda18ab33fea75e94a10fba91c12fc21b7e41a05ce821cea36c76b71b`
