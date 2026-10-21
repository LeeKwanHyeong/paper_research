# MAC9 terminal recovery independent review

Status: **PASS** for the exact eight files bound in INDEPENDENT_QA.json.

The architectural failure is a control-process observation race: nine Validation workers have actual exit0 records, while the preserved v4 dispatcher exited1 before writing its gate. Scientific success and control failure must retain separate evidence.

The recommended design retains the original v4 scientific paths and adds a separate v5 controller. It admits the existing successful Validation evidence, runs the original gate verifier, and starts only fixed Test9. It does not rewrite v4 failure or completed scientific records.

| Option | Implementation cost | Operating cost | Team complexity | Future change cost |
|---|---|---|---|---|
| Keep v4 scientific paths and isolate v5 control | Moderate; explicit admission and collector | One new control root, no model copying | Explicit split between historical failure and new completion | Requires maintaining the narrow provenance adapter |
| Copy v4 tree and relocate every contract/receipt | High; paths and nested SHA bindings change | Extra checkpoint and evidence storage | More relocation rules across existing validators | Higher risk of provenance drift on each relocation |

The first option is appropriate for this single pinned incident. It is not a general retry service.

## Validation

The independent final local run passed **64 tests and14 subtests**, actual process exit0, pytest41.36 seconds. Hashes of all eight reviewed files were equal before and after the run. Tests cover the process-exit race, nonzero exits, preserved source and failure bytes, gate failure preventing Test, exact73 additions, inherited deadlines, no retry, complete27-stage synthetic evidence, one retrieval, SSH0 cache verification, archive safety and pure3seed comparison recomputation.

A separate read-only local check matched the real failed v4 fixture: Validation9 exits0 and nine receipt bindings; v4 dispatcher exit1 with raw launch identity; no Test; correct drained seed42/62 chronology. The pinned old control closure and frozen source989/input9 passed unchanged. No SSH, model execution or deployment was performed by this reviewer.

The comparison regression calculates108 seed differences and36 aggregates through the existing pure comparison functions, preserves adverse seed outcomes and ddof1, records Shared Test5080/cuDNN91900 versus MAC5090/cuDNN92000, and rejects changed results even if their output manifest is updated.

## Risks and limits

Native preflight and final retrieval verification remain necessary. The diagnostic fixture does not contain every original checkpoint/prediction file. This review is not evidence that Test9 or retrieval has completed. A changed boot, source, identity, gate, deadline, prior Test artifact or partial retrieval must stop this attempt. Supervisor own OS exit is not directly observed.

## Remaining sequence

1. Commit the exact eight reviewed files and these two QA artifacts in paper_research.
2. Prepare and seal the new v5 controller; run the authorized native preflight and Test-only continuation once.
3. After actual dispatcher exit0, collect once and verify all scientific27 exits, ownership and SHA evidence.
4. Produce and recheck the local common3seed comparison, then report the result.

Artifacts created: INDEPENDENT_QA.json and INDEPENDENT_QA.md in execution_v5. No source or .agents files were modified by this reviewer.
