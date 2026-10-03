"""Validate saved design identities, scope, evidence and local document references."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []
    def check(name, value):
        checks.append({"check": name, "passed": bool(value)})

    protected = read(OUT / "preserved_inputs.json")
    check("original_scientific_files_manuscript_and_prior_evidence_preserved", all(sha(ROOT / p) == h for p, h in protected.items()))
    contract = read(OUT / "design_contract.json")
    identity = read(OUT / "design_contract_identity.json")
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    check("design_canonical_and_file_hash", canonical == identity["canonical_sha256"] and sha(OUT / "design_contract.json") == identity["file_sha256"])
    check("non_executable_scope", contract["status"] == "design_fixed_non_executable" and not any(contract["authorization"][k] for k in ["new_scientific_fits", "GPU_qualification", "heldout_performance_access", "cloud_rental"]))
    check("resource_boundaries_not_invented", all(v is None for k, v in contract["deployment"].items() if k != "earlier_campaign_budgets_inherited") and contract["training"]["global_deadline"] is None and not contract["deployment"]["earlier_campaign_budgets_inherited"])
    matrix = list(csv.DictReader((OUT / "proposed_run_matrix.csv").open()))
    expected = {(a["model_id"], d["dataset_id"], str(s)) for a in contract["arms"] for d in contract["datasets"] for s in contract["seeds"]}
    actual = {(r["model"], r["dataset"], r["seed"]) for r in matrix}
    check("all_36_conditions_unique_complete", len(matrix) == len(actual) == 36 and actual == expected)
    check("all_rows_are_proposals", all(r["status"] == "proposed_not_executable" for r in matrix))
    check("control_counts_24_and_comparator_12", sum(r["model"].startswith("titantpp_") for r in matrix) == 24 and sum(r["model"] == "deep_renewal_event_native_nb" for r in matrix) == 12)
    check("control_parameter_count", all(a["branches"] * (a["input_width"] * a["bottleneck_width"] + a["bottleneck_width"] * a["output_width"]) == a["additional_parameters"] == 6144 for a in contract["arms"][:2]))
    check("fixed_divisor_and_zero_initialization", all(a["divisor"] == 8 and a["zero_output_init"] and not a["bias"] for a in contract["arms"][:2]))
    check("main_model_and_RMSE_selection_unchanged", contract["representative_model"] == "titantpp_history_mlp" and contract["training"]["selection"] == "strict_first_min_validation_raw_quantity_rmse")
    check("missing_RAF_B_not_invented", contract["reuse"]["no_correction_B_conditions"] == 9 and contract["reuse"]["no_correction_B_raf_present"] is False)
    check("reuse_registry_unchanged", sha(ROOT / contract["reuse"]["registry"]) == contract["reuse"]["registry_sha256"])
    support = read(OUT / "data_support.json")
    check("support_scope_train_validation_only", support["scope"] == "train_validation_only" and not support["heldout_content_read"])
    check("four_data_supports_compatible", len(support["datasets"]) == 4 and all(d["native_shifted_nb_support_compatible_after_existing_loader"] for d in support["datasets"]))
    check("train_validation_target_counts", all(d["train_targets"] > 0 and d["validation_targets"] > 0 for d in contract["datasets"]))
    check("data_identities_same_as_probe", all(any(d["dataset_id"] == s["dataset"] and d["data"]["sha256"] == s["sha256"] for s in support["datasets"]) for d in contract["datasets"]))
    sources = read(OUT / "sources.json")
    check("all_upstream_source_SHAs", all(sha(ROOT / r["local_path"]) == r["sha256"] for r in sources["pinned_upstream_files"]))
    check("frozen_core_and_RAF_source_SHAs", all(sha(ROOT / p) == h for b in sources["frozen_baseline_sources"].values() for p, h in b["selected_model_files"].items()))
    check("upstream_not_executed", sources["upstream_code_imported_or_executed"] is False)
    exposure = read(OUT / "evaluation_population_decision.json")
    check("researcher_reply_preserved", exposure["researcher_attestation"]["exact_reply"] == "기억이 불확실함")
    check("no_unseen_population_claim", exposure["independent_population_id"] is None and set(exposure["classification"].values()) == {"reserved_split_prior_exposure_uncertain"})
    check("prior_incidental_exposure_retained_without_values", exposure["prior_aggregate_exposure"]["record_exists"] and exposure["prior_aggregate_exposure"]["values_reopened_or_copied_this_turn"] is False)
    suites = ET.parse(OUT / "unit_tests.xml").getroot().findall("testsuite")
    count = sum(int(s.attrib["tests"]) for s in suites)
    check("18_CPU_reference_checks_passed", count == 18 and all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == int(s.attrib["skipped"]) == 0 for s in suites))
    baseline = read(ROOT / "reports/titantpp_baseline_reset_20261001_v1/baseline.json")
    check("tracking_updated_without_new_observation", baseline["independent_evaluation"]["researcher_attestation"]["status"] == "answered_uncertain" and baseline["pakdd_extension_design"]["status"] == "design_fixed_non_executable" and baseline["status_observed_kst"] == read(OUT / "before/reports/titantpp_baseline_reset_20261001_v1/baseline.json")["status_observed_kst"])
    broken = []
    for file in OUT.glob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", file.read_text()):
            if link.startswith(("https://", "http://", "#")):
                continue
            path = link.split("#", 1)[0]
            if path == "verification.json":
                continue  # generated by this script below
            if not (file.parent / path).exists():
                broken.append([str(file.relative_to(ROOT)), link])
    check("local_document_links_resolve", not broken)
    checks_failed = [c for c in checks if not c["passed"]]
    artifact_hashes = {str(p.relative_to(ROOT)): sha(p) for p in OUT.rglob("*") if p.is_file() and p.name != "verification.json" and "__pycache__" not in p.parts and "before" not in p.parts and ".pytest_cache" not in p.parts}
    result = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "passed" if not checks_failed else "failed", "checks": checks,
              "checks_passed": len(checks) - len(checks_failed), "checks_total": len(checks),
              "CPU_synthetic_tests_passed": count, "GPU_qualification_passed": False,
              "heldout_content_read": False, "full_native_DeepRenewal_equivalence_tested": False,
              "performance_claim_from_this_work": None, "broken_links": broken,
              "artifact_sha256": artifact_hashes,
              "tracking_sha256": {p: sha(ROOT / p) for p in ["reports/titantpp_baseline_reset_20261001_v1/README.md", "reports/titantpp_baseline_reset_20261001_v1/baseline.json", "reports/titantpp_manuscript_integration_20261001_v1/README.md"]}}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["status", "checks_passed", "checks_total", "CPU_synthetic_tests_passed"]}))
    if checks_failed:
        raise SystemExit(json.dumps(checks_failed))


if __name__ == "__main__":
    main()
