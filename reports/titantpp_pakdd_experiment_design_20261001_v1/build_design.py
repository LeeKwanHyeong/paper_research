"""Assemble a non-executable design from previously audited source/data identities."""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    context = json.loads((OUT / "context.json").read_text())
    registry_path = "reports/titantpp_independent_evaluation_preparation_20261001_v1/evaluation_registry.json"
    registry = read(registry_path)
    support = json.loads((OUT / "data_support.json").read_text())
    receipt = read("reports/titantpp_followup_contract_5080_20260927_v1/upstream_source_receipt.json")
    source_rows = []
    for row in receipt["files"]:
        if row.get("model", row.get("project")) not in ["deep_renewal", "flextpp"] or "local_path" not in row:
            continue
        assert sha(row["local_path"]) == row["sha256"]
        source_rows.append({k: row[k] for k in ["url", "local_path", "sha256"]})
    nb = json.loads((OUT / "upstream/receipt.json").read_text())
    assert sha(nb["path"]) == nb["sha256"]
    source_rows.append({"url": nb["url"], "local_path": nb["path"], "sha256": nb["sha256"]})
    frozen = {}
    for name in ["core", "raf"]:
        b = registry["bundles"][name]
        frozen[name] = {k: b[k] for k in ["source_root", "contract_path", "canonical_sha256", "source_closure_sha256"]}
        frozen[name]["selected_model_files"] = {}
        for rel in ["models/TPPs/CountAwareTitanCoreAblation.py", "models/TPPs/CountAwareTitanMultiLagDetail.py", "data_loader/event_seq_data_module.py"]:
            path = str(Path(b["source_root"]) / rel)
            assert sha(path) == b["source_files"][rel]
            frozen[name]["selected_model_files"][path] = b["source_files"][rel]
    sources = {"primary_literature": [
        {"title": "Forecasting intermittent and sparse time series: A unified probabilistic framework via deep renewal processes", "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764", "role": "task and native interval-size model; original experimental setting"},
        {"title": "Transformers for Mixed-type Event Sequences", "url": "https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html", "role": "related direct comparator, deferred before performance observation"},
        {"title": "FlexTPP repository", "url": "https://github.com/czi-ai/FlexTPP", "role": "README labels implementation unofficial"}],
        "pinned_upstream_files": source_rows, "frozen_baseline_sources": frozen,
        "upstream_code_imported_or_executed": False,
        "read_only_download_note": "Initial sandbox DNS and system certificate lookup failed; certifi CA-verified HTTPS retrieval succeeded. No TLS bypass or environment installation."}
    write("sources.json", sources)
    datasets = []
    for d in support["datasets"]:
        bname = "raf" if d["dataset"] == "raf_spare_parts" else "core"
        spec = next(x for x in registry["bundles"][bname]["datasets"] if x["dataset_id"] == d["dataset"])
        datasets.append({"dataset_id": d["dataset"], "data": {"path": d["path"], "sha256": d["sha256"]},
                         "loader": spec["loader"], "baseline_model_spec": spec["model"],
                         "baseline_train_statistics": spec["statistics"], "optimizer": spec["optimizer"],
                         "steps_per_epoch": spec["steps_per_epoch"],
                         "train_targets": next(g["next_event_targets"] for g in d["groups"] if g["chronological_split"] == "train"),
                         "validation_targets": next(g["next_event_targets"] for g in d["groups"] if g["chronological_split"] == "validation"),
                         "support_checked": True, "heldout_support_checked": False})
    ledger = {"recorded_utc": context["created_utc"],
              "researcher_attestation": {"question": context["researcher_question"], "exact_reply": context["researcher_reply"], "interpretation": "uncertain; not an attestation of either use or non-use", "applies_to": [d["dataset_id"] for d in datasets]},
              "classification": {d["dataset_id"]: "reserved_split_prior_exposure_uncertain" for d in datasets},
              "independent_population_id": None,
              "sources": {"prior_lineage_review": "reports/titantpp_independent_evaluation_preparation_20261001_v1/lineage_review.json", "prior_incidental_aggregate_exposure": "reports/titantpp_followup_contract_5080_20260927_v1/experiment_contract.md#7"},
              "prior_aggregate_exposure": {"record_exists": True, "dataset_level_attribution": "not established by the cited statement", "values_reopened_or_copied_this_turn": False},
              "decision": "Do not label existing splits untouched. Finish validation development then lock all evaluation choices. Existing reserved-split evaluation remains possible with disclosed uncertainty; new provenance-verified data required for an independent-confirmation claim.",
              "past_test_generation_proves_human_use": False,
              "heldout_content_read_this_turn": False,
              "new_evaluation_authorized": False}
    write("evaluation_population_decision.json", ledger)
    arms = [
        {"model_id": "titantpp_current_only_param_matched", "role": "capacity-matched control for explicit predecessor input", "branches": 8, "input_width": 64, "bottleneck_width": 6, "output_width": 64, "bias": False, "additional_parameters": 6144, "thresholds": [1, 2, 4, 8, 16, 32, 64, 128], "divisor": 8, "explicit_predecessor": False, "zero_output_init": True, "caveat": "current state already contains causal history; rank increases from 4 to 6"},
        {"model_id": "titantpp_all_available_history_mlp", "role": "control for progressive branch availability", "branches": 8, "input_width": 128, "bottleneck_width": 4, "output_width": 64, "bias": False, "additional_parameters": 6144, "thresholds": [1] * 8, "divisor": 8, "explicit_predecessor": True, "zero_output_init": True, "predecessor_rule": "all branches gather true lag-one index; never unmask pre-zeroed branch indices", "caveat": "changes active capacity and residual magnitude jointly on short histories"},
        {"model_id": "deep_renewal_event_native_nb", "role": "direct next-event quantity comparator", "hidden_dim": 64, "layers": 2, "dropout": .1, "input_transform": "raw interval-size after shared frozen loader", "time_distribution": "1+NB(mu_dt,alpha_dt)", "quantity_distribution": "1+NB(mu_qty,alpha_qty)", "dispersion": "two global learned scalars with softplus + 1e-5", "quantity_prediction": "1+mu_qty, same mean for both RMSE and MAE", "loss": "target-only time observed-mass NLL + quantity shifted-NB NLL", "instacart_topcode": "recorded D=30 => P(NB>=29)", "label_scope": "native-family PyTorch next-event adapter; not original calendar-horizon replication", "dense_bias_init": 5., "dispersion_bias_init": 2., "lstm_initializer_verification": "pending implementation equivalence checks", "reference_revision": "889a3df86a89a365880b4bc1488bcf4c039f265e"},
    ]
    contract = {"contract_id": "titantpp_pakdd_validation_extension_design_20261001_v1",
                "status": "design_fixed_non_executable", "created_utc": context["created_utc"],
                "authorization": {"design_and_local_CPU_checks": True, "new_scientific_fits": False, "GPU_qualification": False, "heldout_performance_access": False, "cloud_rental": False},
                "representative_model": "titantpp_history_mlp", "scope": "validation_development_before_final_evaluation", "seeds": [42, 52, 62],
                "datasets": datasets, "arms": arms,
                "training": {"max_epochs": 300, "min_epochs": 40, "patience": 40, "batch_size": 128, "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.}, "selection": "strict_first_min_validation_raw_quantity_rmse", "no_warm_start": True, "target_exposure": "all canonical train next-event targets once per epoch", "global_deadline": None, "condition_wall_time_cap": None},
                "reporting": {"same_selected_checkpoint": ["quantity_RMSE", "quantity_MAE", "recorded_time_NLL"], "selected_and_last": True, "seed_aggregation": "mean and sample standard deviation ddof=1", "paired_comparison": "same dataset, seed and canonical target IDs", "unfavorable_and_failed_conditions": "retain", "primary_quantity_prediction": "same forecast for RMSE and MAE", "early_performance_based_subset_exclusion": False, "sum_of_heterogeneous_training_losses_as_rank": False},
                "reuse": {"history_mlp_conditions": 12, "main_table_conditions": 84, "no_correction_B_conditions": 9, "no_correction_B_raf_present": False, "no_retraining_completed_conditions": True, "registry": registry_path, "registry_sha256": sha(registry_path)},
                "new_condition_counts": {"architecture_controls": 24, "deep_renewal": 12, "total_fits": 36, "nonlearned_data_model_rows": 8},
                "nonlearned_baselines": ["last_observed_quantity", "mean_quantity_in_same_observed_window"],
                "deployment": {"host_assignment": None, "runtime_identity": None, "source_closure": None, "launcher": None, "earlier_campaign_budgets_inherited": False},
                "candidate_allocation_not_a_reservation": {"5080": ["yellow_trip_hourly", "intermittent_frozen_5000", "raf_spare_parts"], "5090": ["insta_market_basket"]},
                "pre_execution_requirements": ["production adapters and model identity", "common base initialization and RNG for controls", "MXNet-to-PyTorch LSTM and distribution fidelity review", "native top-coded NB stable log-survival and gradients", "canonical target IDs, loader and batch-prefix parity", "save/resume optimizer/RNG/selection tests", "approved bounded native GPU validation and measured resource caps", "frozen source/runtime, absolute deadlines and explicit campaign launch approval"],
                "evaluation_population": "evaluation_population_decision.json"}
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    write("design_contract.json", contract)
    write("design_contract_identity.json", {"canonical_sha256": canonical, "file_sha256": sha(str(OUT.relative_to(ROOT) / "design_contract.json")), "serialization": "UTF-8 ensure_ascii=False sorted compact JSON of whole design_contract object"})
    fields = ["condition_id", "model", "dataset", "seed", "stage", "proposed_host", "status"]
    with (OUT / "proposed_run_matrix.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for arm in arms:
            for d in datasets:
                for seed in [42, 52, 62]:
                    ds = d["dataset_id"]
                    writer.writerow({"condition_id": f"{ds}__{seed}__{arm['model_id']}", "model": arm["model_id"], "dataset": ds, "seed": seed, "stage": "validation_development", "proposed_host": "5090" if ds == "insta_market_basket" else "5080", "status": "proposed_not_executable"})
    print(json.dumps({"new_fit_designs": 36, "canonical_sha256": canonical, "execution_authorized": False}))


if __name__ == "__main__":
    main()
