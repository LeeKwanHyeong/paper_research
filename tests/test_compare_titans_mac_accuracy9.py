"""Synthetic local comparison calculations and immutable-source boundary tests."""
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics

import pytest

from paper.scripts import compare_titans_mac_accuracy9 as c


def _fixture():
    shared, mac, populations = [], [], {}
    for dataset in c.DATASETS:
        for split in c.SPLITS:
            populations[dataset, split] = {"target_count": 100, "target_identity_sha256": "a"*64,
                "target_quantity_sha256": "b"*64}
            for index, seed in enumerate(c.SEEDS):
                def metric(value):
                    return {"overall": {"count":100,"qty_rmse":value,"qty_mae":value/2,"time_nll":value/10},
                        "tail": {"count":3,"qty_rmse":value*2,"qty_mae":value,"time_nll":value/5},
                        "tail_threshold":200.,"tail_rule":"q > frozen_Train_large_quantity_threshold",
                        "time_metric":"recorded_positive_integer_time_nll"}
                shared.append({"dataset":dataset,"split":split,"seed":seed,"model":c.SHARED,
                    "selected_epoch":index+1,"metrics":metric(1.+index*2)})
                mac.append({"dataset":dataset,"split":split,"seed":seed,"model":c.MAC,
                    "selected_epoch":index+2,"metrics":metric(2.+index*2),"target_identity":populations[dataset,split]})
    return shared,mac,populations


def test_mean_sample_sd_all_six_metrics_and_sign_convention():
    shared,mac,populations=_fixture();result=c.compare_records(shared,mac,populations)
    assert len(result["aggregates"])==36 and len(result["seed_differences"])==108
    row=result["aggregates"][0]
    assert row["shared"]["mean"]==3. and row["shared"]["sample_sd"]==2.
    assert row["mac"]["mean"]==4. and row["mac"]["sample_sd"]==2.
    assert row["shared_minus_mac"]["mean"]==-1. and row["shared_minus_mac"]["sample_sd"]==0.
    assert result["seed_differences"][0]["relative_percent_of_mac"]==-50.
    assert result["seed_differences"][0]["shared_worse"] is False


def test_adverse_seed_and_pairwise_delta_sd_are_retained():
    shared,mac,populations=_fixture();shared[1]["metrics"]["overall"]["qty_rmse"]=8.
    result=c.compare_records(shared,mac,populations);row=result["aggregates"][0]
    assert row["shared_worse_seeds"]==[52]
    assert row["shared_minus_mac"]["mean"]==pytest.approx(2/3)
    assert row["shared_minus_mac"]["sample_sd"]==pytest.approx(statistics.stdev([-1,4,-1]))
    markdown=c.render_markdown(result)
    assert "| 52 |" in markdown and "큰 수량·시간" in markdown and "ddof=1" in markdown
    assert "내부 MLP16을 합치지" in markdown
    assert "5080/cuDNN91900" in markdown and "5090/cuDNN92000" in markdown
    assert "모든 정확도 평가를 같은 GPU·Runtime에서 수행한 비교는 아닙니다" in markdown


@pytest.mark.parametrize("mutation",("duplicate","missing","seed","truth","count","tail","time","nan","negative","epoch"))
def test_scope_target_and_metric_mutations_fail(mutation):
    shared,mac,populations=_fixture()
    if mutation=="duplicate":shared.append(deepcopy(shared[0]))
    elif mutation=="missing":mac.pop()
    elif mutation=="seed":mac[0]["seed"]=72
    elif mutation=="truth":mac[0]["target_identity"]={**mac[0]["target_identity"],"target_quantity_sha256":"z"*64}
    elif mutation=="count":mac[0]["metrics"]["tail"]["count"]=4
    elif mutation=="tail":mac[0]["metrics"]["tail_threshold"]=201.
    elif mutation=="time":mac[0]["metrics"]["time_metric"]="continuous_nll"
    elif mutation=="nan":shared[0]["metrics"]["overall"]["qty_rmse"]=math.nan
    elif mutation=="negative":mac[0]["metrics"]["overall"]["qty_mae"]=-1.
    elif mutation=="epoch":mac[0]["selected_epoch"]=0
    with pytest.raises(ValueError):c.compare_records(shared,mac,populations)


def test_unrelated_historical_models_excluded():
    shared,mac,populations=_fixture()
    shared.append({"model":"titantpp_history_mlp_width16","metrics":{"invalid":math.nan}})
    result=c.compare_records(shared,mac,populations)
    assert result["endpoint_count_per_model"]==18 and len(result["seed_differences"])==108


def test_empty_tail_stays_na_without_zero_imputation():
    shared,mac,populations=_fixture()
    for row in shared+mac:
        row["metrics"]["tail"]={"count":0,"qty_rmse":None,"qty_mae":None,"time_nll":None}
    result=c.compare_records(shared,mac,populations)
    row=next(r for r in result["aggregates"] if r["group"]=="tail")
    assert row["shared"]["mean"] is None and row["shared"]["status"]=="N/A_empty_cohort"
    assert "N/A" in c.render_markdown(result)
    shared[0]["metrics"]["tail"]["qty_rmse"]=0.
    with pytest.raises(ValueError,match="Empty cohort"):c.compare_records(shared,mac,populations)


def test_zero_mac_relative_denominator_is_explicit_na():
    shared,mac,populations=_fixture();mac[0]["metrics"]["overall"]["time_nll"]=0.
    row=next(r for r in c.compare_records(shared,mac,populations)["seed_differences"]
        if r["dataset"]==c.DATASETS[0] and r["split"]=="validation" and r["seed"]==42 and r["group"]=="overall" and r["metric"]=="time_nll")
    assert row["shared_minus_mac"]==.1 and row["shared_worse"] is True
    assert row["relative_percent_of_mac"] is None and row["relative_status"]=="N/A_zero_reference"


def _shared_files(tmp_path,monkeypatch):
    monkeypatch.setattr(c,"ROOT",tmp_path)
    directory=tmp_path/"evidence";directory.mkdir()
    original=tmp_path/"original.json";original.write_text('{"status":"complete"}')
    table={"schema":"shared_final_three_seed_paper_evidence_v1","status":"verified_existing_results_reaggregated",
        "intermittent_final_best_verified":True,"seeds":[42,52,62],"data_variant":"original_uncapped","records":[]}
    verified={"status":"PASS","remote_calls":0,"new_training":0,"new_inference":0,
        "file_bindings":{str(original):c.sha(original)},"final_receipts":{str(original):c.sha(original)}}
    for name,value in (("RESULT_TABLES.json",table),("SOURCE_VERIFICATION.json",verified),("REPRODUCIBILITY.json",{})):
        (directory/name).write_text(json.dumps(value))
    (directory/"MANIFEST.json").write_text(json.dumps({"files":{name:c.sha(directory/name) for name in
        ("RESULT_TABLES.json","SOURCE_VERIFICATION.json","REPRODUCIBILITY.json")}}))
    return directory,original


def test_shared_manifest_and_original_hashes_checked(tmp_path,monkeypatch):
    directory,original=_shared_files(tmp_path,monkeypatch)
    proof=c.verify_shared_evidence(directory,expected_manifest_sha=c.sha(directory/"MANIFEST.json"))[2]
    assert proof["verified_original_bindings"]==1
    original.write_text('changed')
    with pytest.raises(ValueError,match="bound original changed"):
        c.verify_shared_evidence(directory,expected_manifest_sha=c.sha(directory/"MANIFEST.json"))


def test_manifest_rebinding_requires_explicit_expected_sha(tmp_path,monkeypatch):
    directory,_=_shared_files(tmp_path,monkeypatch);expected=c.sha(directory/"MANIFEST.json")
    (directory/"MANIFEST.json").write_text('{}')
    with pytest.raises(ValueError,match="manifest changed"):c.verify_shared_evidence(directory,expected_manifest_sha=expected)


def test_incomplete_check_only_writes_nothing_and_never_verifies_remote(tmp_path,monkeypatch):
    monkeypatch.setattr(c,"BUNDLE",tmp_path/"bundle");monkeypatch.setattr(c,"REPORT",tmp_path/"report")
    monkeypatch.setattr(c.prep,"verify",lambda *a,**kw:pytest.fail("No verification until complete"))
    assert c.run("not-sealed",check_only=True)["status"]=="waiting_for_complete_MAC9"
    assert not (tmp_path/"report").exists()
    with pytest.raises(ValueError,match="incomplete"):c.run("not-sealed")


def test_failed_receipt_cannot_be_promoted_to_comparison(tmp_path,monkeypatch):
    bundle=tmp_path/"bundle";path=bundle/"retrieval/final_receipt.json";path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status":"failed"}))
    monkeypatch.setattr(c,"BUNDLE",bundle)
    with pytest.raises(ValueError,match="incomplete"):c.run("fake",check_only=True)


def test_known_shared_test_runtime_difference_is_preserved_without_rejection():
    training={"python":"3.12.13","torch":"2.11.0+cu130","cuda":"13.0","cudnn":92000,"threads":4,"interop_threads":1}
    reference={**training,"cudnn":91900}
    result=c.evaluation_runtime_context({"hosts":{"5090":{"runtime_expected":training}}},
        {"shared_training_runtime":training,"shared_test_runtime":reference})
    assert result["shared_test_reference"]["host"]=="5080"
    assert result["shared_test_reference"]["runtime"]["cudnn"]==91900
    assert result["mac_training_validation_test"]["runtime"]["cudnn"]==92000
    assert result["all_accuracy_endpoints_same_gpu_runtime"] is False
    assert result["shared_test_reference"]["stored_results_reused"] is True
