#!/usr/bin/env python3
"""Stdlib-only analysis of immutable paired Intermittent Train/Validation scores.

Requires completed original retrieval and six actual joint OS exits. It does
not import a scientific model, read tensor states, infer, fit or access Test.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

KIND = "titantpp_intermittent_time_cross_20261012_v1"
SCHEMA = KIND+"_analysis"
FULL = "titantpp_cnn_gru54"
ARMS = (FULL,FULL+"_correction_disabled",FULL+"_current_only")
SEEDS = (42,52,62)
SPLITS = ("train","validation")
SCORE_NAMES = ("ff","fc","cf","cc","scale_effect","location_effect","interaction","total_effect")


def evaluator():
    here = Path(__file__).resolve().parent
    path = here/"evaluate_titantpp_intermittent_time_cross.py"
    if not path.exists():path = here/"evaluate.py"
    spec=importlib.util.spec_from_file_location("_cross_verified_primitives",path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

E=evaluator()
require,read,sha,canonical,confined,write_exclusive=E.require,E.read,E.sha,E.canonical,E.confined,E.write_exclusive


def finite(value,name):
    require(type(value) in (int,float) and math.isfinite(value),"Nonfinite "+name)
    return float(value)


def close(actual,expected,name,rtol=1e-7,atol=1e-8):
    a,b=finite(actual,name),finite(expected,name)
    require(math.isclose(a,b,rel_tol=rtol,abs_tol=atol),"Reconstruction differs: "+name)


def score_cells(pair, expected_count):
    require(pair["full_arm"] == FULL and pair["control_arm"] in ARMS[1:]
            and pair["seed"] in SEEDS and pair["split"] in SPLITS,"Foreign cross pair")
    cells=pair["cells"]
    require(len({(c["group"],c["key"]) for c in cells}) == len(cells),"Duplicate cross strata")
    all_cells=[c for c in cells if c["group"] == "all"]
    bins=[c for c in cells if c["group"] == "target_gap_bin"]
    require(len(all_cells) == 1 and all_cells[0]["key"] == "all" and all_cells[0]["count"] == expected_count
            and sum(c["count"] for c in bins) == expected_count and len(cells) == len(bins)+1
            and all(c["key"] in ("0","1","2","3","4") for c in bins),"Paired gap coverage differs")
    for cell in cells:
        n=cell["count"]
        require(type(n) is int and n > 0 and set(cell["sums"]) == set(cell["squares"]) == set(cell["means"]) == set(SCORE_NAMES),"Malformed paired sufficient statistics")
        for name in SCORE_NAMES:
            total=finite(cell["sums"][name],name+" sum");squared=finite(cell["squares"][name],name+" squares")
            require(squared >= 0 and squared/n - (total/n)**2 >= -1e-7*max(1.,squared/n),"Impossible paired second moment")
            close(cell["means"][name],total/n,name+" mean")
        for key in ("sums","means"):
            x=cell[key]
            close(x["scale_effect"],x["fc"]-x["ff"],"scale effect",atol=1e-7)
            close(x["location_effect"],x["cf"]-x["ff"],"location effect",atol=1e-7)
            close(x["interaction"],x["cc"]-x["fc"]-x["cf"]+x["ff"],"interaction",atol=1e-7)
            close(x["total_effect"],x["cc"]-x["ff"],"total effect",atol=1e-7)
            close(x["total_effect"],x["scale_effect"]+x["location_effect"]+x["interaction"],"paired decomposition",atol=1e-7)
    for key in ("sums","squares"):
        for name in SCORE_NAMES:
            close(sum(c[key][name] for c in bins),all_cells[0][key][name],"gap coverage "+key+"/"+name)
    return cells


def mean_sd(values):
    require(len(values) == 3 and all(math.isfinite(v) for v in values),"Three-seed descriptive matrix differs")
    return {"mean":statistics.mean(values),"sample_sd":statistics.stdev(values),"seed_count":3}


def process_bridge(actual, receipt, c):
    """Popen command contains interpreter; sys.argv intentionally does not."""
    require(actual.get("exit_code") == 0 and actual["pid"] == receipt["pid"]
            and actual["ppid"] == receipt["ppid"]
            and actual["contract_sha256"] == receipt["contract_sha256"] == c["contract_sha256"],
            "Actual process PID/PPID/contract bridge differs")
    require(actual["argv"] == [c["runtime"]["python"], *receipt["argv"]],
            "Actual process argv/interpreter bridge differs")
    require(finite(actual["completed_unix"], "Actual OS completion") >= receipt.get("completed_unix", 0),
            "Actual OS completion precedes receipt")


def verify_originals(root,c,files):
    v=read(root/"originals/LOCAL_VERIFICATION.json");r=read(root/"originals/receipt.json")
    require(v.get("status") == "verified" and v["contract_sha256"] == r["contract_sha256"] == c["contract_sha256"]
            and v.get("all_archive_members_and_local_files_sha_verified") is True,"Cross binary originals lack local verification")
    manifest_path=root/"originals/manifest.json";archive=root/"originals/originals.tar.gz"
    require(sha(manifest_path) == v["manifest_sha256"] == r["manifest_sha256"]
            and sha(archive) == v["archive_sha256"] == r["archive_sha256"],"Cross retrieval SHA differs")
    manifest=read(manifest_path)
    require(manifest["contract_sha256"] == c["contract_sha256"] and len(manifest["files"]) == v["file_count"] == r["file_count"],"Cross manifest population differs")
    for relative,digest in manifest["files"].items():
        for prefix in ("","originals/verified/"):
            path=confined(root,prefix+relative)
            require(sha(path) == digest,"Cross archived/root bytes changed: "+prefix+relative)
            files[prefix+relative]=digest
    for relative in ("originals/receipt.json","originals/LOCAL_VERIFICATION.json","originals/manifest.json","originals/originals.tar.gz"):
        files[relative]=sha(root/relative)
    require(r.get("status") == "verified_terminal_archived" and r.get("actual_pipeline_exit0") is True
            and r.get("own_processes_absent") is True and r.get("own_gpu_pids_absent") is True,"Cross actual terminal/ownership proof incomplete")
    return v,r


def verify_terminal(root,c,files):
    require(not (root/"control/pipeline_failure.json").exists(),"Preserved pipeline failure prevents completion")
    complete=read(root/"control/pipeline_complete.json");latest=read(root/"control/pipeline_latest.json")
    exit_record=read(root/"control/pipeline_exit.json")
    require(exit_record.get("exit_code") == 0,"Actual pipeline exit0 unconfirmed")
    require(exit_record.get("contract_sha256",c["contract_sha256"]) == c["contract_sha256"],"Actual pipeline exit contract differs")
    require(complete.get("status") == "complete" and complete["contract_sha256"] == c["contract_sha256"]
            and latest.get("status") == latest.get("phase") == "complete" and latest.get("active") == [],"Cross final phase incomplete")
    require(complete.get("model_split_passes") == 18 and complete.get("joint_worker_count") == 6
            and complete.get("actual_all_joint_worker_exit0") is True and complete.get("new_fits") == complete.get("test_inference") == 0,
            "Cross18passes/6physical accounting differs")
    joint_ids={"seed"+str(seed)+"__"+split for seed in SEEDS for split in SPLITS}
    model_ids={j["id"] for j in c["jobs"]}
    require(set(complete["joint_receipts"]) == set(complete["joint_worker_exits"]) == joint_ids
            and set(complete["model_receipts"]) == model_ids
            and all(set(v) == set(SPLITS) for v in complete["model_receipts"].values())
            and set(complete["completed_joint_workers"]) == set(latest["completed_joint_workers"]) == joint_ids
            and latest["model_split_passes_completed"] == 18, "Pipeline omits native joint/model completion entries")
    qualification=read(root/"control/native_qualification.json")
    require(qualification["status"] == "PASS" and qualification["synthetic_only"] is True
            and qualification["contract_sha256"] == c["contract_sha256"] and set(qualification["routes"]) == {"history__"+arm for arm in ARMS},"Cross three-route native qualification incomplete")
    for route,digest in qualification["routes"].items():
        receipt=root/"control/native_probes"/route/"receipt.json";x=read(receipt.with_name("worker_exit.json"))
        require(sha(receipt) == digest and x["exit_code"] == 0,"Cross native actual exit or receipt differs")
        p=read(receipt)
        process_bridge(x,p,c)
        require(p["contract_sha256"] == c["contract_sha256"] and p["status"] == "PASS" and p["native_pinned_contract_checked"] is True
                and p["source_closure"] == E.SOURCE and p["actual_integer_diagonal_eventwise_bitwise_equal"] is True
                and p["hooked_and_unhooked_all_original_outputs_bitwise_equal"] is True
                and p["checkpoint_reads"] == p["optimizer_updates"] == 0 and p["real_data_loaded"] is False,"Native cross route integrity differs")
    return complete,qualification


def summarize(root,expected_contract,expected_operation):
    root=Path(root).resolve();files={}
    c=E.verify_contract(root)
    require(c["contract_sha256"] == expected_contract and sha(root/"operation_integrity.json") == expected_operation,"Explicit cross analysis pins differ")
    E.verify_operations(root,c)
    originals,retrieval=verify_originals(root,c,files)
    complete,qualification=verify_terminal(root,c,files)
    prerequisite=c.get("prerequisite_H1",{})
    require(prerequisite.get("status") == "complete" and prerequisite.get("actual_pipeline_exit0") is True,
            "RAF H1 completion prerequisite unconfirmed")
    rows,models=[],[]
    physical=set()
    for seed in SEEDS:
        for split in SPLITS:
            identity="seed"+str(seed)+"__"+split
            joint=E.verify_joint_receipt(root,c,seed,split)
            joint_path=root/"results"/("seed"+str(seed))/(split+"_joint")
            x=read(joint_path/"worker_exit.json")
            process_bridge(x,joint,c)
            require(qualification["completed_unix"] <= joint["started_unix"],"Real inference preceded native qualification")
            require(complete["joint_receipts"][identity] == sha(joint_path/"receipt.json")
                    and complete["joint_worker_exits"][identity] == sha(joint_path/"worker_exit.json"),"Pipeline joint completion SHA differs")
            physical.add(joint["pid"])
            paired=read(confined(root,joint["paired_scores_path"]))
            for pair in paired["pairs"]:
                cells=score_cells(pair,joint["count"])
                for cell in cells:
                    rows.append({"seed":seed,"split":split,"full_arm":FULL,"control_arm":pair["control_arm"],**cell,
                        "physical_worker_pid":joint["pid"],"target_identity_sha256":paired["target_identity_sha256"],"truth_sha256":paired["truth_sha256"],
                        "negative_effect_means_lower_nll":True,"counterfactual_score_not_new_model":True})
            for job in [j for j in c["jobs"] if j["seed"] == seed]:
                model_path=root/"results"/job["id"]/(split+"_metrics.json")
                require(sha(model_path) == complete["model_receipts"][job["id"]][split], "Pipeline model-pass completion SHA differs")
                r=read(model_path)
                models.append({key:r[key] for key in ("id","seed","arm","split","selected_epoch","checkpoint_sha256","state_sha256","metrics","pid","ppid","argv","started_unix","completed_unix","inference_seconds","target_identity_sha256","truth_sha256")})
                cell=next(row for row in rows if row["seed"] == seed and row["split"] == split and row["group"] == "all" and (job["arm"] == FULL or row["control_arm"] == job["arm"]))
                close(cell["means"]["ff" if job["arm"] == FULL else "cc"],r["metrics"]["time_nll"],"Original diagonal NLL")
    require(len(physical) == 6 and len(models) == 18,"Actual6OS/18pass count differs")
    all_rows=[r for r in rows if r["group"] == "all"]
    require(len(all_rows) == 12,"Cross6pair/2split matrix incomplete")
    means=[]
    for split in SPLITS:
        for arm in ARMS[1:]:
            selected=sorted((r for r in all_rows if r["split"] == split and r["control_arm"] == arm),key=lambda x:x["seed"])
            require([r["seed"] for r in selected] == list(SEEDS),"Good/bad seed coverage differs")
            means.append({"split":split,"control_arm":arm,"metrics":{name:mean_sd([r["means"][name] for r in selected]) for name in SCORE_NAMES}})
    return {"status":"complete","schema":SCHEMA,"contract_sha256":c["contract_sha256"],"operation_integrity_sha256":expected_operation,"created_unix":time.time(),
            "actual_model_split_passes":18,"actual_joint_os_workers_exit0":6,"paired_score_cells":48,"new_cross_cells":24,"new_fits":0,"test_inference":0,"calibration_fits":0,
            "pipeline_actual_exit0":True,"models":models,"rows":rows,"seed_mean_sample_sd":means,"source_files":files,"prerequisite_H1":prerequisite,
            "interpretation":{"law":"D=max(1,round(3*exp(N(mu,sigma)))) with top_code=None; frozen float64 positive_integer_log_mass",
                "negative_effect":"replacement reduces original integer TimeNLL","scale_effect":"Lfc-Lff (actual control sigma, full location)",
                "location_effect":"Lcf-Lff (actual control location, full sigma)","interaction":"Lcc-Lfc-Lcf+Lff; path dependence retained",
                "current_Test_scored":False,"causal_encoder_or_head_training_effect_proven":False,"generalization_proven":False,"calibration_fit":False,
                "raw_arrays_saved":False,"gap_bins":"Train-derived [1,3,5,7]; outcome-defined descriptive strata"}}


def render(payload):
    lines=["# Intermittent 시간 위치·분포 폭 교차 점수 결과", "", "선택9개 모델의 Train·Validation18개 평가를6개 실제 joint worker에서 마쳤다. 각 worker의3개 model-split pass는 같은 PID를 공유하며 원래 수량·시간 여섯 지표와 target/full-history SHA를 재현했다. 새 학습0·새 Test추론0·calibration fit0이다.","",
           "아래 네 점수는 **개별 event의 실제 μ·σ**로 같은 정수 duration 질량을 계산한다. `ff`는 원래 full, `cc`는 원래 대조이며 `fc/cf`는 서로 다른 학습 모델의 출력을 섞은 counterfactual score다. 새 단일 모델이나 causal ablation으로 부르지 않는다.","",
           "| Split | Seed | 대조 | Lff | Lfc | Lcf | Lcc | σ교체 Δ | μ교체 Δ | 상호작용 Δ |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    labels={ARMS[1]:"보정 제거",ARMS[2]:"current-only"}
    for r in sorted((r for r in payload["rows"] if r["group"] == "all"),key=lambda x:(SPLITS.index(x["split"]),x["seed"],ARMS.index(x["control_arm"]))):
        m=r["means"]
        lines.append("| "+" | ".join([r["split"],str(r["seed"]),labels[r["control_arm"]],*[f'{m[n]:.6f}' for n in ("ff","fc","cf","cc","scale_effect","location_effect","interaction")]])+" |")
    lines += ["", "음수 Δ는 원래 full 점수보다 낮은 NLL이며 양수 Δ는 손해다. `σ교체 + μ교체 + 상호작용 = Lcc−Lff`를 모든 구간에서 검증했다. 교체 순서에 따라 조건부 효과가 달라지므로 μ·σ의 기여를 유일한 비율로 분배하지 않는다.","",
              "## 같은 Train 경계를 쓴 duration 구간", "", "| Split | Seed | 대조 | Gap bin | N | σ교체 Δ | μ교체 Δ | 상호작용 Δ |", "|---|---:|---|---|---:|---:|---:|---:|"]
    ranges={"0":"≤1","1":"(1,3]","2":"(3,5]","3":"(5,7]","4":">7"}
    for r in sorted((r for r in payload["rows"] if r["group"] == "target_gap_bin"),key=lambda x:(SPLITS.index(x["split"]),x["seed"],ARMS.index(x["control_arm"]),int(x["key"]))):
        m=r["means"]
        lines.append("| "+" | ".join([r["split"],str(r["seed"]),labels[r["control_arm"]],ranges[r["key"]],str(r["count"]),*[f'{m[n]:.6f}' for n in ("scale_effect","location_effect","interaction")]])+" |")
    lines += ["", "이 구간은 관측된 다음 event의 duration으로 나눈 사후 설명이다. encoder 입력·모델 선택·floor/temperature 탐색·새 Test평가에 사용하지 않았다. Train·Validation 효과만으로 기존 Test의 시간 반전이 해결됐다고 말할 수 없으며, 좁은 σ를 바꾼 배포 모델·학습된 calibration·CNN/GRU의 필요성을 증명하지 않는다.","", "전체/구간별 합·제곱합·3seed 평균/표본SD와 모델 원본 여섯 지표는 `time_cross_conditions.json`에 모두 보존했다."]
    return "\n".join(lines)+"\n"


def analyze(root,expected_contract,expected_operation):
    root=Path(root).resolve()
    payload=summarize(root,expected_contract,expected_operation)
    write_exclusive(root/"analysis_claim.json",{"contract_sha256":expected_contract,"operation_integrity_sha256":expected_operation,"created_unix":time.time(),"raw_predictions_written":False,"scientific_inference_calls":0})
    write_exclusive(root/"analysis/time_cross_conditions.json",payload)
    report=root/"analysis/TIME_CROSS_REPORT.md";report.parent.mkdir(parents=True,exist_ok=True)
    with report.open("x",encoding="utf-8") as stream:stream.write(render(payload))
    receipt={"status":"complete","schema":SCHEMA+"_completion","contract_sha256":expected_contract,"operation_integrity_sha256":expected_operation,"actual_model_split_passes":18,"actual_joint_os_workers_exit0":6,"paired_score_cells":48,"new_cross_cells":24,"new_fits":0,"test_inference":0,"calibration_fits":0,"conditions_sha256":sha(root/"analysis/time_cross_conditions.json"),"report_sha256":sha(report),"completed_unix":time.time()}
    write_exclusive(root/"analysis/completion_receipt.json",receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--contract-sha256",required=True);p.add_argument("--operation-sha256",required=True)
    a=p.parse_args();print(json.dumps(analyze(a.root,a.contract_sha256,a.operation_sha256),ensure_ascii=False),flush=True)

if __name__ == "__main__":main()
