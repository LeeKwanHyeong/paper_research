#!/usr/bin/env python3
"""Stdlib-only RAF Train/Validation native-log/raw sufficient-statistic audit.

No checkpoint deserialization, model import, network call, optimizer, Test
receipt/data lookup, selector, or row-prediction persistence is implemented.
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

CONTRACT = "b1d554f5e80e874b8c7a62aec5b3c113d6b74a2a956f9044964f910cdee56f4c"
KIND = "titantpp_raf_log_raw_20261012_v1"
SOURCE = "c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a"
FULL = "titantpp_cnn_gru54"
ARMS = (FULL, FULL + "_correction_disabled", FULL + "_current_only")
SEEDS = (42, 52, 62)
SPLITS = ("train", "validation")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe(root, relative):
    path = Path(root) / relative
    require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
            and path.resolve().is_relative_to(Path(root).resolve()), "Path escapes RAF bundle")
    return path


def read_bound(root, relative, files, expected=None):
    path = safe(root, relative)
    digest = sha(path)
    require(expected is None or digest == expected, "RAF evidence bytes differ: " + relative)
    files[relative] = digest
    return json.loads(path.read_bytes())


def close(a, b, name, relative=1e-8, absolute=1e-9):
    require(type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=relative, abs_tol=absolute), "RAF arithmetic differs: " + name)


def verify_sums(aggregate):
    cells = aggregate["cells"]
    boundaries = aggregate["quantity_boundaries"]
    require(aggregate["schema"] == "native_per_target_log_raw_sufficient_statistics_v1"
            and boundaries == [2., 30., 60., 200.]
            and aggregate["tail_key"] == "4" and aggregate["raw_rows_persisted"] is False,
            "RAF log/raw schema or fixed bins differ")
    require(set(cells) == {"all", "0", "1", "2", "3", "4"}, "RAF log/raw bins incomplete")
    for forecast in ("model", "last", "mean"):
        whole = cells["all"][forecast]
        n = whole["count"]
        require(n == aggregate["count"] > 0, "RAF log/raw population differs")
        require(sum(cells[str(i)][forecast]["count"] for i in range(5)) == n, "RAF log/raw bin counts differ")
        for field in whole:
            if field.endswith("_sum") or field == "raw_sae":
                close(whole[field], math.fsum(cells[str(i)][forecast][field] for i in range(5)), forecast + "/bin sum/" + field)
        for key, predictions in cells.items():
            cell = predictions[forecast]
            count = cell["count"]
            require(type(count) is int and count >= 0 and all(v is None or (type(v) in (int, float) and math.isfinite(v)) for v in cell.values()), "RAF nonfinite cell")
            for field in ("raw_prediction", "raw_truth", "raw_error", "log_prediction", "log_truth", "log_error"):
                if count:
                    close(cell[field + "_mean"], cell[field + "_sum"] / count, key + "/" + field + "/mean")
                    require(cell[field + "_square_sum"] >= 0., "Negative second moment")
                else:
                    require(cell[field + "_mean"] is None and cell[field + "_sum"] == cell[field + "_square_sum"] == 0., "Empty RAF moment fabricated")
            close(cell["raw_error_sum"], cell["raw_prediction_sum"] - cell["raw_truth_sum"], key + "/signed raw")
            # Signed errors were subtracted in native float32. Their summed
            # roundoff is bounded by epsilon times the input magnitudes;
            # detached float64 sums must not be mistaken for native subtraction.
            roundoff=(2.**-23)*(abs(cell["log_prediction_sum"])+abs(cell["log_truth_sum"]))
            close(cell["log_error_sum"], cell["log_prediction_sum"] - cell["log_truth_sum"], key + "/signed log", absolute=max(1e-9,roundoff))
            for field, numerator, sqrt in (("raw_rmse", "raw_error_square_sum", True),
                    ("raw_mae", "raw_sae", False), ("raw_bias", "raw_error_sum", False),
                    ("log_mse_native", "log_native_loss_sum", False),
                    ("log_rmse_native", "log_native_loss_sum", True), ("log_bias", "log_error_sum", False)):
                if count:
                    expected = cell[numerator] / count
                    close(cell[field], math.sqrt(expected) if sqrt else expected, key + "/" + field)
                else:
                    require(cell[field] is None and cell[numerator] == 0., "Empty RAF performance fabricated")
            require(cell["log_native_loss_sum"] >= 0., "Negative native log loss")
    return True


def contrasts(rows):
    lookup = {(r["arm"], r["seed"], r["split"]): r for r in rows}
    require(set(lookup) == {(a, s, p) for a in ARMS for s in SEEDS for p in SPLITS}, "RAF comparison matrix differs")
    comparisons = []
    for split in SPLITS:
        for seed in SEEDS:
            full = lookup[(FULL, seed, split)]
            for arm in ARMS[1:]:
                control = lookup[(arm, seed, split)]
                require(full["target_identity_sha256"] == control["target_identity_sha256"]
                        and full["truth_history_sha256"] == control["truth_history_sha256"], "RAF paired population differs")
                for key in ("all", "4"):
                    f, c = full["log_raw"]["cells"][key]["model"], control["log_raw"]["cells"][key]["model"]
                    require(f["count"] == c["count"], "RAF paired count differs")
                    delta = {metric: f[metric] - c[metric] for metric in
                        ("raw_rmse", "raw_mae", "raw_bias", "log_mse_native", "log_bias")}
                    comparisons.append({"split": split, "seed": seed, "control_arm": arm,
                        "scope": "all" if key == "all" else "quantity_gt_200", "count": f["count"],
                        "full_minus_control": delta,
                        "raw_worse_log_better": delta["raw_rmse"] > 0. and delta["log_mse_native"] < 0.,
                        "raw_better_log_worse": delta["raw_rmse"] < 0. and delta["log_mse_native"] > 0.})
    return comparisons


def seed_summary(rows):
    output = []
    for split in SPLITS:
        for arm in ARMS:
            group = [r for r in rows if r["split"] == split and r["arm"] == arm]
            require(len(group) == 3, "RAF seed group incomplete")
            summary = {"split": split, "arm": arm, "seed_count": 3, "scopes": {}}
            for key in ("all", "4"):
                scope = {}
                for forecast in ("model", "last", "mean"):
                    scope[forecast] = {}
                    for metric in ("raw_rmse", "raw_mae", "raw_bias", "log_mse_native", "log_bias"):
                        values = [r["log_raw"]["cells"][key][forecast][metric] for r in group]
                        scope[forecast][metric] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}
                summary["scopes"]["all" if key == "all" else "quantity_gt_200"] = scope
            output.append(summary)
    return output


def loss_contributions(aggregate):
    whole=aggregate["cells"]["all"]["model"]
    tail=aggregate["cells"][aggregate["tail_key"]]["model"]
    require(whole["count"]>0 and 0<=tail["count"]<=whole["count"],"RAF contribution counts differ")
    return {"count":whole["count"],"tail_count":tail["count"],
        "tail_target_share":tail["count"]/whole["count"],
        "tail_share_of_quantity_log_loss":tail["log_native_loss_sum"]/whole["log_native_loss_sum"] if whole["log_native_loss_sum"] else None,
        "tail_share_of_raw_quantity_sse":tail["raw_error_square_sum"]/whole["raw_error_square_sum"] if whole["raw_error_square_sum"] else None,
        "quantity_loss_only_not_joint_time_quantity_objective":True,
        "eval_checkpoint_not_training_trajectory":True}


def render_report(c, rows, pairs, summaries):
    arm_name = {FULL: "full", ARMS[1]: "보정 제거", ARMS[2]: "current-only"}
    lines = ["# RAF의 실제 log 수량 목표와 raw 오류 — 완료", "",
        "동일 선택 checkpoint9개×Train·Validation18회 관측이다. 신규 학습0·Test 평가0이며 모델·출력 개입과 선택 epoch 변경은 없다. 실제 per-target float32 log_qty_loss를 관측했고, 충분통계만 저장했다.", "",
        "## 3seed 평균과 표본 SD", "",
        "| split | arm | raw 전체 RMSE | 전체 log MSE | raw 큰 수량 RMSE | 큰 수량 log MSE | 큰 수량 raw bias |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for summary in summaries:
        all_m = summary["scopes"]["all"]["model"]
        tail_m = summary["scopes"]["quantity_gt_200"]["model"]
        def fmt(cell): return f"{cell['mean']:.6f} ± {cell['sample_sd']:.6f}"
        lines.append("| " + " | ".join([summary["split"], arm_name[summary["arm"]],fmt(all_m["raw_rmse"]),fmt(all_m["log_mse_native"]),fmt(tail_m["raw_rmse"]),fmt(tail_m["log_mse_native"]),fmt(tail_m["raw_bias"])]) + " |")
    lines += ["", "## 모든 seed와 두 대조의 관측", "",
        "| split | seed | 대조 | 범위 | full−대조 raw RMSE | full−대조 log MSE |",
        "|---|---:|---|---|---:|---:|"]
    for pair in pairs:
        d=pair["full_minus_control"]
        lines.append(f"| {pair['split']} | {pair['seed']} | {arm_name[pair['control_arm']]} | {pair['scope']} | {d['raw_rmse']:+.6f} | {d['log_mse_native']:+.6f} |")
    lines += ["", "## 같은 대상을 평가한 단순 기준선", "",
        "| split | scope | forecast | raw RMSE | log MSE | raw bias |",
        "|---|---|---|---:|---:|---:|"]
    for split in SPLITS:
        full_summary=next(x for x in summaries if x["split"]==split and x["arm"]==FULL)
        for scope in ("all","quantity_gt_200"):
            for forecast in ("model","last","mean"):
                v=full_summary["scopes"][scope][forecast]
                lines.append(f"| {split} | {scope} | {forecast} | {v['raw_rmse']['mean']:.6f} | {v['log_mse_native']['mean']:.6f} | {v['raw_bias']['mean']:.6f} |")
    lines += ["", "## 먼저 구분해야 하는 두 관측", ""]
    for scope in ("all", "quantity_gt_200"):
        group=[p for p in pairs if p["scope"]==scope]
        opposite=sum(p["raw_worse_log_better"] for p in group)
        reverse=sum(p["raw_better_log_worse"] for p in group)
        both_worse=sum(p["full_minus_control"]["raw_rmse"]>0 and p["full_minus_control"]["log_mse_native"]>0 for p in group)
        lines.append(f"- 같은 {scope} 대상의 full−두 대조12조건: raw RMSE 악화·log MSE 개선 {opposite}/12, raw 개선·log 악화 {reverse}/12, 두 오류 모두 악화 {both_worse}/12다. 나머지·동률도 원본24조건에 보존했다.")
    lines.append("- 전체 log MSE 이득과 큰 수량 raw 손해는 모집단이 다른 두 요약이다. 같은 큰 수량에서 log MSE도 나쁘다면 이를 'log 목표는 큰 수량을 잘 맞췄지만 raw만 악화했다'는 증거로 바꾸지 않는다. 전체/큰 수량의 가중 비중과 동일 부분집합 내 목표 방향을 분리한다.")
    lines += ["", "## 큰 수량의 실제 관측 손실 비중", "",
        "| split | full seed | 큰 수량 / 전체 대상 | 대상 비중 | 수량 log 손실 비중 | raw 수량 SSE 비중 |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["arm"]!=FULL:continue
        v=loss_contributions(row["log_raw"])
        def percentage(x):return f"{100*x:.4f}%" if x is not None else "미정(전체 오류0)"
        lines.append(f"| {row['split']} | {row['seed']} | {v['tail_count']} / {v['count']} | {percentage(v['tail_target_share'])} | {percentage(v['tail_share_of_quantity_log_loss'])} | {percentage(v['tail_share_of_raw_quantity_sse'])} |")
    lines += ["", "위 비중은 같은 선택 모델의 eval 상태에서 관측한 수량 오류다. 시간 손실을 포함한 joint objective 전체 비중이나 학습 중 gradient 기여는 아니며, 두 대조의 동일 충분통계도 계산 원본에 보존했다."]
    lines += ["", "## 해석의 범위", "",
        "- 실제 저장 모델은 고정 GPU/Runtime의 원래 cuDNN eval 경로로 평가했다. 별도 synthetic native qualification의 forward 비개입 검증도 같은 경로다. synthetic gradient hook 검증만 eval-mode cuDNN RNN의 backward 제한 때문에 ATen 경로를 일시 사용하고 즉시 복원했다. 저장 모델 gradient나 학습을 실행하지 않았다.",
        "- actual log 목표의 전체/큰 수량 방향과 raw 오류의 방향은 같은 이벤트에서 비교한다. raw 손해·log 이득의 방향이 다르면 목표 간 상충의 관측으로 기록하며 그 반대와 불리한 조건도 유지한다.",
        "- raw 수량 point 예측은 frozen softplus log location→expm1 raw decode 그대로다. raw 해독의 cap·상수·tail weight·optimizer·학습 gradient를 변경하지 않았다.",
        "- native log loss는 실제 float32 per-target tensor다. signed log 오차 제곱합을 float64로 다시 계산한 값과 마지막 반올림이 다를 수 있어 별도 필드로 보존한다. scalar 평균을 target나 bin으로 분해하지 않았다.",
        "- 큰 수량은 true quantity>200인 사후 부분집합이다. Train에서도 큰 수량 오류가 남는 사실만으로 용량 부족·최적화 실패를 확정하지 않는다. 현재 평가 관측은 손실 변경을 하면 개선된다는 인과 검증, 통계 유의성, 새 자료 일반화, CNN/GRU 필요성의 증명이 아니다.",
        "- Test 자료와 그 수치는 이번 관측·분석에서 열람하지 않았다. 실제 추가 학습·scale fit·모델 선택은 별도 미실행이다.", "",
        "계산 원본: [raf_log_raw_conditions.json](raf_log_raw_conditions.json). 평가 split과 checkpoint/source/target/truth-history SHA는 각 entry에 보존했다.", ""]
    return "\n".join(lines)


def collect(root, expected_operation, expected_contract=CONTRACT):
    root = Path(root).resolve()
    files={}
    c=read_bound(root,"contract.json",files)
    require(c["schema"]==KIND and c["contract_sha256"]==expected_contract
            and canonical({k:v for k,v in c.items() if k!="contract_sha256"})==expected_contract,
            "RAF contract pin differs")
    require(c["sources"]["history"]["closure"]==SOURCE
            and c["h1"]["evaluation_splits"]==list(SPLITS)
            and c["h1"]["new_fits"]==c["h1"]["test_inference"]==0, "RAF scope/source differs")
    seal=read_bound(root,"operation_integrity.json",files,expected_operation)
    require(seal["contract_sha256"]==expected_contract,"RAF operation identity differs")
    for relative,digest in seal["files"].items():
        require(sha(safe(root,relative))==digest,"RAF operation changed: "+relative);files[relative]=digest
    spec=importlib.util.spec_from_file_location("_sealed_raf_log_raw_adapter",safe(root,"control/evaluate.py"))
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    evaluator.verify_contract(root);evaluator.verify_operations(root,c)
    verification=read_bound(root,"originals/LOCAL_VERIFICATION.json",files)
    retrieval=read_bound(root,"originals/receipt.json",files)
    require(verification.get("status")=="verified" and verification["contract_sha256"]==expected_contract
            and verification.get("all_archive_members_and_local_files_sha_verified") is True
            and Path(verification["verified_root"]).resolve()==safe(root,"originals/verified"), "RAF originals not locally verified")
    require(retrieval.get("status")=="verified_terminal_archived" and retrieval.get("contract_sha256")==expected_contract
            and retrieval.get("operation_integrity_sha256")==expected_operation
            and retrieval.get("actual_pipeline_exit0") is True and retrieval.get("actual_worker_exit0_count")==18
            and retrieval.get("native_qualification_pass") is True and retrieval.get("new_fits")==retrieval.get("test_inference")==0
            and retrieval.get("own_processes_absent") is True and retrieval.get("own_gpu_pids_absent") is True,
            "RAF terminal retrieval proof incomplete")
    manifest=read_bound(root,"originals/manifest.json",files,verification["manifest_sha256"])
    files["originals/originals.tar.gz"]=sha(safe(root,"originals/originals.tar.gz"))
    require(files["originals/originals.tar.gz"]==verification["archive_sha256"]==retrieval["archive_sha256"]
            and files["originals/manifest.json"]==retrieval["manifest_sha256"]
            and manifest["contract_sha256"]==expected_contract
            and len(manifest["files"])==verification["file_count"]==retrieval["file_count"], "RAF original archive/manifest differs")
    for relative,digest in manifest["files"].items():
        for prefix in ("","originals/verified/"):
            require(sha(safe(root,prefix+relative))==digest,"RAF immutable local bytes differ: "+prefix+relative)
            files[prefix+relative]=digest
    # Terminal schema bindings are checked below against the actual approved
    # pipeline, never inferred from a tmux session disappearing.
    terminal=verify_terminal(root,c,files)
    rows=[]
    baseline_identities={}
    for job in sorted(c["jobs"],key=lambda j:(j["seed"],j["arm"])):
        for split in SPLITS:
            r=evaluator.validate_receipt(root,c,job,split)
            relative=f"results/{job['id']}/{split}_metrics.json";files[relative]=sha(safe(root,relative))
            require(files[relative]==terminal["receipts"][job["id"]][split],"RAF completed receipt SHA differs")
            t=read_bound(root,r["telemetry_path"],files,r["telemetry_sha256"])
            verify_sums(t["log_raw"])
            x=read_bound(root,f"results/{job['id']}/{split}_worker_exit.json",files,terminal["worker_exits"][job["id"]][split])
            require(x.get("exit_code")==0 and x.get("actual_child_exit_observed") is True
                    and x.get("timed_out") is False and x.get("supervisor_aborted") is False
                    and x.get("job_id")==job["id"] and x.get("split")==split
                    and x.get("contract_sha256")==expected_contract,"RAF actual worker exit differs")
            argv=[c["runtime"]["python"],c["execution"]["remote_root"]+"/control/evaluate.py","--root",c["execution"]["remote_root"],"--job-id",job["id"],"--split",split,"--device","cuda"]
            require(x["argv"]==argv and r["argv"]==argv[1:] and r["pid"]==x["pid"] and r["ppid"]==x["ppid"]
                    and terminal["qualification_completed_unix"]<=r["started_unix"]<=r["completed_unix"]<=x["completed_unix"],"RAF qualification/native process bridge differs")
            require(r["runtime"]["gpu"]["uuid"]==c["runtime"]["gpu_uuid"]
                    and all(r["runtime"][key]==c["runtime"][key] for key in ("torch","cuda","cudnn","numpy","polars")),"RAF native Runtime differs")
            require(r["evaluator_sha256"]==seal["files"]["control/evaluate.py"]
                    and r["telemetry_module_sha256"]==seal["files"]["control/titantpp_history_diagnostic_telemetry.py"]
                    and r["operation_integrity_sha256"]==expected_operation,"RAF worker operation differs")
            baseline={key:{name:t["log_raw"]["cells"][key][name] for name in ("last","mean")} for key in t["log_raw"]["cells"]}
            require(baseline_identities.setdefault(split,baseline)==baseline,"RAF input-only log/raw baselines differ across arms/seeds")
            rows.append({"id":job["id"],"dataset":job["dataset"],"arm":job["arm"],"seed":job["seed"],
                "split":split,"selected_epoch":job["selected_epoch"],"checkpoint_sha256":r["checkpoint_sha256"],
                "state_sha256":r["state_sha256"],"source_closure":r["source_closure"],
                "target_identity_sha256":r["target_identity_sha256"],"truth_history_sha256":r["truth_sha256"],
                "metrics":r["metrics"],"log_raw":t["log_raw"],
                "tail_loss_contributions":loss_contributions(t["log_raw"])})
    require(len(rows)==18,"RAF 18 evaluations incomplete")
    return c,rows,files,terminal


def verify_terminal(root,c,files):
    require(not safe(root,"control/pipeline_failure.json").exists(),"Preserved RAF pipeline failure")
    complete=read_bound(root,"control/pipeline_complete.json",files)
    latest=read_bound(root,"control/pipeline_latest.json",files)
    actual=read_bound(root,"control/pipeline_exit.json",files)
    require(actual.get("exit_code")==0 and actual.get("actual_child_exit_observed") is True,"RAF actual pipeline OS exit missing")
    ids={j["id"] for j in c["jobs"]}
    require(len(ids)==9 and complete.get("status")=="complete" and complete.get("contract_sha256")==c["contract_sha256"]
            and complete.get("train_count")==complete.get("validation_count")==9 and complete.get("model_split_passes")==18
            and complete.get("actual_all_worker_exit0") is True and complete.get("new_fits")==complete.get("test_inference")==0
            and set(complete["receipts"])==set(complete["worker_exits"])==ids
            and all(set(v)==set(SPLITS) for collection in (complete["receipts"],complete["worker_exits"]) for v in collection.values()), "RAF18 completion bridge differs")
    require(latest.get("status")==latest.get("phase")=="complete" and latest.get("active")==[]
            and latest.get("contract_sha256")==c["contract_sha256"]
            and set(latest["train_completed"])==set(latest["validation_completed"])==ids,"RAF last phase incomplete")
    qualification=read_bound(root,"control/native_qualification.json",files)
    require(qualification.get("status")=="PASS" and qualification.get("synthetic_only") is True
            and qualification.get("contract_sha256")==c["contract_sha256"]
            and set(qualification["routes"])=={"history__"+arm for arm in ARMS},"RAF native3 admission incomplete")
    for route,digest in qualification["routes"].items():
        probe=read_bound(root,f"control/native_probes/{route}/receipt.json",files,digest)
        exit=read_bound(root,f"control/native_probes/{route}/worker_exit.json",files)
        require(probe.get("status")=="PASS" and probe.get("arm")==route.removeprefix("history__")
                and probe.get("contract_sha256")==c["contract_sha256"] and probe.get("source_closure")==SOURCE
                and probe.get("native_pinned_contract_checked") is True and probe.get("checkpoint_reads")==probe.get("optimizer_updates")==0
                and probe.get("real_data_loaded") is False and probe.get("held_out_test_evaluated") is False
                and probe.get("hooked_and_unhooked_all_original_outputs_bitwise_equal") is True
                and probe.get("hooked_and_unhooked_fixture_gradients_bitwise_equal") is True
                and probe.get("fixture_gradient_backend")=="aten_cudnn_disabled_then_restored"
                and probe.get("native_forward_qualification_backend_unchanged") is True
                and probe.get("fixture_model_eval_mode_maintained") is True
                and probe.get("actual_native_log_loss_per_target_reconstructed_bitwise") is True
                and all(v is True for v in probe["qualification"].values())
                and exit.get("exit_code")==0 and exit.get("actual_child_exit_observed") is True
                and exit["completed_unix"]<=qualification["completed_unix"],"RAF synthetic native identity differs")
    return {**complete,"actual_pipeline_os_exit0_verified":True,
            "qualification_completed_unix":qualification["completed_unix"],"pipeline_exit_observed_unix":actual["observed_unix"]}


def write_exclusive(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        if isinstance(value,str):stream.write(value)
        else:json.dump(value,stream,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--operation-sha",required=True)
    parser.add_argument("--contract-sha256",default=CONTRACT,
                        help="Explicit pinned canonical contract for a reviewed operational revision")
    args=parser.parse_args(argv)
    root=args.root.resolve()
    require(not (root/"analysis_claim.json").exists() and not (root/"analysis_failure.json").exists()
            and not (root/"analysis").exists(),"Existing RAF analysis/claim/failure forbids overwrite")
    c,rows,files,terminal=collect(root,args.operation_sha,expected_contract=args.contract_sha256)
    write_exclusive(root/"analysis_claim.json",{"schema":"raf_log_raw_analysis_claim_v1","contract_sha256":c["contract_sha256"],"observed_unix":time.time(),"analysis_script_sha256":sha(__file__)})
    pairs=contrasts(rows);summaries=seed_summary(rows)
    payload={"schema":"raf_selected_log_raw_analysis_v1","status":"complete","contract_sha256":c["contract_sha256"],
             "operation_integrity_sha256":args.operation_sha,"new_fits":0,"test_inference":0,
             "actual_worker_exit0_count":18,"rows":rows,"paired_contrasts":pairs,"seed_summary":summaries,
             "source_files":files,"terminal":terminal,"limitations":{"causal_loss_change_effect":False,"fresh_generalization":False,"component_necessity":False,"statistical_significance":False}}
    relative="analysis/raf_log_raw_conditions.json";write_exclusive(root/relative,payload)
    report="analysis/RAF_LOG_RAW_REPORT.md";write_exclusive(root/report,render_report(c,rows,pairs,summaries))
    write_exclusive(root/"analysis/completion_receipt.json",{"status":"complete","contract_sha256":c["contract_sha256"],
        "actual_worker_exit0_count":18,"actual_pipeline_os_exit0_verified":True,"new_fits":0,"test_inference":0,
        "analysis_script_sha256":sha(__file__),"source_files":files,
        "analysis_files":{relative:sha(root/relative),report:sha(root/report)}})
    print(json.dumps({"status":"complete","row_count":len(rows),"paired_contrast_count":len(pairs),"new_fits":0,"test_inference":0}),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
