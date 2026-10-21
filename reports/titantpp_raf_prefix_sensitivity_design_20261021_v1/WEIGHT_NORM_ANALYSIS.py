"""Read selected Shared15 weights on CPU; no model imports or forward/evaluation."""
import datetime
import hashlib
import json
import pathlib

import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha(state):
    # Same tensor-name/dtype/shape/byte convention as runner.py:524-549.
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        assert isinstance(value, torch.Tensor), name
        value = value.detach().cpu().contiguous()
        meta = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(meta).to_bytes(8, "big"))
        digest.update(meta)
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes() if value.numel() else b""
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


root = pathlib.Path(__file__).resolve().parents[2]
out = pathlib.Path(__file__).resolve().parent
source_path = root / "reports/titantpp_raf_shared_error_diagnosis_20261021_v1/SOURCE_MAP.json"
source = json.loads(source_path.read_text())
torch.set_num_threads(1)
torch.set_grad_enabled(False)
cohorts = []
for item in source["shared"]:
    base = (root / item["paths"]["train_receipt.json"]).parent
    matches = list(base.glob("runs/**/best_val_qty_rmse_model.pt"))
    assert len(matches) == 1, matches
    path = matches[0]
    before = sha(path)
    assert before == item["checkpoint_sha256"], path
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["seed"] == item["seed"]
    assert payload["best_epoch"] == item["selected_epoch"]
    state = payload["model_state_dict"]
    canonical = state_sha(state)
    assert canonical == item["state_sha256"] == payload["model_state_sha256"]
    u_key = "multilag_detail.input_projection.weight"
    b_keys = [f"multilag_detail.output_projections.{j}.weight" for j in range(8)]
    u = state[u_key].to(dtype=torch.float64, device="cpu")
    bs = [state[key].to(dtype=torch.float64, device="cpu") for key in b_keys]
    assert tuple(u.shape) == (15, 128)
    assert all(tuple(b.shape) == (64, 15) for b in bs)
    assert bool(torch.isfinite(u).all())
    assert all(bool(torch.isfinite(b).all()) for b in bs)
    uf = float(torch.linalg.matrix_norm(u, ord="fro"))
    uo = float(torch.linalg.matrix_norm(u, ord=2))
    c = torch.zeros((64, 15), dtype=torch.float64)
    prefixes = []
    for k, b in enumerate(bs, start=1):
        c = c + b / 8
        cf = float(torch.linalg.matrix_norm(c, ord="fro"))
        co = float(torch.linalg.matrix_norm(c, ord=2))
        fro_bound = 1.25 * uf * cf
        op_bound = 1.25 * uo * co
        prefixes.append({
            "k": k, "eligibility_history_greater_than": 2 ** (k - 1),
            "actually_used_in_raf_train_validation": k <= 4,
            "C_frobenius": cf, "C_operator": co,
            "frobenius_sufficient_bound": fro_bound,
            "operator_sufficient_bound": op_bound,
            "proposed_s_k_L1": 1 / max(1.0, fro_bound),
            "frobenius_would_bind": fro_bound > 1,
            "operator_product_sufficient_bound_exceeds_L1": op_bound > 1,
            "actual_input_jacobian_measured": False,
        })
    assert sha(path) == before, path
    cohorts.append({
        "seed": item["seed"], "selected_epoch": item["selected_epoch"],
        "checkpoint_path": str(path.relative_to(root)),
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256_before_after": before,
        "canonical_state_sha256": canonical,
        "tensor_keys": [u_key, *b_keys],
        "U_frobenius": uf, "U_operator": uo, "prefixes": prefixes,
    })

used = [p for c in cohorts for p in c["prefixes"] if p["actually_used_in_raf_train_validation"]]
if not any(p["frobenius_would_bind"] for p in used):
    decision = "NO_TRAINING_BASIS_BOUND_INACTIVE_AT_ALL_SELECTED_USED_PREFIXES"
elif not any(p["operator_product_sufficient_bound_exceeds_L1"] for p in used):
    decision = "NO_TRAINING_BASIS_ONLY_FROBENIUS_CONSERVATISM_WOULD_SHRINK"
else:
    decision = "NORM_PREREQUISITE_POSSIBLY_MET_NOT_ACTUAL_AMPLIFICATION_OR_TRAINING_AUTHORIZATION"

result = {
    "schema": "raf_shared15_selected_weight_norms_v1",
    "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_map_sha256": sha(source_path),
    "analysis_source_sha256": sha(pathlib.Path(__file__)),
    "torch_version_cpu_analysis_only": torch.__version__,
    "calculation_dtype": "float64", "device": "cpu",
    "scope": {"selected_shared15_checkpoints_loaded": 3, "model_instances": 0,
              "model_forward": 0, "new_training": 0, "new_evaluation": 0,
              "test_metrics_read": 0, "ssh": 0, "originals_modified": 0},
    "L": 1.0, "gamma": 1.25, "cohorts": cohorts,
    "decision": decision,
    "limitation": "Norm product is a sufficient upper bound for fixed k and weights. It neither measures input-specific Jacobians nor establishes drop causality. Selected weights do not describe every training epoch.",
}
with (out / "WEIGHT_NORMS.json").open("x") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"decision": decision, "used_prefixes": used}, ensure_ascii=False, indent=2))
