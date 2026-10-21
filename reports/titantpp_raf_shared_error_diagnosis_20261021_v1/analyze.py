"""Local-only RAF error accounting from immutable stored Train/Validation rows."""
from pathlib import Path
import gzip
import hashlib
import json
import math
import statistics
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SHARED = "titantpp_history_mlp_shared15_v1"
MODELS = [SHARED, "rmtpp", "thp", "nhp", "sahp", "s2p2_matched_head", "attnhp_matched_head"]
SEEDS = [42, 52, 62]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def metrics(frame):
    q = frame.raw_quantity.to_numpy(dtype=float)
    p = frame.predicted_raw_quantity.to_numpy(dtype=float)
    e = p - q
    if not len(e):
        return {"count": 0}
    return {"count": len(e), "rmse": float(np.sqrt(np.mean(e * e))),
        "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
        "sse": float(np.sum(e * e)), "ae_sum": float(np.sum(np.abs(e))),
        "overprediction_share": float(np.mean(e > 0)),
        "underprediction_share": float(np.mean(e < 0)),
        "true_mean": float(np.mean(q)), "prediction_mean": float(np.mean(p))}


def partitions(frame):
    q, h, last = frame.raw_quantity, frame.history_length, frame.last_quantity
    bounds = [0, 1, 2, 10, 30, 60, 200, float("inf")]
    quantity = {f"{a:g}<q<={b:g}": (q > a) & (q <= b) for a, b in zip(bounds[:-1], bounds[1:])}
    return {"quantity": quantity,
        "history": {"h<=2": h <= 2, "3<=h<=4": (h > 2) & (h <= 4),
                    "5<=h<=8": (h > 4) & (h <= 8), "h>=9": h >= 9},
        "shock": {"rise_at_least5x_last": q >= last * 5,
                  "drop_to_one_fifth_last": q <= last / 5,
                  "middle": (q < last * 5) & (q > last / 5)},
        "tail": {"q<=200": q <= 200, "q>200": q > 200},
        "last_level": {"last<=10": last <= 10, "last>10": last > 10}}


def describe(frame):
    overall = metrics(frame)
    result = {"overall": overall, "partitions": {}}
    for partition, masks in partitions(frame).items():
        parts = {}
        assert sum(int(mask.sum()) for mask in masks.values()) == len(frame)
        for name, mask in masks.items():
            value = metrics(frame.loc[mask])
            if value["count"]:
                value.update(sse_share=value["sse"] / overall["sse"],
                             ae_share=value["ae_sum"] / overall["ae_sum"])
            parts[name] = value
        assert math.isclose(sum(v.get("sse", 0) for v in parts.values()), overall["sse"], rel_tol=1e-12)
        assert math.isclose(sum(v.get("ae_sum", 0) for v in parts.values()), overall["ae_sum"], rel_tol=1e-12)
        result["partitions"][partition] = parts
    return result


def main():
    sources = read(OUT / "SOURCE_MAP.json")
    for path, digest in sources["files_sha256"].items():
        assert sha(ROOT / path) == digest, path
    frames, receipts = {}, {}
    for item in sources["external"]:
        frame = pd.concat([pd.read_parquet(ROOT / part["path"]) for part in item["parts"]], ignore_index=True)
        assert set(frame.split) == {item["split"]}
        assert len(frame) == item["rows"]
        key = item["model"], item["seed"], item["split"]
        frames[key] = frame.rename(columns={"seq": "target_seq"})
        receipts[key] = read(ROOT / item["receipt"])
    for item in sources["shared"]:
        with gzip.open(ROOT / item["paths"]["validation_predictions.jsonl.gz"], "rt") as stream:
            frame = pd.DataFrame(json.loads(line) for line in stream)
        base = frames["rmtpp", item["seed"], "validation"]
        for key in ["entity_id", "target_seq", "raw_quantity", "history_length", "target_index"]:
            assert np.array_equal(frame[key].to_numpy(), base[key].to_numpy()), key
        for key in ["last_quantity", "mean_quantity", "history_quantity_std", "history_quantity_max", "recorded_date"]:
            frame[key] = base[key].to_numpy()
        key = SHARED, item["seed"], "validation"
        frames[key], receipts[key] = frame, read(ROOT / item["paths"]["validation_receipt.json"])

    cohorts = []
    for (model, seed, split), frame in frames.items():
        value = describe(frame)
        ref = receipts[model, seed, split]["metrics"]
        ref = ref.get("overall", ref)
        assert math.isclose(value["overall"]["rmse"], ref["qty_rmse"], rel_tol=1e-10)
        assert math.isclose(value["overall"]["mae"], ref["qty_mae"], rel_tol=1e-10)
        cohorts.append({"model": model, "seed": seed, "split": split, **value})

    comparisons = []
    for seed in SEEDS:
        shared = frames[SHARED, seed, "validation"]
        for model in MODELS[1:]:
            external = frames[model, seed, "validation"]
            for key in ["entity_id", "target_seq", "raw_quantity", "history_length"]:
                assert np.array_equal(shared[key], external[key]), key
            se = shared.predicted_raw_quantity.to_numpy() - shared.raw_quantity.to_numpy()
            ee = external.predicted_raw_quantity.to_numpy() - external.raw_quantity.to_numpy()
            ae_delta, se_delta = np.abs(se) - np.abs(ee), se * se - ee * ee
            value = {"external": model, "seed": seed,
                "mae_delta": float(ae_delta.mean()), "mse_delta": float(se_delta.mean()),
                "rmse_delta": float(np.sqrt(np.mean(se * se)) - np.sqrt(np.mean(ee * ee))),
                "partitions": {}}
            for partition, masks in partitions(shared).items():
                parts = {}
                for name, mask in masks.items():
                    a = mask.to_numpy()
                    parts[name] = {"count": int(a.sum()),
                        "mae_delta_contribution": float(ae_delta[a].sum() / len(a)),
                        "mse_delta_contribution": float(se_delta[a].sum() / len(a)),
                        "shared_mae": float(np.abs(se[a]).mean()) if a.any() else None,
                        "external_mae": float(np.abs(ee[a]).mean()) if a.any() else None}
                assert math.isclose(sum(v["mae_delta_contribution"] for v in parts.values()), value["mae_delta"], abs_tol=1e-12)
                assert math.isclose(sum(v["mse_delta_contribution"] for v in parts.values()), value["mse_delta"], abs_tol=1e-9)
                value["partitions"][partition] = parts
            comparisons.append(value)

    concentration = []
    for model in [SHARED, "rmtpp", "s2p2_matched_head"]:
        copies = []
        for seed in SEEDS:
            frame = frames[model, seed, "validation"].copy()
            frame["ae"] = np.abs(frame.predicted_raw_quantity - frame.raw_quantity)
            frame["se"] = (frame.predicted_raw_quantity - frame.raw_quantity) ** 2
            copies.append(frame[["entity_id", "target_seq", "raw_quantity", "ae", "se"]])
        pooled = pd.concat(copies)
        by_row = pooled.groupby(["entity_id", "target_seq", "raw_quantity"])[["ae", "se"]].mean().reset_index()
        entities = by_row.groupby("entity_id")[["ae", "se"]].sum().sort_values("se", ascending=False)
        concentration.append({"model": model,
            "note": "Mean of seed-specific error sums, not errors of an ensemble prediction",
            "entity_count": len(entities), "top_entity_sse_shares": {str(k): float(entities.se.head(k).sum() / entities.se.sum()) for k in [1, 5, 10, 20, 50]},
            "top_entities": [{"entity": idx, "sse": float(row.se), "ae": float(row.ae)} for idx, row in entities.head(10).iterrows()],
            "largest_errors": by_row.sort_values("se", ascending=False).head(15).to_dict(orient="records")})

    summary = []
    for model in MODELS:
        for split in ["train", "validation"]:
            selected = [c for c in cohorts if c["model"] == model and c["split"] == split]
            if not selected:
                continue
            values = {key: {"mean": statistics.mean(c["overall"][key] for c in selected),
                            "sample_sd": statistics.stdev(c["overall"][key] for c in selected)}
                      for key in ["rmse", "mae", "bias", "overprediction_share"]}
            for name in ["q<=200", "q>200"]:
                values[name] = {key: statistics.mean(c["partitions"]["tail"][name][key] for c in selected)
                                for key in ["rmse", "mae", "bias", "sse_share", "ae_share"]}
            summary.append({"model": model, "split": split, "seed_count": len(selected), **values})
    write("RESULT.json", {"schema": "raf_shared_error_accounting_v1", "scope": "stored Train/Validation only",
        "current_shared_row_seed_count": 3, "validation_target_count": 6690,
        "shared_train_predictions_available": False, "new_training_inference_ssh_test_reads": 0,
        "seed_summary": summary, "cohorts": cohorts, "paired_error_decomposition": comparisons,
        "concentration": concentration, "validation_gate": "all 39 cohort metrics reproduce stored receipts; exact target alignment; additive AE/SSE partition checks passed",
        "causal_identification": False})
    for model in [SHARED, "rmtpp", "s2p2_matched_head"]:
        print(json.dumps(next(s for s in summary if s["model"] == model and s["split"] == "validation"), ensure_ascii=False))
    for model in ["rmtpp", "s2p2_matched_head"]:
        cs = [c for c in comparisons if c["external"] == model]
        print("DECOMPOSITION", model)
        for part in ["quantity", "history", "shock"]:
            print(part, json.dumps({name: {metric: statistics.mean(c["partitions"][part][name][metric] for c in cs) for metric in ["mae_delta_contribution", "mse_delta_contribution"]} for name in cs[0]["partitions"][part]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
