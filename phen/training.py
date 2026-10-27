"""One common optimizer, objective and validation selector for all encoders."""

import math
import json
import os
from pathlib import Path
import random

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from phen.data import target_outputs
from phen.experiment import model_kwargs, record_manifest, write_json
from phen.registry import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
    os.replace(temporary, path)


def run_epoch(model, loader, *, device, training, boundaries, optimizer=None, predictions=False):
    model.train(optimizer is not None)
    sums = {key: 0.0 for key in ("joint_loss", "time_loss", "log_qty_loss", "abs_error", "sq_error", "time_abs_error")}
    strata = [[0, 0.0, 0.0] for _ in range(5)]
    count, records = 0, []
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            device_batch = {key: value.to(device) for key, value in batch.items()}
            outputs = target_outputs(model, device_batch, lambda_log_qty=training["lambda_log_qty"])
            if any(not torch.isfinite(value).all() for value in outputs.values()):
                raise FloatingPointError("Nonfinite prediction or objective")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                outputs["joint_loss"].mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
            values = {key: value.detach().cpu().numpy().astype(np.float64) for key, value in outputs.items()}
            truth, prediction = values["true_qty"], values["pred_qty"]
            error = prediction - truth
            n = len(truth)
            count += n
            for key in ("joint_loss", "time_loss", "log_qty_loss"):
                sums[key] += float(values[key].sum())
            sums["abs_error"] += float(np.abs(error).sum())
            sums["sq_error"] += float(np.square(error).sum())
            sums["time_abs_error"] += float(np.abs(values["pred_dt"] - values["true_dt"]).sum())
            bins = np.searchsorted(boundaries, truth, side="left")
            for index in range(5):
                selected = error[bins == index]
                strata[index][0] += len(selected)
                strata[index][1] += float(np.abs(selected).sum())
                strata[index][2] += float(np.square(selected).sum())
            if predictions:
                for row, sample in enumerate(batch["index"].tolist()):
                    records.append({"sample_index": sample, **{key: float(values[key][row]) for key in
                                    ("true_qty", "pred_qty", "true_dt", "pred_dt", "history_length", "time_loss")}})
    if count == 0:
        raise ValueError("No events evaluated")
    metrics = {key: sums[key] / count for key in ("joint_loss", "time_loss", "log_qty_loss")}
    metrics.update(count=count, quantity_mae=sums["abs_error"] / count,
                   quantity_rmse=math.sqrt(sums["sq_error"] / count), time_mae=sums["time_abs_error"] / count,
                   quantity_strata=[{"count": n, "mae": ae / n if n else None,
                                     "rmse": math.sqrt(se / n) if n else None} for n, ae, se in strata])
    return metrics, records


def save_predictions(path, records, dataset):
    for row in records:
        part, end = dataset.index[row["sample_index"]]
        row["oper_part_no"] = str(dataset.parts[part])
        row["target_seq"] = int(dataset.sequences[part][end + 1])
    pl.DataFrame(records).write_parquet(path)


def train(config, datasets, data_info, *, model_name, seed, device, output):
    generator = set_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = record_manifest(output, config=config, model_name=model_name, seed=seed,
                               device=device, data=data_info, mode="train_validation")
    loaders = {split: DataLoader(dataset, batch_size=config["loader"]["batch_size"],
                                 shuffle=split == "train", num_workers=0,
                                 generator=generator if split == "train" else None)
               for split, dataset in datasets.items()}
    model = build_model(model_name, **model_kwargs(config)).to(device)
    settings = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    best_rmse, best_epoch = math.inf, 0
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        train_metrics, _ = run_epoch(model, loaders["train"], device=device, training=settings,
                                     boundaries=data_info["quantity_boundaries"], optimizer=optimizer)
        validation, _ = run_epoch(model, loaders["validation"], device=device, training=settings,
                                  boundaries=data_info["quantity_boundaries"])
        score = validation["quantity_rmse"]
        improved = score < best_rmse
        if improved:
            best_rmse, best_epoch = score, epoch
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
        payload = {"schema": "phen_checkpoint_v1", "model": model_name, "seed": seed,
                   "config": config, "epoch": epoch, "best_epoch": best_epoch,
                   "best_validation_rmse": best_rmse, "model_state_dict": model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(),
                   "source_files_sha256": manifest["source_files_sha256"], "data": data_info,
                   "torch_rng_state": torch.random.get_rng_state(),
                   "loader_rng_state": generator.get_state()}
        atomic_checkpoint(output / "last.pt", payload)
        if improved:
            atomic_checkpoint(output / "best.pt", payload)
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(history[-1], allow_nan=False) + "\n")
        print(f"epoch={epoch} validation_quantity_rmse={score:.8f} best_epoch={best_epoch}", flush=True)
        if epoch >= settings["min_epochs"] and epoch - best_epoch >= settings["patience"]:
            break
    selected = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    metrics, rows = run_epoch(model, loaders["validation"], device=device, training=settings,
                              boundaries=data_info["quantity_boundaries"], predictions=True)
    save_predictions(output / "validation_predictions.parquet", rows, datasets["validation"])
    summary = {"status": "complete", "model": model_name, "seed": seed, "completed_epochs": epoch,
               "best_epoch": best_epoch, "selected_validation": metrics,
               "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
               "parameter_count": sum(p.numel() for p in model.parameters())}
    write_json(output / "summary.json", summary)
    return summary
