"""Configuration checks and reproducibility records shared by all models."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile

import numpy as np
import polars as pl
import torch

from phen.data import EventWindows, load_frame


ROOT = Path(__file__).resolve().parents[1]


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_config(config):
    if config.get("schema") != "phen_reproduction_v1":
        raise ValueError("Unsupported reproduction configuration")
    if not config.get("dataset_id") or Path(config["dataset_id"]).name != config["dataset_id"] or config["dataset_id"] in (".", ".."):
        raise ValueError("dataset_id must be a safe directory name")
    seeds = config.get("seeds")
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must contain unique integers in [0, 2**32)")
    loader, train = config["loader"], config["training"]
    for key, minimum in (("batch_size", 1), ("lookback", 1), ("max_seq_len", 2)):
        if type(loader[key]) is not int or loader[key] < minimum:
            raise ValueError("Invalid loader setting: " + key)
    for key in ("epochs", "min_epochs", "patience"):
        if type(train[key]) is not int or train[key] < 1:
            raise ValueError("Invalid training setting: " + key)
    if train["min_epochs"] > train["epochs"]:
        raise ValueError("min_epochs exceeds epochs")
    for key in ("lr", "grad_clip", "lambda_log_qty"):
        if not math.isfinite(train[key]) or train[key] <= 0:
            raise ValueError("Invalid training setting: " + key)
    if not math.isfinite(train["weight_decay"]) or train["weight_decay"] < 0:
        raise ValueError("Invalid weight_decay")
    if "quantity_boundaries" in config:
        boundaries = config["quantity_boundaries"]
        if (len(boundaries) != 4 or not all(math.isfinite(x) and x >= 0 for x in boundaries)
                or any(a > b for a, b in zip(boundaries, boundaries[1:]))):
            raise ValueError("Quantity boundaries must be four ordered finite nonnegative values")
    return config


def prepare_data(config, data_path):
    digest = sha_file(data_path)
    accepted = config.get("accepted_data_sha256", [])
    if accepted and digest not in accepted:
        raise ValueError("Input SHA256 differs from the frozen configuration; use a new reviewed configuration")
    frame = load_frame(data_path)
    loader = config["loader"]
    datasets = {split: EventWindows(frame, target_split=split, lookback=loader["lookback"],
                                   max_seq_len=loader["max_seq_len"])
                for split in ("train", "validation")}
    populations = {split: dataset.population() for split, dataset in datasets.items()}
    for split, expected in config.get("expected_populations", {}).items():
        if split not in populations or populations[split] != expected:
            raise ValueError("Frozen target population differs: " + split)
    quantities = frame.filter(pl.col("chronological_split") == "train")["demand_qty"].to_numpy().astype(np.float64)
    log_qty = np.log1p(quantities)
    statistics = {"train_log_mean": float(log_qty.mean()), "train_log_std": float(log_qty.std())}
    for key, expected in config["statistics"].items():
        if key in statistics and not math.isclose(statistics[key], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Train-only initialization statistic differs: " + key)
    if statistics["train_log_mean"] <= 0:
        raise ValueError("Quantity initialization requires positive train log mean")
    train = datasets["train"]
    durations = np.array([train.durations[p][i + 1] for p, i in train.index], dtype=np.float64)
    time_scale = float(np.median(durations))
    scaled = np.log(durations / time_scale)
    temporal = {"time_scale": time_scale, "time_initial_location": float(scaled.mean()),
                "time_initial_scale": float(scaled.std())}
    for key, expected in temporal.items():
        if not math.isclose(config["model"][key], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Train-target time statistic differs: " + key)
    observation = config["model"]["time_observation_contract"]
    top_code = observation["top_code"]
    for dataset in datasets.values():
        gaps = np.array([dataset.durations[p][i + 1] for p, i in dataset.index])
        if top_code is not None and (gaps > top_code).any():
            raise ValueError("Duration target exceeds configured top code")
    boundaries = (config["quantity_boundaries"] if "quantity_boundaries" in config
                  else np.quantile(quantities, [.5, .9, .95, .99]).tolist())
    return datasets, {"data_path": str(Path(data_path).resolve()), "data_sha256": digest,
                      "populations": populations, "statistics": statistics,
                      "temporal_statistics": temporal, "held_out_materialized": False,
                      "quantity_boundaries": boundaries,
                      "quantity_boundaries_source": "frozen_configuration" if "quantity_boundaries" in config else "train_quantiles"}


def model_kwargs(config):
    return {**config["model"], "train_log_mean": config["statistics"]["train_log_mean"],
            "max_seq_len": config["loader"]["max_seq_len"]}


def source_files():
    files = [p for p in (ROOT / "phen").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts]
    files += list((ROOT / "scripts/phen").glob("*.sh"))
    return sorted(files)


def record_manifest(output, *, config, model_name, seed, device, data, mode):
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
    files = source_files()
    with tarfile.open(output / "source.tar.gz", "x:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    manifest = {"schema": "phen_run_v1", "config": config, "model": model_name,
                "seed": seed, "mode": mode, "device": device, "data": data,
                "git_revision": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
                "source_status": git("status", "--porcelain", "--", "phen", "scripts/phen"),
                "source_files_sha256": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
                "source_archive_sha256": sha_file(output / "source.tar.gz"),
                "runtime": {"python": sys.version, "platform": platform.platform(),
                            "torch": str(torch.__version__), "numpy": np.__version__, "polars": pl.__version__,
                            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                            "environment": {k: os.environ.get(k) for k in ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")}},
                "selection": "earliest_strict_finite_validation_raw_quantity_rmse_minimum",
                "historical_training_trajectory_reproduced": False}
    write_json(output / "manifest.json", manifest)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    with (output / "pip_freeze.txt").open("x") as stream:
        stream.write(freeze)
    return manifest
