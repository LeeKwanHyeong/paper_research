"""CLI for one model/seed, without historical campaign controllers."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phen.data import EventWindows, load_frame
from phen.experiment import (ROOT, model_kwargs, prepare_data, record_manifest,
                             sha_file, validate_config, write_json)
from phen.registry import MODELS, build_model
from phen.training import run_epoch, save_predictions, set_seed, train


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", choices=MODELS, default="phen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path, help="Relocate an input without changing its frozen SHA")
    parser.add_argument("--device", default="cpu", help="Explicit cpu or cuda device; never auto-select a GPU")
    parser.add_argument("--output-root", type=Path, default=ROOT / "search_artifacts/phen")
    parser.add_argument("--mode", choices=("train", "preflight", "evaluate"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-held-out", action="store_true", help="Explicitly unlock test evaluation")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--min-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Read configuration only, without data or output writes")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if args.split == "test" and (args.mode != "evaluate" or not args.allow_held_out):
        raise PermissionError("Test requires --mode evaluate --allow-held-out and a selected checkpoint")
    if args.mode == "evaluate" and args.checkpoint is None:
        raise ValueError("Evaluation requires --checkpoint")
    if args.mode != "evaluate" and (args.checkpoint is not None or args.allow_held_out):
        raise ValueError("Checkpoint and held-out options are evaluation-only")
    config = json.loads(args.config.read_text())
    for argument, key in ((args.epochs, "epochs"), (args.min_epochs, "min_epochs"), (args.patience, "patience")):
        if argument is not None:
            config["training"][key] = argument
    validate_config(config)
    path = args.data or ROOT / config["data_path"]
    output = args.output_root.resolve() / config["dataset_id"] / args.model / f"seed_{args.seed}"
    if output.is_relative_to(ROOT / "phen") or output.is_relative_to(ROOT / "scripts/phen"):
        raise ValueError("Output must be outside source directories to keep source snapshots clean")
    if args.mode == "evaluate":
        output = output / ("evaluation_" + args.split)
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "model": args.model, "seed": args.seed, "device": args.device,
                          "data_path": str(path), "output": str(output), "config": config,
                          "data_read": False, "training_executed": False}, indent=2))
        return
    if args.mode != "preflight" and output.exists():
        raise FileExistsError("Preserve existing run; select a new --output-root: " + str(output))
    datasets, info = prepare_data(config, path)
    if args.mode == "preflight":
        print(json.dumps(info, indent=2))
        return
    if args.mode == "train":
        result = train(config, datasets, info, model_name=args.model, seed=args.seed,
                       device=args.device, output=output)
    else:
        set_seed(args.seed)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if payload.get("schema") != "phen_checkpoint_v1" or payload["model"] != args.model or payload["seed"] != args.seed:
            raise ValueError("Checkpoint schema, model or seed differs")
        if args.split == "test" and payload["epoch"] != payload["best_epoch"]:
            raise ValueError("Test requires the validation-selected best checkpoint, not a last-epoch diagnostic")
        for key in ("dataset_id", "model", "loader", "statistics"):
            if payload["config"][key] != config[key]:
                raise ValueError("Checkpoint configuration differs: " + key)
        if payload["data"]["quantity_boundaries"] != info["quantity_boundaries"]:
            raise ValueError("Checkpoint quantity reporting boundaries differ")
        if payload["config"]["training"]["lambda_log_qty"] != config["training"]["lambda_log_qty"]:
            raise ValueError("Checkpoint quantity-loss weight differs")
        if info["populations"] != payload["data"]["populations"]:
            raise ValueError("Checkpoint target populations differ")
        for relative, expected in payload["source_files_sha256"].items():
            if relative.endswith(".py") and "/tests/" not in relative and sha_file(ROOT / relative) != expected:
                raise ValueError("Checkpoint model/runtime source changed: " + relative)
        if args.split == "test":
            frame = load_frame(path, include_test=True)
            datasets["test"] = EventWindows(frame, target_split="test", lookback=config["loader"]["lookback"],
                                            max_seq_len=config["loader"]["max_seq_len"])
            info = {**info, "held_out_materialized": True,
                    "test_population": datasets["test"].population()}
        output.mkdir(parents=True, exist_ok=False)
        record_manifest(output, config=config, model_name=args.model, seed=args.seed,
                        device=args.device, data=info, mode="evaluate_" + args.split)
        model = build_model(args.model, **model_kwargs(config)).to(args.device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        loader = DataLoader(datasets[args.split], batch_size=config["loader"]["batch_size"], shuffle=False)
        metrics, rows = run_epoch(model, loader, device=args.device, training=config["training"],
                                  boundaries=info["quantity_boundaries"], predictions=True)
        save_predictions(output / (args.split + "_predictions.parquet"), rows, datasets[args.split])
        result = {"status": "complete", "split": args.split, "metrics": metrics,
                  "checkpoint_sha256": sha_file(args.checkpoint), "selected_epoch": payload["epoch"],
                  "held_out_test_evaluated": args.split == "test"}
        write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
