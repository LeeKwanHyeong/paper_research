#!/usr/bin/env python3
"""Validation-only replay of normalized integer-duration likelihood conventions."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
from models.TPPs.CausalLogDurationAdapter import CausalLogDurationAdapter
from paper.scripts.audit_duration_observation_likelihood import (
    _conditional_parameters,
    _standardized_log_boundary,
    centered_integer_log_mass,
)
from paper.scripts.run_hard_lmm_causal_duration_adapter import (
    FrozenBaseTimeCache,
    GlobalScaleControl,
    _batch_terms,
    compact_masked_history,
    load_admitted_frame,
    make_indexed_loader,
    prepare_count_frame,
)


OUTPUT = Path(__file__).resolve().with_name(
    "taxi_normalized_integer_likelihood_replay_20260906.json"
)
DATA = ROOT / "sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet"
AUDIT_SCRIPT = ROOT / "paper/scripts/audit_duration_observation_likelihood.py"
ADAPTER_SCRIPT = ROOT / "paper/scripts/run_hard_lmm_causal_duration_adapter.py"
ADAPTER_MODULE = ROOT / "models/TPPs/CausalLogDurationAdapter.py"
DATASET_MODULE = ROOT / "data_loader/event_seq_data_module.py"
EXPECTED_MODEL_SOURCE_REVISION = "17b215a5c74453940a627dee049a5ee91796f005"
EXPECTED_DATA_SHA256 = "b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46"
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "4a005d4a77a89f7ca793d8de56afb9267a3ca4a5e60c53e09465c0494d60ed85"
)
EXPECTED_VALIDATION_TARGET_IDENTITY_SHA256 = (
    "19493e9265c6b733b03a08fafc391a86abf2f46b4efb0fc140c8d48534e407fb"
)
EXPECTED_SOURCE_FILE_SHA256 = {
    "audit": "a55a882de22822e68c0cfadcefc98ba3abbbbadac5ec230a8b7aaaaf40487a27",
    "runner": "f40f646feb47a1eae113421b680c78ca72d59f92570b144cf177fed211042b1d",
    "adapter": "8d969ecc71fd30ad5cd3f38ac22019d72fc08c93d140274e78605bbe4c75844e",
    "dataset": "7c11d9ace6c4e4d3bb03861aae92393aa946a38452d702dc7625210b339897ca",
}

ROLE_INPUTS = {
    "A": (
        ROOT / "search_artifacts/duration_observation_likelihood_audit_20260906/remote_taxi_matched/A/cache/validation_features.pt",
        ROOT / "search_artifacts/duration_observation_likelihood_audit_20260906/remote_taxi_matched_full/A/best_validation_proper_time_nll_model.pt",
    ),
    "B": (
        ROOT / "search_artifacts/hard_lmm_time_adapter_diagnostic_20260906/cache/yellow_trip_hourly/cache/validation_features.pt",
        ROOT / "search_artifacts/hard_lmm_frozen_lognormal_duration_seed42_5090_20260906_c04d32b/full/yellow_trip_hourly/best_validation_proper_time_nll_model.pt",
    ),
    "rmtpp": (
        ROOT / "search_artifacts/duration_observation_likelihood_audit_20260906/remote_taxi_matched/rmtpp/cache/validation_features.pt",
        ROOT / "search_artifacts/duration_observation_likelihood_audit_20260906/remote_taxi_matched_full/rmtpp/best_validation_proper_time_nll_model.pt",
    ),
    "thp": (
        ROOT / "search_artifacts/duration_observation_likelihood_audit_20260906/remote_taxi_matched/thp/cache/validation_features.pt",
        ROOT / "search_artifacts/duration_observation_likelihood_audit_20260906/remote_taxi_matched_full/thp/best_validation_proper_time_nll_model.pt",
    ),
}

ADAPTER_INPUTS = {
    "causal_scale_adapter": ROOT / "search_artifacts/hard_lmm_causal_duration_adapter_seed42_5090_20260906_17b215a/seed42/yellow_trip_hourly/candidate/best_validation_continuous_nll_model.pt",
    "global_scale_control": ROOT / "search_artifacts/hard_lmm_causal_duration_adapter_seed42_5090_20260906_17b215a/seed42/yellow_trip_hourly/global_scale_control/best_validation_continuous_nll_model.pt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


def likelihoods(
    target: torch.Tensor, location: torch.Tensor, scale: torch.Tensor, time_scale: float
) -> dict[str, float]:
    target = target.to(torch.float64)
    location = location.to(torch.float64)
    scale = scale.to(torch.float64)
    legacy_logp = centered_integer_log_mass(
        target_dt=target,
        location=location,
        scale=scale,
        time_scale=time_scale,
        top_code=None,
    )
    one = target == 1.0

    # D=max(1, round(T)): all sub-0.5 mass is folded into the first category.
    fold_logp = legacy_logp.clone()
    upper_z = _standardized_log_boundary(
        torch.full_like(target[one], 1.5),
        location[one],
        scale[one],
        time_scale=time_scale,
    )
    fold_logp[one] = torch.special.log_ndtr(upper_z)

    # D=round(T) conditional on D>=1: renormalize the legacy bins by S(0.5).
    half_z = _standardized_log_boundary(
        torch.full_like(target, 0.5),
        location,
        scale,
        time_scale=time_scale,
    )
    log_survival_half = torch.special.log_ndtr(-half_z)
    truncated_logp = legacy_logp - log_survival_half

    return {
        "count": int(target.numel()),
        "dt_equals_1_count": int(one.sum().item()),
        "legacy_subnormalized_centered_score": float(-legacy_logp.mean().item()),
        "fold_to_one_normalized_nll": float(-fold_logp.mean().item()),
        "zero_truncated_normalized_nll": float(-truncated_logp.mean().item()),
        "mean_missing_mass_below_half": float(
            torch.special.ndtr(half_z).mean().item()
        ),
    }


def selected_role_replay() -> tuple[dict[str, dict], dict[str, object]]:
    rows: dict[str, dict] = {}
    inputs: dict[str, object] = {}
    canonical_target = None
    b_terms = None
    for role, (cache_path, checkpoint_path) in ROLE_INPUTS.items():
        cache = load(cache_path)
        checkpoint = load(checkpoint_path)
        identity = cache["identity"]
        assert identity["dataset"] == "yellow_trip_hourly"
        assert identity["split"] == "validation"
        assert identity["held_out_test_evaluated"] is False
        assert identity["data_sha256"] == EXPECTED_DATA_SHA256
        assert identity["split_manifest_sha256"] == EXPECTED_SPLIT_MANIFEST_SHA256
        assert (
            identity["target_identity_sha256"]
            == EXPECTED_VALIDATION_TARGET_IDENTITY_SHA256
        )
        assert identity["target_count"] == 8268
        assert checkpoint["evaluation_scope"] == "validation_only"
        assert checkpoint["held_out_test_evaluated"] is False
        hidden = cache["time_hidden"]
        target = cache["target_dt"]
        state = checkpoint["model_state_dict"]
        head = checkpoint["encoder_config"].get("time_head", {})
        statistics = checkpoint["train_time_statistics"]
        time_scale = float(statistics.get("time_scale", head.get("time_scale")))
        sigma_floor = float(head.get("time_sigma_floor", 0.001))
        location, scale = _conditional_parameters(
            hidden=hidden, state=state, sigma_floor=sigma_floor
        )
        if canonical_target is None:
            canonical_target = target
        else:
            assert torch.equal(target, canonical_target)
        rows[role] = likelihoods(target, location, scale, time_scale)
        rows[role]["selected_continuous_density_nll"] = float(
            checkpoint["selected_metric_value"]
        )
        inputs[role] = {
            "cache_path": repo_relative(cache_path),
            "cache_file_sha256": sha256_file(cache_path),
            "cache_identity_sha256": cache["cache_sha256"],
            "checkpoint_path": repo_relative(checkpoint_path),
            "checkpoint_file_sha256": sha256_file(checkpoint_path),
            "checkpoint_state_sha256": checkpoint["model_state_sha256"],
        }
        if role == "B":
            b_terms = (target, location, scale, time_scale, sigma_floor)
    assert b_terms is not None
    return rows, {"files": inputs, "B_terms": b_terms}


@torch.no_grad()
def adapter_replay(b_terms: tuple) -> tuple[dict[str, dict], dict[str, object]]:
    target, location, base_scale, time_scale, sigma_floor = b_terms
    frame = prepare_count_frame(load_admitted_frame(DATA))
    dataset = RMTPPWeekLookbackDataset(
        frame,
        lookback_weeks=168,
        max_seq_len=256,
        mode="all",
        split_col="chronological_split",
        target_splits={"validation"},
    )
    assert len(dataset) == target.numel() == 8268
    base_cache = FrozenBaseTimeCache(
        location=location.to(torch.float64).contiguous(),
        sigma=base_scale.to(torch.float64).contiguous(),
        median=(time_scale * torch.exp(location.to(torch.float64))).contiguous(),
        target_dt=target.detach().cpu().contiguous(),
    )
    base_cache.validate()

    modules = {}
    adapter_payload = load(ADAPTER_INPUTS["causal_scale_adapter"])
    adapter_state = adapter_payload["module_state_dict"]
    candidate = CausalLogDurationAdapter(
        log_duration_mean=float(adapter_state["log_duration_mean"].item()),
        log_duration_std=float(adapter_state["log_duration_std"].item()),
        sigma_floor=float(adapter_state["sigma_floor"].item()),
    )
    candidate.load_state_dict(adapter_state, strict=True)
    modules["causal_scale_adapter"] = (candidate, adapter_payload)

    control_payload = load(ADAPTER_INPUTS["global_scale_control"])
    control = GlobalScaleControl(sigma_floor=sigma_floor)
    control.load_state_dict(control_payload["module_state_dict"], strict=True)
    modules["global_scale_control"] = (control, control_payload)

    adjusted_scales = {name: [] for name in modules}
    locations = []
    targets = []
    for batch in make_indexed_loader(
        dataset, batch_size=4096, count=len(dataset), shuffle=False, seed=None
    ):
        _, history_dt, history_mask, batch_target, batch_location, batch_scale = (
            _batch_terms(batch=batch, base_cache=base_cache, device=torch.device("cpu"))
        )
        compact, lengths = compact_masked_history(history_dt, history_mask)
        locations.append(batch_location.to(torch.float64))
        targets.append(batch_target.to(torch.float64))
        for name, (module, _) in modules.items():
            module.eval()
            adjusted_scales[name].append(
                module.adjusted_scale(
                    batch_scale.to(torch.float64), compact, lengths
                ).to(torch.float64)
            )
    all_target = torch.cat(targets)
    all_location = torch.cat(locations)
    assert torch.equal(all_target.to(target.dtype), target)

    rows = {}
    inputs = {}
    for name, (_, payload) in modules.items():
        scale = torch.cat(adjusted_scales[name])
        rows[name] = likelihoods(all_target, all_location, scale, time_scale)
        rows[name]["selected_continuous_density_nll"] = float(
            payload["selected_metric_value"]
        )
        path = ADAPTER_INPUTS[name]
        inputs[name] = {
            "checkpoint_path": repo_relative(path),
            "checkpoint_file_sha256": sha256_file(path),
            "module_state_sha256": payload["module_state_sha256"],
            "best_epoch": int(payload["best_epoch"]),
        }
    return rows, inputs


def main() -> None:
    source_paths = {
        "audit": AUDIT_SCRIPT,
        "runner": ADAPTER_SCRIPT,
        "adapter": ADAPTER_MODULE,
        "dataset": DATASET_MODULE,
    }
    actual_source_hashes = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    assert actual_source_hashes == EXPECTED_SOURCE_FILE_SHA256

    role_rows, bundle = selected_role_replay()
    b_terms = bundle.pop("B_terms")
    adapter_rows, adapter_inputs = adapter_replay(b_terms)
    rows = {**role_rows, **adapter_rows}

    # Guard against silent replay drift from the already recorded legacy score.
    expected_legacy = {
        "A": 0.8244428630817175,
        "B": 0.8999188466342886,
        "rmtpp": 1.0363887672357808,
        "thp": 0.867298955371786,
        "causal_scale_adapter": 0.9069289202951727,
        "global_scale_control": 0.9112638440974633,
    }
    for role, expected in expected_legacy.items():
        actual = rows[role]["legacy_subnormalized_centered_score"]
        tolerance = 2e-12 if role in ROLE_INPUTS else 1e-6
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance), (role, actual, expected)
        rows[role]["legacy_replay_reference"] = expected
        rows[role]["legacy_replay_absolute_error"] = abs(actual - expected)

    payload = {
        "schema_version": 1,
        "scope": {
            "dataset": "yellow_trip_hourly",
            "split": "validation_only",
            "target_count": 8268,
            "held_out_test_evaluated": False,
            "training_or_checkpoint_selection": False,
        },
        "source_revision": EXPECTED_MODEL_SOURCE_REVISION,
        "likelihood_conventions": {
            "legacy_subnormalized": "p(d)=F(d+0.5)-F(max(0.5,d-0.5)); total mass S(0.5)",
            "fold_to_one_normalized": "D=max(1,round(T)); p(1)=F(1.5), p(d)=F(d+0.5)-F(d-0.5) for d>=2",
            "zero_truncated_normalized": "D=round(T) conditioned on D>=1; divide every legacy bin by S(0.5)",
        },
        "rows": rows,
        "inputs": {
            **bundle,
            "adapters": adapter_inputs,
            "data": {"path": repo_relative(DATA), "sha256": sha256_file(DATA)},
            "source_files": {
                repo_relative(path): sha256_file(path)
                for path in (AUDIT_SCRIPT, ADAPTER_SCRIPT, ADAPTER_MODULE, DATASET_MODULE)
            },
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, sort_keys=True))
    print(f"output={OUTPUT}")
    print(f"output_sha256={sha256_file(OUTPUT)}")


if __name__ == "__main__":
    main()
