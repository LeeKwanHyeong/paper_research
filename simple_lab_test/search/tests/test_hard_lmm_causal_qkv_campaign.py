from __future__ import annotations

import json
import sys

import pytest

from paper.scripts import run_hard_lmm_causal_qkv_campaign as campaign
from paper.scripts import profile_hard_lmm_causal_qkv as profile
from paper.scripts import run_count_aware_tpp_backbone_control as trainer


def test_campaign_rejects_5080_and_incomplete_source(tmp_path):
    base = ["--output-root", str(tmp_path), "--source-revision", "a" * 40]
    with pytest.raises(SystemExit):
        campaign.parse_args([*base, "--host-role", "5080"])
    with pytest.raises(SystemExit):
        campaign.parse_args(["--output-root", str(tmp_path), "--source-revision", "abc"])


def test_cost_policy_exactly_matches_frozen_contract():
    contract = campaign.read_json(campaign.CONTRACT_PATH)
    cost = contract["cost_gate"]
    assert cost == dict(batch_size=profile.BATCH_SIZE, lengths=list(profile.LENGTHS),
                        warmup_steps=profile.WARMUP_STEPS, measured_steps=profile.MEASURED_STEPS,
                        repeats=profile.REPEATS, max_step_ratio=profile.STEP_LIMIT,
                        max_peak_allocated_ratio=profile.PEAK_LIMIT)


@pytest.mark.parametrize("dataset", list(campaign.DATASETS))
@pytest.mark.parametrize("phase_name", ["e1", "seed42_screening"])
def test_actual_trainer_accepts_every_generated_command(monkeypatch, tmp_path, dataset, phase_name):
    contract = campaign.read_json(campaign.CONTRACT_PATH)
    command = campaign.job_command(
        python=sys.executable, candidate=contract["candidate"], dataset=dataset,
        output=tmp_path, source_revision="a" * 40,
        phase=contract["phases"][phase_name], host_role="5090",
    )
    monkeypatch.setattr(sys, "argv", command[2:])
    args = trainer.parse_args()
    variants = trainer.normalize_quantity_variants(args.quantity_variants)
    trainer.validate_model_role_contract(
        model_role=args.model_role, backbones=(args.backbones,), quantity_variants=variants,
        time_head_mode=args.time_head_mode, lambda_tail=args.lambda_tail,
    )
    assert args.model_role == "hard_lmm_causal_qkv_candidate"
    assert args.checkpoint_monitor == "validation_raw_quantity_rmse"
    assert args.quantile_adaptive_strength == 0
    assert args.time_intercept_limit == 300
    assert args.device == "cuda"
    assert args.max_train_batches is args.max_val_batches is args.max_series is None


@pytest.mark.parametrize("failure", ["cost", "e1", "screening", "none"])
def test_conditional_sequence_never_runs_past_failed_prerequisite(monkeypatch, tmp_path, failure):
    from paper.scripts import audit_hard_lmm_causal_qkv as audit_module

    monkeypatch.setattr(campaign, "gpu_preflight", lambda _: {"gpu_name": "RTX 5090"})
    monkeypatch.setattr(campaign, "verify_inputs", lambda _: {})
    calls = []

    def run(command, **kwargs):
        if str(campaign.PROFILER) in command:
            path = tmp_path / "cuda_cost_profile.json"
            path.write_text(json.dumps({"status": "failed" if failure == "cost" else "passed",
                                        "cost_gate": {"status": "passed"}}))
        else:
            calls.append(command[command.index("--dataset-contract") + 1])

    def audit(output, **kwargs):
        if failure == "e1":
            raise ValueError("e1 contract failed")
        metrics = dict(raw_rmse=0.1, overall_mae=0.1, body_mae=0.1,
                       gt_p99_mae=0.1, clamped_time_loss=-10)
        if failure == "screening" and output.name.startswith("seed42"):
            metrics["raw_rmse"] = 10000
        return {"status": "passed", "metrics": metrics}

    monkeypatch.setattr(campaign.subprocess, "run", run)
    monkeypatch.setattr(audit_module, "audit_causal_qkv_job", audit)
    argv = ["--output-root", str(tmp_path), "--source-revision", "a" * 40]
    if failure in ("cost", "e1"):
        with pytest.raises(ValueError):
            campaign.main(argv)
    else:
        campaign.main(argv)
    status = campaign.read_json(tmp_path / "campaign_status.json")
    expected_count = {"cost": 0, "e1": 1, "screening": 4, "none": 6}[failure]
    assert len(calls) == expected_count
    assert status["held_out_test_evaluated"] is False
    assert status["status"] == {
        "cost": "failed_execution", "e1": "failed_execution",
        "screening": "stopped_seed42_gate_failed", "none": "seed42_all_datasets_passed",
    }[failure]
    if failure == "screening":
        assert calls[-1] == "insta_market_basket"
    with pytest.raises(ValueError, match="exists"):
        campaign.main(argv)
