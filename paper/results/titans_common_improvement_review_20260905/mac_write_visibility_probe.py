#!/usr/bin/env python3
"""Synthetic, random-initialization audit of existing B1 MAC write visibility.

No training, real data, GPU, checkpoint, or model-source changes are performed.
Reproduce with:
/usr/local/bin/python3 -s -B /tmp/titans_common_improvement_review/mac_write_visibility_probe.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import torch


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/Users/igwanhyeong/PycharmProjects/paper_research"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from models.Titan.common.titans_mac import TitansMACEncoder

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(314159)
    model = TitansMACEncoder(
        input_dim=2, d_model=64, n_layers=2, n_heads=4, d_ff=128,
        persistent_memory_size=16, segment_size=16, max_len=32, dropout=0.1,
    ).to(device="cpu", dtype=torch.float32).eval()
    model.neural_memory.compile_cuda_scan = False
    model.neural_memory.gradient_max_norm = 1.0
    initial_parameter_digest = parameter_digest(model)
    generator = torch.Generator(device="cpu").manual_seed(271828)
    raw_dts = torch.randint(1, 13, (1, 32), generator=generator).float()
    raw_quantities = torch.randint(0, 51, (1, 32), generator=generator).float()
    output_probe = torch.randn(64, generator=generator)
    rows = []
    writer_prefixes = (
        "key_projection.", "value_projection.", "update_rate_projection.",
        "momentum_projection.", "forgetting_projection.",
    )

    for history_length in (16, 17):
        mask = torch.arange(32).unsqueeze(0) <= history_length
        write_mask = torch.arange(32).unsqueeze(0) < history_length
        history_quantities = raw_quantities.clone()
        history_quantities[:, history_length] = 0.0
        inputs = torch.stack((torch.log1p(raw_dts), torch.log1p(history_quantities)), dim=-1)
        inputs = inputs * mask.unsqueeze(-1)
        history_position = history_length - 1
        with torch.no_grad():
            full, full_state, full_diagnostics = model.forward_with_state(
                inputs, mask=mask, write_mask=write_mask,
            )
            no_update, no_update_state, no_update_diagnostics = model.forward_with_state(
                inputs, mask=mask, write_mask=torch.zeros_like(write_mask),
            )
        full_selected = full[:, history_position]
        no_update_selected = no_update[:, history_position]
        delta = full_selected - no_update_selected

        # Probe the actual hidden state, avoiding the independent known confound
        # that the common quantity head is initialized with all-zero weights.
        model.zero_grad(set_to_none=True)
        encoded = model(inputs, mask=mask, write_mask=write_mask)
        scalar_probe = (encoded[:, history_position] * output_probe).sum()
        scalar_probe.backward()
        gradient_summary = {}
        for name, parameter in model.neural_memory.named_parameters():
            gradient = parameter.grad
            gradient_summary[name] = {
                "is_none": gradient is None,
                "finite": None if gradient is None else bool(torch.isfinite(gradient).all()),
                "l2_norm": None if gradient is None else float(gradient.double().norm()),
                "max_abs": None if gradient is None else float(gradient.abs().max()),
                "nonzero_elements": None if gradient is None else int(torch.count_nonzero(gradient)),
            }
        writer_gradients = {
            name: item for name, item in gradient_summary.items()
            if name.startswith(writer_prefixes)
        }
        row = {
            "history_length": history_length,
            "target_position_zero_based": history_length,
            "prediction_history_position_zero_based": history_position,
            "prediction_segment_start_zero_based": (history_position // 16) * 16,
            "prior_visible_write_count": (history_position // 16) * 16,
            "input_sequence_length_with_right_padding": 32,
            "valid_token_count_including_masked_target": int(mask.sum()),
            "full_write_count": int(full_diagnostics["write_applied"].sum()),
            "no_update_write_count": int(no_update_diagnostics["write_applied"].sum()),
            "full_vs_no_update_selected_state_bitwise_equal": torch.equal(full_selected, no_update_selected),
            "full_vs_no_update_selected_state_max_abs": float(delta.abs().max()),
            "full_vs_no_update_selected_state_l2": float(delta.double().norm()),
            "final_memory_full_vs_no_update_max_abs": max(
                float((a - b).abs().max())
                for a, b in zip(full_state.memory_tensors(), no_update_state.memory_tensors(), strict=True)
            ),
            "full_selected_state_sha256": tensor_digest(full_selected),
            "no_update_selected_state_sha256": tensor_digest(no_update_selected),
            "synthetic_input_sha256": tensor_digest(inputs),
            "writer_gradient_summary": writer_gradients,
            "all_memory_parameter_gradient_summary": gradient_summary,
        }
        assert row["full_write_count"] == history_length
        assert row["no_update_write_count"] == 0
        if history_length == 16:
            assert row["full_vs_no_update_selected_state_bitwise_equal"]
            # Appended/padded later segments remain in the concatenation graph;
            # autograd can allocate exactly-zero gradients instead of None.
            assert all(item["is_none"] or item["l2_norm"] == 0.0 for item in writer_gradients.values())
        else:
            assert row["full_vs_no_update_selected_state_max_abs"] > 1e-8
            assert all(not item["is_none"] and item["finite"] and item["l2_norm"] > 0 for item in writer_gradients.values())
        rows.append(row)

    assert parameter_digest(model) == initial_parameter_digest
    sources = [
        "models/Titan/common/titans_mac.py",
        "models/Titan/common/titans_memory_stability.py",
        "paper/scripts/count_aware_tpp_backbone/core.py",
        "data_loader/event_seq_data_module.py",
    ]
    report = {
        "status": "PASS",
        "scope": "Synthetic random-initialized existing B1 MAC; structural visibility/gradient audit only",
        "real_data_used": False,
        "training_performed": False,
        "optimizer_steps": 0,
        "gpu_used": False,
        "checkpoint_loaded": False,
        "model_parameters_unchanged": True,
        "interpretation": [
            "H=16 writes occur, but the last observed prediction state uses initial segment-start memory, so those writes cannot affect that target state.",
            "H=17 prediction crosses one segment boundary, allowing the first 16 observed writes to affect its selected state and writer gradients.",
            "The synthetic scalar loss probes encoder visibility, not benchmark quantity/time metrics or trained performance.",
            "This cannot establish that the visibility boundary causes Instacart errors or that a new schedule improves any dataset.",
        ],
        "environment": {
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "torch": torch.__version__,
            "device": "cpu", "dtype": "float32", "torch_num_threads": torch.get_num_threads(),
            "model_mode": "eval", "dropout_active": False,
            "compiled_scan": False, "inner_gradient_max_norm": 1.0,
        },
        "model_configuration": {
            "input_dim": 2, "hidden_dim": 64, "layers": 2, "heads": 4,
            "ffn_dim": 128, "persistent_tokens": 16, "segment_size": 16,
            "max_len": 32, "configured_dropout": 0.1,
        },
        "seeds": {"initialization": 314159, "synthetic_inputs_and_probe": 271828},
        "model_parameters_sha256": initial_parameter_digest,
        "repo": str(args.repo),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
        "source_sha256": {name: file_digest(args.repo / name) for name in sources},
        "script_sha256": file_digest(Path(__file__)),
        "reproduce_command": f"/usr/local/bin/python3 -s -B {Path(__file__).resolve()}",
        "cases": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "output": str(args.output),
        "cases": [{
            "H": row["history_length"],
            "write_count": row["full_write_count"],
            "prior_visible_write_count": row["prior_visible_write_count"],
            "output_equal": row["full_vs_no_update_selected_state_bitwise_equal"],
            "output_max_abs_difference": row["full_vs_no_update_selected_state_max_abs"],
            "writer_grad_l2": {name: value["l2_norm"] for name, value in row["writer_gradient_summary"].items()},
        } for row in rows],
    }, indent=2))


if __name__ == "__main__":
    main()
