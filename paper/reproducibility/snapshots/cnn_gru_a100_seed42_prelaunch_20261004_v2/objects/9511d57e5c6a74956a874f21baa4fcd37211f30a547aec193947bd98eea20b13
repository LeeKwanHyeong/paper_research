"""Pinned numerical runtime for isolated J/Q/T CPU and CUDA qualification.

Configuration precedes CUDA initialization. Identity reads the actual device and
library state; it is an execution precondition, never a substitute for the
separate continuous-versus-resumed CUDA qualification experiment.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import torch

CUBLAS_WORKSPACE = ":4096:8"
RUNTIME_SCHEMA = "jqt_strict_runtime_v2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _device(device: str | torch.device) -> torch.device:
    parsed = torch.device(device)
    _require(str(parsed) in {"cpu", "cuda:0"}, "J/Q/T runtime supports only cpu or explicit cuda:0")
    return parsed


def configure_runtime(device: str = "cpu", threads: int = 4) -> None:
    """Set fixed numerical flags before data/model work, without seeding RNG."""
    parsed = _device(device)
    _require(type(threads) is int and threads > 0, "Runtime threads must be a positive integer")
    if parsed.type == "cuda":
        initialized = torch.cuda.is_initialized()
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        _require(workspace in {None, CUBLAS_WORKSPACE}, f"CUBLAS_WORKSPACE_CONFIG must be {CUBLAS_WORKSPACE}")
        _require(not initialized or workspace == CUBLAS_WORKSPACE, "CUDA initialized before deterministic CUBLAS workspace was configured")
        tf32_override = os.environ.get("NVIDIA_TF32_OVERRIDE")
        _require(tf32_override in {None, "0"}, "NVIDIA_TF32_OVERRIDE must not force TF32")
        _require(not initialized or tf32_override == "0", "CUDA initialized before NVIDIA_TF32_OVERRIDE=0 was configured")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise ValueError("Configure runtime before starting inter-op work; inter-op threads must be 1") from error
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for name in ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
        if hasattr(torch.backends.cuda.matmul, name):
            setattr(torch.backends.cuda.matmul, name, False)
    if parsed.type == "cuda":
        _require(torch.cuda.is_available(), "Requested CUDA runtime is unavailable")
        torch.cuda.set_device(parsed)


def _canonical_uuid(value: str) -> str:
    return value.strip().lower().removeprefix("gpu-").replace("-", "")


def _gpu_identity(index: int) -> dict:
    properties = torch.cuda.get_device_properties(index)
    uuid = str(getattr(properties, "uuid", ""))
    _require(bool(uuid), "CUDA device UUID is required to pin the actual GPU")
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,driver_version,pci.bus_id", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Cannot establish the NVIDIA driver/device identity with nvidia-smi") from error
    rows = [[field.strip() for field in row] for row in csv.reader(io.StringIO(completed.stdout))]
    matches = [row for row in rows if len(row) == 3 and _canonical_uuid(row[0]) == _canonical_uuid(uuid)]
    _require(len(matches) == 1, "CUDA device UUID does not uniquely match nvidia-smi")
    matched = matches[0]
    _require(bool(matched[1]) and matched[1] != "N/A", "NVIDIA driver version is unavailable")
    return {
        "logical_index": index,
        "uuid": matched[0],
        "name": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "multiprocessor_count": int(properties.multi_processor_count),
        "driver_version": matched[1],
        "pci_bus_id": matched[2],
        "visible_device_count": torch.cuda.device_count(),
    }


def runtime_identity(device: str = "cpu") -> dict:
    """Read and validate the configured runtime, including actual CUDA hardware."""
    parsed = _device(device)
    _require(torch.are_deterministic_algorithms_enabled(), "Deterministic algorithms must be enabled")
    _require(not torch.is_deterministic_algorithms_warn_only_enabled(), "Deterministic warn_only must be False")
    _require(torch.get_num_interop_threads() == 1, "Inter-op threads must be 1")
    _require(not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic, "cuDNN benchmark must be off and deterministic must be on")
    _require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32, "TF32 must be disabled")
    _require(torch.get_float32_matmul_precision() == "highest", "Float32 matmul precision must be highest")
    _require(torch.get_default_dtype() == torch.float32, "Default model dtype must be float32")
    _require(not torch.is_autocast_enabled(parsed.type), "J/Q/T runtime excludes autocast")
    reduced = {name: getattr(torch.backends.cuda.matmul, name, None) for name in ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")}
    _require(all(value is None or value is False for value in reduced.values()), "Reduced-precision reduction must be disabled")
    gpu = None
    if parsed.type == "cuda":
        _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_WORKSPACE, "Deterministic CUBLAS workspace configuration drift")
        _require(os.environ.get("NVIDIA_TF32_OVERRIDE") == "0", "NVIDIA TF32 environment configuration drift")
        _require(torch.cuda.is_available(), "Requested CUDA runtime is unavailable")
        _require(torch.cuda.current_device() == 0, "Active CUDA device must be cuda:0")
        gpu = _gpu_identity(0)
    return {
        "schema": RUNTIME_SCHEMA,
        "device": str(parsed), "device_type": parsed.type,
        "python": platform.python_version(),
        "torch": str(torch.__version__), "numpy": np.__version__, "polars": pl.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "reduced_precision_reduction": reduced,
        "environment": {name: os.environ.get(name) for name in (
            "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED",
        )},
        "runtime_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
