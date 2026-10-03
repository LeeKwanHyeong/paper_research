"""Reproduce the frozen bundle's root sentinel failure outside the checkout."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[3]
BUNDLE = (
    ROOT
    / "search_artifacts/successor_episode_parallel_20260914_v1/source_bundle.tar.gz"
)


@pytest.fixture
def extracted_bundle(tmp_path):
    assert not tmp_path.resolve().is_relative_to(ROOT)
    destination = tmp_path / "deployment"
    with tarfile.open(BUNDLE, "r:gz") as archive:
        archive.extractall(destination, filter="data")
    frozen = destination / "source"
    assert (frozen / "models").is_dir()
    assert (frozen / "utils").is_dir()
    assert not (frozen / "sample_data").exists()
    outside = tmp_path / "unrelated_working_directory"
    outside.mkdir()
    return frozen, outside


def isolated_python(frozen, outside, body):
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PROJECT_ROOT"):
        environment.pop(name, None)
    environment.update(CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
    bootstrap = """
import importlib
import sys
from pathlib import Path

frozen, checkout = (Path(value).resolve() for value in sys.argv[1:])
assert sys.flags.isolated == 1
assert not Path.cwd().is_relative_to(checkout)
assert all(not Path(value).resolve().is_relative_to(checkout) for value in sys.path)
sys.path.insert(0, str(frozen))
from simple_lab_test.common import pathing
assert Path(pathing.__file__).resolve().is_relative_to(frozen)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", bootstrap + textwrap.dedent(body),
         str(frozen), str(ROOT)],
        cwd=outside,
        env=environment,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_original_bundle_reproduces_missing_root_before_training_import(extracted_bundle):
    frozen, outside = extracted_bundle
    output = isolated_python(frozen, outside, """
        def expect_missing_root(operation):
            try:
                operation()
            except RuntimeError as error:
                assert "Could not locate the paper_research project root" in str(error)
                assert str(frozen) in str(error)
                assert str(checkout) not in str(error)
            else:
                raise AssertionError("Missing sample_data unexpectedly resolved a root")

        expect_missing_root(lambda: pathing.resolve_project_root(pathing.__file__))
        expect_missing_root(lambda: importlib.import_module(
            "paper.scripts.count_aware_tpp_backbone.training"))
        print("missing_root_reproduced")
    """)
    assert output.strip() == "missing_root_reproduced"


def test_empty_sample_data_repairs_frozen_training_and_diagnostic_imports(extracted_bundle):
    frozen, outside = extracted_bundle
    (frozen / "sample_data").mkdir()
    output = isolated_python(frozen, outside, """
        assert pathing.resolve_project_root(pathing.__file__) == frozen
        for name in (
            "paper.scripts.count_aware_tpp_backbone.training",
            "paper.scripts.time_quantity_diagnostic",
            "paper.scripts.run_time_quantity_diagnostic",
            "paper.scripts.verify_successor_episode_cost",
        ):
            module = importlib.import_module(name)
            assert Path(module.__file__).resolve().is_relative_to(frozen), name

        from simple_lab_test.search.common import experiment_utils, benchmark_utils
        assert experiment_utils.PROJECT_ROOT == frozen
        assert benchmark_utils.PROJECT_ROOT == frozen
        first_party = {"models", "utils", "data_loader", "paper", "simple_lab_test"}
        for name, module in tuple(sys.modules.items()):
            if name.split(".", 1)[0] not in first_party:
                continue
            location = getattr(module, "__file__", None)
            if location is not None:
                assert Path(location).resolve().is_relative_to(frozen), (name, location)
            for location in getattr(module, "__path__", ()):
                assert Path(location).resolve().is_relative_to(frozen), (name, location)
        assert all(not Path(value).resolve().is_relative_to(checkout) for value in sys.path)
        assert list((frozen / "sample_data").iterdir()) == []
        import torch
        assert not torch.cuda.is_initialized()
        print("isolated_frozen_imports_passed")
    """)
    assert output.strip() == "isolated_frozen_imports_passed"
