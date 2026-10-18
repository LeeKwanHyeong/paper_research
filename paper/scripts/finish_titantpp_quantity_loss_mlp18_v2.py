"""Finite completion for v2, preserving the failed v1 operation and code."""
import hashlib
import importlib.util
from pathlib import Path

from paper.scripts import finish_titantpp_quantity_loss_mlp18 as previous
from paper.scripts.compare_titantpp_quantity_loss_mlp18_v2 import manager_binding

PARENT_SHA = "97dad83a27fceedb7f73098a6043a32e6ddea2b5dce254a04a5fee5ac1ee31e2"
path = Path(previous.__file__).resolve()
if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
    raise ValueError("Preserved v1 completion source changed")
spec = importlib.util.spec_from_file_location(__name__ + "_private_parent", path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
original_wait_program = adapter.wait_program


def wait_program(*args, **kwargs):
    program = original_wait_program(*args, **kwargs)
    original = "from paper.scripts import titantpp_quantity_loss_mlp18_runtime as r"
    if program.count(original) != 1:
        raise ValueError("Completion runtime binding is not unique")
    return program.replace(original, "from paper.scripts import titantpp_quantity_loss_mlp18_runtime_v2 as r", 1)


adapter.wait_program = wait_program
adapter.__file__ = __file__


def main():
    with manager_binding():
        return adapter.main()


if __name__ == "__main__":
    main()
