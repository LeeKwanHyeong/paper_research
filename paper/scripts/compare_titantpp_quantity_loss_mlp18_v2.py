"""Apply the unchanged quantity-loss aggregation to the isolated v2 bundle."""
from contextlib import contextmanager
import hashlib
import sys

from paper import scripts as package
from paper.scripts import compare_titantpp_quantity_loss_mlp18 as previous

PARENT_SHA = "f507e6915143aa4a176b0b543011debea816ed950b39e98db031d3d08995da36"


@contextmanager
def manager_binding():
    from paper.scripts import manage_titantpp_quantity_loss_mlp18_v2 as manager
    name = "paper.scripts.manage_titantpp_quantity_loss_mlp18"
    old_module = sys.modules.get(name)
    sentinel = object()
    old_attribute = getattr(package, "manage_titantpp_quantity_loss_mlp18", sentinel)
    sys.modules[name] = manager
    package.manage_titantpp_quantity_loss_mlp18 = manager
    try:
        yield manager
    finally:
        if old_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old_module
        if old_attribute is sentinel:
            delattr(package, "manage_titantpp_quantity_loss_mlp18")
        else:
            package.manage_titantpp_quantity_loss_mlp18 = old_attribute


def main():
    from pathlib import Path
    if hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Preserved v1 comparison source changed")
    with manager_binding():
        return previous.main()


if __name__ == "__main__":
    main()
