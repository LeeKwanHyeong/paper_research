"""
Notebook startup helpers for local and remote Jupyter sessions.

Why this module exists:
- Jupyter kernels often start from a working directory outside the repository.
- Several notebooks need the same `PROJECT_ROOT` + `sys.path` bootstrap logic.
- Server paths can drift over time, so we keep the fallback rules in one place.

Typical notebook usage:

```python
from pathlib import Path
import sys

SERVER_PROJECT_ROOT = Path('~/workspace/paper_research').expanduser().resolve()
if str(SERVER_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_PROJECT_ROOT))

from utils.notebook_bootstrap import bootstrap_notebook

PROJECT_ROOT, DIR = bootstrap_notebook(preferred_root=SERVER_PROJECT_ROOT)
```
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_SENTINELS = ("models", "utils", "sample_data")
DEFAULT_SERVER_ROOTS = (
    Path("~/workspace/paper_research").expanduser(),
    Path("~/workspace/paper_research/paper_research").expanduser(),
)


def _is_valid_project_root(candidate: Path) -> bool:
    """
    Verify that a directory looks like the `paper_research` repository root.
    """
    return all((candidate / name).exists() for name in ROOT_SENTINELS)


def _iter_search_candidates(
    *,
    preferred_root: str | Path | None = None,
    start: str | Path | None = None,
) -> list[Path]:
    """
    Build a deterministic candidate list for root resolution.

    Search order:
    1. explicit preferred root provided by the notebook
    2. common Linux server roots used in this project
    3. current working directory and its parents
    """
    candidates: list[Path] = []

    if preferred_root is not None:
        candidates.append(Path(preferred_root).expanduser())

    candidates.extend(DEFAULT_SERVER_ROOTS)

    anchor = Path(start).expanduser() if start is not None else Path.cwd()
    anchor = anchor.resolve()
    base = anchor if anchor.is_dir() else anchor.parent
    candidates.extend([base, *base.parents])

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            deduped.append(resolved)
    return deduped


def resolve_project_root(
    *,
    preferred_root: str | Path | None = None,
    start: str | Path | None = None,
) -> Path:
    """
    Resolve the project root from a preferred path and a small fallback search.
    """
    candidates = _iter_search_candidates(preferred_root=preferred_root, start=start)
    for candidate in candidates:
        if _is_valid_project_root(candidate):
            return candidate

    checked = "\n".join(str(path) for path in candidates)
    raise RuntimeError(
        "Could not locate the paper_research project root. Checked:\n"
        f"{checked}"
    )


def bootstrap_notebook(
    *,
    preferred_root: str | Path | None = None,
    start: str | Path | None = None,
    verbose: bool = True,
) -> tuple[Path, str]:
    """
    Resolve the project root, register it on `sys.path`, and return notebook vars.

    Returns:
    - `PROJECT_ROOT` as `Path`
    - `DIR` as `str` with a trailing slash for legacy notebook compatibility
    """
    project_root = resolve_project_root(preferred_root=preferred_root, start=start)
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    dir_str = root_str + "/"
    if verbose:
        print("PROJECT_ROOT =", project_root)
    return project_root, dir_str
