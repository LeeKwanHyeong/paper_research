"""Check Python syntax without importing models or running experiment scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    "models",
    "data_loader",
    "utils",
    "simple_lab_test",
    "paper/scripts",
    "paper/tests",
    "paper/reproducibility",
    "scripts",
)


def source_paths(root: Path, *, changed: bool = False) -> list[Path]:
    commands = [
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", *SOURCE_ROOTS]
    ]
    if changed:
        commands.append(
            [
                "git",
                "diff",
                "HEAD",
                "--name-only",
                "--diff-filter=ACMR",
                "-z",
                "--",
                *SOURCE_ROOTS,
            ]
        )
    else:
        commands.append(["git", "ls-files", "--cached", "-z", "--", *SOURCE_ROOTS])
    names = set()
    for command in commands:
        output = subprocess.check_output(command, cwd=root).decode("utf-8")
        names.update(name for name in output.split("\0") if name.endswith(".py"))
    return [root / name for name in sorted(names) if (root / name).is_file()]


def check_paths(paths: list[Path], root: Path) -> dict:
    errors = []
    for path in paths:
        try:
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Source path resolves outside the repository")
            # Compilation checks syntax, including invalid return/await placement.
            # It does not execute code, import modules, or write bytecode files.
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        except (SyntaxError, ValueError, OSError) as error:
            errors.append({"path": str(path.relative_to(root)), "error": str(error)})
    return {
        "status": "failed" if errors else "passed",
        "checked_files": len(paths),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--changed", action="store_true", help="Check pending source changes only"
    )
    args = parser.parse_args()
    result = check_paths(source_paths(ROOT, changed=args.changed), ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
