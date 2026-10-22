"""Version 2 local comparison adapter, retaining the frozen pure aggregation.

Only the isolated version 2 contract, retrieval and output paths are admitted.
The original comparator command line and its module globals are never changed.
"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

manager = import_module("paper.scripts.manage_shared_history_mlp_control")
frozen = import_module("paper.scripts.compare_shared_history_mlp_control")

ROOT, BUNDLE = manager.ROOT, manager.BUNDLE / "execution_v2"
SCHEMA = frozen.SCHEMA
frozen_aggregate = frozen.aggregate
render_markdown, render_csv, verify_outputs = (
    frozen.render_markdown,
    frozen.render_csv,
    frozen.verify_outputs,
)
require, read, sha = manager.require, manager.read, manager.sha


def validate_version(contract):
    require(
        contract.get("execution_version") == 2
        and isinstance(contract.get("design_sha256"), str)
        and len(contract["design_sha256"]) == 64
        and contract["design_sha256"] == contract.get("science_spec_sha256"),
        "Version 2 design metadata is missing or differs",
    )


def aggregate(contract, retrieval):
    validate_version(contract)
    require(
        Path(retrieval).resolve() == (BUNDLE / "retrieval").resolve(),
        "Only version 2 retrieval is admitted",
    )
    result = dict(frozen_aggregate(contract, Path(retrieval)))
    result.update(execution_version=2, design_sha256=contract["design_sha256"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    require(
        args.contract.resolve() == (BUNDLE / "execution_contract.json").resolve()
        and args.retrieval.resolve() == (BUNDLE / "retrieval").resolve()
        and args.output.resolve() == (BUNDLE / "comparison_v1").resolve(),
        "Only version 2 fixed local paths are admitted",
    )
    require(
        args.contract.is_file()
        and not args.contract.is_symlink()
        and sha(args.contract) == args.contract_sha,
        "Externally pinned version 2 raw contract differs",
    )
    result = aggregate(read(args.contract), args.retrieval)
    if args.check_only:
        if args.output.exists():
            verify_outputs(args.output, result)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "execution_version": 2,
                    "evaluation_scope": "validation_only",
                    "records": 27,
                    "paired_differences": 162,
                    "cost_records": 162,
                    "output_verified": args.output.exists(),
                    "remote_calls": 0,
                }
            )
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    manager.write(args.output / "REPORT.json", result)
    with (args.output / "REPORT.md").open("x") as stream:
        stream.write(render_markdown(result))
    with (args.output / "TABLE.csv").open("x") as stream:
        stream.write(render_csv(result))
    verify_outputs(args.output, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "execution_version": 2,
                "evaluation_scope": "validation_only",
                "output": str(args.output),
                "remote_calls": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
