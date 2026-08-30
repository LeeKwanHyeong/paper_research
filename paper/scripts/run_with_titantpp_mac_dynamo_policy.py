#!/usr/bin/env python3
"""Run a frozen TitanTPP-MAC entrypoint with a bounded Dynamo policy."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


DEFAULT_RECOMPILE_LIMIT = 64
DEFAULT_ACCUMULATED_RECOMPILE_LIMIT = 512


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompile-limit",
        type=int,
        default=DEFAULT_RECOMPILE_LIMIT,
    )
    parser.add_argument(
        "--accumulated-recompile-limit",
        type=int,
        default=DEFAULT_ACCUMULATED_RECOMPILE_LIMIT,
    )
    parser.add_argument("runner", type=Path)
    return parser.parse_known_args()


def main() -> None:
    args, runner_args = parse_args()
    if not args.runner.is_file():
        raise FileNotFoundError(args.runner)
    if args.recompile_limit < 1:
        raise ValueError("recompile_limit must be positive")
    if args.accumulated_recompile_limit < args.recompile_limit:
        raise ValueError(
            "accumulated_recompile_limit must be at least recompile_limit"
        )

    import torch._dynamo.config as dynamo_config

    dynamo_config.recompile_limit = args.recompile_limit
    dynamo_config.accumulated_recompile_limit = (
        args.accumulated_recompile_limit
    )
    print(
        "[titantpp-mac-dynamo-policy] "
        f"recompile_limit={dynamo_config.recompile_limit} "
        "accumulated_recompile_limit="
        f"{dynamo_config.accumulated_recompile_limit}",
        flush=True,
    )
    sys.argv = [str(args.runner), *runner_args]
    runpy.run_path(str(args.runner), run_name="__main__")


if __name__ == "__main__":
    main()
