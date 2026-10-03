#!/usr/bin/env python3
"""Run a fixed three-arm dataset partition on its qualified native GPU host."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts import successor_episode_parallel_common as common
from paper.scripts import run_successor_episode_comparison as comparison
from paper.scripts import run_observed_slot_partition as shared


def production(contract, permit, host_name, qualification, budget):
    return comparison.production(contract, permit, host_name, qualification, budget, execution_common=common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("execute", "_worker"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--permit", type=Path, required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--authority-fd", type=int)
    args = parser.parse_args()
    if args.command == "execute":
        common.require(args.owner_pid is None and args.authority_fd is None, "Supervisor cannot inherit worker authority")
        shared.supervisor(args.contract, args.permit, args.host, common=common, entrypoint=__file__, assignment_validator=common.validate_assignment)
    else:
        common.require(args.owner_pid is not None and args.authority_fd is not None, "Inherited worker authority required")
        shared.worker(args.contract, args.permit, args.host, args.owner_pid, args.authority_fd,
            common=common, entrypoint=__file__, production=production, assignment_validator=common.validate_assignment)


if __name__ == "__main__":
    main()
