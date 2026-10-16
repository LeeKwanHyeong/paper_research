"""Preserve frozen finalizer and recover validated results under a separate seal."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT / 'search_artifacts/titantpp_contextual_retrieval_pilot_repair_20261007_v1'

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def load_recovery(bundle):
    observer = module(PROJECT / 'paper/scripts/observe_titantpp_retrieval_results.py', '_result_recovery_observer')
    profile = observer.validate_recovery_profile(bundle)
    finalizer = module(PROJECT / 'paper/scripts/finalize_titantpp_retrieval_repair.py', '_original_result_recovery_finalizer')
    original_load = finalizer.load_bundle
    def load_bundle(path):
        c, old_observer, integrity = original_load(path)
        observer.preflight(path)
        observer.profile_eligibility = old_observer.profile_eligibility
        return c, observer, {**integrity, 'result_recovery_profile_sha256': observer.sha(Path(path) / observer.RECOVERY_PROFILE),
            'result_recovery_observer_sha256': observer.sha(PROJECT / 'paper/scripts/observe_titantpp_retrieval_results.py'),
            'result_recovery_finalizer_sha256': observer.sha(Path(__file__))}
    finalizer.load_bundle = load_bundle
    return finalizer, profile

def run(bundle=BUNDLE, **kwargs):
    finalizer, _ = load_recovery(Path(bundle).resolve())
    return finalizer.run(bundle, **kwargs)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    args = parser.parse_args()
    print(json.dumps(run(args.bundle), ensure_ascii=False, allow_nan=False))

if __name__ == '__main__':
    main()
