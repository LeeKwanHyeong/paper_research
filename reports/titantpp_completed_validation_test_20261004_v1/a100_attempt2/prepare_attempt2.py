"""Preserve attempt 1 and prepare an explicitly authorized environment-only attempt.

No inference, GPU calls, network access, or changes to scientific artifacts.
"""
from pathlib import Path
import copy
import hashlib
import json
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'a100'
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main():
    copied = ['prepare.py', 'evaluate.py', 'run_all.py', 'verify.py', 'test_contract.py',
              'collect.py', 'evaluation_registry.json', 'selected_binding.json',
              'comparison_binding.json', 'dataset_manifest.json', 'local_checkpoint_preflight.json']
    for name in copied:
        with (HERE / name).open('xb') as stream:
            stream.write((PARENT / name).read_bytes())
    contract = json.loads((PARENT / 'execution_contract.json').read_text())
    contract['resources']['root'] += '_attempt2'
    contract['evaluator']['path'] = str((HERE / 'evaluate.py').relative_to(ROOT))
    diagnostic = json.loads((PARENT / 'runtime_environment_diagnostic.json').read_text())
    evidence = ['execution_contract.json', 'first_attempt_failure_evidence.json',
                'runtime_environment_diagnostic.json', 'launch_receipt.json']
    contract['technical_attempt'] = {
        'number': 2,
        'reason': 'Attempt 1 inherited no LD_LIBRARY_PATH; existing CUDA NVRTC builtins were not discoverable. The first Validation inference failed before any prediction row; Test was not entered.',
        'parent_evidence': {str((PARENT / name).relative_to(ROOT)): sha(PARENT / name) for name in evidence},
        'parent_remote_root': json.loads((PARENT / 'execution_contract.json').read_text())['resources']['root'],
        'first_attempt_prediction_rows': 0,
        'first_attempt_test_rows': 0,
        'authorization': 'Explicit parent release for one bounded technical inference attempt within the user-approved completed Validation/Test evaluation and existing 5080 transfer scope.',
        'automatic_retry': False,
        'run_all_verify_evaluate_byte_identical_to_parent': True,
        'scientific_source_checkpoint_selection_loader_target_changes': False,
        'runtime_installation_changes': False,
        'per_process_environment': {'LD_LIBRARY_PATH': diagnostic['required_existing_library_directory']},
        'existing_runtime_libraries': diagnostic['nvrtc_libraries'],
        'library_environment_reference': 'Same existing cu13/lib path used by the approved width16 evaluator; no installation or shared Runtime mutation.',
        'gpu_execution_requires_parent_release_after_deep_renewal': True,
    }
    write(HERE / 'execution_contract.json', contract)
    manifest = json.loads((PARENT / 'deployment_manifest.json').read_text())
    manifest['remote_root'] = contract['resources']['root']
    old = str(PARENT.relative_to(ROOT)) + '/'
    new = str(HERE.relative_to(ROOT)) + '/'
    files = {}
    for path, digest in manifest['files'].items():
        replaced = path.replace(old, new)
        files[replaced] = sha(ROOT / replaced) if path.startswith(old) else digest
    for name in ['prepare_attempt2.py', 'collect.py']:
        files[str((HERE / name).relative_to(ROOT))] = sha(HERE / name)
    manifest['files'] = files
    manifest['technical_attempt'] = 2
    manifest['parent_contract_sha256'] = sha(PARENT / 'execution_contract.json')
    write(HERE / 'deployment_manifest.json', manifest)
    receipt = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'conditions': 12, 'inference_splits_required': 24,
        'contract_sha256': sha(HERE / 'execution_contract.json'),
        'byte_identical_copies': {name: sha(HERE / name) for name in copied},
        'parent_checkpoint_preflight_reused': True,
        'dataset_source_selected_population_identity_unchanged': True,
        'inference_started': False, 'instacart': 'not_scheduled',
    }
    write(HERE / 'preparation_receipt.json', receipt)
    print(json.dumps({'prepared': str(HERE), 'contract_sha256': receipt['contract_sha256'],
                      'manifest_files': len(files)}))


if __name__ == '__main__':
    main()
