"""Local bounded Intermittent readiness/preparation hook for the hourly driver.

Remote observation, original retrieval, deployment, and launch belong to the
calling driver. This hook never connects to a server or starts evaluation.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location('width8_12_test_prepare', Path(__file__).with_name('prepare.py'))
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)

HERE, ROOT = prepare.HERE, prepare.ROOT
EXPECTED = {('intermittent_frozen_5000', model, seed) for model in prepare.MODELS for seed in (42, 52, 62)}
REQUIRED = ('best_val_qty_rmse_model.pt', 'history.json', 'endpoint_replays.json', 'initialization.json', 'input_receipt.json', 'status.json')


def readiness(original_roots, accepted_contracts):
    found = {}
    for root in original_roots:
        for terminal_path in Path(root).rglob('terminal_manifest.json'):
            terminal = prepare.read(terminal_path)
            key = prepare.scientific_job(terminal['job'])
            if key not in EXPECTED:
                continue
            prepare.require(key not in found, 'Duplicate Intermittent original condition')
            prepare.require(terminal['status'] == 'complete' and terminal['scientific_success'] is True,
                            'Intermittent terminal is not scientifically complete')
            prepare.require(terminal['contract_sha256'] in accepted_contracts, 'Unbound Intermittent training contract')
            for name in REQUIRED:
                relatives = [relative for relative in terminal['files'] if Path(relative).name == name]
                prepare.require(len(relatives) == 1, f'Missing original binding: {name}')
                path = (terminal_path.parent / relatives[0]).resolve()
                prepare.require(path.is_relative_to(terminal_path.parent.resolve()), 'Original binding escapes job folder')
                prepare.require(path.is_file() and prepare.sha(path) == terminal['files'][relatives[0]], f'Original file SHA mismatch: {name}')
            found[key] = str(terminal_path)
    missing = sorted(EXPECTED - set(found))
    return {'status': 'ready' if not missing else 'pending', 'required_conditions': 6,
            'verified_terminal_conditions': len(found), 'missing_conditions': missing,
            'terminal_manifests': sorted(found.values()), 'new_training': False,
            'remote_calls': 0, 'evaluation_started': False}


def deployment_request(campaign, destination=None):
    registry = prepare.read(campaign / 'evaluation_registry.json')
    bindings = prepare.read(campaign / 'selected_binding.json')
    contract = prepare.read(campaign / 'execution_contract.json')
    paths = set()
    for bundle in registry['bundles'].values():
        for relative, digest in bundle['source_files'].items():
            path = ROOT / bundle['source_root'] / relative
            prepare.require(prepare.sha(path) == digest, f'Frozen source changed: {relative}')
            paths.add(path)
        marker = ROOT / bundle['source_root'] / 'sample_data/.keep'
        prepare.require(marker.is_file(), 'Frozen project-root ancillary marker is missing')
        paths.add(marker)
    for binding in bindings['rows']:
        paths.add(ROOT / binding['terminal_manifest_path'])
        for relative, digest in binding['original_files'].items():
            path = ROOT / relative
            prepare.require(prepare.sha(path) == digest, f'Original file changed: {relative}')
            paths.add(path)
    for name in ('evaluate.py', 'pipeline.py', 'prepare.py', 'preflight.py'):
        paths.add(HERE / name)
    for path in campaign.glob('*.json'):
        if path.name != 'deployment_request.json':
            paths.add(path)
    remote = Path(contract['resources']['root'])
    relative = campaign.relative_to(ROOT)
    request = {'schema': 'titantpp_width8_12_test_deployment_v1', 'root': str(remote),
               'files': {str(path.relative_to(ROOT)): prepare.sha(path) for path in sorted(paths)},
               'file_count': len(paths), 'bytes': sum(path.stat().st_size for path in paths),
               'environment': contract['environment'], 'dataset_transfer_required': False,
               'dataset_manifest': prepare.read(campaign / 'dataset_manifest.json'),
               'command': [contract['resources']['python'], str(remote / HERE.relative_to(ROOT) / 'pipeline.py'),
                           '--root', str(remote), '--campaign', str(remote / relative), '--device', 'cuda'],
               'preflight_command': [contract['resources']['python'], str(remote / HERE.relative_to(ROOT) / 'preflight.py'),
                                     '--root', str(remote), '--campaign', str(remote / relative)],
               'path_exists_preflight_required_before_launch': True,
               'conditions': len(registry['rows']), 'remote_execution_started_here': False}
    destination = campaign / 'deployment_request.json' if destination is None else Path(destination)
    with destination.open('x') as stream:
        json.dump(request, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return request


def main(args):
    contracts = [prepare.TRAINING / 'execution_contract.json', *args.training_contract]
    accepted = {prepare.sha_json(prepare.read(path)) for path in contracts}
    result = readiness(args.original_root, accepted)
    if result['status'] != 'ready' or not args.prepare:
        return result
    campaign = Path(args.output).resolve()
    prepare.require(campaign.is_relative_to(HERE), 'Followup output must remain in this report directory')
    if campaign.exists():
        # Repeated scheduler runs report state; they never prepare, retry, or launch again.
        for name in ('inference_completion.json', 'pipeline_status.json'):
            path = campaign / name
            if path.exists():
                status = prepare.read(path)
                return {**result, 'status': status['status'], 'campaign': str(campaign), 'existing_attempt': True,
                        'evaluation_started': True, 'automatic_retry': False}
        prepare.require((campaign / 'deployment_request.json').is_file(), 'Existing campaign has no deployment request; preserve it for inspection')
        return {**result, 'status': 'prepared', 'campaign': str(campaign), 'existing_attempt': True,
                'deployment_request': str(campaign / 'deployment_request.json'), 'automatic_retry': False}
    with contextlib.redirect_stdout(io.StringIO()):
        prepare.prepare(SimpleNamespace(datasets=['intermittent_frozen_5000'], training_contract=args.training_contract,
                                        original_root=args.original_root, job_ids=None, output=campaign,
                                        remote_root=args.remote_root, dataset_root=args.dataset_root))
    request = deployment_request(campaign)
    return {**result, 'status': 'prepared', 'campaign': str(campaign),
            'deployment_request': str(campaign / 'deployment_request.json'),
            'pipeline_command': request['command'], 'automatic_retry': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', required=True, action='append', type=Path)
    parser.add_argument('--training-contract', action='append', type=Path, default=[])
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare', action='store_true')
    result = main(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(3 if result['status'] == 'pending' else 0)
