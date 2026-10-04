"""Build a reviewable handoff package locally; never connects to either host."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / 'search_artifacts/titantpp_history_width8_12_dual_20261004_v1'
BUNDLE = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / 'paper/scripts/capacity_handoff_runtime.py'
spec = importlib.util.spec_from_file_location('capacity_handoff_runtime', ADAPTER)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def save(path, value):
    if path.exists():
        r.require(r.read(path) == value, 'Existing preparation changed: ' + str(path))
    else:
        r.write(path, value, exclusive=True)


def copy_checked(source, destination, digest):
    r.require(source.is_file() and not source.is_symlink() and r.sha(source) == digest, 'Input SHA changed: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        r.require(r.sha(destination) == digest, 'Prepared file changed: ' + str(destination))
    else:
        with source.open('rb') as source_stream, destination.open('xb') as output:
            shutil.copyfileobj(source_stream, output)
    r.require(r.sha(destination) == digest, 'Copied SHA changed')


def prepare(user_instruction):
    r.require(bool(user_instruction.strip()), 'Explicit transfer instruction required')
    parent = r.read(PARENT / 'execution_contract.json')
    c = r.derive_contract(parent)
    r.validate_contract(parent, c)
    parent_approval, parent_start = r.read(PARENT / 'approval.json'), r.read(PARENT / 'start_permit.json')
    approval = {'approved': True, 'contract_sha256': r.canonical(c), 'parent_contract_sha256': r.PARENT_SHA,
                'hosts': ['5080', '5090'], 'transfer_job_ids': [j['id'] for j in r.target_jobs(parent)],
                'scope': 'Move only unstarted Intermittent seed62 width8/12 from5090 to dedicated5080 root; validation only; no source/model/loss/selection/deadline changes; no retry',
                'user_instruction': user_instruction}
    start = {**parent_start, 'contract_sha256': r.canonical(c), 'approval_sha256': r.canonical(approval)}
    meta = {'schema': r.NAME, 'parent_contract_sha256': r.PARENT_SHA, 'contract_sha256': r.canonical(c),
            'source_closure_sha256': r.SOURCE_SHA, 'job_ids': approval['transfer_job_ids'],
            'execution_host': '5080', 'origin_host': '5090', 'root': r.DESTINATION,
            'adapter_sha256': r.sha(ADAPTER), 'contract_diff': r.contract_diff(parent, c),
            'parent_start_permit_sha256': r.canonical(parent_start), 'approval_sha256': r.canonical(approval),
            'start_permit_sha256': r.canonical(start), 'selection_and_loss_changed': False,
            'cross_gpu_efficiency_claim': False}
    for name, value in [('parent_execution_contract.json', parent), ('execution_contract.json', c),
                        ('parent_approval.json', parent_approval), ('parent_start_permit.json', parent_start),
                        ('parent_current.json', r.read(PARENT / 'current.json')), ('approval.json', approval),
                        ('start_permit.json', start), ('handoff_contract.json', meta)]:
        save(BUNDLE / name, value)
    copy_checked(ADAPTER, BUNDLE / 'control/capacity_handoff_runtime.py', r.sha(ADAPTER))
    for name, digest in parent['source']['files'].items():
        copy_checked(PARENT / 'frozen_source' / name, BUNDLE / 'source' / name, digest)
    copy_checked(PARENT / 'frozen_source/sample_data/.keep', BUNDLE / 'source/sample_data/.keep', r.sha(PARENT / 'frozen_source/sample_data/.keep'))
    for row in parent['reuse']:
        if row['dataset'] == r.DATASET and row['seed'] == 62:
            for name, digest in row['file_sha256'].items():
                copy_checked(PARENT / 'references/width4' / r.DATASET / '62' / name,
                             BUNDLE / 'references/width4' / r.DATASET / '62' / name, digest)
    data = next(d for d in parent['datasets'] if d['dataset_id'] == r.DATASET)
    for kind in ('data', 'split_manifest'):
        identity = data['inherited_data_identity'][kind]
        relative = identity['path'].split('/paper_research/', 1)[1]
        copy_checked(ROOT / relative, BUNDLE / 'data' / Path(identity['path']).name, identity['sha256'])
    r.load(BUNDLE)
    files = {str(p.relative_to(BUNDLE)): r.sha(p) for p in sorted(BUNDLE.rglob('*'))
             if p.is_file() and p.name not in ('deployment.tar.gz', 'deployment_manifest.json', 'preparation_receipt.json')
             and not any(part in ('__pycache__', '.pytest_cache', 'tests') for part in p.relative_to(BUNDLE).parts)}
    save(BUNDLE / 'deployment_manifest.json', files)
    archive = BUNDLE / 'deployment.tar.gz'
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as stream:
            for name in [*sorted(files), 'deployment_manifest.json']:
                stream.add(BUNDLE / name, arcname=name, recursive=False)
    return {'status': 'prepared_local_only', 'bundle': str(BUNDLE), 'destination_root': r.DESTINATION,
            'contract_sha256': r.canonical(c), 'adapter_sha256': r.sha(ADAPTER), 'source_files': 114,
            'deadline_unix': start['deadline_unix'], 'deploy_archive': str(archive), 'archive_sha256': r.sha(archive),
            'remote_qualification_passed': False, 'source_reserved': False, 'training_started': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user-instruction', required=True)
    args = parser.parse_args()
    print(__import__('json').dumps(prepare(args.user_instruction), ensure_ascii=False))
