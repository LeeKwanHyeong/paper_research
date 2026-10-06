"""Seal the already approved six full130 head fits after local code/tests are committed."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
import tempfile
import time

BUNDLE = Path(__file__).resolve().parents[1]
PROJECT = BUNDLE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    assert not (BUNDLE/'execution_contract.json').exists(), 'Contract already sealed; review rather than retry'
    c = json.loads((BUNDLE/'draft_contract.json').read_text())
    assert json.loads((BUNDLE/'preflight_receipt.json').read_text())['status'] == 'passed'
    r = c['runtime']; root = c['root']
    r.update(root=root, operation_root=root+'/operation', output_dir=root+'/run',
        tmux='titantpp_time_full130_5080_20261006')
    r['environment'].update(MPLCONFIGDIR=root+'/cache/matplotlib', XDG_CACHE_HOME=root+'/cache')
    sources = {'operation/controller.py': BUNDLE/'control/controller.py',
        'operation/titantpp_time_head_full_refit_runtime.py': PROJECT/'paper/scripts/titantpp_time_head_full_refit_runtime.py'}
    operation_files = {rel: sha(path) for rel, path in sources.items()}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
    c['operation'] = {'files': operation_files, 'files_sha256': digest(operation_files), 'git_revision': head}
    c['runtime_sha256'] = operation_files['operation/titantpp_time_head_full_refit_runtime.py']
    c['fit_split'] = 'train'; c['selection_split'] = 'validation'
    csha = digest(c)
    approval = json.loads((BUNDLE/'approval.json').read_text()); approval['contract_sha256'] = csha
    start = c.get('lease_started_unix', time.time())
    permit = {'contract_sha256': csha, 'approval_sha256': digest(approval),
        'issued_at_unix': start, 'deadline_unix': start+c['limits']['total_wall_seconds']}
    current = {'campaign': c['campaign'], 'host': '5080', 'root': root,
        'canonical_sha256': csha, 'contract': 'execution_contract.json',
        'status': 'sealed_native_qualification_pending', 'fit_count': 6, 'stage': 'time_head_full130',
        'execution_attempt': c.get('execution_attempt', 1)}
    write(BUNDLE/'execution_contract.json', c); write(BUNDLE/'approval.json', approval)
    write(BUNDLE/'start_permit.json', permit); write(BUNDLE/'current.json', current)
    with tempfile.TemporaryDirectory() as temp:
        staging = Path(temp)
        for rel, path in sources.items():
            target = staging/rel; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        for name in ('execution_contract.json', 'approval.json', 'start_permit.json', 'current.json'):
            (staging/name).write_bytes((BUNDLE/name).read_bytes())
        payload = BUNDLE/'deployment_payload.tar.gz'
        with tarfile.open(payload, 'w:gz') as archive:
            for path in sorted(staging.rglob('*')):
                if path.is_file(): archive.add(path, arcname=str(path.relative_to(staging)))
    write(BUNDLE/'deployment_payload_receipt.json', {'archive_sha256': sha(payload),
        'contract_sha256': csha, 'operation_git_revision': head, 'starts_training': False,
        'files': {rel: sha(path) for rel, path in sources.items()}, 'created_unix': time.time()})
    print(json.dumps({'contract_sha256': csha, 'payload_sha256': sha(payload),
        'operation_git_revision': head, 'deadline_unix': permit['deadline_unix']}))


if __name__ == '__main__': main()
