"""Reuse deleted PRO4500 Pod originals; copy only the authorized terminal fit.

No network, prediction, GPU access, or edits to prior originals are performed.
"""
import io
import json
import shutil
import tarfile
import time
from pathlib import Path

from collect import ROOT, REPORT, DEST, read, canonical, sha


def main():
    scope = read(REPORT / 'scope.json')['hosts']['pro4500']
    bundle = Path(scope['bundle'])
    src = bundle / 'retrieved/original'
    cleanup = read(bundle / 'control/cleanup_receipt.json')
    full_receipt = read(bundle / 'retrieved/retrieval_receipt.json')
    assert cleanup['pod_id'] == 'eu8yh29yympuf4' and cleanup['deleted_and_absent']
    assert full_receipt['status'] == 'sha_verified'
    assert sha(bundle / 'retrieved/original.tar.gz') == full_receipt['archive_sha256']
    full_manifest = read(src / 'retrieval_manifest.json')
    c = read(src / 'execution_contract.json')
    assert canonical(c) == scope['contract_sha256']
    assert canonical(c['source']['files']) == scope['source_closure_sha256']
    files = {}

    def add(path, name=None, expected=None):
        path = Path(path)
        name = name or str(path.relative_to(src))
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert path.is_file() and not path.is_symlink()
        digest = sha(path)
        assert digest == (expected or full_manifest['files'][name]), name
        files[name] = (path, digest)

    for rel, digest in c['source']['files'].items():
        add(src / 'source' / rel, expected=digest)
    for rel in ('execution_contract.json', 'approval.json', 'start_permit.json',
                'training_permit.json', 'qualification/receipt.json',
                'source_reservation.json'):
        add(src / rel)
    status = read(src / 'status.json')
    completed = status.get('completed', status.get('outcomes', {}))
    for pinned in scope['jobs']:
        job = pinned['job']
        assert job in c['jobs']
        run = src / 'run' / job['id']
        manifest = read(run / 'terminal_manifest.json')
        assert manifest['scientific_success'] and manifest['job'] == job
        assert manifest['contract_sha256'] == scope['contract_sha256']
        add(run / 'terminal_manifest.json', expected=pinned['terminal_manifest_sha256'])
        assert completed[job['id']]['terminal_manifest_sha256'] == pinned['terminal_manifest_sha256']
        add(src / 'claims' / (job['id'] + '.json'))
        for rel, digest in manifest['files'].items():
            add(run / rel, expected=digest)
    # The terminal Pod archive omits reused baseline data. Reuse the exact
    # exposure file already recovered and audited in the prior CPU audit.
    ref = next(x for x in c['reuse'] if x['dataset'] == 'intermittent_frozen_5000' and x['seed'] == 62)
    name = 'reference_exposure/intermittent_frozen_5000__62__titantpp_history_mlp.json'
    baseline = ROOT / 'search_artifacts/titantpp_completed22_audit_20261002_v1/retrieved/pro4500/original' / name
    add(baseline, name, ref['file_sha256']['exposure.json'])
    out = DEST / 'pro4500'
    out.mkdir(parents=True, exist_ok=False)
    manifest = {rel: {'sha256': digest, 'bytes': path.stat().st_size, 'local_source_path': str(path)}
                for rel, (path, digest) in files.items()}
    receipt = {'collected_unix': time.time(), 'host': 'pro4500',
               'root': str(src), 'contract_sha256': canonical(c),
               'source_closure_sha256': c['source']['files_sha256'],
               'source_files': len(c['source']['files']), 'jobs': len(scope['jobs']),
               'pinned_jobs': scope['jobs'], 'host_status_at_collection': status,
               'terminal_completion_index': completed,
               'scope': 'authorized_terminal_subset_reused_from_deleted_pod',
               'running_jobs_excluded': True, 'remote_writes': False,
               'source_archive': str(bundle / 'retrieved/original.tar.gz'),
               'source_archive_sha256': full_receipt['archive_sha256'],
               'cleanup_receipt': str(bundle / 'control/cleanup_receipt.json'),
               'reference_map': {name: {'local_source_path': str(baseline), 'sha256': sha(baseline)}},
               'files': manifest}
    with tarfile.open(out / 'original.tar', 'w') as tar:
        for rel, (path, digest) in sorted(files.items()):
            tar.add(path, arcname=rel, recursive=False)
            assert sha(path) == digest
        raw = json.dumps(receipt, ensure_ascii=False, indent=2).encode()
        info = tarfile.TarInfo('collection_manifest.json')
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    original = out / 'original'
    original.mkdir()
    with tarfile.open(out / 'original.tar') as tar:
        tar.extractall(original, filter='data')
    for rel, record in manifest.items():
        assert sha(original / rel) == record['sha256']
    result = {'status': 'retrieved_all_file_sha_verified', 'host': 'pro4500',
              'collected_unix': receipt['collected_unix'], 'verified_unix': time.time(),
              'files': len(files), 'bytes': sum(v['bytes'] for v in manifest.values()),
              'source_files': len(c['source']['files']), 'jobs': len(scope['jobs']),
              'archive_sha256': sha(out / 'original.tar'), 'binary_cpu_audit': 'pending',
              'remote_writes': False, 'running_jobs_excluded': True,
              'local_originals_reused': True, 'script_sha256': sha(__file__),
              'scope_sha256': sha(REPORT / 'scope.json')}
    (out / 'retrieval_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
