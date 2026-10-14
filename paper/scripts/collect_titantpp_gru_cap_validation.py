"""Recover nine terminal cap fits once, then summarize their fixed Validation results.

Only SHA-bound, explicitly terminal jobs may enter the read-only SSH transport.
An existing incomplete collection intent blocks automatic retry. This program
never trains, evaluates, changes a remote file, or reads held-out predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shlex
import shutil
import subprocess
import tarfile
import time
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT / 'search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1/execution'
MONITOR = PROJECT / 'reports/titantpp_gru_residual_cap_validation_20261014_v1/hourly_monitor'
REPORT = MONITOR.parent
MAX_BYTES = 8 * 1024 ** 3
MAX_FILES = 4096
HOSTS = ('5080', '5090')


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def safe(root: Path, name: str) -> Path:
    require(isinstance(name, str) and name and '\\' not in name and '\x00' not in name,
        'Invalid artifact path')
    parts = PurePosixPath(name).parts
    require(not name.startswith('/') and all(x not in ('', '.', '..') for x in parts)
        and str(PurePosixPath(name)) == name, 'Noncanonical artifact path')
    root = Path(root).resolve()
    path = root
    for part in parts:
        path = path / part
        require(not path.is_symlink(), 'Artifact symlink rejected')
    require(path.resolve().is_relative_to(root), 'Artifact escaped its root')
    return path


def digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(x in '0123456789abcdef' for x in value)


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def rel(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def unique_write(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    return sha(path)


def locate(root: Path, terminal: dict[str, Any], suffix: str) -> tuple[Path, str]:
    files = terminal.get("files")
    require(isinstance(files, dict), "Terminal manifest files map is absent")
    matches = [(name, digest) for name, digest in files.items()
               if isinstance(name, str) and name.endswith(suffix) and isinstance(digest, str)]
    require(len(matches) == 1, "Terminal manifest has ambiguous required artifact: " + suffix)
    name, digest = matches[0]
    path = safe(root, name)
    require(path.is_file() and sha(path) == digest, "Retrieved artifact differs: " + name)
    return path, digest


def normalize(contract_path: Path, retrieval_root: Path, output_index: Path) -> dict[str, Any]:
    from paper.scripts import summarize_titantpp_gru_cap_validation as summary
    contract = summary.read(contract_path)
    require(contract.get("held_out_test_evaluated") is False, "Collector accepts Validation-only contract")
    jobs = contract.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 9, "Expected nine frozen candidate jobs")
    root = Path(retrieval_root).resolve()
    bindings: dict[str, Any] = {}
    for job in jobs:
        job_id = job.get("id")
        require(isinstance(job_id, str), "Frozen job id missing")
        terminal_path = safe(root, job_id + "/terminal_manifest.json")
        require(terminal_path.is_file(), "Terminal manifest is missing for " + job_id)
        terminal_sha = summary.sha(terminal_path)
        terminal = summary.read(terminal_path)
        require(terminal.get("status") == "complete" and terminal.get("scientific_success") is True
                and terminal.get("job") == job and terminal.get("held_out_test_evaluated") is False
                and terminal.get("evaluation_scope") == "validation_only"
                and terminal.get("contract_sha256") == summary.canonical(contract),
                "Terminal identity differs for " + job_id)
        history, history_sha = locate(terminal_path.parent, terminal, "history.json")
        checkpoint, checkpoint_sha = locate(terminal_path.parent, terminal, "best_val_qty_rmse_model.pt")
        endpoint, endpoint_sha = locate(terminal_path.parent, terminal, "endpoint_replays.json")
        inputs, input_sha = locate(terminal_path.parent, terminal, "input_receipt.json")
        endpoint_value, input_value = summary.read(endpoint), summary.read(inputs)
        selected_epoch, _ = summary.strict_selected_epoch(summary.selection_history(history))
        selected = endpoint_value.get("selected")
        require(endpoint_value.get("status") == "complete" and endpoint_value.get("job") == job
                and endpoint_value.get("best_epoch") == selected_epoch
                and endpoint_value.get("evaluation_scope") == "validation_only"
                and endpoint_value.get("held_out_test_evaluated") is False
                and isinstance(selected, dict) and selected.get("held_out_test_evaluated") is False,
                "Endpoint replay differs for " + job_id)
        population = input_value.get("populations", {}).get("validation")
        require(input_value.get("held_out_materialized") is False and isinstance(population, dict)
                and selected.get("count") == population.get("target_count"),
                "Input population differs for " + job_id)
        derived = terminal_path.parent / "derived_validation_replay.json"
        derived_sha = unique_write(derived, {
            "schema": "titantpp_gru_residual_cap_validation_local_normalized_replay_v1",
            "job": job, "source_endpoint_replays_sha256": endpoint_sha,
            "history_sha256": history_sha, "checkpoint_sha256": checkpoint_sha,
            "state_tensor_sha256": selected.get("state_sha256"), "selected_epoch": selected_epoch,
            "validation_population": population, "metrics": summary.endpoint_metrics(selected),
            "held_out_test_evaluated": False,
        })
        bindings[job_id] = {
            "terminal_manifest_path": rel(root, terminal_path), "terminal_manifest_sha256": terminal_sha,
            "history_path": rel(root, history), "history_sha256": history_sha,
            "checkpoint_path": rel(root, checkpoint), "checkpoint_sha256": checkpoint_sha,
            "endpoint_replays_path": rel(root, endpoint), "endpoint_replays_sha256": endpoint_sha,
            "input_receipt_path": rel(root, inputs), "input_receipt_sha256": input_sha,
            "derived_validation_replay_path": rel(root, derived), "derived_validation_replay_sha256": derived_sha,
            "state_tensor_sha256": selected["state_sha256"],
        }
    index = {"schema": summary.INDEX_KIND, "contract_raw_sha256": summary.sha(contract_path), "jobs": bindings,
             "held_out_test_evaluated": False}
    unique_write(output_index, index)
    return index


def verify_seal(bundle: Path, monitor: Path, integrity_sha: str) -> dict[str, Any]:
    path = bundle / 'collection_integrity.json'
    require(digest(integrity_sha) and sha(path) == integrity_sha, 'Collection integrity SHA changed; SSH0')
    seal = read(path)
    require(seal.get('schema') == 'titantpp_gru_cap_collection_integrity_v1'
        and canonical(seal['files']) == seal['files_sha256'], 'Collection dependency closure changed; SSH0')
    required = {'paper/scripts/collect_titantpp_gru_cap_validation.py',
        'paper/scripts/summarize_titantpp_gru_cap_validation.py',
        'paper/scripts/observe_titantpp_gru_cap_validation.py',
        'paper/scripts/launch_titantpp_gru_cap_validation.py'}
    monitor_name = str((monitor / 'monitor_integrity.json').relative_to(PROJECT))
    required.add(monitor_name)
    require(required <= set(seal['files']), 'Collection dependency seal incomplete; SSH0')
    for name, expected in seal['files'].items():
        require(digest(expected) and sha(safe(PROJECT, name)) == expected,
            'Collection dependency SHA changed: ' + name + '; SSH0')
    monitor_seal = read(monitor / 'monitor_integrity.json')
    require(seal['monitor_integrity_raw_sha256'] == seal['files'][monitor_name]
        and all(seal['files'].get(name) == value for name, value in monitor_seal['files'].items()),
        'Monitor dependency closure is not pinned by collection seal; SSH0')
    require(sha(bundle / 'execution_contract.json') == seal['contract_raw_sha256'],
        'Execution contract raw SHA changed; SSH0')
    return seal


def terminal_files(c: dict[str, Any], host: str, snapshot: dict[str, Any], launcher: Any) -> dict[str, str]:
    """Build the exact allowlist from already verified terminal snapshot bytes."""
    files = launcher.decode_snapshot(snapshot, c, host)
    expected = {}
    for job in [x for x in c['jobs'] if x['host'] == host]:
        name = 'run/' + job['id'] + '/terminal_manifest.json'
        raw = files[name]
        manifest = json.loads(raw)
        require(manifest.get('status') == 'complete' and manifest.get('scientific_success') is True
            and manifest.get('held_out_test_evaluated') is False
            and manifest.get('evaluation_scope') == 'validation_only'
            and manifest.get('contract_sha256') == canonical(c) and manifest.get('job') == job,
            'Terminal manifest identity changed; SSH0')
        expected[job['id'] + '/terminal_manifest.json'] = hashlib.sha256(raw).hexdigest()
        require(isinstance(manifest.get('files'), dict) and manifest['files'], 'Empty terminal manifest; SSH0')
        for member, value in manifest['files'].items():
            safe(Path('/'), member)
            require(digest(value) and member != 'terminal_manifest.json', 'Invalid terminal member; SSH0')
            require(not any(x.lower().startswith(('test_', 'heldout', 'held_out'))
                or x.lower() in ('test', 'test.json', 'test.csv') for x in PurePosixPath(member).parts),
                'Held-out artifact outside collection approval; SSH0')
            key = job['id'] + '/' + member
            require(key not in expected, 'Duplicate terminal member; SSH0')
            expected[key] = value
    require(0 < len(expected) <= MAX_FILES, 'Invalid terminal member count; SSH0')
    return expected


def completion_gate(bundle: Path, monitor: Path, round_path: Path, round_sha: str,
                    integrity_sha: str) -> dict[str, Any]:
    seal = verify_seal(bundle, monitor, integrity_sha)
    # Imports occur only after every local helper and dependency has been SHA checked.
    from paper.scripts import observe_titantpp_gru_cap_validation as observer
    from paper.scripts import launch_titantpp_gru_cap_validation as launch
    require(round_path.resolve().is_relative_to(monitor.resolve() / 'rounds')
        and not round_path.is_symlink() and digest(round_sha) and sha(round_path) == round_sha,
        'Round path/raw SHA changed; SSH0')
    result = read(round_path)
    c = read(bundle / 'execution_contract.json')
    observer.verify_monitor(monitor, seal['monitor_integrity_raw_sha256'], c, bundle)
    c, _ = launch.check_bundle(bundle)
    require(result.get('schema') == 'titantpp_gru_cap_validation_monitor_v1'
        and result.get('contract_sha256') == canonical(c)
        and result.get('monitor_integrity_raw_sha256') == seal['monitor_integrity_raw_sha256']
        and result.get('held_out_test_evaluated') is False, 'Round identity changed; SSH0')
    counts = observer.unique_counts(c, result['hosts'])
    require(counts == result['counts'] and result['all_training_terminal_verified']
        is counts['all_training_terminal_verified'], 'Round count closure mismatch; SSH0')
    if not result['all_training_terminal_verified']:
        return {'status': 'waiting_for_terminal_training', 'remote_calls': 0,
            'original_retrieval': 0, 'Test': 0, 'counts': counts}
    require(counts == {'total': 9, 'completed': 9, 'running': 0, 'waiting': 0, 'unconfirmed': 0,
        'all_training_terminal_verified': True}, 'Nine terminal jobs required; SSH0')
    launcher = launch.launcher_for(bundle)
    expected = {}
    cache_evidence = {}
    # Reverify both caches before allowing even the first host transport.
    for host in HOSTS:
        cached = observer.load_cache(bundle, monitor, c, host, launcher)
        require(cached is not None, 'Verified terminal cache absent; SSH0')
        path, snapshot, report = cached
        source = result['hosts'][host]
        require(source['snapshot_path'] == str(path) and source['snapshot_raw_sha256'] == sha(path)
            and source['observed_unix'] == snapshot['observed_unix']
            and source['status'] == observer.TERMINAL and source['owner_process_absence_verified'] is True,
            'Round and terminal cache disagree; SSH0')
        require({r['job']['id'] for r in report['jobs']} == {r['job']['id'] for r in source['jobs']}
            and all(r['status'] == 'complete' for r in report['jobs']), 'Cached terminal jobs differ; SSH0')
        expected[host] = terminal_files(c, host, snapshot, launcher)
        cache = observer.cache_path(monitor, host)
        cache_evidence[host] = {'path': str(cache), 'raw_sha256': sha(cache),
            'snapshot_path': str(path), 'snapshot_raw_sha256': sha(path),
            'original_observed_unix': snapshot['observed_unix']}
    return {'status': 'terminal_verified', 'contract': c, 'expected': expected,
        'terminal_cache_evidence': cache_evidence, 'round_path': str(round_path), 'round_raw_sha256': round_sha}


REMOTE_SCRIPT = r'''
import hashlib,json,os,pathlib,stat,sys,tarfile
p=json.load(sys.stdin)
root=pathlib.Path(p['root'])
assert root.is_absolute() and not root.is_symlink() and root.resolve()==root
assert p['host'] in ('5080','5090')
assert root.name=='titantpp_gru_residual_cap_validation_20261014_v1_'+p['host']
assert 0<len(p['files'])<=4096
def safe(name):
    q=pathlib.PurePosixPath(name)
    assert not q.is_absolute() and str(q)==name and '\\' not in name and '..' not in q.parts
    x=root/'run'
    assert not x.is_symlink()
    for part in q.parts:
        x=x/part
        assert not x.is_symlink()
    assert x.resolve().is_relative_to(root/'run') and x.is_file()
    return x
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
total=0
for name,digest in p['files'].items():
    x=safe(name);total+=x.stat().st_size
    assert total<=p['maximum_bytes'] and sha(x)==digest
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as out:
    for name in sorted(p['files']):
        x=safe(name)
        with x.open('rb') as f:
            s=os.fstat(f.fileno());assert stat.S_ISREG(s.st_mode)
            info=tarfile.TarInfo(name);info.size=s.st_size;info.mode=0o444;info.mtime=s.st_mtime
            out.addfile(info,f)
'''


def retrieve_once(host: str, spec: dict[str, Any], expected: dict[str, str], destination: Path) -> dict[str, Any]:
    """One read-only SSH; raw stdout is an exclusive archive and stderr is retained."""
    require(host in HOSTS and str(spec['root']).endswith('_' + host), 'Foreign retrieval host')
    destination.mkdir(parents=True, exist_ok=False)
    require(shutil.disk_usage(destination).free >= MAX_BYTES + 1024 ** 3, 'Insufficient local archive space')
    archive, stderr = destination / 'originals.tar.gz', destination / 'stderr.raw'
    command = ' '.join(shlex.quote(x) for x in ['env', 'PYTHONDONTWRITEBYTECODE=1',
        spec['python'], '-c', REMOTE_SCRIPT])
    payload = json.dumps({'host': host, 'root': spec['root'], 'files': expected,
        'maximum_bytes': MAX_BYTES}, ensure_ascii=False, allow_nan=False).encode()
    started, timed_out, launch_error, rc = time.time(), False, None, None
    with archive.open('xb') as out, stderr.open('xb') as err:
        try:
            process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, command],
                stdin=subprocess.PIPE, stdout=out, stderr=err)
            try:
                process.communicate(payload, timeout=600)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()  # Only this local SSH transport, never a remote training process.
                process.communicate()
            rc = process.returncode
        except OSError as error:
            launch_error = type(error).__name__ + ': ' + str(error)
    receipt = {'host': host, 'remote_calls': 1, 'remote_mutations': 0, 'started_unix': started,
        'ended_unix': time.time(), 'actual_transport_exit': rc, 'timeout': timed_out,
        'launch_error': launch_error, 'archive_path': str(archive), 'archive_raw_sha256': sha(archive),
        'stdout_bytes': archive.stat().st_size, 'stderr_path': str(stderr),
        'stderr_raw_sha256': sha(stderr), 'stderr_bytes': stderr.stat().st_size,
        'truncation': False, 'automatic_retry': False}
    unique_write(destination / 'transport_receipt.json', receipt)
    require(rc == 0 and not timed_out and launch_error is None, 'Retrieval transport failed; raw diagnostics preserved; no retry')
    require(archive.stat().st_size <= MAX_BYTES + 1024 ** 2, 'Archive exceeds retrieval byte limit')
    return receipt


def verify_archive(path: Path, expected: dict[str, str]) -> dict[str, int]:
    seen, total = set(), 0
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            safe(Path('/'), member.name)
            require(member.isfile() and not member.issym() and not member.islnk()
                and member.name in expected and member.name not in seen,
                'Unexpected, duplicate, or nonregular archive member')
            require(0 <= member.size <= MAX_BYTES, 'Invalid archive member size')
            total += member.size
            require(total <= MAX_BYTES and len(seen) < MAX_FILES, 'Archive expansion limit exceeded')
            stream = archive.extractfile(member)
            h = hashlib.sha256()
            count = 0
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block); count += len(block)
            require(count == member.size and h.hexdigest() == expected[member.name], 'Archive member SHA differs')
            seen.add(member.name)
    require(seen == set(expected), 'Archive member set incomplete')
    return {'file_count': len(seen), 'uncompressed_bytes': total}


def extract_verified(path: Path, expected: dict[str, str], root: Path) -> None:
    # A full first pass verifies all members before creating any recovered original.
    verify_archive(path, expected)
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            target = safe(root, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as out, archive.extractfile(member) as stream:
                shutil.copyfileobj(stream, out, length=1024 * 1024)
            require(sha(target) == expected[member.name], 'Archive changed during extraction')


def finish_local(bundle: Path, report: Path, retrieval: Path, gate: dict[str, Any],
                 integrity_sha: str) -> dict[str, Any]:
    from paper.scripts import summarize_titantpp_gru_cap_validation as summary
    original = retrieval / 'originals'
    expected = {name: value for host in HOSTS for name, value in gate['expected'][host].items()}
    require(len(expected) == sum(len(v) for v in gate['expected'].values()), 'Duplicate cross-host job files')
    for name, value in expected.items():
        require(sha(safe(original, name)) == value, 'Recovered original changed')
    index_path = retrieval / 'retrieval_index.json'
    normalize(bundle / 'execution_contract.json', original, index_path)
    result = summary.summarize(bundle / 'execution_contract.json', original, index_path)
    require(result['status'] == 'complete_validation_only' and result['candidate_fit_count'] == 9
        and result['baseline_reference_count'] == 18 and result['held_out_test_evaluated'] is False,
        'Final Validation summary is incomplete')
    json_path, md_path = report / 'THREE_SEED_VALIDATION.json', report / 'THREE_SEED_VALIDATION.md'
    unique_write(json_path, result)
    summary.write_exclusive(md_path, summary.markdown(result))
    local_files = {str(p): sha(p) for p in sorted(retrieval.rglob('*')) if p.is_file()}
    local_files.update({str(json_path): sha(json_path), str(md_path): sha(md_path)})
    receipt = {'schema': 'titantpp_gru_cap_collection_final_v1', 'status': 'complete',
        'candidate_fit_count': 9, 'seed_count': 3, 'baseline_reference_count': 18,
        'contract_raw_sha256': sha(bundle / 'execution_contract.json'),
        'contract_sha256': canonical(gate['contract']), 'collection_integrity_raw_sha256': integrity_sha,
        'round_path': gate['round_path'], 'round_raw_sha256': gate['round_raw_sha256'],
        'terminal_cache_evidence': gate['terminal_cache_evidence'],
        'held_out_test_evaluated': False, 'remote_mutations': 0, 'automatic_retry': False,
        'original_files_sha256': canonical(expected), 'original_file_count': len(expected),
        'local_files': local_files, 'local_files_sha256': canonical(local_files),
        'summary_json_path': str(json_path), 'summary_json_raw_sha256': sha(json_path),
        'summary_markdown_path': str(md_path), 'summary_markdown_raw_sha256': sha(md_path),
        'completed_unix': time.time()}
    unique_write(retrieval / 'final_receipt.json', receipt)
    return receipt


def verify_final(bundle: Path, report: Path, retrieval: Path, gate: dict[str, Any],
                 integrity_sha: str) -> dict[str, Any]:
    from paper.scripts import summarize_titantpp_gru_cap_validation as summary
    receipt = read(retrieval / 'final_receipt.json')
    require(receipt.get('schema') == 'titantpp_gru_cap_collection_final_v1'
        and receipt['status'] == 'complete' and receipt['candidate_fit_count'] == 9 and receipt['seed_count'] == 3
        and receipt['baseline_reference_count'] == 18 and receipt['held_out_test_evaluated'] is False
        and receipt['remote_mutations'] == 0 and receipt['automatic_retry'] is False
        and receipt['contract_raw_sha256'] == sha(bundle / 'execution_contract.json')
        and receipt['contract_sha256'] == canonical(gate['contract'])
        and receipt['collection_integrity_raw_sha256'] == integrity_sha
        and canonical(receipt['local_files']) == receipt['local_files_sha256'], 'Final receipt identity changed; SSH0')
    summary_paths = (report / 'THREE_SEED_VALIDATION.json', report / 'THREE_SEED_VALIDATION.md')
    inventory = {str(p) for p in retrieval.rglob('*') if p.is_file()
        and p != retrieval / 'final_receipt.json'} | {str(p) for p in summary_paths}
    require(set(receipt['local_files']) == inventory, 'Final artifact inventory differs; SSH0')
    for name, value in receipt['local_files'].items():
        path = Path(name)
        require(path == path.resolve() and (path.is_relative_to(retrieval) or path in summary_paths),
            'Final receipt path escaped; SSH0')
        require(not path.is_symlink() and sha(path) == value, 'Final local artifact SHA changed; SSH0')
    for host in HOSTS:
        verify_archive(retrieval / 'archives' / host / 'originals.tar.gz', gate['expected'][host])
        for name, value in gate['expected'][host].items():
            require(sha(safe(retrieval / 'originals', name)) == value, 'Final original SHA changed; SSH0')
    result = summary.summarize(bundle / 'execution_contract.json', retrieval / 'originals', retrieval / 'retrieval_index.json')
    require(result == read(report / 'THREE_SEED_VALIDATION.json')
        and summary.markdown(result) == (report / 'THREE_SEED_VALIDATION.md').read_text(),
        'Final derived summary changed; SSH0')
    return {**receipt, 'cached_complete_reverified': True, 'remote_calls_this_call': 0}


def collect(round_path: Path, round_sha: str, integrity_sha: str, *, bundle: Path = BUNDLE,
            monitor: Path = MONITOR, report: Path = REPORT, acquire=retrieve_once) -> dict[str, Any]:
    bundle, monitor, report = Path(bundle).resolve(), Path(monitor).resolve(), Path(report).resolve()
    gate = completion_gate(bundle, monitor, Path(round_path), round_sha, integrity_sha)
    if gate['status'] != 'terminal_verified':
        return gate
    retrieval = bundle / 'retrieval'
    if (retrieval / 'final_receipt.json').exists():
        return verify_final(bundle, report, retrieval, gate, integrity_sha)
    require(not retrieval.exists(), 'Existing incomplete collection intent preserved; no automatic retry; SSH0')
    require(not (report / 'THREE_SEED_VALIDATION.json').exists()
        and not (report / 'THREE_SEED_VALIDATION.md').exists(), 'Existing Validation report preserved; SSH0')
    retrieval.mkdir(parents=True, exist_ok=False)
    unique_write(retrieval / 'intent.json', {'round_path': str(round_path), 'round_raw_sha256': round_sha,
        'collection_integrity_raw_sha256': integrity_sha, 'started_unix': time.time(),
        'pid': os.getpid(), 'automatic_retry': False, 'maximum_read_only_SSH_per_host': 1,
        'remote_mutations': 0, 'held_out_test_evaluated': False})
    try:
        receipts = {}
        for host in HOSTS:
            receipts[host] = acquire(host, gate['contract']['hosts'][host], gate['expected'][host],
                retrieval / 'archives' / host)
            verify_archive(Path(receipts[host]['archive_path']), gate['expected'][host])
        # Both complete archives must verify before local original extraction starts.
        original = retrieval / 'originals'
        original.mkdir(exist_ok=False)
        for host in HOSTS:
            extract_verified(Path(receipts[host]['archive_path']), gate['expected'][host], original)
        return finish_local(bundle, report, retrieval, gate, integrity_sha)
    except Exception as error:
        unique_write(retrieval / 'failure.json', {'status': 'failed_no_automatic_retry',
            'error': type(error).__name__ + ': ' + str(error), 'failed_unix': time.time(),
            'remote_mutations': 0, 'held_out_test_evaluated': False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True, type=Path)
    parser.add_argument('--round-sha', required=True)
    parser.add_argument('--integrity-sha', required=True)
    args = parser.parse_args()
    result = collect(args.round, args.round_sha, args.integrity_sha)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
