"""Local-only archive verification and collection; no remote, model or training imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import traceback

HERE = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def record(path, remote=None):
    path = Path(path)
    return {'path': str(remote or path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_file(path, ref):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or linked file: ' + str(path))
    require(path.stat().st_size == ref['bytes'] and sha(path) == ref['sha256'], 'File SHA/bytes mismatch: ' + str(path))
    return path


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and path.as_posix() == name
            and all(part not in ('.', '..', '') for part in name.split('/')),
            'Unsafe archive member: ' + name)
    return path


def safe_extract(archive, output):
    require(not output.exists(), 'Existing extraction/partial preserved: ' + str(output))
    # Inspect every member before creating any target file.
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        require(0 < len(members) <= 20000, 'Unexpected archive member count')
        names = [m.name for m in members]
        require(len(names) == len(set(names)), 'Duplicate archive member')
        require(sum(m.size for m in members) <= 5_000_000_000, 'Archive uncompressed size exceeds bound')
        for member in members:
            safe_name(member.name)
            require(member.isfile() and not member.issym() and not member.islnk() and not member.isdev(), 'Only regular archive files admitted')
        output.mkdir(parents=True, exist_ok=False)
        for member in members:
            dest = output / member.name
            require(dest.resolve().is_relative_to(output.resolve()), 'Archive member escaped output')
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, dest.open('xb') as target:
                shutil.copyfileobj(source, target)
    return set(names)


def find_pack(here, host, mode, archive, remote_root, explicit=None):
    candidates = [Path(explicit)] if explicit else sorted((here / 'transports').glob(f'*_{host}_pack_{mode}.stdout'))
    matching = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text())
            if payload['archive']['path'] != str(remote_root / (mode + '_results.tar.gz')):
                continue
            if payload['archive']['sha256'] != sha(archive) or payload['archive']['bytes'] != archive.stat().st_size:
                continue
            require(payload['manifest']['path'] == str(remote_root / (mode + '_RESULT_MEMBERS.json')), 'Pack manifest path differs')
            transport = path.with_suffix('.receipt.json')
            if transport.exists():
                receipt = json.loads(transport.read_text())
                require(receipt['actual_exit'] == 0 and not receipt['timed_out'], 'Pack transport failed')
                verify_file(path, receipt['stdout'])
            else:
                require(explicit is not None, 'Automatic pack discovery requires transport receipt')
            matching.append((path, payload))
        except (KeyError, json.JSONDecodeError):
            continue
    require(len(matching) == 1, 'Require exactly one matching archive pack record; found ' + str(len(matching)))
    return matching[0]


def verify_members(originals, mode, payload, remote_root):
    name = mode + '_RESULT_MEMBERS.json'
    manifest_path = verify_file(originals / name, payload['manifest'])
    manifest = json.loads(manifest_path.read_text())
    expected = set(manifest['files']) | {name}
    actual = {p.relative_to(originals).as_posix() for p in originals.rglob('*') if p.is_file()}
    require(actual == expected, 'Archive and MEMBERS inventories differ')
    for relative, ref in manifest['files'].items():
        safe_name(relative)
        require(ref['path'] == str(remote_root / relative), 'MEMBERS remote path differs: ' + relative)
        verify_file(originals / relative, ref)
    return manifest_path, manifest


def verify_exit(value, controller=False):
    if controller:
        require(value['actual_collector_exit'] == 0 and value['post_guard_passed'] is True, 'Controller exit/post-guard failed')
    else:
        require(value['actual_child_exit'] == 0 and not value.get('timed_out', False), 'Child exit/timeout failed')


def verify_guard(observed, frozen):
    for field in ('boot_id', 'gpu_uuid', 'protected'):
        require(observed[field] == frozen[field], 'Guard identity/protected state differs: ' + field)
    require(observed['gpu_free_mib'] >= frozen['minimum_free_mib'], 'Guard free memory below frozen bound')
    allowed = {p['pid'] for p in frozen['protected'] if p['state'] == 'T'}
    require(set(observed['compute_pids']) <= allowed, 'Unadmitted compute process in guard')


def local_remote_ref(originals, remote_root, ref):
    remote = Path(ref['path'])
    require(remote.is_relative_to(remote_root), 'Remote reference escaped attempt root')
    path = originals / remote.relative_to(remote_root)
    verify_file(path, ref)
    return record(path)


def verify_chunk_payload(local_chunk, scalar_ref, tolerances):
    import numpy as np
    import pandas as pd
    obs = pd.read_parquet(local_chunk['observations']['path'])
    with np.load(local_chunk['head_hidden']['path'], allow_pickle=False) as stored:
        require(set(stored.files) == {'head_hidden', 'target_index'}, 'Hidden NPZ fields differ')
        hidden, ids = stored['head_hidden'], stored['target_index']
        require(ids.dtype == np.int64 and hidden.dtype == np.float32 and hidden.shape == (len(ids), 64)
                and np.isfinite(hidden).all(), 'Hidden dtype/shape/values differ')
        require(np.array_equal(ids, obs.target_index.to_numpy()), 'Hidden/observation row identity differs')
    require(len(obs) == local_chunk['rows'], 'Chunk row count differs')
    require(hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest() == local_chunk['target_index_sha256'], 'Chunk target index digest differs')
    require(np.all((ids >= 0) & (ids < len(scalar_ref['target_index']))), 'Chunk target outside reference population')
    for field in ('target_index', 'entity_id', 'target_seq', 'raw_quantity', 'history_length', 'target_gap'):
        actual = obs[field].astype(str).to_numpy() if field == 'entity_id' else obs[field].to_numpy()
        require(np.array_equal(actual, scalar_ref[field][ids]), 'Chunk/reference identity differs: ' + field)
    require(np.array_equal(obs.true_quantity, obs.raw_quantity), 'Native true quantity differs')
    maximum = {}
    for field in ('pred_quantity', 'time_nll', 'time_location_float64', 'time_sigma_float64'):
        actual, expected = obs[field].to_numpy(dtype=np.float64), scalar_ref[field][ids]
        require(np.isfinite(actual).all() and np.isfinite(expected).all(), 'Nonfinite compact observation')
        error = np.abs(actual - expected)
        maximum[field] = float(error.max())
        if field in tolerances:
            require(np.all(error <= tolerances[field]['atol'] + tolerances[field]['rtol'] * np.abs(expected)), 'Saved-observation parity failed: ' + field)
    return ids, maximum


def collect(here, host, mode, archive=None, pack_json=None):
    import numpy as np
    execution = here / 'execution_v1' / host
    config = json.loads((execution / 'PREPARED.json').read_text())
    remote_root = Path(config['remote_root'])
    archive = Path(archive or execution / (mode + '_results.tar.gz'))
    pack_path, payload = find_pack(here, host, mode, archive, remote_root, pack_json)
    verify_file(archive, payload['archive'])
    destination = execution / 'retrieval' / mode
    require(not destination.exists(), 'Existing collection/partial preserved')
    originals = destination / 'originals'
    try:
        safe_extract(archive, originals)
        members_path, members = verify_members(originals, mode, payload, remote_root)
        stage = execution / 'stage'
        spec_path = stage / f'{host}_{mode.upper()}_SPEC.json'
        spec = json.loads(spec_path.read_text())
        spec_digest = sha(spec_path)
        require(spec['mode'] == mode, 'Local execution spec mode differs')
        require(spec['allowed_output_root'] == str(remote_root), 'Local execution root differs')
        require(spec['comparison_tolerances'] == {'pred_quantity': {'atol': 1e-5, 'rtol': 1e-5}, 'time_nll': {'atol': 1e-4, 'rtol': 1e-4}}, 'Parity tolerance changed')
        verify_file(stage / spec['runner_filename'], {'bytes': (stage / spec['runner_filename']).stat().st_size, 'sha256': spec['runner_sha256']})
        guard = json.loads((stage / (host + '_GUARD.json')).read_text())
        verify_guard(members['guard'], guard)
        controller_path = originals / 'control' / (mode + '.ACTUAL_EXIT.json')
        controller = json.loads(controller_path.read_text()); verify_exit(controller, True)
        require(controller['spec']['path'] == str(remote_root / spec_path.name), 'Controller used another spec path')
        verify_file(spec_path, controller['spec'])
        for key in ('stdout', 'stderr'):
            local_remote_ref(originals, remote_root, controller[key])
        for point in ('before_guard', 'after_guard'):
            verify_guard(json.loads((originals / 'control' / f'{mode}.{point}.json').read_text()), guard)
        claim = json.loads((originals / 'control' / f'{mode}.claim.json').read_text())
        require(claim['host'] == host and claim['mode'] == mode and claim['spec_sha256'] == spec_digest, 'Controller claim differs')
        require(claim['manifest_sha256'] == sha(stage / 'DEPLOY_MANIFEST.json'), 'Deployment manifest changed')
        if mode == 'qualification':
            runtime = json.loads((originals / 'control/qualification.runtime.ACTUAL_EXIT.json').read_text())
            require(runtime['actual_exit'] == 0 and not runtime.get('timed_out', False), 'CUDA primitive qualification failed')
        out = originals / ('output_' + mode)
        supervisor = json.loads((out / 'COMPLETE.json').read_text())
        require(supervisor['spec_sha256'] == spec_digest and supervisor['collector_sha256'] == spec['runner_sha256']
                and supervisor['actual_child_exits_all_zero'], 'Collector supervisor binding/exit differs')
        require(len(supervisor['receipts']) == len(spec['jobs']), 'Supervisor receipt count differs')
        for ref in supervisor['receipts']: local_remote_ref(originals, remote_root, ref)
        gates, cohorts, checked_files = {}, [], set()
        for job in spec['jobs']:
            child_path = out / (job['id'] + '.ACTUAL_EXIT.json')
            child = json.loads(child_path.read_text()); verify_exit(child)
            require(child['job'] == job['id'] and child['spec_sha256'] == spec_digest and child['collector_sha256'] == spec['runner_sha256'], 'Child binding differs')
            for key in ('stdout', 'stderr'): local_remote_ref(originals, remote_root, child[key])
            job_dir = out / job['id']; complete_path = job_dir / 'COMPLETE.json'
            result = json.loads(complete_path.read_text())
            require(result['status'] == ('complete' if mode == 'full' else 'qualification_complete'), 'Job status differs')
            for field, expected in (('seed', job['seed']), ('checkpoint_kind', job['checkpoint_kind']), ('epoch', job['expected_epoch']),
                                    ('checkpoint_sha256', job['checkpoint_sha256']), ('canonical_checkpoint_state_sha256', job['expected_model_state_sha256']),
                                    ('source_root', job['source_root']), ('runner_sha256', spec['runner_sha256']), ('spec_semantic_sha256', json_sha(spec))):
                require(result[field] == expected, 'Job identity differs: ' + field)
            require(result['parameter_buffer_sha_before'] == result['parameter_buffer_sha_after'] and result['input_and_model_immutability_verified']
                    and result['context_state_unchanged'] and result['parameter_grad_fields_all_none'], 'Model/source/checkpoint immutability failed')
            require(result['held_out_materialized'] is False and result['new_training_epochs'] == result['optimizer_steps'] == 0, 'Unadmitted heldout/training execution')
            require(result['runtime']['eval'] and not result['runtime']['autograd'] and not result['runtime']['optimizer_created'], 'Runtime was not frozen evaluation')
            require(all(relative in job['source_files'] for relative in result['actual_imports'].values()), 'Unbound source imported')
            require(set(result['splits']) == set(job['splits']), 'Job split inventory differs')
            gates[job['id']] = {'job_complete': record(complete_path, remote_root / ('output_' + mode) / job['id'] / 'COMPLETE.json'),
                                'actual_exit': record(child_path, remote_root / ('output_' + mode) / child_path.name)}
            for split, split_result in result['splits'].items():
                split_path = job_dir / (split + '_manifest.json')
                require(json.loads(split_path.read_text()) == split_result, 'Split manifest differs from COMPLETE')
                expected_rows = spec['population_counts'][split] if mode == 'full' else 2048
                require(split_result['rows'] == split_result['native_reference_rows_checked'] == expected_rows and split_result['head_hidden_saved'], 'Incomplete hidden/parity rows')
                require(sum(x['rows'] for x in split_result['chunks']) == expected_rows, 'Chunk rows differ')
                local_remote_ref(originals, remote_root, split_result['progress'])
                saved_ref = job['reference_results'][split]
                scalar_path = stage / 'references' / Path(saved_ref['path']).name
                verify_file(scalar_path, saved_ref)
                with np.load(scalar_path, allow_pickle=False) as stored:
                    scalar_ref = {key: stored[key].copy() for key in stored.files}
                chunks, all_ids, maximum = [], [], {}
                offset = 0
                for number, chunk in enumerate(split_result['chunks']):
                    require(chunk['chunk'] == number and chunk['row_offset'] == offset and chunk['split'] == split, 'Chunk sequence differs')
                    local = {**chunk, 'remote_observations': chunk['observations'], 'remote_head_hidden': chunk['head_hidden']}
                    for key in ('observations', 'head_hidden'):
                        local[key] = local_remote_ref(originals, remote_root, chunk[key]); checked_files.add(local[key]['path'])
                    ids, errors = verify_chunk_payload(local, scalar_ref, spec['comparison_tolerances'])
                    all_ids.append(ids); offset += len(ids); chunks.append(local)
                    for key, value in errors.items(): maximum[key] = max(maximum.get(key, 0.), value)
                ids = np.concatenate(all_ids)
                require(len(np.unique(ids)) == expected_rows, 'Duplicate targets in collected cohort')
                sample_ref = job['reference_samples'][split]
                previous = here.parent / 'phen_intermittent_full_observations_20261026_v1'
                sample_local = previous / ('execution_v8' if job['seed'] == 52 else 'execution_v6') / 'bundle/references' / Path(sample_ref['path']).name
                sample = json.loads(verify_file(sample_local, sample_ref).read_text())
                expected_sample = np.array([row['target_index'] for row in sample['targets']], dtype=np.int64)
                require(len(expected_sample) == 2048 and np.array_equal(ids[:2048], expected_sample), 'Collected smoke order differs from frozen sample')
                if mode == 'full':
                    require(np.array_equal(np.sort(ids), np.arange(expected_rows)), 'Incomplete full target population')
                    rest = np.ones(expected_rows, dtype=bool); rest[expected_sample] = False
                    require(np.array_equal(ids[2048:], np.flatnonzero(rest)), 'Remaining full target order differs')
                require(set(maximum) == set(split_result['reference_max_absolute_errors']), 'Parity summary fields differ')
                for key in maximum:
                    require(math.isclose(maximum[key], split_result['reference_max_absolute_errors'][key], rel_tol=1e-12, abs_tol=1e-12), 'Native parity summary differs: ' + key)
                cohorts.append({'seed': job['seed'], 'checkpoint': job['checkpoint_kind'], 'epoch': job['expected_epoch'], 'split': split,
                    'rows': expected_rows, 'checkpoint_sha256': job['checkpoint_sha256'], 'chunks': chunks,
                    'source_refs': {'archive': record(archive), 'members': record(members_path), 'controller_exit': record(controller_path),
                                    'child_exit': record(child_path), 'job_complete': record(complete_path), 'split_manifest': record(split_path)},
                    'parity_max_absolute_errors': maximum, 'actual_child_exit': 0, 'model_state_unchanged': True})
        require(len(cohorts) == 3, 'Expected exactly three cohorts per host')
        result = {'status': 'COLLECTED_AND_VERIFIED', 'host': host, 'mode': mode, 'remote_root': str(remote_root),
                  'archive': record(archive), 'pack_record': record(pack_path), 'members': record(members_path), 'spec': record(spec_path),
                  'originals': str(originals.resolve()), 'members_verified': len(members['files']), 'cohorts': cohorts,
                  'qualification_gates': gates if mode == 'qualification' else None,
                  'actual_controller_exit': 0, 'post_guard_passed': True, 'all_child_exits_zero': True,
                  'training': False, 'inference': False, 'remote_calls': 0}
        if mode == 'qualification': save(destination / 'qualification_gates.json', gates)
        save(destination / 'COLLECTED.json', result)
        return result
    except BaseException:
        if destination.exists() and not (destination / 'FAILURE.json').exists():
            save(destination / 'FAILURE.json', {'status': 'failed_preserved', 'traceback': traceback.format_exc(), 'retry_attempted': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=('5090', '5080'), required=True)
    parser.add_argument('--mode', choices=('qualification', 'full'), required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--pack-json', type=Path)
    args = parser.parse_args()
    result = collect(HERE, args.host, args.mode, args.archive, args.pack_json)
    print(json.dumps({key: value for key, value in result.items() if key not in ('cohorts', 'qualification_gates')}, indent=2))
