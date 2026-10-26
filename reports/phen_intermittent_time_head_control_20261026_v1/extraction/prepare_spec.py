"""Generate a reviewable execution spec; no remote connection or inference."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parents[1] / 'phen_intermittent_full_observations_20261026_v1'


def remap(value, old, new):
    if isinstance(value, str): return value.replace(old, new)
    if isinstance(value, list): return [remap(v, old, new) for v in value]
    if isinstance(value, dict): return {k: remap(v, old, new) for k, v in value.items()}
    return value


def record(path, remote_path=None):
    path = Path(path)
    return {'path': str(remote_path or path.resolve()), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(seed, remote_root, science_spec, remote_science_spec, references, mode, gate=None):
    prior_path = PRIOR / ('execution_v8/5090_FULL_SPEC_v8.json' if seed == 52 else 'execution_v6/5080_FULL_SPEC_v6.json')
    prior = json.loads(prior_path.read_text())
    jobs = deepcopy(prior['jobs'])
    if seed == 42:
        older = json.loads((PRIOR / 'execution_v4/5080_FULL_SPEC_v4.json').read_text())
        best = next(j for j in older['jobs'] if j['seed'] == 42 and j['checkpoint_kind'] == 'best')
        jobs.insert(0, remap(best, '_execution_v4', '_execution_v6'))
    remote_root = Path(remote_root)
    assert remote_root.is_absolute() and 'phen_intermittent_time_head_control_' in str(remote_root)
    allowed = ('execution_device', 'batch_size', 'chunk_rows', 'num_threads', 'population_counts',
               'input_order', 'reference_samples', 'working_dir', 'expected_device_name', 'expected_device_uuid', 'split_manifest_record')
    spec = {key: deepcopy(prior[key]) for key in allowed}
    spec.update(schema='phen_time_head_hidden_extraction_v1', runner_filename='collector_hidden.py',
                allowed_output_root=str(remote_root),
                runner_sha256=record(ROOT / 'collector_hidden.py')['sha256'], mode=mode,
                qualification_batches=16, child_timeout_seconds=900 if mode == 'qualification' else 7200,
                science_spec=record(science_spec, remote_science_spec),
                comparison_tolerances={'pred_quantity': {'atol': 1e-5, 'rtol': 1e-5}, 'time_nll': {'atol': 1e-4, 'rtol': 1e-4}})
    manifest = json.loads(Path(references).read_text())
    spec['reference_manifest_provenance'] = record(references)
    for job in jobs:
        job['splits'] = ['validation'] if job['checkpoint_kind'] == 'best' else ['train', 'validation']
        job['reference_results'] = {}
        for split in job['splits']:
            ref = next(r for r in manifest['references'] if (r['seed'], r['checkpoint'], r['split']) == (seed, job['checkpoint_kind'], split))
            assert ref['checkpoint_sha256'] == job['checkpoint_sha256']
            job['reference_results'][split] = {**ref['reference'], 'path': str(remote_root / 'references' / Path(ref['reference']['path']).name)}
        job['reference_precision'] = {r['split']: r['mu_sigma_precision'] for r in manifest['references'] if (r['seed'], r['checkpoint']) == (seed, job['checkpoint_kind'])}
    spec['jobs'] = jobs
    if mode == 'full':
        assert gate is not None, 'Full spec requires collected/hash-bound qualification job COMPLETE + ACTUAL_EXIT refs'
        spec['qualification_gate'] = json.loads(Path(gate).read_text())
        assert set(spec['qualification_gate']) == {j['id'] for j in jobs}
    previous_root = str(Path(prior['working_dir']).parent)
    environment = remap(deepcopy(prior['process_environment']), previous_root, str(remote_root))
    guard_path = PRIOR / ('execution_v8/5090_GUARD.json' if seed == 52 else 'execution_v6/5080_GUARD.json')
    old_guard = json.loads(guard_path.read_text())
    context = {'host': '5090' if seed == 52 else '5080', 'seed': seed, 'remote_root': str(remote_root),
               'python': old_guard['python'], 'tmux': old_guard['tmux'], 'process_environment': environment,
               'existing_bundle': prior['working_dir'], 'prior_spec': record(prior_path), 'prior_guard': record(guard_path),
               'protected': old_guard['protected'], 'boot_id_recorded_previously': old_guard['boot_id'],
               'gpu_uuid_recorded_previously': old_guard['gpu_uuid'], 'minimum_free_mib': old_guard['minimum_free_mib'],
               'runtime_libraries': old_guard['runtime_libraries'],
               'warning': 'Read-only historical paths/identities; root must revalidate current host/protected-process/GPU state before remote launch. Never modify previous root or signal protected processes.'}
    return spec, context


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(52, 42), required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--science-spec', type=Path, required=True)
    parser.add_argument('--remote-science-spec', required=True)
    parser.add_argument('--references', type=Path, default=ROOT / 'references_v1/REFERENCES.json')
    parser.add_argument('--mode', choices=('qualification', 'full'), required=True)
    parser.add_argument('--qualification-gates', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec, context = prepare(args.seed, args.remote_root, args.science_spec, args.remote_science_spec, args.references, args.mode, args.qualification_gates)
    with args.output.open('x') as stream: json.dump(spec, stream, indent=2); stream.write('\n')
    with args.output.with_suffix('.context.json').open('x') as stream: json.dump(context, stream, indent=2); stream.write('\n')
    print(json.dumps({'spec': record(args.output), 'context': record(args.output.with_suffix('.context.json'))}, indent=2))
