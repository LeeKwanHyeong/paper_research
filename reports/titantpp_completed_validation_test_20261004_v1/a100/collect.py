"""Retrieve completed A100 reevaluation evidence and recompute all per-seed metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
from datetime import datetime, timezone

from verify import (read, sha, save, require, inspect_predictions, compare_identity,
                    check_receipt, validation_gate)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REMOTE_READER = r'''
import hashlib,io,json,sys,tarfile,time
from pathlib import Path
b=Path(sys.argv[1]);completion=json.loads((b/'completion_receipt.json').read_text())
assert completion['status']=='complete' and completion['validation_conditions']==12 and completion['test_conditions']==12
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()
names=['completion_receipt.json','results.json','validation_gate.json','campaign_started.json','execution.lock']
paths=[b/n for n in names if (b/n).is_file()]
paths+=sorted(p for name in ('runs','logs') for p in (b/name).rglob('*') if p.is_file() and not p.is_symlink())
files={str(p.relative_to(b)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
assert sum(v['bytes'] for v in files.values())<15*1024**3
manifest=json.dumps({'files':files,'observed_unix':time.time(),'root':str(b)},sort_keys=True).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tf:
 info=tarfile.TarInfo('collection_manifest.json');info.size=len(manifest);tf.addfile(info,io.BytesIO(manifest))
 for p in paths:tf.add(p,arcname=str(p.relative_to(b)),recursive=False)
'''


def retrieve():
    import tarfile
    contract = read(HERE / 'execution_contract.json')
    remote = str(Path(contract['resources']['root']) / HERE.relative_to(ROOT))
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '5080',
               shlex.join(['python3', '-c', REMOTE_READER, remote])]
    require(not (HERE / 'retrieval_receipt.json').exists(), 'Original retrieval receipt already exists')
    require(not (HERE / 'runs').exists(), 'No implicit overwrite or partial retrieval retry')
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    seen, manifest = set(), None
    try:
        with tarfile.open(fileobj=child.stdout, mode='r|') as archive:
            for member in archive:
                require(member.isfile(), 'Only regular files may be retrieved')
                require(member.size <= 128 * 1024**2, 'Unexpectedly large individual evidence file')
                payload = archive.extractfile(member).read()
                if manifest is None:
                    require(member.name == 'collection_manifest.json', 'Missing collection manifest')
                    manifest = json.loads(payload)
                    continue
                require(member.name in manifest['files'] and member.name not in seen, 'Unexpected retrieval file')
                relative = Path(member.name)
                require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact path')
                require(relative.parts[0] in ('runs', 'logs') or member.name in
                        ('completion_receipt.json', 'results.json', 'validation_gate.json',
                         'campaign_started.json', 'execution.lock'), 'Unapproved artifact path')
                expected = manifest['files'][member.name]
                require(len(payload) == expected['bytes'] and hashlib.sha256(payload).hexdigest() == expected['sha256'],
                        'Retrieved evidence SHA mismatch')
                target = HERE / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(payload)
                seen.add(member.name)
        stderr = child.stderr.read().decode(errors='replace')
        require(child.wait(timeout=30) == 0, 'Read-only retrieval failed: ' + stderr)
        require(manifest is not None and seen == set(manifest['files']), 'Incomplete artifact retrieval')
        save(HERE / 'retrieval_receipt.json', {'status': 'sha_verified', 'remote_mutations': False,
            'remote_root': remote, 'retrieved_utc': datetime.now(timezone.utc).isoformat(), **manifest})
    except BaseException:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        raise


def analyze():
    contract = read(HERE / 'execution_contract.json')
    completion = read(HERE / 'completion_receipt.json')
    require(completion['status'] == 'complete' and completion['contract_sha256'] == sha(HERE / 'execution_contract.json'),
            'Completion contract mismatch')
    require(completion['results_sha256'] == sha(HERE / 'results.json'), 'Remote results SHA mismatch')
    registry = read(HERE / 'evaluation_registry.json')
    bindings = {(r['dataset'], r['model'], r['seed']): r for r in read(HERE / 'selected_binding.json')['rows']}
    baseline_bindings = {r['dataset']: r for r in read(HERE / 'comparison_binding.json')['rows']}
    baseline_cache, rows, evidence = {}, [], []
    for row in registry['rows']:
        binding = bindings[(row['dataset'], row['model'], row['seed'])]
        name = f"{row['dataset']}__{row['model']}__seed{row['seed']}"
        for split in ('validation', 'test'):
            folder = HERE / 'runs' / split / name
            receipt, groups = inspect_predictions(folder, binding['tail_threshold'])
            check_receipt(receipt, row, contract, ROOT)
            if split == 'validation':
                for view, reference in [('overall', binding['validation_reference']),
                                        ('tail', binding['validation_tail_reference'])]:
                    validation_gate(groups[view], reference, contract['validation_gate']['rtol'],
                                    contract['validation_gate']['atol'])
            else:
                if row['dataset'] not in baseline_cache:
                    baseline = baseline_bindings[row['dataset']]
                    require(sha(ROOT / baseline['folder'] / 'receipt.json') == baseline['receipt_sha256'],
                            'Baseline receipt changed')
                    baseline_cache[row['dataset']] = inspect_predictions(ROOT / baseline['folder'], binding['tail_threshold'])
                compare_identity(receipt, baseline_cache[row['dataset']][0])
            for view, metrics in groups.items():
                rows.append({'dataset': row['dataset'], 'model': row['model'], 'seed': 42,
                    'split': split, 'view': view, 'threshold': binding['tail_threshold'] if view == 'tail' else None,
                    'threshold_rule': 'raw_quantity_strictly_greater_than_train_threshold' if view == 'tail' else '',
                    **metrics, 'selected_epoch': row['selected_epoch'], 'checkpoint_sha256': row['checkpoint_file_sha256'],
                    'source_bundle': row['evaluator_source_bundle'],
                    'source_closure_sha256': registry['bundles'][row['evaluator_source_bundle']]['source_closure_sha256'],
                    'comparison_family': 'A100_seed42_exploratory', 'training_gpu': 'A100 SXM', 'evaluation_gpu': 'RTX 5080',
                    'time_nll_definition': 'recorded_positive_integer_lognormal_round_clamp_mass_nll; top-coded survival where configured',
                    'receipt_path': str((folder / 'receipt.json').relative_to(ROOT)),
                    'receipt_sha256': sha(folder / 'receipt.json'),
                    'source': str((folder / 'receipt.json').relative_to(ROOT)),
                    'source_sha256': sha(folder / 'receipt.json')})
            evidence.append({'dataset': row['dataset'], 'model': row['model'], 'split': split,
                'receipt_sha256': sha(folder / 'receipt.json'), 'independently_recomputed_parts': len(receipt['parts']),
                'target_identity_sha256': receipt['target_identity_sha256'], 'truth_sha256': receipt['truth_sha256']})
    require(len(rows) == 48 and len(evidence) == 24, 'Missing candidate split/view')
    with (HERE / 'metrics_per_seed.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    save(HERE / 'analysis_receipt.json', {'status': 'complete', 'metric_rows': len(rows),
        'complete_split_conditions': len(evidence), 'metrics_sha256': sha(HERE / 'metrics_per_seed.csv'),
        'all_raw_prediction_parts_reverified': True, 'source_gate': 'frozen evaluator checks before/after; CPU strict-load preflight retained',
        'evidence': evidence, 'sample_sd': None, 'not_independent_untouched_test': True,
        'not_scheduled': read(HERE / 'selected_binding.json')['not_scheduled']})
    print(json.dumps({'status': 'complete', 'metric_rows': len(rows), 'split_conditions': len(evidence)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieve', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    if not (args.retrieve or args.analyze):
        parser.error('Select --retrieve and/or --analyze')
    if args.retrieve:
        retrieve()
    if args.analyze:
        analyze()
