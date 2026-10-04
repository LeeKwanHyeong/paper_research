"""Read-only original artifact retrieval and immutable width16 evaluation registry."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shlex
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'reports/titantpp_width16_test_20261003_v1'
OBSERVATION = '20261004T045346227283Z'
MODEL = 'titantpp_history_mlp_width16'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def inventory():
    rows, bundles = [], {}
    for name in ('dual', 'replication'):
        folder = ROOT / f'search_artifacts/titantpp_history_width16_{name}_20261003_v1'
        contract = read(folder / 'execution_contract.json')
        source = folder / 'frozen_source'
        for relative, digest in contract['source']['files'].items():
            assert sha(source / relative) == digest, relative
        bundles[name] = {'source_root': str(source.relative_to(ROOT)),
                         'source_files': contract['source']['files'],
                         'source_closure_sha256': contract['source']['files_sha256'],
                         'datasets': contract['datasets']}
        observed_path = folder / 'observations' / OBSERVATION / 'summary.json'
        for observation in read(observed_path)['observations']:
            for record in observation['jobs']:
                job, terminal = record['job'], record['terminal_manifest']
                assert terminal['scientific_success'] and terminal['status'] == 'complete'
                assert terminal['job'] == job
                selected = record['endpoint_replays']['selected']
                reuse = job['seed'] == 42 and job['dataset'] in ('yellow_trip_hourly', 'raf_spare_parts')
                local = OLD / 'original' / job['id'] if reuse else HERE / 'original' / job['id']
                rows.append({'job': job, 'bundle': name, 'terminal': terminal,
                    'terminal_manifest_sha256': record['terminal_manifest_sha256'],
                    'observed_path': str(observed_path.relative_to(ROOT)),
                    'observed_sha256': sha(observed_path),
                    'remote_folder': observation['root'] + '/run/' + job['id'],
                    'local_folder': str(local.relative_to(ROOT)),
                    'selected_epoch': record['endpoint_replays']['best_epoch'],
                    'state_tensor_sha256': selected['state_sha256'],
                    'validation_reference': {k: selected[k] for k in ('count', 'qty_rmse', 'qty_mae', 'time_nll')},
                    'reuse_existing_prediction_splits': ['validation', 'test'] if reuse else []})
    assert len(rows) == 12
    assert len({(row['job']['dataset'], row['job']['seed']) for row in rows}) == 12
    return rows, bundles


def retrieve(rows):
    receipts = []
    grouped = {}
    for row in rows:
        if row['reuse_existing_prediction_splits']:
            continue
        grouped.setdefault(row['job']['host'], []).append(row)
    # Each remote call only reads pre-existing terminal-bound files and emits a tar stream.
    remote_code = '''import hashlib,io,json,sys,tarfile
from pathlib import Path
items=json.loads(sys.argv[1])
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|") as archive:
 for item in items:
  path=Path(item["remote"]);data=path.read_bytes()
  assert hashlib.sha256(data).hexdigest()==item["sha256"],str(path)
  info=tarfile.TarInfo(item["local"]);info.size=len(data);info.mode=0o644
  archive.addfile(info,io.BytesIO(data))
'''
    wanted = {'initialization.json', 'input_receipt.json', 'status.json',
              'best_val_qty_rmse_model.pt', 'history.json', 'endpoint_replays.json',
              'selected_train_validation_diagnostic.json', 'summary.json'}
    for host, group in grouped.items():
        items = []
        for row in group:
            for relative, digest in row['terminal']['files'].items():
                if Path(relative).name not in wanted:
                    continue
                items.append({'remote': row['remote_folder'] + '/' + relative,
                    'local': str(Path('original') / row['job']['id'] / relative), 'sha256': digest})
            items.append({'remote': row['remote_folder'] + '/terminal_manifest.json',
                'local': str(Path('original') / row['job']['id'] / 'terminal_manifest.json'),
                'sha256': row['terminal_manifest_sha256']})
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
                   shlex.join(['python3', '-c', remote_code, json.dumps(items)])]
        result = subprocess.run(command, check=False, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError(f'{host} read-only retrieval failed: {result.stderr.decode(errors="replace")}')
        output = result.stdout
        expected = {item['local']: item for item in items}
        payloads = {}
        with tarfile.open(fileobj=io.BytesIO(output), mode='r:') as archive:
            for member in archive.getmembers():
                assert member.isfile() and member.name in expected and member.name not in payloads
                payload = archive.extractfile(member).read()
                assert hashlib.sha256(payload).hexdigest() == expected[member.name]['sha256']
                payloads[member.name] = payload
        assert set(payloads) == set(expected)
        for relative, payload in payloads.items():
            destination = HERE / relative
            assert not destination.exists()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(payload)
            receipts.append({'host': host, **expected[relative], 'bytes': len(payload)})
    write('retrieval_receipt.json', {'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'remote_mutations': False, 'last_checkpoint_binary_retrieved': False, 'files': receipts})


def prepare(rows, bundles):
    registry_rows, selected_binding = [], []
    for row in rows:
        folder = ROOT / row['local_folder']
        selected_relative = next(p for p in row['terminal']['files'] if p.endswith('/best_val_qty_rmse_model.pt'))
        checkpoint = folder / selected_relative
        history = checkpoint.with_name('history.json')
        for path in (checkpoint, history, checkpoint.with_name('endpoint_replays.json')):
            assert sha(path) == row['terminal']['files'][str(path.relative_to(folder))]
        finite = [entry for entry in read(history)['history'] if math.isfinite(entry['val_qty_rmse'])]
        selected = min(finite, key=lambda entry: entry['val_qty_rmse'])
        assert selected['epoch'] == row['selected_epoch']
        assert math.isclose(selected['val_qty_rmse'], row['validation_reference']['qty_rmse'], rel_tol=1e-12, abs_tol=1e-12)
        registry_rows.append({'dataset': row['job']['dataset'], 'model': MODEL,
            'seed': row['job']['seed'], 'endpoint': 'selected', 'evaluator_source_bundle': row['bundle'],
            'checkpoint_path': str(checkpoint.relative_to(ROOT)), 'checkpoint_file_sha256': sha(checkpoint),
            'state_tensor_sha256': row['state_tensor_sha256'], 'selected_epoch': row['selected_epoch'],
            'validation_count': row['validation_reference']['count']})
        selected_binding.append({'dataset': row['job']['dataset'], 'seed': row['job']['seed'],
            'selected_epoch': row['selected_epoch'], 'history_sha256': sha(history),
            'first_validation_minimum_confirmed': True, 'validation_reference': row['validation_reference'],
            'reuse_existing_prediction_splits': row['reuse_existing_prediction_splits']})
    write('evaluation_registry.json', {'rows': registry_rows, 'bundles': bundles})
    write('selected_binding.json', {'selection_unchanged': True, 'rows': selected_binding})
    for filename, source in [('evaluate.py', OLD / 'evaluate.py'),
                             ('dataset_manifest.json', ROOT / 'reports/titantpp_legacy_evaluation_20261003_v1/dataset_manifest.json')]:
        with (HERE / filename).open('xb') as stream:
            stream.write(source.read_bytes())
    write('preparation_receipt.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'registry_sha256': sha(HERE / 'evaluation_registry.json'),
        'dataset_manifest_sha256': sha(HERE / 'dataset_manifest.json'),
        'evaluator_sha256': sha(HERE / 'evaluate.py'),
        'evaluator_byte_identical_to_existing': sha(HERE / 'evaluate.py') == sha(OLD / 'evaluate.py'),
        'conditions': 12, 'new_inference_splits_required': 20, 'existing_splits_to_reuse': 4,
        'remote_evaluation_started': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieve', action='store_true')
    args = parser.parse_args()
    rows, bundles = inventory()
    if (HERE / 'inventory.json').exists():
        assert read(HERE / 'inventory.json') == {'rows': rows}
    else:
        write('inventory.json', {'rows': rows})
    if args.retrieve:
        retrieve(rows)
        prepare(rows, bundles)
    print(json.dumps({'conditions': len(rows), 'retrieval_requested': args.retrieve}))
