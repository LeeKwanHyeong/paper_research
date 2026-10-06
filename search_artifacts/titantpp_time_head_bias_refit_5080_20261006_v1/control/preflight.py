"""Read-only host/immutable-original gate for the separately approved head refit."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    c = json.load(sys.stdin)
    parent = Path(c['parent_root'])
    p = json.loads((parent/'execution_contract.json').read_text())
    assert digest(p) == c['parent_contract_sha256']
    assert p['hosts']['5080']['root'] == str(parent)
    assert json.loads((parent/'status.json').read_text())['status'] == 'complete'
    assert json.loads((parent/'supervisor_process_exit.json').read_text())['returncode'] == 0
    lines = subprocess.check_output(['ps', '-eo', 'pid,ppid,args'], text=True).splitlines()
    assert not any(str(parent/'operation/campaign.py') in line and
        ('--mode fit' in line or '--mode dispatch' in line) for line in lines)
    assert not any(str(Path(c['root'])/'operation') in line for line in lines)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip()
    assert c['runtime']['gpu_uuid'] in gpu and '5080' in gpu
    gpu_pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader'], text=True).strip()
    assert not gpu_pids, 'GPU has another owner; do not launch'
    assert digest(c['source']['files']) == c['source']['files_sha256']
    for rel, expected in c['source']['files'].items():
        assert sha(Path(c['source']['root'])/rel) == expected, rel
    for d in c['datasets']:
        for ref in (d['inherited_data_identity']['data'], d['inherited_data_identity']['split_manifest']):
            assert sha(Path(c['data_root'])/ref['path']) == ref['sha256']
    checks = []
    for j in c['jobs']:
        cp = j['checkpoint']; binding = j['source_terminal']
        assert sha(cp['path']) == cp['sha256'], j['id']
        assert sha(binding['path']) == binding['sha256']
        terminal = json.loads(Path(binding['path']).read_text())
        assert terminal['scientific_success'] is True
        assert terminal['held_out_test_evaluated'] is False
        assert any(rel.endswith('best_val_qty_rmse_model.pt') and value == cp['sha256']
            for rel, value in terminal['files'].items())
        ref = j['source_endpoint']
        assert sha(ref['path']) == ref['sha256']
        ep = json.loads(Path(ref['path']).read_text())
        assert ep['evaluation_scope'] == 'validation_only'
        assert ep['held_out_test_evaluated'] is False
        assert ep['best_epoch'] == cp['epoch']
        assert ep['selected']['state_sha256'] == cp['state_sha256']
        checks.append({'job': j['id'], 'checkpoint_sha256': cp['sha256'],
            'source_epoch': cp['epoch'], 'source_terminal_sha256': binding['sha256']})
    print(json.dumps({'status': 'passed', 'actual_observed_utc':
        datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host': '5080', 'parent_contract_sha256': c['parent_contract_sha256'],
        'GPU': gpu, 'gpu_pids': [], 'source_files_verified': len(c['source']['files']),
        'jobs': checks, 'starts_training': False, 'held_out_test_evaluated': False}))


if __name__ == '__main__':
    main()
