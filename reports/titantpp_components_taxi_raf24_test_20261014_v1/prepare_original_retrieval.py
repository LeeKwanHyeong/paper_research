"""Build an exact360 read-only retrieval packet from sealed local originals."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
TRAINING = PROJECT / 'search_artifacts/titantpp_cnn_gru_components_20261012_v5/execution_contract.json'
CACHE = PROJECT / 'reports/titantpp_cnn_gru_components_20261012_v5/hourly_monitor_v2/terminal_cache/5080.json'
SNAPSHOT = PROJECT / 'search_artifacts/titantpp_cnn_gru_components_20261012_v5/control/status/5080/1ecf99983b614d9b8f3984cad020c2fe.json'
TRAINING_RAW = '591a31ed862148c1f7343fef33a65da9d263ec9b80769a240d1ff43cdb6b49f4'
CACHE_RAW = 'a732d32a5b5976123da6fe2c6ae92844040f7d0cc6e4c05130e1b60e742b3ce1'
SNAPSHOT_RAW = '352b65c7f5c55a11fae7720b67c26c437b6999f662cfa0b8297fe0b2c33556f2'
CANONICAL = 'e60badde6d9d902500aa67764f6834213ea1775797508f31d204ae67a5e63074'
SOURCE = '15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53'
REMOTE = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_components_20261012_v5_5080'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decoded(snapshot, name):
    item = snapshot['files'][name]
    raw = base64.b64decode(item['base64'], validate=True)
    require(hashlib.sha256(raw).hexdigest() == item['raw_sha256'], 'Snapshot original raw SHA differs: ' + name)
    return json.loads(raw), item['raw_sha256']


def create_packet():
    for path, digest in ((TRAINING, TRAINING_RAW), (CACHE, CACHE_RAW), (SNAPSHOT, SNAPSHOT_RAW)):
        require(sha(path) == digest, 'Pinned original changed: ' + str(path))
    training, cache, snapshot = (json.loads(path.read_text()) for path in (TRAINING, CACHE, SNAPSHOT))
    canonical = hashlib.sha256(json.dumps(training, sort_keys=True, ensure_ascii=False,
                              allow_nan=False, separators=(',', ':')).encode()).hexdigest()
    require(canonical == CANONICAL and training['source']['files_sha256'] == SOURCE, 'Training source/contract differs')
    require(cache['root'] == snapshot['root'] == REMOTE and snapshot['contract_sha256'] == CANONICAL
            and snapshot['source_files_sha256'] == SOURCE and snapshot['gpu_pids'] == [], 'Original terminal root differs')
    for name in ('phases/train_exit.json', 'phases/train_complete.json'):
        exit_receipt, _ = decoded(snapshot, name)
        require(exit_receipt['contract_sha256'] == CANONICAL, 'Original dispatcher exit contract differs')
        key = 'os_exit_code' if name.endswith('train_exit.json') else 'subprocess_os_exit_code'
        require(type(exit_receipt[key]) is int and exit_receipt[key] == 0, 'Original dispatcher actual exit0 differs')
    jobs = [j for j in training['jobs'] if j['host'] == '5080']
    require(len(jobs) == 24 and {j['dataset'] for j in jobs} == {'yellow_trip_hourly', 'raf_spare_parts'},
            'Original Taxi/RAF24 scope differs')
    files = {}
    for job in jobs:
        prefix = 'run/' + job['id'] + '/'
        terminal, digest = decoded(snapshot, prefix + 'terminal_manifest.json')
        require(terminal['job'] == job and terminal['contract_sha256'] == CANONICAL
                and terminal['scientific_success'] is True and terminal['status'] == 'complete'
                and terminal['held_out_test_evaluated'] is False
                and terminal['evaluation_scope'] == 'validation_only', 'Terminal scientific identity differs')
        worker_path = 'worker_exits/' + job['id'] + '.json'
        worker, worker_digest = decoded(snapshot, worker_path)
        require(worker['job'] == job and worker['contract_sha256'] == CANONICAL
                and worker['actual_child_exit_observed'] is True and type(worker['returncode']) is int
                and worker['returncode'] == 0, 'Original scientific worker actual exit0 differs')
        files[prefix + 'terminal_manifest.json'] = digest
        files[worker_path] = worker_digest
        verification = snapshot['terminal_hashes'][job['id']]
        require(verification['all_files_verified'] is True
                and set(verification['files']) == set(terminal['files'])
                and all(value is True for value in verification['files'].values()), 'Original terminal SHA verification incomplete')
        for relative, raw_sha in terminal['files'].items():
            name = prefix + relative
            if name in snapshot['terminal_hashes']:
                require(snapshot['terminal_hashes'][name] == raw_sha, 'Terminal binary hash differs')
            if name in snapshot['files']:
                item = snapshot['files'][name]
                require(hashlib.sha256(base64.b64decode(item['base64'], validate=True)).hexdigest()
                        == item['raw_sha256'] == raw_sha, 'Snapshot terminal file hash differs')
            files[name] = raw_sha
    require(len(files) == 360, 'Exactly360 original files required')
    return {'schema': 'titantpp_components_taxi_raf24_original_retrieval_v1',
            'remote_original_root': REMOTE, 'training_contract_raw_sha256': TRAINING_RAW,
            'training_contract_canonical_sha256': CANONICAL, 'source131_closure': SOURCE,
            'terminal_cache_raw_sha256': CACHE_RAW, 'terminal_snapshot_raw_sha256': SNAPSHOT_RAW,
            'expected_files': files, 'job_ids': sorted(j['id'] for j in jobs),
            'original_root_writes': 0, 'new_training': False, 'model_predictions': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    packet = create_packet()
    raw = (json.dumps(packet, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as handle:
            handle.write(raw)
    print(json.dumps({'status': 'prepared', 'files': len(packet['expected_files']),
                      'conditions': len(packet['job_ids']), 'packet_raw_sha256': hashlib.sha256(raw).hexdigest(),
                      'remote_calls': 0, 'model_predictions': 0}))


if __name__ == '__main__':
    main()
