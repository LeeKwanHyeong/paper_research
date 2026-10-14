"""One approved manual2 observation and, only after terminal verification, collection.

The original observer and collector remain unchanged. Their pinned source is
loaded into isolated namespaces with only the approved launcher/root routing.
No qualification, training, repair, or retry is performed by this wrapper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import ModuleType


PROJECT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT / 'search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1/execution_manual2'
OUTPUT = PROJECT / 'reports/titantpp_gru_residual_cap_validation_20261014_v1/hourly_monitor_manual2'
REPORT = OUTPUT.parent
OBSERVER = 'paper/scripts/observe_titantpp_gru_cap_validation.py'
COLLECTOR = 'paper/scripts/collect_titantpp_gru_cap_validation.py'
PINNED = {
    OBSERVER: '89f5794e7e85a585cbdd1793b1320c08240245a519ef39836158743df5421c04',
    COLLECTOR: '1dd9bcb96643999f0cff5f2531dcde78477440a2c45bb762e7a89f228bf6fa1e',
}
REQUIRED = {'paper/scripts/advance_titantpp_gru_cap_manual2.py',
    'paper/scripts/launch_titantpp_gru_cap_manual2.py'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_extra_seal(path, expected):
    require(sha(path) == expected, 'Manual2 seal rawSHA changed; remote0')
    seal = read(path)
    require(REQUIRED <= set(seal['files']), 'Manual2 wrapper/launcher missing from seal; remote0')
    for name in REQUIRED:
        require(sha(PROJECT / name) == seal['files'][name], 'Manual2 routing code SHA changed; remote0')
    return seal


def replace_once(text, old, new):
    require(text.count(old) == 1, 'Pinned routing target is not unique; remote0')
    return text.replace(old, new, 1)


def load_namespace(name, path, extra, replacements):
    raw = (PROJECT / path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PINNED[path], 'Original helper SHA changed; remote0')
    text = raw.decode('utf-8')
    for old, new in replacements:
        text = replace_once(text, old, new)
    module = ModuleType(name)
    module.__dict__.update(__file__=str(PROJECT / path), **extra)
    exec(compile(text, str(PROJECT / path), 'exec'), module.__dict__)
    return module


def routed_helpers(launcher):
    observer = load_namespace('_cap_manual2_observer', OBSERVER, {'_manual_launcher': launcher}, [
        ('from paper.scripts import launch_titantpp_gru_cap_validation as launch',
         'launch = _manual_launcher')])
    original_monitor_check = observer.verify_monitor

    def monitor_check(output, digest, c, bundle=BUNDLE):
        require(Path(output).resolve() == OUTPUT and Path(bundle).resolve() == BUNDLE,
            'Manual2 observation path drift; remote0')
        verify_extra_seal(Path(output) / 'monitor_integrity.json', digest)
        return original_monitor_check(output, digest, c, bundle)

    observer.verify_monitor = monitor_check
    collector = load_namespace('_cap_manual2_collector', COLLECTOR,
        {'_manual_launcher': launcher, '_manual_observer': observer}, [
        ('from paper.scripts import launch_titantpp_gru_cap_validation as launch',
         'launch = _manual_launcher'),
        ('from paper.scripts import observe_titantpp_gru_cap_validation as observer',
         'observer = _manual_observer'),
        ("assert root.name=='titantpp_gru_residual_cap_validation_20261014_v1_'+p['host']",
         "assert root.name=='titantpp_gru_residual_cap_validation_20261014_v1_'+p['host']+'_manual2'"),
        ("str(spec['root']).endswith('_' + host)", "str(spec['root']).endswith('_' + host + '_manual2')")])
    original_collection_check = collector.verify_seal

    def collection_check(bundle, monitor, integrity_sha):
        require(Path(bundle).resolve() == BUNDLE and Path(monitor).resolve() == OUTPUT,
            'Manual2 collection path drift; remote0')
        verify_extra_seal(Path(bundle) / 'collection_integrity.json', integrity_sha)
        return original_collection_check(bundle, monitor, integrity_sha)

    collector.verify_seal = collection_check
    return observer, collector


def checked_round(result):
    require(Path(result['round_path']).resolve().is_relative_to(OUTPUT / 'rounds')
        and Path(result['latest_path']).resolve() == OUTPUT / 'latest.json'
        and Path(result['markdown_path']).resolve() == Path(result['round_path']).with_suffix('.md'),
        'Manual2 round output path drift; no collection')
    for name, checksum in (('round_path', 'round_raw_sha256'), ('markdown_path', 'markdown_raw_sha256'),
                           ('latest_path', 'latest_raw_sha256')):
        require(sha(result[name]) == result[checksum], 'Manual2 round artifact SHA changed; no collection')
    latest = read(result['latest_path'])
    require(all(latest.get(k) == result[k] for k in ('round_path', 'round_raw_sha256',
        'markdown_path', 'markdown_raw_sha256', 'contract_sha256')), 'Manual2 latest pointer differs; no collection')


def advance(monitor_integrity_sha, collection_integrity_sha):
    # Both seals are checked before even the observer's single permitted SSH.
    verify_extra_seal(OUTPUT / 'monitor_integrity.json', monitor_integrity_sha)
    verify_extra_seal(BUNDLE / 'collection_integrity.json', collection_integrity_sha)
    from paper.scripts import launch_titantpp_gru_cap_manual2 as launcher
    observer, collector = routed_helpers(launcher)
    collector.verify_seal(BUNDLE, OUTPUT, collection_integrity_sha)
    result = observer.observe(BUNDLE, OUTPUT, monitor_integrity_sha=monitor_integrity_sha)
    checked_round(result)
    collection = collector.collect(Path(result['round_path']), result['round_raw_sha256'],
        collection_integrity_sha, bundle=BUNDLE, monitor=OUTPUT, report=REPORT)
    return {'observation': result, 'collection': collection, 'automatic_retry': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monitor-integrity-sha', required=True)
    parser.add_argument('--collection-integrity-sha', required=True)
    args = parser.parse_args()
    print(json.dumps(advance(args.monitor_integrity_sha, args.collection_integrity_sha),
        ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
