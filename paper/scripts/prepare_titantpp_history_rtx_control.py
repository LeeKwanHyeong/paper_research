"""Seal per-host RTX controls from the verified A100 qualification pipeline.

No SSH, CUDA, training, resource allocation, or shared Runtime change. Native
qualification and three-worker admission remain mandatory before dispatcher fit.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

PROJECT = Path(__file__).resolve().parents[2]
PARENT = PROJECT / 'search_artifacts/titantpp_contribution_history_recovery_20261010_v1'
BUNDLE = PROJECT / 'search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1'
PARENT_CONTROL_SHA = 'e1ab1ba7e3765fed14079430372e40a8f938bb547e4e191566d28ed4377e8d58'
PARENT_CONTRACT_SHA = 'da077e96e61255286c5cb35b54dc09ec8144e2d2f410e6eb46fea4681ee17899'
CONTRACT_SHA = '518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False) + '\n').encode()


def safe(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe sealed path')
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Sealed path escape')
    return path


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = payload.encode() if isinstance(payload, str) else payload
    if path.exists():
        require(path.read_bytes() == payload, 'Existing immutable control differs: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(payload)


def substitute(text, old, new):
    require(old in text, 'Missing control transform: ' + old[:100])
    return text.replace(old, new)


def require_prelaunch(bundle):
    """No re-sealing after an attempt, claim, permit, or scientific run exists."""
    patterns = ('launch_receipt*.json', 'launch_intent*.json', 'deployment_receipt*.json',
        'deploy_intent*.json', 'deploy_receipt*.json')
    require(not any(any((bundle / 'deployment').glob(pattern)) for pattern in patterns),
        'Remote deployment already attempted; controls are immutable')
    for name in ('run', 'pipeline.claim', 'supervisor.claim', 'qualification.claim',
                 'training_permit.json', 'start_permit_5080.json', 'start_permit_5090.json'):
        require(not (bundle / name).exists(), 'Preserved execution/permit blocks re-sealing: ' + name)


def reseal_prelaunch(args):
    """Preserve the old local seal and bind approved project fixes before deploy.

Remote controls, source, operation, and package bytes cannot change in this
procedure. A different deployed artifact requires a separate approved bundle.
    """
    bundle = Path(args.bundle).resolve()
    require_prelaunch(bundle)
    seal_path = bundle / 'control/local_tools_integrity.json'
    previous_bytes = seal_path.read_bytes()
    previous_sha = hashlib.sha256(previous_bytes).hexdigest()
    seal = json.loads(previous_bytes)
    require(seal['contract_sha256'] == CONTRACT_SHA
        and canonical(read(bundle / 'execution_contract.json')) == CONTRACT_SHA,
        'Prelaunch contract binding drift')
    for name, digest in seal['files'].items():
        require(sha(safe(bundle, name)) == digest, 'Remote control/package drift; no reseal: ' + name)
    for name, digest in read(bundle / 'execution_contract.json')['operation']['files'].items():
        require(sha(safe(bundle, name)) == digest, 'Operation drift; no reseal: ' + name)
    project_files = dict(seal['project_files'])
    project_files['paper/tests/test_titantpp_history_rtx_deployment.py'] = ''
    project_files = {name: sha(safe(PROJECT, name)) for name in project_files}
    require(project_files != seal['project_files'], 'Project files unchanged; no reseal needed')
    preserved = bundle / 'control/preserved_prelaunch' / ('local_tools_integrity.' + previous_sha + '.json')
    write_new(preserved, previous_bytes)
    require(sha(preserved) == previous_sha, 'Prior local seal preservation failed')
    updated = {**seal, 'project_files': project_files, 'prelaunch_reseal': {
        'prior_integrity_raw_sha256': previous_sha,
        'reason': 'Approved strict A100 ownership-release proof verification and its regression tests',
        'no_remote_deploy_attempt_or_run_verified': True,
        'remote_control_and_package_bytes_unchanged': True}}
    payload = encoded(updated)
    next_sha = hashlib.sha256(payload).hexdigest()
    # Check again immediately before replacing only the explicitly unlaunched
    # local index; an immutable raw copy of the previous index already exists.
    require_prelaunch(bundle)
    require(seal_path.read_bytes() == previous_bytes, 'Local seal changed during re-sealing')
    temporary = seal_path.with_name(seal_path.name + '.prelaunch_reseal.' + next_sha)
    write_new(temporary, payload)
    temporary.replace(seal_path)
    receipt = {'contract_sha256': CONTRACT_SHA, 'status': 'verified_prelaunch_resealed',
        'previous_integrity_raw_sha256': previous_sha, 'integrity_raw_sha256': next_sha,
        'preserved_previous_seal': str(preserved.relative_to(bundle)),
        'project_files': project_files, 'remote_control_and_package_bytes_unchanged': True,
        'network_calls': 0, 'GPU_calls': 0}
    write_new(bundle / 'control' / ('prelaunch_reseal_receipt.' + next_sha + '.json'), encoded(receipt))
    return receipt


def read_original_controls():
    integrity_path = PARENT / 'control/integrity_receipt.json'
    require(sha(integrity_path) == PARENT_CONTROL_SHA, 'Parent control seal drift')
    seal = read(integrity_path)
    require(seal['contract_sha256'] == PARENT_CONTRACT_SHA, 'Parent control contract drift')
    originals = {}
    for name in ('pipeline.py', 'concurrency_admission.py'):
        relative = 'control/' + name
        path = PARENT / relative
        require(sha(path) == seal['files'][relative], 'Parent control file drift: ' + relative)
        originals[name] = path.read_text()
    return originals


def transform_pipeline(original, host):
    require(host in ('5080', '5090'), 'Foreign RTX host')
    text = original.replace("'a100'", 'HOST')
    text = substitute(text,
        "require(set(c['hosts'])=={HOST} and c['hosts'][HOST]['root']==str(root),'Foreign owned root')",
        "require(set(c['hosts'])=={'5080','5090'} and c['hosts'][HOST]['root']==str(root),'Foreign owned root')")
    text = substitute(text, "hard=min(start['deadline_unix'],c['budget']['hard_deadline_unix'])",
        "hard=start['deadline_unix']")
    text = text.replace("len(c['jobs'])", "len([j for j in c['jobs'] if j['host']==HOST])")
    # A stdlib operation imported for preflight must also have a stable module
    # identity when its robust GPU helper resolves sys.modules[__name__].
    text = substitute(text,
        "spec=importlib.util.spec_from_file_location('sealed_a100_history_operation',path)",
        "spec=importlib.util.spec_from_file_location('sealed_rtx_history_operation_'+HOST,path)")
    text = substitute(text,
        'module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)',
        'module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)')
    text = substitute(text, "    op=load_operation(root);op.ROOT=Path(h['source_root']).resolve()",
        "    sys.path.insert(0,str(root/'operation'))\n    op=load_operation(root);op.ROOT=Path(h['source_root']).resolve()")
    text = substitute(text, 'import traceback\n', 'import traceback\n\nHOST = ' + repr(host) + '\n')
    text = text.replace('One-attempt native qualification -> history27 dispatch in an owned A100 root.',
        'One-attempt native qualification -> three-worker history dispatch in an owned RTX root.')
    text = text.replace('This\nprocess does not implement a provider billing stop; a separately verified budget\ncontroller is required.',
        'Each\nphase and fit is bounded by the approved host and condition wall-clock lease.')
    require("c['budget']" not in text and "'a100'" not in text, 'Unconverted provider authority')
    ast.parse(text)
    return text


def transform_admission(original, host):
    require(host in ('5080', '5090'), 'Foreign RTX host')
    text = original.replace("'a100'", 'HOST')
    text = substitute(text,
        'from pipeline import read, write, require, sha_json, sha_file, stop_owned',
        'from pipeline import HOST, read, write, require, sha_json, sha_file, stop_owned')
    text = substitute(text, "deadline = min(deadline, c['budget']['hard_deadline_unix'])",
        "deadline = min(deadline, read(root / 'start_permit.json')['deadline_unix'])")
    text = substitute(text, '    spec.loader.exec_module(op)',
        '    sys.modules[spec.name] = op\n    spec.loader.exec_module(op)')
    require("c['budget']" not in text and "'a100'" not in text, 'Unconverted provider deadline')
    ast.parse(text)
    # The synthetic model inputs, optimizer updates, finite checks, barrier,
    # ownership witnesses, and admission criteria only change host selection.
    for name in ('guard', 'wait_file', 'validate_ready', 'gpu_state'):
        node = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == name)
        transformed = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
        require(ast.get_source_segment(original, node).replace("'a100'", 'HOST')
            == ast.get_source_segment(text, transformed), 'Admission scientific gate changed: ' + name)
    return text


def write_package(bundle, host, mapping):
    target = bundle / 'deployment' / ('package_' + host + '.tar.gz')
    raw = io.BytesIO()
    files = {}
    with tarfile.open(fileobj=raw, mode='w') as archive:
        for remote_name, local_path in sorted(mapping.items()):
            files[remote_name] = sha(local_path)
            info = tarfile.TarInfo(remote_name)
            info.size = local_path.stat().st_size
            info.mode = 0o644
            with local_path.open('rb') as stream:
                archive.addfile(info, stream)
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode='wb', mtime=0) as stream:
        stream.write(raw.getvalue())
    write_new(target, compressed.getvalue())
    manifest = {'host': host, 'contract_sha256': CONTRACT_SHA, 'files': files,
        'archive_raw_sha256': sha(target), 'file_count': len(files),
        'release_and_start_permit_copied_separately_before_launch': True,
        'network_calls': 0, 'GPU_calls': 0}
    write_new(bundle / 'deployment' / ('package_manifest_' + host + '.json'), encoded(manifest))
    return manifest


def prepare(args):
    bundle = Path(args.bundle).resolve()
    # Re-sealing after a remote deployment would invalidate a running owner.
    require_prelaunch(bundle)
    contract = read(bundle / 'execution_contract.json')
    require(canonical(contract) == CONTRACT_SHA, 'Prepared RTX contract drift')
    require(read(bundle / 'approval.json')['contract_sha256'] == CONTRACT_SHA,
        'Prepared approval drift')
    originals = read_original_controls()
    host_metadata = {}
    all_files = {}
    for host in ('5080', '5090'):
        local_root = bundle / 'control/hosts' / host
        pipeline = transform_pipeline(originals['pipeline.py'], host)
        admission = transform_admission(originals['concurrency_admission.py'], host)
        write_new(local_root / 'pipeline.py', pipeline)
        write_new(local_root / 'concurrency_admission.py', admission)
        mapping = {name: safe(bundle, name) for name in contract['operation']['files']}
        mapping.update({'control/pipeline.py': local_root / 'pipeline.py',
            'control/concurrency_admission.py': local_root / 'concurrency_admission.py'})
        control_seal = {'contract_sha256': CONTRACT_SHA, 'host': host,
            'remote_root': contract['hosts'][host]['root'],
            'files': {name: sha(path) for name, path in sorted(mapping.items())},
            'parent_control_integrity_raw_sha256': PARENT_CONTROL_SHA,
            'native_qualification_and_concurrency_required': True,
            'A100_owner_release_required': True, 'automatic_retry': False}
        write_new(local_root / 'integrity_receipt.json', encoded(control_seal))
        seal_raw = sha(local_root / 'integrity_receipt.json')
        mapping['control/integrity_receipt.json'] = local_root / 'integrity_receipt.json'
        for name in {**contract['source']['files'], **contract['input_files']}:
            path = safe(bundle / 'source', name)
            expected = {**contract['source']['files'], **contract['input_files']}[name]
            require(sha(path) == expected, 'Sealed scientific source/input drift: ' + name)
            mapping['source/' + name] = path
        for name, digest in contract['operation']['files'].items():
            require(sha(safe(bundle, name)) == digest, 'Sealed operation drift: ' + name)
        mapping.update({name: bundle / name for name in ('execution_contract.json', 'approval.json')})
        manifest = write_package(bundle, host, mapping)
        host_metadata[host] = {'root': contract['hosts'][host]['root'],
            'integrity_raw_sha256': seal_raw,
            'pipeline_raw_sha256': sha(local_root / 'pipeline.py'),
            'admission_raw_sha256': sha(local_root / 'concurrency_admission.py'),
            'package_raw_sha256': manifest['archive_raw_sha256'],
            'package_path': str(bundle / 'deployment' / ('package_' + host + '.tar.gz')),
            'file_count': manifest['file_count'],
            'scientific_jobs': sum(j['host'] == host for j in contract['jobs']),
            'native_runtime': contract['hosts'][host]['runtime_expected']}
        for path in sorted(local_root.iterdir()):
            all_files[str(path.relative_to(bundle))] = sha(path)
        for name in ('package_' + host + '.tar.gz', 'package_manifest_' + host + '.json'):
            all_files['deployment/' + name] = sha(bundle / 'deployment' / name)
    local_seal = {'contract_canonical': CONTRACT_SHA, 'contract_sha256': CONTRACT_SHA,
        'hosts': host_metadata,
        'files': all_files, 'project_files': {
            'paper/scripts/prepare_titantpp_history_rtx_control.py': sha(Path(__file__)),
            'paper/scripts/deploy_titantpp_history_rtx_handoff.py': sha(PROJECT / 'paper/scripts/deploy_titantpp_history_rtx_handoff.py'),
            'paper/tests/test_titantpp_history_rtx_control.py': sha(PROJECT / 'paper/tests/test_titantpp_history_rtx_control.py')},
        'ownership_release_and_permits_not_fabricated': True,
        'status': 'prepared_not_launched', 'remote_calls': 0}
    write_new(bundle / 'control/local_tools_integrity.json', encoded(local_seal))
    return {'contract_sha256': CONTRACT_SHA, 'hosts': host_metadata,
        'local_tools_integrity_raw_sha256': sha(bundle / 'control/local_tools_integrity.json'),
        'network_calls': 0, 'GPU_calls': 0, 'status': 'prepared_not_launched'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--self-test', action='store_true')
    group.add_argument('--reseal-prelaunch', action='store_true')
    parser.add_argument('--bundle', default=str(BUNDLE))
    args = parser.parse_args()
    if args.self_test:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(PROJECT / 'paper/tests'),
            pattern='test_titantpp_history_rtx_control.py')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    result = reseal_prelaunch(args) if args.reseal_prelaunch else prepare(args)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
