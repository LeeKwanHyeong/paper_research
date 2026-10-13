"""Prepare only Intermittent seed62's four fresh fits in an isolated5080 root.

Science131, inputs, trainer, native qualifier and phase wrapper are byte copies.
Only campaign validate/authorization bodies change, plus queue-proof validators.
No remote operation, qualification or training is performed by preparation.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from paper.scripts import titantpp_components_seed62_handoff as handoff

PROJECT = handoff.PROJECT
PARENT = PROJECT / 'search_artifacts' / handoff.PARENT_SCHEMA
TOOLS = ('paper/scripts/prepare_titantpp_components_seed62_handoff.py',
         'paper/scripts/titantpp_components_seed62_handoff.py',
         'paper/tests/test_titantpp_components_seed62_handoff.py')


def functions(text):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def queue_validator_source():
    text = (PROJECT / TOOLS[1]).read_text(); tree = ast.parse(text)
    names = {'require', 'canonical', 'sha', 'read', 'transferred_jobs', 'descriptor',
             'check_parent_contract', 'validate_contract', 'verify_running_parent', 'validate_queue_proof'}
    constants = {'SCHEMA', 'PARENT_SCHEMA', 'PARENT_SHA', 'SOURCE_SHA', 'OP_SHA', 'ROOT_PARENT',
                 'PARENT_ROOT', 'DESTINATION_ROOT', 'ARMS', 'DATASET', 'STARTED', 'DEADLINE', 'SCIENCE_FIELDS'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
             or isinstance(n, ast.Assign) and any(isinstance(v, ast.Name) and v.id in constants for v in n.targets)]
    require = handoff.require
    require({n.name for n in nodes if isinstance(n, ast.FunctionDef)} == names, 'Queue validator extraction incomplete')
    header = 'import base64, hashlib, json\nfrom pathlib import Path\nfrom copy import deepcopy\n'
    return header + '\n'.join(ast.unparse(n) for n in nodes) + '\n'


def adapter_source(original):
    tree = ast.parse(original)
    old = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    handoff.require('validate' in old and 'authorization' in old, 'Campaign functions missing')
    validate = '''def validate(c, *, source_root=None, operation_root=None, verify_files=True):
    namespace = {}
    exec(_HANDOFF_QUEUE_CHECK_SOURCE, namespace)
    root = Path(c["hosts"]["5080"]["root"])
    parent = read(root / "ownership/parent_execution_contract.json")
    namespace["validate_contract"](c, parent)
    if verify_files:
        sr = Path(source_root or ROOT)
        op = Path(operation_root or Path(__file__).resolve().parent)
        require(os.environ.get("SOURCE_REVISION") == FROZEN_REVISION, "Source environment drift")
        for name, digest in {**c["source"]["files"], **c["input_files"]}.items():
            require(sha_file(safe_relative(sr, name)) == digest, "Source/input integrity drift: " + name)
        for name, digest in c["operation"]["files"].items():
            require(sha_file(safe_relative(op.parent, name)) == digest, "Operation integrity drift: " + name)
    return c
'''
    authorization = deepcopy(old['authorization'])
    authorization.body.insert(1, ast.parse('_seed62_validate_queue_exclusion(c, root)').body[0])
    replacements = {'validate': validate, 'authorization': ast.unparse(authorization) + '\n'}
    lines = original.splitlines(keepends=True)
    for name in sorted(replacements, key=lambda k: old[k].lineno, reverse=True):
        node = old[name]; lines[node.lineno - 1:node.end_lineno] = [replacements[name]]
    result = ''.join(lines)
    # Appended definitions are inactive at import. Original dispatch/fit/etc stay exact ASTs.
    extension = '\n_HANDOFF_QUEUE_CHECK_SOURCE = ' + repr(queue_validator_source()) + '\n'
    extension += '''
def _seed62_validate_queue_exclusion(c, root):
    namespace = {}
    exec(_HANDOFF_QUEUE_CHECK_SOURCE, namespace)
    parent = read(root / "ownership/parent_execution_contract.json")
    source = root / "ownership/queue_exclusion_source.json"
    proof = read(source)
    namespace["validate_queue_proof"](proof, c, parent)
    approval = read(root / "approval.json")
    require(approval.get("queue_exclusion_source_raw_sha256") == sha_file(source),
            "Native approval queue exclusion rawSHA changed")
'''
    parsed = ast.parse(result)
    guard = parsed.body[-1]
    handoff.require(isinstance(guard, ast.If) and isinstance(guard.test, ast.Compare)
                    and ast.unparse(guard.test) == "__name__ == '__main__'", 'Expected final main guard missing')
    result_lines = result.splitlines(keepends=True)
    result_lines[guard.lineno - 1:guard.lineno - 1] = [extension + '\n']
    result = ''.join(result_lines)
    before, after = functions(original), functions(result)
    handoff.require(all(after[name] == fingerprint for name, fingerprint in before.items()
                        if name not in ('validate', 'authorization')), 'Scientific dispatch/fit/qualification function changed')
    handoff.require(set(after) - set(before) == {'_seed62_validate_queue_exclusion'}, 'Unexpected adapter functions')
    return result


def make_contract(parent, operations, helper_sha):
    handoff.check_parent_contract(parent)
    c = deepcopy(parent); c['schema'] = handoff.SCHEMA
    c['assignments'] = {'5080': [[handoff.DATASET, 62]], '5090': []}
    c['jobs'] = handoff.transferred_jobs('5080')
    c['seed62_handoff'] = handoff.descriptor(parent, helper_sha)
    c['operational_replacement'] = {'parent_contract_sha256': handoff.PARENT_SHA,
        'host_leases': {'5080': {'started_at_unix': handoff.STARTED, 'deadline_unix': handoff.DEADLINE}},
        'parent_originals_preserved': True, 'parent42_and52_natural_progress_preserved': True}
    for host, spec in c['hosts'].items():
        root = str(handoff.ROOT_PARENT / (handoff.SCHEMA + '_' + host))
        spec.update(root=root, source_root=root + '/source', operation_root=root + '/operation',
                    output_dir=root + '/run', tmux=handoff.SCHEMA + '_' + host,
                    assigned_datasets=[handoff.DATASET] if host == '5080' else [])
        spec['environment'].update(PYTHONPATH=root + '/source', MPLCONFIGDIR=root + '/cache/matplotlib', XDG_CACHE_HOME=root + '/cache')
    c['operation'] = {'files': operations, 'files_sha256': handoff.canonical(operations)}
    handoff.validate_contract(c, parent)
    return c


def prepare(destination):
    from paper.scripts import launch_titantpp_cnn_gru_components_v5 as original
    handoff.require(handoff.sha(PARENT / 'local_tools_integrity.json') == handoff.LOCAL40_SHA, 'Original local40 raw seal changed')
    parent, checked = original.check_bundle(PARENT)
    handoff.require(checked['contract_sha256'] == handoff.PARENT_SHA, 'Original parent pointer changed')
    destination = Path(destination).resolve()
    handoff.require(destination == PROJECT / 'search_artifacts' / handoff.SCHEMA, 'Unexpected new local bundle')
    handoff.require(not destination.exists(), 'Existing handoff preparation preserved; no retry')
    campaign = adapter_source((PARENT / 'operation/campaign.py').read_text())
    operations = deepcopy(parent['operation']['files'])
    operations['operation/campaign.py'] = hashlib.sha256(campaign.encode()).hexdigest()
    c = make_contract(parent, operations, handoff.sha(PROJECT / TOOLS[1]))
    destination.mkdir(parents=True, exist_ok=False)
    for name, digest in {**c['source']['files'], **c['input_files']}.items():
        target = handoff.safe(destination / 'source', name); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(handoff.safe(PARENT / 'source', name), target)
        handoff.require(handoff.sha(target) == digest, 'Science/input copy mismatch')
    for name, digest in operations.items():
        target = handoff.safe(destination, name); target.parent.mkdir(parents=True, exist_ok=True)
        if name == 'operation/campaign.py': target.write_text(campaign)
        else: shutil.copyfile(handoff.safe(PARENT, name), target)
        handoff.require(handoff.sha(target) == digest, 'Adapter copy mismatch')
    parent_path = destination / 'ownership/parent_execution_contract.json'
    parent_path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(PARENT / 'execution_contract.json', parent_path)
    handoff.write_exclusive(destination / 'execution_contract.json', c)
    handoff.write_exclusive(destination / 'current.json', {'schema': handoff.SCHEMA, 'contract_sha256': handoff.canonical(c),
        'bundle': str(destination), 'status': 'prepared_seed62_queue_exclusion'})
    handoff.write_exclusive(destination / 'launch_hold.json', {'hold': True, 'contract_sha256': handoff.canonical(c),
        'reason': 'Explicit queue exclusion proof, launch authorization and own native qualification required'})
    tools = deepcopy(handoff.read(PARENT / 'local_tools_integrity.json')['files'])
    tools.update({name: handoff.sha(PROJECT / name) for name in TOOLS})
    handoff.write_exclusive(destination / 'local_tools_integrity.json', {'files': tools, 'files_sha256': handoff.canonical(tools)})
    handoff.write_exclusive(destination / 'preparation_pending.json', {'status': 'awaiting_queue_exclusion_proof',
        'contract_sha256': handoff.canonical(c), 'contract_raw_sha256': handoff.sha(destination / 'execution_contract.json'),
        'parent_contract_raw_sha256': handoff.sha(parent_path),
        'local_tools_integrity_raw_sha256': handoff.sha(destination / 'local_tools_integrity.json'),
        'remote_calls': 0, 'scientific_fits': 0, 'native_qualification': 'not_run', 'lease_extended': False})
    handoff.check_bundle(destination, require_queue_exclusion=False)
    return c


def bind_queue_exclusion(bundle, source):
    c, _ = handoff.check_bundle(bundle, require_queue_exclusion=False)
    bundle = Path(bundle); source = Path(source)
    parent = handoff.read(bundle / 'ownership/parent_execution_contract.json')
    proof = handoff.read(source); handoff.validate_queue_proof(proof, c, parent)
    target = bundle / 'ownership/queue_exclusion_source.json'
    handoff.require(not target.exists() and not (bundle / 'deployment.tar.gz').exists()
                    and not (bundle / 'preparation_receipt.json').exists(), 'Existing proof/payload preserved; no rebind')
    with target.open('xb') as stream: stream.write(source.read_bytes())
    receipt = {**handoff.read(bundle / 'preparation_pending.json'), 'status': 'prepared_on_hold_with_queue_exclusion',
        'queue_exclusion_source_raw_sha256': handoff.sha(target), 'queue_exclusion_verified': True,
        'parent_full_owner_released': False, 'source_files': 131, 'input_files': 7, 'operation_files': 4}
    members = handoff.expected_members(c, receipt)
    with tarfile.open(bundle / 'deployment.tar.gz', 'x:gz') as archive:
        for name in members: archive.add(handoff.safe(bundle, name), arcname=name, recursive=False)
    receipt['payload_sha256'] = handoff.sha(bundle / 'deployment.tar.gz')
    handoff.write_exclusive(bundle / 'preparation_receipt.json', receipt)
    handoff.check_bundle(bundle)
    return receipt


def main():
    p = argparse.ArgumentParser(); p.add_argument('--destination', type=Path, default=PROJECT / 'search_artifacts' / handoff.SCHEMA)
    p.add_argument('--queue-proof', type=Path)
    a = p.parse_args()
    result = bind_queue_exclusion(a.destination, a.queue_proof) if a.queue_proof else prepare(a.destination)
    print(json.dumps({'contract_sha256': handoff.canonical(result) if not a.queue_proof else result['contract_sha256'],
        'remote_calls': 0, 'scientific_fits': 0, 'lease_extended': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
