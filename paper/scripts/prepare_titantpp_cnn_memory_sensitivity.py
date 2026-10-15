"""Create a separate, Train/Validation-only diagnostic bundle; never remote."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

PROJECT = Path(__file__).resolve().parents[2]
NAME = 'titantpp_cnn_memory_sensitivity_20261015_v1'
ARM = 'titantpp_cnn_gru54_cnn_only'
SOURCE = '15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def prepare(root):
    root = root.resolve()
    name = root.name
    assert name in tuple(NAME.replace('_v1', '_v'+str(v)) for v in (1, 2, 3))
    root.mkdir(parents=True, exist_ok=False)
    files, operations, parents, jobs, datasets, runtimes = {}, {}, {}, [], {}, {}
    def transfer(src, relative, digest, manifest):
        assert sha(src) == digest, 'Frozen source file changed: ' + str(src)
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert sha(dest) == digest
        else:
            with src.open('rb') as inp, dest.open('xb') as out:
                shutil.copyfileobj(inp, out)
        assert sha(dest) == digest
        manifest[relative] = digest
    source = None
    for cohort in ('taxi_raf24', 'inter8', 'inter62'):
        parent = PROJECT / f'search_artifacts/titantpp_components_{cohort}_test_20261014_v1'
        c = json.loads((parent / 'contract.json').read_text())
        assert c['contract_sha256'] == canonical({k: v for k, v in c.items() if k != 'contract_sha256'})
        # Extract only frozen checkpoint/Train/Validation metadata, no Test receipt.
        parents[cohort] = dict(path=str(parent), contract_raw_sha256=sha(parent / 'contract.json'),
                               contract_canonical_sha256=c['contract_sha256'])
        src = c['sources']['components']
        assert canonical(src['files']) == src['closure'] == SOURCE
        if source is None:
            source = copy.deepcopy(src)
            for path, digest in src['files'].items():
                transfer(parent / src['root'] / path, 'source/' + path, digest, files)
        else:
            assert src == source
        helper = f'operation/helper_{cohort}.py'
        transfer(parent / 'operation/evaluate.py', helper,
                 c['evaluator_integrity']['files']['operation/evaluate.py'], operations)
        transfer(parent / 'source/sample_data/.frozen_root_sentinel',
                 'source/sample_data/.frozen_root_sentinel',
                 c['evaluator_integrity']['files']['source/sample_data/.frozen_root_sentinel'], operations)
        runtimes.update(c['runtimes'])
        for dataset, spec in c['datasets'].items():
            path = f'inputs/data/{dataset}.parquet'
            transfer(parent / spec['training_data_path'], path, spec['training_data_sha256'], files)
            value = dict(path=path, sha256=spec['training_data_sha256'], definition=spec['definition'])
            if dataset in datasets:
                assert datasets[dataset] == value
            datasets[dataset] = value
        for original in c['jobs']:
            if original['arm'] != ARM:
                continue
            old = copy.deepcopy(original)
            jid = old['id']
            dest = f'inputs/jobs/{jid}/selected.pt'
            transfer(parent / old['checkpoint']['relative_path'], dest, old['checkpoint']['sha256'], files)
            old['checkpoint']['relative_path'] = dest
            for key in ('checkpoint_metadata', 'selection_history', 'worker_exit', 'terminal_manifest'):
                original_path = original[key + '_path']
                dest = f'inputs/jobs/{jid}/{key}{Path(original_path).suffix}'
                transfer(parent / original_path, dest, original[key + '_sha256'], files)
                old[key + '_path'] = dest
            # Only these fields are consumed by original validate_payload and new inference.
            retained = ('cohort', 'id', 'arm', 'seed', 'initial_state_sha256', 'source_revision_history',
                'selected_epoch', 'selected_metric_value', 'checkpoint_state_sha256', 'checkpoint',
                'checkpoint_metadata_path', 'checkpoint_metadata_sha256', 'selection_history_path',
                'selection_history_sha256', 'worker_exit_path', 'worker_exit_sha256',
                'terminal_manifest_path', 'terminal_manifest_sha256', 'validation_reference')
            jobs.append(dict(id=jid, dataset=old['dataset'], seed=old['seed'], arm=ARM,
                host=old['evaluation_host'], helper=helper, original={k: old[k] for k in retained}))
    assert len(jobs) == len({j['id'] for j in jobs}) == 9
    for local, dest in [('evaluate_titantpp_cnn_memory_sensitivity.py', 'evaluate.py'),
                        ('titantpp_cnn_memory_control.py', 'control.py'),
                        ('titantpp_cnn_memory_dispatch.py', 'dispatch.py'),
                        ('evaluate_titantpp_frozen_rows.py', 'row_utils.py')]:
        p = PROJECT / 'paper/scripts' / local
        transfer(p, 'operation/' + dest, sha(p), operations)
    now = time.time()
    prior_execution = json.loads((PROJECT / 'search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1/execution_manual2/execution_contract.json').read_text())
    execution = {}
    for host in runtimes:
        previous = prior_execution['hosts'][host]
        native_root = f'/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/{name}_{host}'
        environment = dict(previous['environment'])
        environment.update(PYTHONPATH=native_root+'/source', MPLCONFIGDIR=native_root+'/cache/matplotlib',
                           XDG_CACHE_HOME=native_root+'/cache')
        execution[host] = dict(tmux_binary=previous['tmux_binary'], environment=environment,
                               tmux=f'cnn_memory_sensitivity_20261015_{name.rsplit("_",1)[1]}_{host}')
    c = dict(schema='titantpp_cnn_memory_sensitivity_v1', created_unix=now, deadline_unix=now+48*3600,
        approval=dict(user_instruction='이거 먼저 진행하자. 그리고 학습은 직전 실험과 동일하게 5080/5090에서 세개씩 병렬로 돌릴 수 있게 학습하는 방식으로',
            scope='fixed CNN Train/Validation memory intervention first; conditional subsequent controlled training',
            diagnostic_authorized=True, automatic_retry=False, no_existing_artifact_overwrite=True),
        allowed_splits=['train', 'validation'], new_training=False, frozen_selection=True,
        alphas=[1., .5, .25, 0.], max_parallel_per_host=3, source=source,
        datasets=datasets, jobs=sorted(jobs, key=lambda j:(j['host'],j['dataset'],j['seed'])),
        runtimes=runtimes, execution=execution, parents=parents, operation_files=operations, input_files=files,
        remote_roots={h:f'/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/{name}_{h}' for h in runtimes},
        analysis=dict(primary_metric='raw_quantity_rmse', secondary='raw_quantity_mae',
            aggregation='three seed mean and sample SD; paired seed and event differences',
            features='observed quantities and encoded/retrieval telemetry only; no future target features',
            strata='Train feature quartiles, frozen before inspecting Validation attenuation outcomes',
            fixed_attenuation_support='At least one alpha<1 lowers dataset mean Validation RMSE by >=1% and improves >=2/3 seeds; all six metrics and all datasets retained',
            adaptive_support='Additionally an observed-feature stratum has attenuation benefit and another has harm in >=2/3 Validation seeds using Train quartile cuts; no target-defined gate',
            stop='No fixed support: do not launch gate training. Fixed support alone: report constant attenuation evidence, do not claim adaptive need.',
            inferential_limit='Post-Test exploratory diagnostic; inference intervention is not newly trained model performance'))
    if name != NAME:
        previous_name = NAME.replace('_v1', '_v2') if name.endswith('_v3') else NAME
        prior = PROJECT/'search_artifacts'/previous_name/'contract.json'
        old = json.loads(prior.read_text())
        c['deadline_unix'] = old['deadline_unix']
        c['operational_repair'] = dict(prior_contract_raw_sha256=sha(prior),
            prior_bundle=previous_name,
            reason=('Push down Train/Validation split filter before materializing historical mixed input' if name.endswith('_v3')
                    else 'Missing empty sample_data root sentinel blocked source import before data/model inference'),
            original_roots_and_failures_preserved=True, same_science_selection_inputs=True)
    with (root / 'contract.json').open('x') as stream:
        json.dump(c, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(root=str(root), contract_sha256=sha(root/'contract.json'), jobs=len(jobs),
                         files=len(files), operations=len(operations)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=PROJECT / 'search_artifacts' / NAME)
    prepare(parser.parse_args().root)
