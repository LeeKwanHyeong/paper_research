#!/usr/bin/env python3
"""One final, source-bound dual-timescale candidate on 5090, validation only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts.run_hard_lmm_backbone_candidate_campaign import (
    audit_job, evaluate_gate, gpu_preflight, job_command, read_json, require, sha256,
)

CONTRACT = ROOT / 'paper/contracts/hard_lmm_dual_timescale_v1.json'
TESTS = ['test_dual_timescale_memory.py', 'test_dual_timescale_training_route.py',
         'test_dual_timescale_campaign.py']
BACKBONE = 'titantpp_dual_timescale_memory'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_manifest(path, revision):
    manifest = read_json(path)
    require(manifest['source_revision'] == revision and len(revision) == 40, 'Source revision drift')
    require(manifest['host_role'] == '5090', 'Host binding drift')
    require(manifest['evaluation_scope'] == 'validation_only', 'Evaluation scope drift')
    require(manifest['contract_sha256'] == sha256(CONTRACT), 'Contract drift')
    for name, digest in manifest['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source path')
        require(sha256(ROOT / relative) == digest, f'Source/input hash drift: {name}')
    for folder in ('models', 'paper/scripts', 'data_loader', 'simple_lab_test', 'utils'):
        extras = {str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*.py')} - set(manifest['files'])
        require(not extras, f'Unregistered Python source: {extras}')
    c = read_json(CONTRACT)
    require(c['host_role'] == '5090' and c['candidate']['backbone'] == BACKBONE, 'Candidate drift')
    require(c['held_out_test_evaluated'] is False and c['additional_seeds_authorized'] is False, 'Scope drift')
    require(c['common_training']['seed'] == 42, 'Seed drift')
    for row in c['data_bindings']:
        for key, digest in [('data_path', 'data_sha256'), ('split_manifest_path', 'split_manifest_sha256')]:
            require(sha256(ROOT / row[key]) == row[digest], 'Dataset drift')
    b = c['B_validation_reference']
    require(sha256(ROOT / b['source']) == b['sha256'], 'B reference drift')
    return c


def checked_cuda_xml(path):
    cases = ET.parse(path).getroot().findall('.//testcase')
    require(cases and all(x.find('failure') is None and x.find('error') is None for x in cases), 'CUDA tests failed')
    cuda = [x for x in cases if 'cuda' in x.get('name', '').lower()]
    require(cuda and all(x.find('skipped') is None for x in cuda), 'CUDA contracts skipped or absent')
    return {'tests': len(cases), 'executed_cuda_cases': len(cuda), 'xml_sha256': sha256(path)}


def full_audit(output, c, dataset, revision, phase):
    import torch
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    result = audit_job(output, candidate=c['candidate'], dataset=dataset,
                       source_revision=revision, expected_epochs=phase['epochs'])
    folder = Path(result['summary']).parent
    summary = read_json(folder / 'summary.json')
    launch = read_json(output / 'launch_contract.json')
    history = read_json(folder / 'history.json')['history']
    require(set(launch['split_rows']) == {'train', 'validation'}, 'Test rows materialized')
    for key, expected in {'batch_size':128,'lr':0.001,'lambda_log_qty':1.0,'lambda_tail':0.0,
                          'grad_clip':1.0,'hidden_dim':64,'model_role':c['candidate']['model_role']}.items():
        require(launch.get(key) == expected, f'Training contract drift: {key}')
    early = launch['early_stopping']
    require(early['monitor'] == 'validation_raw_quantity_rmse' and early['min_epochs'] == phase['minimum_epochs']
            and early['patience'] == phase['patience'], 'Early stopping drift')
    require(phase['minimum_epochs'] <= len(history) <= phase['epochs'], 'Epoch completion drift')
    if len(history) < phase['epochs']:
        require(history[-1]['epoch'] - summary['best_epoch'] >= phase['patience'], 'Premature stop')
    for row in history:
        for key, value in row.items():
            if isinstance(value, float):
                require(math.isfinite(value), f'Nonfinite history: {key}')
    selected = torch.load(folder / 'best_val_qty_rmse_model.pt', map_location='cpu', weights_only=False)
    last = torch.load(folder / 'last_epoch_state.pt', map_location='cpu', weights_only=False)
    for payload in (selected, last):
        validate_checkpoint_route(payload, BACKBONE)
        require(payload['source_revision_history'] == [revision], 'Resume source drift')
        require(payload['held_out_test_evaluated'] is False, 'Checkpoint scope drift')
        require(payload['model_state_sha256'] == canonical_state_dict_sha256(payload['model_state_dict']), 'Checkpoint state hash drift')
    require(last['best_state_sha256'] == selected['model_state_sha256'] == summary['checkpoint_state_sha256'], 'Best/last state mismatch')
    require(last['optimizer_state_dict']['state'] and last['rng_state'], 'Missing resume state')
    require(last['checkpoint_monitor'] == 'validation_raw_quantity_rmse', 'Resume selector drift')
    result.update(checkpoint_state_sha256=selected['model_state_sha256'],
                  last_checkpoint_sha256=sha256(folder/'last_epoch_state.pt'),
                  elapsed_seconds=summary['elapsed_seconds'],
                  early_stopping=early, source_revision=revision)
    # The model route tests prove exact load/next-step restoration. Bind the
    # actual selected model's schema and learned memory gain here as well.
    gains = {name: float(value) for name, value in selected['model_state_dict'].items()
             if 'alpha' in name and value.numel() == 1}
    require(gains and all(math.isfinite(v) for v in gains.values()), 'Missing memory gain')
    result['selected_memory_gains'] = gains
    if phase['epochs'] == 300:
        require(any(v != 0 for v in gains.values()), 'No learned memory path in selected model')
    return result


def run(args):
    import platform
    import torch
    c = verify_manifest(args.manifest, args.source_revision)
    out = args.output_root.resolve()
    require(ROOT != out and ROOT not in out.parents and out not in ROOT.parents, 'Source/output must be disjoint')
    out.mkdir(parents=True, exist_ok=True)
    identity = {'contract_sha256':sha256(CONTRACT),'source_revision':args.source_revision,
                'manifest_sha256':sha256(args.manifest),'host_role':'5090','held_out_test_evaluated':False,
                'runtime':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
                           'host':platform.node(),'python_executable':str(Path(sys.executable).resolve())}}
    if (out/'identity.json').exists():
        require(read_json(out/'identity.json') == identity, 'Campaign identity drift')
    save(out/'identity.json', identity)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONHASHSEED='42',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT))
    state = {'status':'preflight','updated_at':now(),'jobs':{}, **identity}
    if (out/'status.json').exists():
        prior = read_json(out/'status.json')
        if prior['status'] in ('stopped_seed42_gate_failed','seed42_passed_pending_replicates'):
            return prior
        state['jobs'] = prior.get('jobs', {})
    lock = open('/tmp/paper_research_dual_timescale_gpu0.lock','a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Inherit the lock in long-lived children so parent death cannot allow duplicates.
    def launch(command, log):
        with open(log, 'a') as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, pass_fds=(lock.fileno(),))
    try:
        state['gpu_preflight'] = gpu_preflight('RTX 5090')
        state['status'] = 'cuda_contracts'
        save(out/'status.json',state)
        if not (out/'cuda_tests_receipt.json').exists():
            launch([sys.executable,'-s','-m','pytest','-q','-p','no:cacheprovider',
                    *[str(ROOT/'simple_lab_test/search/tests'/name) for name in TESTS],
                    '--junitxml='+str(out/'cuda_tests.xml')], out/'cuda_tests.log')
            save(out/'cuda_tests_receipt.json',checked_cuda_xml(out/'cuda_tests.xml'))
        else:
            require(checked_cuda_xml(out/'cuda_tests.xml') == read_json(out/'cuda_tests_receipt.json'), 'CUDA proof drift')
        if not (out/'cost.json').exists():
            launch([sys.executable,'-s',str(ROOT/'paper/scripts/profile_dual_timescale.py'),
                    '--output',str(out/'cost.json')], out/'cost.log')
        require(read_json(out/'cost.json')['status']=='passed','Cost gate failed')
        for phase_name in ('e1','seed42_screening'):
            phase = c['phases'][phase_name]
            for dataset in phase['datasets_in_order']:
                job = phase_name+'_'+dataset
                destination=out/'jobs'/job
                destination.mkdir(parents=True,exist_ok=True)
                state.update(status='running_'+phase_name,current_job=job,updated_at=now())
                save(out/'status.json',state)
                verify_manifest(args.manifest,args.source_revision)
                complete = (destination/'launch_contract.json').exists() and read_json(destination/'launch_contract.json').get('status')=='complete'
                if not complete:
                    launch(job_command(python=sys.executable,candidate=c['candidate'],dataset=dataset,
                                       output=destination,source_revision=args.source_revision,phase=phase,
                                       host_role='5090'),out/(job+'.log'))
                receipt=full_audit(destination,c,dataset,args.source_revision,phase)
                if phase_name=='seed42_screening':
                    receipt['gate']=evaluate_gate(receipt['metrics'],c['B_validation_reference']['datasets'][dataset])
                save(out/(job+'_audit.json'),receipt)
                state['jobs'][job]=receipt
                save(out/'status.json',state)
                if phase_name=='seed42_screening' and receipt['gate']['status']!='passed':
                    state.update(status='stopped_seed42_gate_failed',final_model='TitanTPP(B)',
                                 failed_dataset=dataset,updated_at=now())
                    save(out/'status.json',state)
                    return state
        state.update(status='seed42_passed_pending_replicates',updated_at=now())
        save(out/'status.json',state)
        return state
    except BaseException as error:
        state.update(status='execution_failed',error=repr(error),updated_at=now())
        save(out/'status.json',state)
        raise
    finally:
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--source-revision',required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    print(json.dumps(run(parser.parse_args()),ensure_ascii=False))
