"""Bounded validation replications on 5080; immutable seed42 predecessors stay intact."""
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tarfile

from paper.scripts import local_detail_benchmark_contract as original
from paper.scripts.observed_slot_parallel_common import require, sha_file, sha_json, source_manifest, write_json

ROOT = original.ROOT
SCHEMA = 'local_detail_benchmark_replication_v1'
BASE = 'paper/contracts/local_detail_benchmark_observed_time_v4.json'
BASE_SHA = '45c8b03a23a2c2aa5f0889708c2a0f68d7f958636b5e8e5ebd12c0b327180da5'
LINEAGE = 'paper/contracts/local_detail_replication_seed42_lineage_v1.json'
RUN_NAME = 'local_detail_replication_seed52_62_5080_20260924_v1'
ENTRYPOINTS = (*original.ENTRYPOINTS, 'paper/scripts/local_detail_replication_contract.py')
EXTRA = (*original.EXTRA, BASE, LINEAGE)
read = original.read


def protocol():
    parent = read(ROOT / BASE)
    require(sha_json(parent) == BASE_SHA, 'Frozen seed42 comparison changed')
    c = {k: deepcopy(parent[k]) for k in ('model_role','arms','datasets','architecture','design_sha256','comparison','limits','policy')}
    c.update(schema=SCHEMA, approved=False, approval_required=True, seed=None,
             training={**deepcopy(parent['training']), 'seed':None}, qualification_seed=52,
             total_arms=24, endpoint_replays=48, parent_contract_sha256=BASE_SHA)
    c['hosts'] = {'5080':deepcopy(parent['hosts']['5080'])}
    spec = c['hosts']['5080'];spec['root'] = str(Path(spec['root']).parent/RUN_NAME)
    spec.update(source_root=spec['root']+'/source', output_dir=spec['root']+'/run',
                tmux=RUN_NAME, assigned_datasets=['yellow_trip_hourly','intermittent_frozen_5000'])
    c['process_environment'] = {'5080':{**parent['process_environment']['5080'],'XDG_CACHE_HOME':spec['root']+'/cache'}}
    c['library_sha256'] = {'5080':deepcopy(parent['library_sha256']['5080'])}
    c['policy'].update(additional_seeds=[52,62], other_hosts='preserve_running_5090_without_mutation')
    c['limits']['max_gpu_hours_aggregate'] = 240
    c['replication'] = {
        'seeds':[52,62], 'report_seeds':[42,52,62],
        'jobs':[{'host':'5080','dataset':ds,'seed':seed} for ds in spec['assigned_datasets'] for seed in (52,62)],
        'lineage':read(ROOT/LINEAGE), 'reused_seed42_arms':12, 'reused_seed42_replays':24,
        'aggregation':'report all three seeds, mean and sample SD, paired per-seed changes; never keep only favorable seeds',
        'interpretation':'Quantity-Aware quantity accuracy is primary; report time NLL, failed gates and strata without suppressing regressions',
        'instacart':'current seed42 benchmark continues on5090; additional seeds remain future work, not excluded from final assessment',
        'structure':'B and ungated local-detail candidate fixed; no new architecture or head/loss tuning',
    }
    c['maximum_optimizer_steps'] = sum(d['expected_global_steps'] for d in c['datasets']
        if d['dataset_id'] in spec['assigned_datasets'])*len(c['arms'])*2
    c['comparison']['scope'] = 'seed52/62 validation replications; aggregate with frozen seed42; held-out remains locked'
    c['launch'] = {
        'allowed_execution_hosts':['5080'], 'fixed_started_at_unix':parent['launch']['fixed_started_at_unix'],
        'fixed_deadline_unix':parent['launch']['fixed_deadline_unix'],
        'clock_origin':'retain original benchmark common clock and absolute deadline; no extension',
        'root_must_be_new':True, 'requires_both_native_qualifications':False,
        'qualification_scope':'5080 synthetic seed52 qualification before any seed52/62 training',
        'deadline_action':'stop only owned worker group, preserve all artifacts, no automatic retry/resume'}
    return c


def validate_contract(c, *, verify_source=True):
    for key,value in protocol().items():
        require(c.get(key)==value, 'Frozen replication protocol changed: '+key)
    source=c['source']
    require(len(source['base_git_revision'])==40 and sha_json(source['files'])==source['files_sha256'], 'Source closure inconsistent')
    if verify_source:
        require(source_manifest(ROOT,entrypoints=ENTRYPOINTS,extra_files=EXTRA)==source['files'], 'Frozen replication source changed')
    require(c['verification']['status']=='passed_synthetic_cpu' and c['verification']['failed']==0, 'CPU integration required')


def verify_predecessors(c, host):
    require(host=='5080','Replication host differs')
    for old in c['replication']['lineage']['datasets']:
        root=Path(old['root'])
        require(sha_json(read(root/'frozen_execution/execution_contract.json'))==old['contract_sha256'], 'Seed42 predecessor contract changed')
        require(read(root/'run/status.json')['status']==old['expected_run_status']
                and read(root/'dispatch_status.json')['status']==old['expected_dispatch_status'], 'Seed42 predecessor status differs')
        require(read(root/'run'/old['dataset']/'paired_comparison.json')['status']=='complete', 'Seed42 dataset incomplete')
        for name,digest in old['files'].items():
            require(sha_file(root/name)==digest,'Seed42 result changed: '+name)
        require(subprocess.run([c['hosts'][host]['tmux_binary'],'has-session','-t',old['tmux']],capture_output=True).returncode!=0,
                'Seed42 predecessor tmux remains active')
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:command=path.read_bytes()
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            require(str(root).encode() not in command,'Seed42 predecessor owned process remains active')


def freeze(contract_path, output, verification_path, cost_path):
    require(not contract_path.exists() and not output.exists(),'Fresh contract and bundle paths required')
    files=source_manifest(ROOT,entrypoints=ENTRYPOINTS,extra_files=EXTRA)
    verification,cost=read(verification_path),read(cost_path)
    require(verification['tested_source_files_sha256']==sha_json(files) and verification['passed']>0,'Verification predates source')
    c={**protocol(),'status':'frozen_for_user_requested_5080_replication', 'verification':verification,'cost_estimate':cost,
       'source':{'base_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 'working_tree':'isolated content-hashed closure; unrelated dirty worktree preserved','files':files,'files_sha256':sha_json(files)},
       'evidence_sha256':{'verification':sha_file(verification_path),'cost':sha_file(cost_path)}}
    validate_contract(c);output.mkdir(parents=True);write_json(contract_path,c,exclusive=True)
    with tarfile.open(output/'source_bundle.tar.gz','w:gz') as archive:
        entry=tarfile.TarInfo('source/sample_data');entry.type=tarfile.DIRTYPE;entry.mode=0o755;archive.addfile(entry)
        for name in files:
            raw=(ROOT/name).read_bytes();entry=tarfile.TarInfo('source/'+name);entry.size=len(raw);entry.mode=0o644;archive.addfile(entry,io.BytesIO(raw))
        raw=(json.dumps(c,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
        entry=tarfile.TarInfo('frozen_execution/execution_contract.json');entry.size=len(raw);archive.addfile(entry,io.BytesIO(raw))
    receipt={'status':'prepared_not_launched','contract_sha256':sha_json(c),'source_files_sha256':sha_json(files),
             'bundle_sha256':sha_file(output/'source_bundle.tar.gz'),'new_training_arms':24,'new_replays':48,'gpu_executed':False}
    write_json(output/'preparation_receipt.json',receipt,exclusive=True)
    return receipt
