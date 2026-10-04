"""Bounded validation replications on 5090; immutable seed42 predecessors stay intact."""
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tarfile

from paper.scripts import local_detail_benchmark_contract as original
from paper.scripts.observed_slot_parallel_common import require, sha_file, sha_json, source_manifest, write_json

ROOT = original.ROOT
SCHEMA = 'local_detail_instacart_replication_v1'
BASE = 'paper/contracts/local_detail_benchmark_observed_time_v3.json'
BASE_SHA = '6220992f788973bc3adc12224060fe4fa8cc0e5ecd0ad0dcf2128abb9108e18a'
RUN_NAME = 'local_detail_replication_seed52_62_5090_20260924_v1'
ENTRYPOINTS = (*original.ENTRYPOINTS, 'paper/scripts/local_detail_instacart_replication_contract.py', 'paper/scripts/local_detail_instacart_queue.py')
EXTRA = (*original.EXTRA, BASE, 'paper/contracts/local_detail_benchmark_observed_time_v4.json', 'paper/contracts/local_detail_replication_seed42_lineage_v1.json')
read = original.read


def protocol():
    parent = read(ROOT / BASE)
    require(sha_json(parent) == BASE_SHA, 'Frozen seed42 comparison changed')
    c = {k: deepcopy(parent[k]) for k in ('model_role','arms','datasets','architecture','design_sha256','comparison','limits','policy')}
    c.update(schema=SCHEMA, approved=False, approval_required=True, seed=None,
             training={**deepcopy(parent['training']), 'seed':None}, qualification_seed=52,
             total_arms=12, endpoint_replays=24, parent_contract_sha256=BASE_SHA)
    c['hosts'] = {'5090':deepcopy(parent['hosts']['5090'])}
    spec = c['hosts']['5090'];spec['root'] = str(Path(spec['root']).parent/RUN_NAME)
    spec.update(source_root=spec['root']+'/source', output_dir=spec['root']+'/run',
                tmux=RUN_NAME, assigned_datasets=['insta_market_basket'])
    c['process_environment'] = {'5090':{**parent['process_environment']['5090'],'XDG_CACHE_HOME':spec['root']+'/cache'}}
    c['library_sha256'] = {'5090':deepcopy(parent['library_sha256']['5090'])}
    c['policy'].update(additional_seeds=[52,62], other_hosts='preserve_running_5080_and_seed42_5090_without_mutation')
    c['limits']['max_gpu_hours_aggregate'] = 240
    c['replication'] = {
        'seeds':[52,62], 'report_seeds':[42,52,62],
        'jobs':[{'host':'5090','dataset':ds,'seed':seed} for ds in spec['assigned_datasets'] for seed in (52,62)],
        'dependency':{'root':parent['hosts']['5090']['root'], 'tmux':parent['hosts']['5090']['tmux'], 'contract_sha256':BASE_SHA, 'source_files_sha256':parent['source']['files_sha256'], 'dataset':'insta_market_basket', 'seed':42, 'required_arms':6, 'required_replays':12}, 'reused_seed42_arms':6, 'reused_seed42_replays':12,
        'aggregation':'report all three seeds, mean and sample SD, paired per-seed changes; never keep only favorable seeds',
        'interpretation':'Quantity-Aware quantity accuracy is primary; report time NLL, failed gates and strata without suppressing regressions',
        'instacart':'start seed52 then seed62 only after seed42 six-model completion, endpoint audit and owned-process exit; same5090 environment',
        'structure':'B and ungated local-detail candidate fixed; no new architecture or head/loss tuning',
    }
    c['maximum_optimizer_steps'] = sum(d['expected_global_steps'] for d in c['datasets']
        if d['dataset_id'] in spec['assigned_datasets'])*len(c['arms'])*2
    c['comparison']['scope'] = 'seed52/62 validation replications; aggregate with frozen seed42; held-out remains locked'
    c['launch'] = {
        'allowed_execution_hosts':['5090'], 'fixed_started_at_unix':parent['launch']['fixed_started_at_unix'],
        'fixed_deadline_unix':parent['launch']['fixed_deadline_unix'],
        'clock_origin':'retain original benchmark common clock and absolute deadline; no extension',
        'root_must_be_new':True, 'requires_both_native_qualifications':False,
        'qualification_scope':'5090 synthetic seed52 qualification before any seed52/62 training',
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


def dependency_status(c, host):
    """One metadata observation. Completion alone does not authorize dispatch."""
    require(host == '5090', 'Replication host differs')
    old = c['replication']['dependency']; root = Path(old['root'])
    prior = read(root/'frozen_execution/execution_contract.json')
    require(sha_json(prior) == old['contract_sha256'], 'Seed42 predecessor contract changed')
    require(prior['source']['files_sha256'] == old['source_files_sha256'], 'Predecessor source identity changed')
    run, dispatch = read(root/'run/status.json'), read(root/'dispatch_status.json')
    states = (run['status'], dispatch['status'])
    require(all(x in ('training','complete') for x in states), 'Seed42 predecessor failed or unexpected state')
    return {'status':'ready_for_audit' if states == ('complete','complete') else 'waiting_for_seed42',
            'run':run, 'dispatch':dispatch}


def audit_completed_dataset(root, prior, dataset, host):
    """Validate small JSON evidence only; no data or checkpoint loading."""
    import math
    from paper.scripts import run_local_detail_benchmark as r
    root = Path(root); dest = root/'run'/dataset
    data = next(d for d in prior['datasets'] if d['dataset_id'] == dataset)
    paired, part = read(dest/'paired_comparison.json'), read(root/'run/partition_summary.json')
    require(paired['status'] == part['status'] == 'complete' and paired['host'] == part['host'] == host,
            'Predecessor partition incomplete')
    require(paired['evaluation_scope'] == 'validation_only' and paired['held_out_test_evaluated'] is False
            and part['held_out_test_evaluated'] is False, 'Predecessor evaluation scope differs')
    require(set(paired['arms']) == set(prior['arms']) and set(part['datasets']) == {dataset}
            and part['datasets'][dataset] == paired and part['contract_sha256'] == sha_json(prior)
            and part['deadline_unix'] == prior['launch']['fixed_deadline_unix'], 'Predecessor partition identity differs')
    manifest = read(root/'run/wrapper_manifest.json')
    require(manifest['authority']['contract_sha256'] == sha_json(prior) and manifest['authority']['host'] == host
            and manifest['resume'] is False and manifest['retry'] is False, 'Predecessor wrapper identity differs')
    initial = read(dest/'initialization.json'); exposures={}; total=0; files={}
    for arm in prior['arms']:
        run=dest/'runs'/arm/r.VARIANT/'seed_42'
        summary=read(run/'summary.json'); history=read(run/'history.json')['history']; exposure=read(run/'exposure.json')
        steps=r.audit_arm(history,summary,exposure,data,prior['training']); endpoints=read(run/'endpoint_replays.json')
        require(summary['seed'] == 42 and summary['backbone'] == arm, 'Predecessor arm/seed differs')
        require(endpoints == paired['arms'][arm] and endpoints['global_steps'] == steps
                and endpoints['completed_epochs'] == len(history) and endpoints['best_epoch'] == summary['best_epoch']
                and endpoints['initial_state_sha256'] == summary['initial_state_sha256'] == initial[arm], 'Predecessor endpoint metadata differs')
        for label,epoch in (('selected',summary['best_epoch']),('last',len(history))):
            ep=endpoints[label]; r.audit_replay_accounting(ep)
            require(ep['count'] == data['inherited_data_identity']['populations']['validation']['target_count'], 'Replay population differs')
            require(ep['evaluation_scope'] == 'validation_only' and ep['held_out_test_evaluated'] is False and ep['time_metric'] == r.TIME_METRIC, 'Replay split/time metric differs')
            expected_path=summary['checkpoint_path'] if label == 'selected' else str(Path(summary['checkpoint_path']).parent/'last_epoch_state.pt')
            require(ep['checkpoint_path'] == expected_path, 'Replay checkpoint path differs')
            for metric,key in (('qty_rmse','val_qty_rmse'),('qty_mae','val_qty_mae'),('time_nll','val_time_nll')):
                require(math.isclose(ep[metric],history[epoch-1][key],rel_tol=1e-10,abs_tol=1e-8), 'Replay/history differs')
        exposures[arm]=exposure; total+=steps
        for name in ('summary.json','history.json','exposure.json','endpoint_replays.json','epoch_timing.json'):
            path=run/name; files[str(path.relative_to(root))]=sha_file(path)
    require(r.audit_batch_prefixes(exposures) and paired['batch_prefix_equal'] is True, 'Predecessor batch prefix differs')
    require(part['total_optimizer_steps'] == total and part['endpoint_replays'] == 12, 'Predecessor totals differ')
    for path in (dest/'paired_comparison.json',dest/'initialization.json',dest/'input_receipt.json',root/'run/partition_summary.json',root/'run/wrapper_manifest.json'):
        files[str(path.relative_to(root))]=sha_file(path)
    return {'status':'passed','arms':6,'endpoint_replays':12,'optimizer_steps':total,'files':files,'checkpoint_loaded':False}


def verify_predecessors(c, host):
    require(dependency_status(c,host)['status'] == 'ready_for_audit', 'Seed42 predecessor still running')
    old=c['replication']['dependency']; root=Path(old['root']); prior=read(root/'frozen_execution/execution_contract.json')
    require({n:sha_file(root/'source'/n) for n in prior['source']['files']} == prior['source']['files'], 'Predecessor source changed')
    require(subprocess.run([c['hosts'][host]['tmux_binary'],'has-session','-t',old['tmux']],capture_output=True).returncode != 0,
            'Seed42 predecessor tmux remains active')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:command=path.read_bytes()
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        require(str(root).encode() not in command, 'Seed42 predecessor owned process remains active')
    return audit_completed_dataset(root,prior,old['dataset'],host)


def freeze(contract_path, output, verification_path, cost_path):
    require(not contract_path.exists() and not output.exists(),'Fresh contract and bundle paths required')
    files=source_manifest(ROOT,entrypoints=ENTRYPOINTS,extra_files=EXTRA)
    verification,cost=read(verification_path),read(cost_path)
    require(verification['tested_source_files_sha256']==sha_json(files) and verification['passed']>0,'Verification predates source')
    c={**protocol(),'status':'frozen_for_user_requested_5090_replication', 'verification':verification,'cost_estimate':cost,
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
             'bundle_sha256':sha_file(output/'source_bundle.tar.gz'),'new_training_arms':12,'new_replays':24,'gpu_executed':False}
    write_json(output/'preparation_receipt.json',receipt,exclusive=True)
    return receipt
