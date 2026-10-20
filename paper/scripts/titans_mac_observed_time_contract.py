"""Fail-closed identity and authority for the nine-fit MAC comparison."""
from pathlib import Path
import math
import time
from paper.scripts import observed_slot_parallel_common as common

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'titans_mac_observed_time_execution_v1'
ROLE = 'observed_time_titans_mac_clip1_v1'
ARM = 'titantpp_titans_mac'
DESIGN_SHA = 'd59d51c219a2a300a1d295615f578c28239a490a9f50f4aed3c492acafee0b12'
DESIGN_PATH = 'paper/contracts/titans_mac_observed_time_comparison_v1.json'
require = common.require


def read(path):
    import json
    return json.loads(Path(path).read_text())


def limits_from_design(d):
    p = d['limits_proposed_not_approved']
    return dict(total_wall_seconds=264*3600, max_gpu_hours_aggregate=480,
        host_wall_seconds={h: n*3600 for h,n in p['host_wall_hours'].items()},
        per_dataset_arm_wall_seconds={'yellow_trip_hourly':8*3600,
            'intermittent_frozen_5000':64*3600, 'insta_market_basket':84*3600},
        qualification_seconds_per_host=1800, max_concurrent_gpu_jobs_per_host=1,
        max_file_bytes=p['file_bytes_max'], per_host_output_bytes=p['output_bytes_per_host'],
        min_free_bytes=p['minimum_disk_free_bytes'], minimum_free_vram_mib=12000)


def validate_execution_contract(c, verify_source=True):
    d = read(ROOT/DESIGN_PATH)
    require(common.sha_json(d)==DESIGN_SHA and c['design_sha256']==DESIGN_SHA, 'Design changed')
    require(c['schema']==SCHEMA and c['model_role']==ROLE and c['execution_arms']==[ARM], 'Wrong MAC role')
    require(c['approved'] is False and c['status']=='frozen_awaiting_separate_gpu_approval', 'Approval must be separate')
    for key in ('hosts','jobs','training','architecture','analysis','failure_policy','state_and_causality'):
        require(c[key]==d[key], 'Design scope changed: '+key)
    require(c['new_fits']==9 and c['endpoint_replays']==18 and c['limits']==limits_from_design(d), 'Scope/budget changed')
    require(c['execution_isolation']=={'no_progress_timeout_seconds':1800,'stack_seconds':300,
        'fresh_process_per_fit':True,'automatic_retry':False,'automatic_resume':False}, 'Isolation changed')
    require(c['evaluation_scope']=='validation_only' and c['held_out_locked'] is True, 'Split changed')
    require(len(c['datasets'])==3, 'Dataset count changed')
    for actual,planned in zip(c['datasets'],d['datasets']):
        expected={**planned,'epochs':300,'expected_global_steps':300*planned['steps_per_epoch']}
        require(actual==expected, 'Data/head/optimizer changed')
    require(len(c['reuse'])==18 and {(r['dataset'],r['seed'],r['arm']) for r in c['reuse']}==
        {(x['dataset_id'],s,a) for x in d['datasets'] for s in (42,52,62) for a in ('titantpp','titantpp_local_detail')},
        'Reused baseline coverage changed')
    require(common.sha_json(c['reuse'])==c['reuse_sha256'], 'Baseline evidence changed')
    require(c['source']['files_sha256']==common.sha_json(c['source']['files']), 'Invalid source closure')
    require(c['source']['parent_files_sha256']==d['parent']['source_closure_sha256'], 'Wrong source parent')
    if verify_source:
        for rel,digest in c['source']['files'].items():
            path=ROOT/rel
            require(path.resolve().is_relative_to(ROOT) and not path.is_symlink() and
                    common.sha_file(path)==digest, 'Source drift: '+rel)
    return c


def validate_permit(c, approval, permit, host, now=None):
    now=time.time() if now is None else now
    require(host in ('5080','5090') and approval.get('approved') is True and
        approval.get('hosts')==['5080','5090'] and bool(approval.get('user_instruction')) and
        approval.get('contract_sha256')==common.sha_json(c), 'Explicit new GPU approval required')
    require(approval.get('core_results_reviewed') is True and
        isinstance(approval.get('core_terminal_audit_sha256'),str) and
        len(approval['core_terminal_audit_sha256'])==64, 'Core terminal review required')
    require(permit.get('schema')=='titans_mac_start_permit_v1' and permit.get('host')==host and
        permit.get('contract_sha256')==common.sha_json(c) and
        permit.get('approval_sha256')==common.sha_json(approval), 'Start permit identity mismatch')
    keys=('campaign_started_at_unix','campaign_deadline_unix','started_at_unix','deadline_unix')
    require(all(type(permit.get(k)) in (int,float) and math.isfinite(permit[k]) for k in keys), 'Invalid clock')
    t0,global_end,start,end=(permit[k] for k in keys)
    require(global_end==t0+c['limits']['total_wall_seconds'] and t0<=start<=now<end and
        end==min(start+c['limits']['host_wall_seconds'][host],global_end), 'Expired or extended deadline')
    return c['hosts'][host]


def verify_core_terminal_review(c,approval,host):
    """Local collected evidence must precede the new GPU launch decision."""
    path=Path(c['hosts'][host]['root'])/'prior_core_terminal_audit.json'
    receipt=read(path)
    require(common.sha_file(path)==approval['core_terminal_audit_sha256'], 'Prior terminal audit bytes changed')
    require(receipt.get('campaign_contract_sha256')=='eff125f587f7a9a8097d5d43b3d688eac481abb78bb6006b5aa4ab1b0ec2a88d'
        and receipt.get('collected') is True and receipt.get('user_results_reported') is True
        and set(receipt.get('hosts',{}))=={'5080','5090'}, 'Core terminal evidence incomplete')
    for row in receipt['hosts'].values():
        require(row.get('terminal') is True and row.get('owned_processes_exited') is True and
            isinstance(row.get('terminal_evidence_sha256'),str) and len(row['terminal_evidence_sha256'])==64,
            'Core owned processes/terminal evidence unresolved')


def verify_qualification(c,approval,permit,host,q):
    require(q.get('status')=='passed' and q.get('device')=='cuda:0' and q.get('host')==host and
        q.get('contract_sha256')==common.sha_json(c) and q.get('approval_sha256')==common.sha_json(approval) and
        q.get('permit_sha256')==common.sha_json(permit) and q.get('source_files_sha256')==c['source']['files_sha256'],
        'Native qualification identity mismatch')
    require(q.get('real_data_loaded') is False and q.get('held_out_evaluated') is False and
        q.get('synthetic_optimizer_updates')==60, 'Native workload/scope mismatch')
    require(q.get('checks')==dict.fromkeys(('finite_updates','causality','padding','fresh_state','restore','common_head'),True),
        'Native correctness evidence incomplete')
    spec=c['hosts'][host]; runtime=q['runtime']
    require(runtime['gpu']['uuid']==spec['gpu_uuid'] and all(runtime.get(k)==v for k,v in spec['runtime_expected'].items()),
        'Native runtime mismatch')
    require(q.get('process_environment')==c['process_environment'][host] and q.get('library_sha256')==c['library_sha256'][host],
        'Native environment mismatch')
    rows=q['costs']; require([r['dataset'] for r in rows]==spec['assigned_datasets'], 'Native dataset coverage mismatch')
    for row in rows:
        d=next(d for d in c['datasets'] if d['dataset_id']==row['dataset'])
        require(row['batch_size']==128 and row['length']==d['loader']['max_seq_len'] and
            row['updates']==60//len(rows) and len(row['step_seconds'])==row['updates'] and
            all(math.isfinite(t) and t>0 for t in row['step_seconds']) and
            0<row['peak_allocated_bytes']<=runtime['gpu']['total_memory_bytes'], 'Invalid native measurement')
    require(set(q['initialization'])==set(spec['assigned_datasets']) and
        all(set(q['initialization'][d])=={'42','52','62'} for d in spec['assigned_datasets']), 'Native initialization coverage')
