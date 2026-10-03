"""Local-only construction/validation of a non-executable research proposal."""
import copy
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CORE = ROOT / 'search_artifacts/titantpp_core_ablation_20260928_v1/frozen_execution/execution_contract.json'
DEST = ROOT / 'paper/contracts/titans_mac_observed_time_comparison_v1.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    core = json.loads(CORE.read_text())
    expected_core = 'eff125f587f7a9a8097d5d43b3d688eac481abb78bb6006b5aa4ab1b0ec2a88d'
    assert canonical(core) == expected_core
    assert canonical(core['source']['files']) == 'e21c45b5f095d258e62998ce818b07bef38ff2cfff6bc408b3594bd1adba9265'
    assert all(sha(ROOT / f) == h for f, h in core['source']['files'].items())
    probe = json.loads((OUT / 'factory_probe.json').read_text())
    assert probe['status'] == 'passed' and len(probe['rows']) == 3
    assert '83 passed, 1 skipped' in (OUT / 'existing_cpu_tests.log').read_text()
    prior_csv = ROOT / 'reports/titans_mac_reuse_audit_20260928_v1/stable_mac_conditions.csv'
    historical = list(csv.DictReader(prior_csv.open()))
    # Use only the old observation from the proposed SAME host; never GPU-name ratios.
    mapping = [('5080', 'yellow_trip_hourly', 62, 8),
               ('5080', 'intermittent_frozen_5000', 62, 64),
               ('5090', 'insta_market_basket', 52, 84)]
    costs, jobs = [], []
    for host, dataset, oldseed, cap in mapping:
        r = next(r for r in historical if r['dataset'] == dataset and int(r['seed']) == oldseed)
        s = json.loads((ROOT / r['source']).read_text())
        h = json.loads((ROOT / r['source']).with_name('history.json').read_text())['history']
        assert len(h) == int(r['completed_epochs']) and s['elapsed_seconds'] == float(r['fit_seconds'])
        proxy = s['elapsed_seconds'] / len(h)
        costs.append({'host': host, 'dataset': dataset, 'historical_seed': oldseed,
                      'historical_completed_epochs': len(h), 'historical_fit_hours': s['elapsed_seconds'] / 3600,
                      'fit_seconds_divided_by_completed_epochs': proxy,
                      'three_seed_40_epoch_proxy_hours': proxy * 3 * 40 / 3600,
                      'three_seed_300_epoch_proxy_hours': proxy * 3 * 300 / 3600,
                      'proposed_per_fit_and_two_replays_cap_hours': cap,
                      'source': r['source'], 'summary_sha256': sha(ROOT / r['source']),
                      'history_sha256': sha((ROOT / r['source']).with_name('history.json'))})
        for seed in (42, 52, 62):
            jobs.append({'host': host, 'dataset': dataset, 'seed': seed,
                         'arm': 'titantpp_titans_mac', 'id': f'{dataset}__{seed}__titantpp_titans_mac',
                         'maximum_fit_and_endpoint_hours': cap})
    hosts = copy.deepcopy(core['hosts'])
    for host in hosts:
        spec = hosts[host]
        root = f'/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titans_mac_observed_time_{host}_20260928_v1'
        spec.update(root=root, source_root=root + '/source', output_dir=root + '/run',
                    tmux=f'titans_mac_observed_time_{host}_20260928_v1',
                    maximum_concurrent_workers=1, proposed_wall_cap_hours={'5080': 216, '5090': 264}[host],
                    live_reverified=False, runtime_reference='inherited core native receipt; verify again before launch')
        spec['environment'].update(CUDA_VISIBLE_DEVICES='0')
    datasets = [{k: copy.deepcopy(d[k]) for k in (
        'dataset_id', 'inherited_data_identity', 'loader', 'model', 'optimizer',
        'statistics', 'time_statistics', 'steps_per_epoch', 'quantity_boundaries_all_train_rows',
        'history_boundaries', 'additional_history_boundaries')} for d in core['datasets']]
    plan = {
        'schema': 'titans_mac_observed_time_comparison_design_v1',
        'status': 'reviewable_design_not_executable', 'approved': False,
        'scope_authorization': 'User authorized contract preparation and local evidence checks only.',
        'scientific_fit_count': 9, 'validation_endpoint_replay_count': 18,
        'evaluation_scope': 'validation_only', 'held_out_locked': True,
        'model_label': 'Titans-MAC adapted to count-aware events, inner-clip1, common observed-time head',
        'model_role_proposed': 'observed_time_titans_mac_clip1_v1',
        'parent': {'contract_path': str(CORE.relative_to(ROOT)), 'canonical_sha256': expected_core,
                   'source_closure_sha256': core['source']['files_sha256'],
                   'parent_source_files': core['source']['files'],
                   'working_branch_at_review': 'codex/hard-lmm-causal-qkv'},
        'source': {
            'status': 'implementation_not_frozen', 'execution_closure_sha256': None,
            'preserve_live_core_97_files': True,
            'unchanged_model_and_training_math': [
                'models/TPPs/CountAwareFactory.py', 'models/TPPs/CountAwareTPP.py',
                'models/Titan/common/titans_mac.py', 'models/Titan/common/titans_memory_stability.py',
                'paper/scripts/count_aware_tpp_backbone/core.py', 'paper/scripts/count_aware_tpp_backbone/training.py'],
            'additional_existing_files': {f: sha(ROOT / f) for f in [
                'models/Titan/common/titans_mac_optimized.py',
                'paper/scripts/run_with_titantpp_mac_dynamo_policy.py']},
            'planned_new_files': ['paper/scripts/titans_mac_observed_time_contract.py',
                                 'paper/scripts/titans_mac_observed_time_runtime.py',
                                 'paper/scripts/run_titans_mac_observed_time.py'],
            'planned_copy_only_edits': {
                'paper/scripts/count_aware_tpp_backbone/observed_time.py':
                    'Add explicit MAC-only role with mandatory inner clip1 and identical observation/selector checks; preserve existing role allowlists.',
                'paper/scripts/count_aware_tpp_backbone/constants.py':
                    'Register the new role without changing historical primary/B012 roles.'},
            'implementation_location': 'New campaign-local source copy; parent originals and running remote source/Runtime remain unchanged.',
            'launch_requirement': 'Freeze complete import closure, parent-to-child diff, tests, new execution-contract SHA and source SHA before requesting run approval.'},
        'architecture': {'backbone': 'titantpp_titans_mac', 'hidden_dim': 64, 'layers': 2,
                         'heads': 4, 'feedforward_dim': 128, 'dropout': 0.1,
                         'persistent_tokens': 16, 'segment_size': 16,
                         'static_hard_lmm': False, 'history_full_or_mlp_correction': False,
                         'inner_gradient_clip': {'max_norm': 1.,
                             'scope': 'joint L2 of four associative gradients per row per observed token',
                             'before': 'momentum then adaptive forgetting', 'differentiable': True},
                         'optimized_scan': 'existing fixed_shape_compiled_state_only_cuda batch128 chunk16',
                         'dynamo_process_limits': {'recompile_limit': 64, 'accumulated_recompile_limit': 512},
                         'silent_eager_fallback': False},
        'state_and_causality': {
            'lifetime': 'fresh state per sample/window; no carry across batches, epochs, train/validation or series',
            'read_write': 'segment-start memory read, causal prediction, observed valid-token writes for later segments',
            'target': 'predict from last observed history position; withhold target quantity and prohibit target write; target dt used for likelihood only',
            'padding': 'neither prediction nor memory update/position count changes with padding',
            'streaming_api': 'test series reset separately; persistent cross-window state is not used in this campaign',
            'short_history_limit': 'histories within the first segment have no earlier-segment online writes to read; report this subgroup',
            'replay': 'eval/no_grad still performs analytical inner updates on observed history; reset state on every forward; outer weights unchanged'},
        'training': {**core['training'], 'outer_optimizer': 'AdamW', 'learning_rate': .001,
                     'weight_decay': .01, 'outer_gradient_clip': 1., 'lambda_log_qty': 1.,
                     'lambda_tail': 0., 'automatic_mixed_precision': False,
                     'selection_loss': 'recorded positive-integer duration NLL + log1p quantity MSE; select only raw quantity RMSE',
                     'initialization': 'same seed and train-only statistics; identical common head tensor values checked natively; different encoder tensors/count/RNG consumption disclosed',
                     'batch_order': 'dedicated seed generator; hash actual B/Full-prefix through min completed epochs; no dropping final batch',
                     'hpo_trials': 0},
        'datasets': datasets, 'jobs': jobs, 'hosts': hosts,
        'limits_proposed_not_approved': {
            'aggregate_gpu_reservation_hours': 480, 'host_wall_hours': {'5080': 216, '5090': 264},
            'clock': 'per-host clock begins immediately before native CUDA qualification; includes qualification, preparation, all fits and replays',
            'calendar_ceiling_hours_from_first_host_start': 264,
            'start_unix': None, 'deadline_unix': None,
            'deadline_rule': 'At launch freeze T0+264h globally and min(host_start+host_cap, global_deadline) per host; no extension for late start or wait.',
            'per_arm_includes_two_endpoint_replays': True,
            'per_host_synthetic_qualification_seconds': 1800, 'per_host_synthetic_updates': 60,
            'output_bytes_per_host': 17179869184, 'file_bytes_max': 67108864,
            'minimum_disk_free_bytes': 21474836480, 'minimum_free_vram_mib': 12000,
            'other_cuda_compute_processes_allowed_at_start': False,
            'no_progress_stack_seconds': 300, 'no_progress_stop_seconds': 1800,
            'compile_phase_hard_cap_seconds': 1800,
            'effective_arm_cap': 'minimum of dataset arm cap, host deadline, campaign deadline; truncated arms remain incomplete',
            'not_an_eta': True, 'prior_240_or_480_hour_authorizations_reused': False},
        'start_gates': [
            'Production wrapper/role tests and source/contract freeze complete, followed by explicit new GPU approval.',
            'Existing core campaign terminal and collected; user sees its results before MAC launch decision. No automatic launch on resource availability.',
            'Both hosts: verify exact Runtime/library hashes/GPU, exclusive claims, empty output root, no foreign CUDA workload; never stop other jobs.',
            'Approved synthetic native qualification on assigned dataset heads/context lengths; CPU tests are not CUDA qualification.',
            'Each scientific condition first full train+validation epoch must pass finite metrics, population/steps, clipping policy, checkpoint identity and memory-state checks before continuing. These are real epoch1 updates in the same fit, not a discarded pilot or extra fit.',
            'CUDA failures, OOM, source/Runtime drift, identity mismatch or deadlines stop that host; never shrink batch, alter compile mode, skip batch/seed, retry or resume.'
        ],
        'failure_policy': {'automatic_retry': False, 'automatic_resume': False,
                           'stop_owned_child_only': True, 'other_host_on_single_host_failure': 'continue its authorized jobs and read-only monitoring',
                           'partial_updates': 'record known completed batches and unknown partial updates explicitly',
                           'nonempty_root_or_claim': 'stop before launching any child',
                           'ssh_response_lost': 'read claims/PID/tmux before deciding; communication failure is not training failure'},
        'reused_results': {'B_Full_fits': 18, 'external_RMTPP_THP_NHP_SAHP_fits': 36,
                           'core_new_fits_pending_campaign_completion': 36,
                           'automatic_retraining': False,
                           'historical_MAC_seed42_unclipped_and_seed52_62_legacy': 'context only; never pool as the new 3-seed result',
                           'logical_unique_total_if_all_complete': 99,
                           'logical_endpoint_records_if_all_complete': 198},
        'analysis': {
            'primary_contrast': 'B versus MAC per dataset, seed42/52/62; candidate_minus_MAC raw quantity RMSE',
            'secondary_predeclared_contrasts': ['Full versus MAC', 'history MLP versus MAC'],
            'core_other_arms': 'report all; no selecting only favorable models, datasets, seeds or endpoints',
            'metrics': ['same-selected-checkpoint raw RMSE/MAE', 'recorded_time_nll', 'body/tail and nonempty quantity bins',
                        'existing history bins', 'first40 and last30 RMSE mean/sample SD', 'selected and last endpoint separately',
                        'parameters, actual epochs/steps/targets, fit/replay/wall time, allocator memory'],
            'additional_history_bins': [15, 31, 63, 127],
            'additional_history_bins_note': 'integer observed length, right=True: <=15,16..31,32..63,64..127,>=128. MAC-only diagnostic unless other-model equivalent records exist; no unauthorized new B/Full/MLP replay.',
            'aggregation': 'within-dataset three-seed mean and sample SD ddof=1, matched seed differences; no pooled raw-RMSE average across datasets',
            'inference': 'three seeds do not establish general significance; no held-out generalization claim',
            'performance_guard_reference': copy.deepcopy(core['comparison']['acceptance']),
            'guard_application': 'Secondary diagnostic applied explicitly as candidate/MAC for B, Full, MLP; NLL uses absolute difference; empty/zero-denominator/SD-degenerate cells NA, not pass. Existing Full/B gate remains unchanged.',
            'cost_limitations': 'fixed epochs/early-stop rule is not equal GPU time or parameter count; no borrowing 15.5% core concurrency benefit; no subtracting overlap from observed times',
            'claims_not_isolated': ['pure memory effect', 'memory x history interaction', 'verbatim original Titans reproduction', 'latest TPP superiority', 'held-out generalization']},
        'termination_evidence': ['approval and start permit', 'parent/child source and Runtime hash receipts',
                                 'qualification', 'input and initialization identity', 'history/summary/exposure',
                                 'selected/last byte and tensor SHA + replay agreement', 'worker receipt/partition/status/logs',
                                 'owned PID/tmux exit', 'failure/unstarted rows', 'validation final report + machine-readable ledger'],
        'current_readiness': {'contract_prepared': True, 'existing_tests': '83 passed, 1 skipped (CUDA unavailable)',
                              'cpu_factory_probe': 'three dataset head configurations, synthetic batch2 len34 seed42 passed',
                              'wrapper_implemented': False, 'native_qualification': False,
                              'source_frozen_for_new_execution': False, 'gpu_approved': False,
                              'deployed': False, 'launched': False},
    }
    save(DEST, plan)
    save(OUT / 'cost_estimate.json', {'method': 'historical fit elapsed / completed epochs; amortized proxy, NOT individually timed epochs or current ETA',
         'rows': costs, 'range_for_3seeds_each_40_to_300_epochs_gpu_hours': [sum(r[f'three_seed_{e}_epoch_proxy_hours'] for r in costs) for e in (40, 300)],
         'exclusions': ['future wrapper/head/Runtime effects', 'new compilation', 'qualification', 'endpoint replays', 'queue/wait'],
         'proposed_cap_gpu_hours': 480})
    with (OUT / 'execution_matrix.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0])); w.writeheader(); w.writerows(jobs)
    sources = [str(CORE.relative_to(ROOT)), str(prior_csv.relative_to(ROOT)),
        'paper/contracts/count_aware_titantpp_mac_inner_gradient_stability_v1.json',
        'paper/scripts/count_aware_titantpp_mac_contract.py',
        'paper/scripts/count_aware_titantpp_mac_runtime.py',
        'reports/titans_mac_reuse_audit_20260928_v1/report.md',
        'reports/local_detail_3seed_final_20260927_v1/terminal_audit.json',
        *plan['source']['additional_existing_files'], *core['source']['files']]
    sources.extend(r['source'] for r in costs)
    sources.extend(str(Path(r['source']).with_name('history.json')) for r in costs)
    sources = sorted(set(sources))
    save(OUT / 'sources.json', {f: sha(ROOT / f) for f in sources})
    checks = {
        'parent_canonical_and_97_sources_unchanged': True,
        'exact_nine_unique_dataset_seed_jobs': len(jobs) == len({j['id'] for j in jobs}) == 9,
        'six_5080_three_5090_jobs': [sum(j['host'] == h for j in jobs) for h in ['5080', '5090']] == [6, 3],
        'two_validation_endpoints_per_job': plan['validation_endpoint_replay_count'] == len(jobs) * 2,
        'cap_equals_host_caps': sum(plan['limits_proposed_not_approved']['host_wall_hours'].values()) == 480,
        'all_common_dataset_fields_equal_parent': all(all(d[k] == next(x for x in core['datasets'] if x['dataset_id'] == d['dataset_id'])[k] for k in d) for d in datasets),
        'no_approval_or_dates_implied': plan['approved'] is False and plan['limits_proposed_not_approved']['deadline_unix'] is None,
        'not_execution_ready': not any(plan['current_readiness'][k] for k in ['wrapper_implemented','native_qualification','gpu_approved','deployed','launched']),
        'old_and_current_conditions_not_double_counted': 18 + 36 + 36 + 9 == plan['reused_results']['logical_unique_total_if_all_complete'],
        'actual_local_probe_passed': probe['status'] == 'passed',
    }
    assert all(checks.values()), checks
    save(OUT / 'verification.json', {'status': 'passed', 'checks': checks,
        'design_contract_path': str(DEST.relative_to(ROOT)), 'design_canonical_sha256': canonical(plan),
        'execution_contract_sha256': None, 'gpu_started': False, 'held_out_accessed': False,
        'tests_log_sha256': sha(OUT / 'existing_cpu_tests.log'), 'factory_probe_sha256': sha(OUT / 'factory_probe.json')})
    print(json.dumps({'contract': str(DEST), 'sha256': canonical(plan), 'checks_passed': len(checks),
                      'cost_proxies': json.loads((OUT / 'cost_estimate.json').read_text())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
