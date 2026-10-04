"""Bind existing Deep Renewal selected binaries; no deserialization or inference."""
from pathlib import Path
import copy
import csv
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARM = 'deep_renewal_event_native_nb'

def load(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))

def write(name, x):
    (OUT / name).write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')

def main():
    registry_path = ROOT / 'reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json'
    original = load(registry_path)
    jobs = {}
    audit_csv = ROOT / 'reports/titantpp_completed22_audit_20261002_v1/condition_registry.csv'
    for r in csv.DictReader(audit_csv.open()):
        if r['model'] == ARM and r['training_state'] == 'complete':
            jobs[(r['dataset'], int(r['seed']))] = dict(root=Path(r['original_job_path']), host=r['host'], audit=r['audit'])
    final_path = ROOT / 'reports/titantpp_5090_final_sync_20261003_v1/registry.json'
    for r in load(final_path)['conditions']:
        if r['job']['arm'] == ARM:
            jobs[(r['job']['dataset'], int(r['job']['seed']))] = dict(root=Path(r['original_root']), host='5090', audit=r['audit_path'])
    jobs[('intermittent_frozen_5000', 62)] = dict(
        root=ROOT / 'search_artifacts/titantpp_intermittent_seed62_runpodpro4500_20261002_v1_retry1/retrieved/original/run/intermittent_frozen_5000__62__deep_renewal_event_native_nb',
        host='pro4500', audit=None)
    assert len(jobs) == 12
    bundles = {k: copy.deepcopy(original['bundles'][k]) for k in ('extension5080', 'extension5090', 'extensionPRO4500')}
    checks = []
    for key, b in bundles.items():
        for f, expected in b['source_files'].items():
            p = ROOT / b['source_root'] / f
            assert sha(p) == expected, str(p)
        # Preserve original training contract and scientific-source closure verbatim.
        b['factory_config_overrides'] = dict(quantity_variant='shifted_nb_quantity_nll', time_head_mode='shifted_nb_duration')
        b['factory_config_override_source'] = 'paper/scripts/run_pakdd_extension.py::build_model'
        checks.append(dict(bundle=key, source_files_verified=len(b['source_files']), source_closure_sha256=b['source_closure_sha256']))
    rows, refs, metrics = [], {}, []
    for (dataset, seed), entry in sorted(jobs.items()):
        jobroot = entry['root']
        run = jobroot / 'runs' / ARM / 'shifted_nb_quantity_nll' / f'seed_{seed}'
        cp = run / 'best_val_qty_rmse_model.pt'
        ep_path, sm_path, tm_path = run / 'endpoint_replays.json', run / 'summary.json', jobroot / 'terminal_manifest.json'
        ep, sm, tm = load(ep_path), load(sm_path), load(tm_path)
        assert tm['scientific_success'] and tm['status'] == 'complete'
        assert ep['status'] == 'complete' and ep['evaluation_scope'] == 'validation_only' and ep['held_out_test_evaluated'] is False
        val = ep['selected']
        assert val['evaluation_scope'] == 'validation_only' and val['held_out_test_evaluated'] is False
        for p in (cp, ep_path, sm_path, run/'history.json', jobroot/'input_receipt.json', jobroot/'initialization.json'):
            assert sha(p) == tm['files'][str(p.relative_to(jobroot))], str(p)
        assert val['state_sha256'] == sm['checkpoint_state_sha256']
        assert sm['variant'] == 'shifted_nb_quantity_nll'
        assert sm['encoder_config']['time_head']['mode'] == 'shifted_nb_duration'
        key = 'extensionPRO4500' if entry['host'] == 'pro4500' else 'extension' + entry['host']
        b = bundles[key]
        assert tm['contract_sha256'] == b['canonical_sha256']
        assert sm['interface_meta']['source_files_sha256'] == b['source_closure_sha256']
        cfg = next(d for d in b['datasets'] if d['dataset_id'] == dataset)
        r = dict(dataset=dataset, model=ARM, label='Deep Renewal (native shifted-NB; research only)', seed=seed, host=entry['host'],
                 endpoint='selected', selection='strict_first_min_validation_raw_quantity_rmse', selected_epoch=ep['best_epoch'],
                 completed_epochs=ep['completed_epochs'], checkpoint_path=rel(cp), checkpoint_file_sha256=sha(cp),
                 checkpoint_bytes=cp.stat().st_size, state_tensor_sha256=val['state_sha256'], local_binary_present=True,
                 recorded_remote_checkpoint_path=val['checkpoint_path'], evaluator_source_bundle=key,
                 frozen_source_directory=b['source_root'], source_closure_sha256=b['source_closure_sha256'],
                 contract_canonical_sha256=b['canonical_sha256'], original_contract_path=b['contract_path'],
                 validation_replay_path=rel(ep_path), validation_replay_sha256=sha(ep_path), validation_count=val['count'],
                 validation_reference=val, source_summary_path=rel(sm_path), source_summary_sha256=sha(sm_path),
                 history_path=rel(run/'history.json'), history_sha256=sha(run/'history.json'),
                 terminal_manifest_path=rel(tm_path), terminal_manifest_sha256=sha(tm_path),
                 input_receipt_path=rel(jobroot/'input_receipt.json'), input_receipt_sha256=sha(jobroot/'input_receipt.json'),
                 initialization_path=rel(jobroot/'initialization.json'), initialization_sha256=sha(jobroot/'initialization.json'),
                 initial_state_sha256=ep['initial_state_sha256'], original_source_revision=sm['source_revision'],
                 encoder_config=sm['encoder_config'], interface_meta=sm['interface_meta'], loader=cfg['loader'],
                 statistics=cfg['statistics'], time_statistics=cfg['time_statistics'], inherited_data_identity=cfg['inherited_data_identity'],
                 native_config_overrides=b['factory_config_overrides'], prior_cpu_audit_path=rel(entry['audit']) if entry['audit'] else None,
                 prior_cpu_audit_sha256=sha(entry['audit']) if entry['audit'] else None,
                 binary_status='prior_CPU_audit_reused_and_current_bytes_verified' if entry['audit'] else 'current_bytes_verified_CPU_deserialization_not_yet_recorded',
                 manuscript_comparison_included=False, comparison_family='native_deep_renewal_research_only',
                 time_nll_definition='recorded_positive_integer_shifted_negative_binomial_mass_nll; top-coded survival where configured')
        rows.append(r)
        refs[f'{dataset}__{ARM}__seed{seed}'] = {k: val[k] for k in ('count','qty_mae','qty_rmse','time_nll','state_sha256','evaluation_scope','held_out_test_evaluated')}
        metrics.append({k:r[k] for k in ('dataset','model','seed','selected_epoch','comparison_family','time_nll_definition')} | dict(split='validation', **{k:val[k] for k in ('count','qty_mae','qty_rmse','time_nll')}, source=rel(ep_path), source_sha256=sha(ep_path), provenance='selected_endpoint_replay_not_new_prediction', view='overall'))
    write('evaluation_registry.json', dict(status='12_selected_byte_identities_verified; 11_prior_CPU_audits_reused; 1_CPU_audit_not_recorded', rows=rows, bundles=bundles, manuscript_comparison_included=False))
    write('validation_references.json', dict(scope='native_Deep_Renewal_original_selected_validation', selection='strict_first_min_validation_raw_quantity_rmse', relative_tolerance=1e-5, absolute_tolerance=1e-5, references=refs))
    manifest_source = ROOT / 'reports/titantpp_legacy_evaluation_20261003_v1/dataset_manifest.json'
    manifest = load(manifest_source)
    manifest['source_manifest_path'] = rel(manifest_source)
    manifest['source_manifest_sha256'] = sha(manifest_source)
    manifest['comparison_family'] = 'native_deep_renewal_research_only'
    write('dataset_manifest.json', manifest)
    write('selected_binding.json', dict(status='passed_byte_and_record_bindings', created_utc=datetime.now(timezone.utc).isoformat(), checkpoint_count=12, terminal_record_bindings_verified=12, checkpoint_state_hashes_source='original selected endpoint replay and summary; no new deserialization', source_checks=checks, new_inference_calls=0, new_training_calls=0, remote_calls=0, rows=[{k:r[k] for k in ('dataset','model','seed','checkpoint_path','checkpoint_file_sha256','state_tensor_sha256','selected_epoch','validation_replay_path','validation_replay_sha256','terminal_manifest_path','terminal_manifest_sha256','source_closure_sha256','evaluator_source_bundle','binary_status')} for r in rows]))
    with (OUT/'validation_metrics_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    print(json.dumps(dict(rows=len(rows), path=rel(OUT/'evaluation_registry.json'), source_checks=checks)))

if __name__ == '__main__':
    main()
