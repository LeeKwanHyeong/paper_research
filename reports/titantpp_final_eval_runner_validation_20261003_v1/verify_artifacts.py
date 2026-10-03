"""Check saved preparation receipts only; no model/data inference or remote I/O."""
from pathlib import Path
import datetime
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BINDING = ROOT / 'reports/titantpp_final_eval_checkpoint_binding_20261003_v1'
POPULATION = ROOT / 'reports/titantpp_final_eval_population_eligibility_20261003_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    checks = {}
    original = read(OUT / 'preserved_inputs.json')
    checks['historical_inputs_preserved'] = all(
        sha(ROOT / item['path']) == item['sha256'] for item in original['inputs'])
    registry_sha = sha(BINDING / 'evaluation_registry.json')
    registry = read(BINDING / 'evaluation_registry.json')
    audit = read(BINDING / 'verification.json')
    checks['bound_registry_receipt'] = (
        audit['status'] == 'passed' and audit['evaluation_registry_sha256'] == registry_sha
        and len(registry['rows']) == 108 and audit['still_missing'] == 0)
    smoke_dir = OUT / 'validation_smoke/attempt1'
    smoke = read(smoke_dir / 'combined.json')
    bundles = [read(smoke_dir / (name + '.json')) for name in
               ('extension5080', 'extensionPRO4500', 'extension5090')]
    checks['new_inference_code_identity'] = all(
        b['runner_sha256'] == sha(OUT / 'evaluate_validation.py')
        and b['helper_sha256'] == sha(OUT / 'sample_contract.py')
        and b['registry_sha256'] == registry_sha for b in bundles)
    checks['new24_validation_checks_passed'] = (
        sum(len(b['checks']) for b in bundles) == 24
        and all(b['status'] == 'passed' and not b['heldout_read'] for b in bundles)
        and all(all(c[key] for key in (
            'target_quantity_perturbation_preserves_prediction_and_time_loss',
            'target_gap_perturbation_preserves_quantity_prediction',
            'single_vs_batch_matches', 'parameters_unchanged', 'finite_outputs'))
                for b in bundles for c in b['checks']))
    checks['combined_receipt_counts_and_scope'] = (
        smoke['status'] == 'passed' and smoke['learned_conditions'] == 108
        and smoke['reused_learned_conditions'] == 84 and smoke['new_learned_conditions'] == 24
        and smoke['registry_sha256'] == registry_sha
        and smoke['prediction_rows'] == len(smoke['records']) == 638
        and sum(smoke['targets_by_dataset'].values()) == 22
        and smoke['simple_prediction_rows'] == 44 and not smoke['heldout_read']
        and not smoke['full_endpoint_replay'] and not smoke['native_gpu_tested']
        and all(r['split'] == 'validation' for r in smoke['records']))
    checks['prior_inference_immutable_reuse'] = (
        sha(ROOT / smoke['prior_smoke_path']) == smoke['prior_smoke_sha256'])
    suites = ET.parse(OUT / 'unit_tests.xml').getroot().findall('testsuite')
    checks['unit_tests_31_passed'] = (
        sum(int(s.attrib['tests']) for s in suites) == 31
        and all(int(s.attrib[k]) == 0 for s in suites for k in ('errors', 'failures', 'skipped')))
    stats = read(smoke_dir / 'statistics.json')
    checks['statistics_are_sample_only'] = (
        stats['scope'] == 'validation_sample_correctness_only'
        and not stats['paper_confidence_intervals'] and not stats['heldout_read'])
    protocol = read(OUT / 'effective_protocol.json')
    checks['effective_protocol_scope_and_registry'] = (
        protocol['checkpoint_manifest_sha256'] == registry_sha
        and protocol['validation_preparation_authorized'] and not protocol['execution_authorized']
        and protocol['readiness']['all_checkpoint_binaries_bound']
        and not protocol['readiness']['clean_population_eligible']
        and not protocol['readiness']['full_population_streaming_verified'])
    checks['restored_intermittent_site_primary'] = (
        protocol['resampling']['units']['intermittent_frozen_5000'] == 'whole_site_cd_cluster'
        and protocol['resampling']['intermittent_sensitivity_unit'] == 'whole_oper_part_no_cluster')
    population = read(POPULATION / 'eligibility.json')
    checks['population_not_misrepresented_as_clean'] = (
        population['counts']['PASS'] == 0
        and not any(d['ready_for_independent_evaluation'] for d in population['datasets'])
        and not population['new_evaluation'] and not population['heldout_metric_or_prediction_files_opened'])
    checks['manuscript_preserved'] = (
        sha(ROOT / 'paper/titantpp_pakdd_2027_draft/main.tex') ==
        '6e6e43cdecbffe45108c71cd342df296862d013011d9e7cac2b778e2c9971a20')
    review = read(OUT / 'independent_review.json')
    checks['independent_review_passed'] = review['status'] == 'passed' and review['findings'] == []
    result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'passed' if all(checks.values()) else 'failed',
              'scope': 'saved_preparation_artifact_consistency_not_independent_evaluation',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'heldout_read': False, 'new_inference': False}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    files = [OUT / name for name in (
        'README.md', 'amendment.json', 'effective_protocol.json', 'runtime.json',
        'preserved_inputs.json', 'historical_verifier_rerun.json', 'evaluate_validation.py',
        'sample_contract.py', 'paired_statistics.py', 'test_evaluator.py', 'unit_tests.xml',
        'independent_review.json', 'verify_artifacts.py', 'verification.json')]
    files += sorted(smoke_dir.glob('*.json'))
    files += [BINDING / name for name in ('evaluation_registry.json', 'verification.json')]
    files += [POPULATION / name for name in (
        'eligibility.json', 'effective_rule_reconciliation.json', 'README.md')]
    source_manifest = {'created_utc': result['created_utc'], 'files': {
        str(p.relative_to(ROOT)): sha(p) for p in files}}
    (OUT / 'sources.json').write_text(json.dumps(source_manifest, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'passed': result['passed'],
                      'total': result['total'], 'failed': [k for k,v in checks.items() if not v]}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
