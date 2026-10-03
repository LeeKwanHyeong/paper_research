"""Bounded filename / execution-contract metadata search; never read result files."""
from pathlib import Path
import subprocess, json, re, hashlib
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
roots = ['sample_data', 'benchmark_data', 'search_artifacts']
files = subprocess.check_output(['rg', '--files', '--hidden', *roots], cwd=ROOT, text=True).splitlines()
date_pattern = re.compile(r'(?<!\d)2015[-_/]?02(?:\D|$)|(?<!\d)2015.*(?:february|feb\b)', re.I)
filename_matches = [x for x in files if date_pattern.search(x)]
basenames = {'launch_contract.json', 'experiment_manifest.json', 'source_manifest.json', 'download_manifest.json'}
candidates = [x for x in files if Path(x).name in basenames]
allowed = {'dataset', 'dataset_name', 'data_path', 'input_path', 'source_path', 'source_url', 'download_url',
           'url', 'file_name', 'filename', 'dataset_root', 'start_date', 'end_date', 'date_start', 'date_end',
           'split_manifest_path', 'source_revision', 'evaluation_scope', 'held_out_test_evaluated'}
results, errors, selected = [], [], []


def walk(x, pointer=''):
    if isinstance(x, dict):
        for k, v in x.items():
            at = pointer + '/' + k
            if k in allowed and isinstance(v, (str, int, float, bool)):
                yield at, v
            if isinstance(v, (list, dict)):
                yield from walk(v, at)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from walk(v, pointer + '/' + str(i))


for rel in candidates:
    try:
        p = ROOT / rel
        obj = json.loads(p.read_text())
        values = list(walk(obj))
        if any(date_pattern.search(str(v)) for _, v in values):
            results.append({'path': rel, 'matching_metadata': [{'pointer': k, 'value': v} for k, v in values if date_pattern.search(str(v))]})
        if 'taxi' in rel.lower() or any('taxi' in str(v).lower() or 'yellow_trip' in str(v).lower() for _, v in values):
            selected.append({'path': rel, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                             'source_path_or_period_fields': [{'pointer': k, 'value': v} for k, v in values
                                                            if k.rsplit('/', 1)[-1] in allowed - {'source_revision'}]})
    except (ValueError, OSError) as e:
        errors.append({'path': rel, 'error': type(e).__name__})
code_evidence = json.loads((OUT / 'local_metadata.json').read_text())['notebook_code_only']
code_matches = []
for nb in code_evidence:
    if 'yellow_trip' in nb['path']:
        for cell in nb['code_cells']:
            if date_pattern.search(cell['source']):
                code_matches.append({'path': nb['path'], 'cell_index': cell['cell_index']})
report = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'scope': roots,
          'filenames_scanned': len(files), 'execution_or_source_contracts_parsed': len(candidates),
          'metadata_key_allowlist': sorted(allowed), 'filename_date_matches': filename_matches,
          'contract_date_matches': results, 'current_taxi_notebook_code_date_matches': code_matches,
          'taxi_related_contracts': selected, 'errors': errors,
          'date_query': '2015-02 / 201502 /2015 February filename and explicit metadata fields',
          'heldout_metric_prediction_or_mixed_result_files_opened': False,
          'notebook_outputs_read': False,
          'interpretation': 'No match is bounded negative evidence only. Generic filenames, deleted/renamed files, notebook outputs, other machines, external storage and researcher access are outside coverage.',
          'prior_access_status': 'user_unsure', 'nonaccess_attested': False}
(OUT / 'taxi_access_search.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['filenames_scanned', 'execution_or_source_contracts_parsed',
                                      'filename_date_matches', 'contract_date_matches', 'current_taxi_notebook_code_date_matches', 'errors']}, ensure_ascii=False))
