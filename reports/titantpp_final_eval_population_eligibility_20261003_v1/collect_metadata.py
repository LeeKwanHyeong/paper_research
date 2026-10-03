"""Read-only population metadata audit. Never loads target/gap columns or models.

Only table materialization: selected_series site_cd/oper_part_no identifiers.
Parquet footer statistics are inspected only for an explicit calendar-date allowlist.
Notebook outputs are excluded; only code source is hashed/excerpted.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def file_record(rel):
    p = ROOT / rel
    return {'path': rel, 'exists': p.exists(), 'size_bytes': p.stat().st_size if p.exists() else None,
            'sha256': sha(p) if p.exists() else None}


def footer(rel, date_columns=()):
    p = ROOT / rel
    pf = pq.ParquetFile(p)
    dates = {}
    for col in date_columns:
        lo, hi = [], []
        for i in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(i)
            for j in range(rg.num_columns):
                c = rg.column(j)
                if c.path_in_schema != col:
                    continue
                s = c.statistics
                if s is not None and s.has_min_max:
                    lo.append(s.min)
                    hi.append(s.max)
        dates[col] = {'min': str(min(lo)) if lo else None, 'max': str(max(hi)) if hi else None,
                      'row_groups_with_date_statistics': len(lo)}
    return {'path': rel, 'size_bytes': p.stat().st_size,
            'column_names': pf.schema.names, 'calendar_date_footer_only': dates,
            'data_rows_loaded': False, 'target_or_gap_statistics_read': False}


def run():
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'source/schema/date/provenance metadata; no evaluation',
                'calendar_date_footer_allowlist': ['tpep_pickup_datetime', 'time_bucket', 'demand_dt'],
                'materialized_column_allowlist': ['site_cd', 'oper_part_no']}
    metadata['taxi'] = [footer('sample_data/new_york_taxi/yellow_trip.parquet', ['tpep_pickup_datetime']),
                        footer('sample_data/new_york_taxi/yellow_trip_hourly.parquet', ['time_bucket', 'demand_dt'])]
    metadata['intermittent'] = {}
    sample_rel = 'sample_data/intermittent_v2/intermittent_frozen_5000_sampling_manifest.json'
    sm = json.loads((ROOT / sample_rel).read_text())
    metadata['intermittent']['sampling_manifest_allowed'] = {k: sm[k] for k in ['created_at', 'source', 'sampling'] if k in sm}
    raw = Path(sm['source']['path'])
    metadata['intermittent']['raw_source_exists'] = raw.exists()
    selected_rel = 'sample_data/intermittent_v2/intermittent_frozen_5000_selected_series.parquet'
    selected = pq.read_table(ROOT / selected_rel, columns=['site_cd', 'oper_part_no']).to_pylist()
    pairs = sorted((str(x['site_cd']), str(x['oper_part_no'])) for x in selected)
    # Do not release identifiers; bind the full mapping by a canonical hash.
    metadata['intermittent']['selected_mapping'] = {
        'path': selected_rel, 'file_sha256': sha(ROOT / selected_rel),
        'rows': len(pairs), 'unique_sites': len({x[0] for x in pairs}),
        'unique_series': len({x[1] for x in pairs}),
        'canonical_pair_json_sha256': hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest(),
        'columns_read': ['site_cd', 'oper_part_no'], 'identifiers_released': False}
    metadata['intermittent']['events'] = footer('sample_data/intermittent_v2/intermittent_frozen_5000_events.parquet', ['demand_dt'])
    metadata['instacart_raw_stat_only'] = [
        {'path': str(p.relative_to(ROOT)), 'exists': p.exists(), 'size_bytes': p.stat().st_size if p.exists() else None}
        for p in [ROOT / 'sample_data/insta_market_basket' / x for x in
                  ['orders.csv', 'order_products__prior.csv', 'order_products__train.csv']]]
    manifest = json.loads((ROOT / 'benchmark_data/manifests/raf_spare_parts_v1.json').read_text())
    contract = json.loads((ROOT / 'benchmark_data/contracts/raf_spare_parts_v1.json').read_text())
    metadata['raf'] = {'source': manifest['source'],
                       'date_metadata': {k: manifest['audit'][k] for k in ['date_start', 'date_end', 'months'] if k in manifest['audit']},
                       'event_definition': contract['event_definition'], 'split': contract['split'],
                       'source_rights_local_assertions_not_authority_grant': contract['source'],
                       'raw_stat_and_hash': file_record(contract['source']['path']), 'workbook_rows_opened': False}
    sources = [
        'TEST_SESSION_PROTOCOL.md',
        'paper/scripts/build_intermittent_frozen_subset.py',
        'simple_lab_test/notebooks/preprocessing/tpp_split_utils.py',
        'simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb',
        'simple_lab_test/notebooks/preprocessing/insta_market_basket.ipynb',
        'simple_lab_test/notebooks/preprocessing/insta_market_basket_tpp_splits.ipynb',
        'benchmark_data/scripts/audit_and_prepare_raf.py',
        'benchmark_data/contracts/raf_spare_parts_v1.json',
        'benchmark_data/manifests/raf_spare_parts_v1.json',
        sample_rel,
        'reports/titantpp_pakdd_extension_preparation_20261001_v1/lineage.json',
        'reports/titantpp_independent_evaluation_preparation_20261001_v1/lineage_review.json',
        'reports/titantpp_independent_evaluation_preparation_20261001_v1/protocol_effective.json',
        'reports/titantpp_independent_evaluation_preparation_20261001_v1/amendment.json',
    ]
    metadata['source_files'] = [file_record(r) for r in sources]
    metadata['notebook_code_only'] = []
    for r in sources:
        if r.endswith('.ipynb') and (ROOT / r).exists():
            nb = json.loads((ROOT / r).read_text())
            cells = [{'cell_index': i, 'source': ''.join(c.get('source', []))}
                     for i, c in enumerate(nb.get('cells', [])) if c.get('cell_type') == 'code']
            metadata['notebook_code_only'].append({'path': r, 'outputs_read_or_saved': False,
                'code_sha256': hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest(),
                'code_cells': cells})
    lineage = json.loads((ROOT / 'reports/titantpp_pakdd_extension_preparation_20261001_v1/lineage.json').read_text())
    metadata['prior_access_metadata_reused'] = {
        'path': 'reports/titantpp_pakdd_extension_preparation_20261001_v1/lineage.json',
        'datasets': {k: {x: d[x] for x in ['classification', 'historical_test_access', 'final_evaluation_role',
                                             'raw_current_source_available', 'raw_legacy_source_available'] if x in d}
                     for k, d in lineage['datasets'].items()},
        'incidental_manifest_target_summaries_previously_exposed': lineage['access_note']['incidental_manifest_target_summaries_exposed'],
        'no_target_summaries_reopened': True}
    prep = json.loads((ROOT / 'reports/titantpp_independent_evaluation_preparation_20261001_v1/lineage_review.json').read_text())
    metadata['preparation_lineage_keys'] = sorted(prep)
    metadata['no_new_evaluation'] = True
    (OUT / 'local_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'saved': str(OUT / 'local_metadata.json'), 'sites': len({x[0] for x in pairs}),
                      'series': len(pairs), 'intermittent_source_exists': raw.exists(),
                      'intermittent_calendar_footer': metadata['intermittent']['events']['calendar_date_footer_only']}, ensure_ascii=False))


if __name__ == '__main__':
    run()
