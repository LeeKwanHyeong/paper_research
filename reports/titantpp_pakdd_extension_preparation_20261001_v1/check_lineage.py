"""Read event identity columns only; never load held-out targets or predictions."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(frame):
    rows = frame.sort(frame.columns).write_csv().encode()
    return {'rows': frame.height, 'unique': frame.unique().height,
            'columns': frame.columns, 'canonical_csv_sha256': hashlib.sha256(rows).hexdigest()}
def keys(path, split=None, columns=('oper_part_no','seq','demand_dt')):
    q = pl.scan_parquet(ROOT / path)
    if split:
        q = q.filter(pl.col('chronological_split') == split)
    return q.select(list(columns)).collect()
def overlap(a,b,cols):
    a,b = a.select(cols).unique(),b.select(cols).unique()
    n=a.join(b,on=cols,how='inner').height
    return {'left':a.height,'right':b.height,'intersection':n,
            'left_only':a.height-n,'right_only':b.height-n}

result = {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope':'event identifiers, split membership and source only', 'datasets':{}}
for name,stem,historical in [
    ('yellow_trip_hourly','sample_data/new_york_taxi/yellow_trip_hourly',
     'search_artifacts/model_enhancement_v3_taxi_short_e50_0710/experiment_manifest.json'),
    ('insta_market_basket','sample_data/insta_market_basket/instacart_marked_target',
     'search_artifacts/insta_ablation_baseline_e200/experiment_manifest.json')]:
    combined=stem+'_with_split.parquet'; separate=stem+'_test.parquet'
    a,b=keys(combined,'test'),keys(separate)
    manifest=stem+'_split_manifest.json'; m=json.loads((ROOT/manifest).read_text())
    result['datasets'][name] = {'classification':'same_event_keys_in_available_artifacts',
      'historical_execution_bytes':'not recoverable from manifest; no historical data digest',
      'historical_test_access':'confirmed by prior access review',
      'historical_manifest':{'path':historical,'sha256':sha(ROOT/historical)},
      'split_config':m['config'],'manifest_created_at':m['created_at'],
      'split_manifest_sha256':sha(ROOT/manifest),
      'current_data_sha256':sha(ROOT/combined),'available_legacy_test_sha256':sha(ROOT/separate),
      'current_test_identity':identity(a),'available_legacy_test_identity':identity(b),
      'overlap':overlap(a,b,a.columns),
      'final_evaluation_role':'previously exposed legacy test; not untouched independent confirmation'}
old='sample_data/head_office/marked_target_with_split.parquet'
new='sample_data/intermittent_v2/intermittent_frozen_5000_with_split.parquet'
a,b=keys(old,'test'),keys(new,'test')
# This candidate normalization is diagnostic, not an asserted entity mapping.
part=pl.col('oper_part_no').str.split('::').list.last().alias('candidate_part')
aa=a.with_columns(part);bb=b.with_columns(part)
result['datasets']['intermittent_frozen_5000']={
 'classification':'unresolved_source_entity_mapping',
 'available_legacy_test_identity':identity(a),'current_test_identity':identity(b),
 'literal_overlap':overlap(a,b,a.columns),
 'candidate_part_date_overlap':overlap(aa,bb,['candidate_part','demand_dt']),
 'candidate_mapping_is_verified':False,
 'legacy_event_rule':'notebook candidate: demand>8, merge bursts with gaps<=3 weeks; exact build provenance unbound',
 'current_event_rule':'positive source rows; site_cd::part_no; fixed 5000 stratified series; no burst merging',
 'raw_current_source_available':Path('/Users/igwanhyeong/data/demand_engine/data_v2/intermittent.parquet').exists(),
 'raw_legacy_source_available':(ROOT/'sample_data/tb_master_target.parquet').exists(),
 'final_evaluation_role':'legacy holdout with unresolved prior exposure; no untouched independence claim'}
raf='benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_with_split.parquet'
r=keys(raf,'test')
result['datasets']['raf_spare_parts']={
 'classification':'separate_dataset_from_the_identified_legacy_tests',
 'test_identity':identity(r),'current_data_sha256':sha(ROOT/raf),
 'historical_test_access':'no confirmed access in bounded prior search; not proof of nonaccess',
 'final_evaluation_role':'fixed temporal holdout; disclose incomplete historical access evidence',
 'train_months':'1996-01..2000-10','validation_months':'2000-11..2001-11','test_months':'2001-12..2002-12'}
split='simple_lab_test/notebooks/preprocessing/tpp_split_utils.py'
oldsource=subprocess.check_output(['git','show','3407f09:'+split],cwd=ROOT)
result['split_source']={'path':split,'historical_revision':'3407f09',
 'historical_sha256':hashlib.sha256(oldsource).hexdigest(),'current_sha256':sha(ROOT/split),
 'unchanged':oldsource==(ROOT/split).read_bytes()}
result['evaluation_decision']={
 'validation_extension':'all four frozen train/validation datasets; no split changes',
 'final_existing_test':'all four fixed test populations, after model freeze and separate evaluation permit; legacy holdout description',
 'untouched_independent_test_claim':False,
 'new_independent_confirmation':{
  'Taxi':'a later non-overlapping raw calendar period after all current event timestamps; same frozen spatial aggregation',
  'Intermittent':'new extraction after current per-site date cutoffs, or provenance-verified external sites/parts',
  'Instacart':'no verified later release available locally; use separately sourced basket histories or an external event-quantity dataset',
  'RAF':'this 84-month archive cannot supply later periods; independently sourced parts/months needed for new confirmation'},
 'new_data_acquisition_or_evaluation_started':False}
result['access_note']={
 'heldout_performance_or_predictions_read':False,'heldout_quantity_gap_columns_materialized':False,
 'incidental_manifest_target_summaries_exposed':True,
 'description':'Initial manifest inspection accidentally emitted existing summary fields including heldout target-distribution aggregates. No performance or predictions; subsequent reads are allowlisted. This is recorded rather than called a pristine unseen split.'}
(OUT/'lineage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:{x:v[x] for x in ('classification','overlap','candidate_part_date_overlap') if x in v} for k,v in result['datasets'].items()},indent=2))
