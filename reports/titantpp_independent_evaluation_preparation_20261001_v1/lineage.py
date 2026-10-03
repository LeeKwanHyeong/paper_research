"""Split provenance from contracts and train/entity metadata; no held-out outcomes."""
import hashlib,json,subprocess
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import polars as pl
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 old=read(ROOT/'reports/titantpp_followup_contract_5080_20260927_v1/heldout_provenance_audit.json')
 historical=[]
 for r in old['matching_campaign_records']:
  p=ROOT/r['manifest'];identical=p.exists() and sha(p)==r['sha256']
  examples=[{'path':x['path'],'recorded_bytes':x['bytes'],'exists':(ROOT/x['path']).is_file(),'current_bytes':(ROOT/x['path']).stat().st_size if (ROOT/x['path']).is_file() else None} for x in r['artifact_examples_metadata_only']]
  historical.append({'manifest':r['manifest'],'manifest_unchanged':identical,'datasets':r['dataset_metadata'],'artifact_examples_stat_only':examples})
 sample_path=ROOT/'sample_data/intermittent_v2/intermittent_frozen_5000_sampling_manifest.json'
 sample=read(sample_path);selected_path=Path(sample['artifacts']['selected_series']['path'])
 assert sha(selected_path)==sample['artifacts']['selected_series']['sha256']
 mapping=pl.read_parquet(selected_path,columns=['site_cd','oper_part_no']).with_columns(pl.col('site_cd').cast(pl.String),pl.col('oper_part_no').cast(pl.String))
 assert mapping['oper_part_no'].n_unique()==mapping.height==5000 and mapping['site_cd'].n_unique()==50
 mapping.sort(['site_cd','oper_part_no']).write_csv(OUT/'intermittent_entity_site_mapping.csv')
 oldids=set(pl.scan_parquet(ROOT/'sample_data/head_office/marked_target_with_split.parquet').filter(pl.col('chronological_split')=='train').select(pl.col('oper_part_no').cast(pl.String)).unique().collect()['oper_part_no'])
 newids=set(mapping['oper_part_no'])
 current_source=Path(sample['source']['path'])
 inter={'source':sample['source'],'source_file_currently_available':current_source.exists(),'source_sha_current':sha(current_source) if current_source.is_file() else None,'sampling':sample['sampling'],'sampling_manifest_sha256':sha(sample_path),'selected_series_sha256':sha(selected_path),'metadata_columns_read':['site_cd','oper_part_no'],'old_dataset_columns_read':['oper_part_no'],'old_dataset_filter':'chronological_split == train','old_train_entities':len(oldids),'new_selected_entities':len(newids),'sites':50,'exact_old_train_entity_id_overlap':len(oldids&newids),'identity_caveat':'IDs have different construction and legacy source identity is incomplete; zero string overlap does not establish event-level or human-exposure independence.','bootstrap_amendment':'Use whole site_cd clusters for the primary conditional interval; whole oper_part_no is a declared sensitivity analysis.','mapping_sha256':sha(OUT/'intermittent_entity_site_mapping.csv')}
 paths=subprocess.check_output(['rg','--files','search_artifacts','paper/results','-g','launch_contract.json','-g','experiment_manifest.json'],cwd=ROOT,text=True).splitlines()
 raf=[]
 for rel in paths:
  if 'raf' not in rel or not rel.endswith('launch_contract.json'):continue
  p=ROOT/rel;j=read(p)
  allow=['dataset','data_path','data_sha256','split_manifest_path','split_manifest_sha256','evaluation_scope','held_out_test_evaluated','source_revision']
  raf.append({'path':rel,'sha256':sha(p),'metadata':{k:j[k] for k in allow if k in j}})
 contract=read(ROOT/'benchmark_data/contracts/raf_spare_parts_v1.json');manifest=read(ROOT/'benchmark_data/manifests/raf_spare_parts_v1.json')
 raf_prov={'source_file_identity':manifest['source'],'split':contract['split'],'declared_policy':contract['held_out_policy'],'historical_launch_contracts':raf,'scope_counts':dict(Counter((r['metadata'].get('evaluation_scope','unknown')+':'+str(r['metadata'].get('held_out_test_evaluated'))) for r in raf)),'coverage_limit':'All 25 path-matched local launch contracts available through rg, not all historical machines, notebooks, copies, human views or differently named campaigns.'}
 result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'metadata_lineage_review_complete_independence_unresolved','historical_manifests_rechecked':historical,'intermittent':inter,'raf':raf_prov,'researcher_attestation':{'status':'pending','note':'A request to continue preparation is not an attestation of no prior test viewing or model-selection use.'},'independence_status':{d:'not_established' for d in ['yellow_trip_hourly','intermittent_frozen_5000','insta_market_basket','raf_spare_parts']},'access_scope':{'heldout_metric_or_prediction_files_opened':False,'heldout_quantity_or_gap_columns_read':False,'train_entity_identifiers_read':True,'sampling_entity_site_metadata_read':True,'new_evaluation':False}}
 (OUT/'lineage_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'historical_records':len(historical),'raf_launch_contracts':len(raf),'raf_scope_counts':raf_prov['scope_counts'],'intermittent_sites':50,'independence':'not_established'},ensure_ascii=False))
if __name__=='__main__':main()
