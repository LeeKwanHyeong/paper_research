from pathlib import Path
import hashlib,json,math,datetime
root=Path('/Users/igwanhyeong/PycharmProjects/paper_research');r=root/'reports/titantpp_backbone_post_attention_ffn_raf6_20261016_v1';b=root/'search_artifacts/titantpp_backbone_post_attention_ffn_raf6_20261016_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
files={};checks=0
for p in (b/'followup_integrity.json',b/'retrieval/final_receipt.json',b/'evaluation_v1/bootstrap_integrity.json',b/'evaluation_v1/final_receipt.json',r/'EVALUATION_READY.json'):
 files[str(p.relative_to(root))]=sha(p)
 for n,h in read(p)['files'].items():
  q=Path(n);q=q if q.is_absolute() else root/q
  assert sha(q)==h,n;files[str(q.relative_to(root))]=h;checks+=1
keys={'whole_qty_rmse':(None,'qty_rmse'),'whole_qty_mae':(None,'qty_mae'),'tail_qty_rmse':('tail','qty_rmse'),'tail_qty_mae':('tail','qty_mae'),'whole_time_nll':(None,'time_nll'),'tail_time_nll':('tail','time_nll')}
current='titantpp_cnn_ffn_current_v1';temporal='titantpp_cnn_ffn_temporal_v1';cnn='titantpp_cnn';arms=[cnn,current,temporal];seeds=[42,52,62]
def v(row,key):
 group,m=keys[key];return (row['metrics'] if group is None else row['metrics'][group])[m]
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-11),(a,b)
result={};means=sdcount=delta=0
for split in ('validation','test'):
 p=r/'evaluation_v1'/('THREE_SEED_'+split.upper()+'.json');d=read(p);rows={(x['arm'],x['seed']):x for x in d['seed_results']}
 assert set(rows)=={(a,s) for a in arms for s in seeds} and len(d['seed_results'])==9
 for g in d['grouped']:
  for k,saved in g['metrics'].items():
   vv=[v(rows[(g['arm'],s)],k) for s in seeds];m=math.fsum(vv)/3;sd=math.sqrt(math.fsum((x-m)**2 for x in vv)/2)
   close(m,saved['mean']);close(sd,saved['sample_sd']);means+=1;sdcount+=1
 assert len(d['seed_differences'])==9
 for x in d['seed_differences']:
  a=rows[(x['candidate'],x['seed'])];z=rows[(x['baseline'],x['seed'])]
  for k,saved in x['metrics'].items():
   dv=v(a,k)-v(z,k);close(dv,saved['candidate_minus_baseline']);close(100*dv/v(z,k),saved['relative_percent']);delta+=1
 result[split]=d
c=read(r/'evaluation_v1/EXTERNAL_COMPARISON.json');reg=read(r/'EXTERNAL_BASELINES.json');criteria=[]
for a in (current,temporal):
 for split in ('validation','test'):
  rows=[x for x in result[split]['seed_results'] if x['arm']==a]
  for m in ('qty_rmse','qty_mae'):
   cm=math.fsum(x['metrics'][m] for x in rows)/3
   cs=c['summaries'][a][split][m];close(cm,cs['candidate']['mean'])
   rr=[x for x in reg['rows'] if x['split']==split];models={x['model'] for x in rr};ext={model:math.fsum(x['metrics'][m] for x in rr if x['model']==model)/3 for model in models if model!='historical_CNN'}
   best=min(ext.values());close(best,cs['external_frontier']['mean'])
   hm=math.fsum(x['metrics'][m] for x in rr if x['model']=='historical_CNN')/3;close(hm,cs['historical_CNN']['mean'])
   passed=cm<hm and cm<best;assert passed==cs['user_performance_pass'];criteria.append({'arm':a,'split':split,'metric':m,'passed':passed})
 assert c['user_performance_gate'][a]==all(x['passed'] for x in criteria if x['arm']==a)
assert c['candidate_selection']['selected_candidate'] is None
final=read(b/'evaluation_v1/final_receipt.json');assert final['candidate_test_count']==6 and final['validation_replay_count']==6 and final['actual_native_worker_exit0_count']==12 and final['actual_pipeline_exit0_count']==2
out=r/'manual_completion_v1';out.mkdir()
with (out/'verify.py').open('xb') as f:f.write(Path(__file__).read_bytes())
receipt={'status':'PASS_local_stdlib_recalculation','checked_kst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'training_terminal_observed_kst':'2026-10-16T08:32:31.831268+09:00','evaluation_completed_kst':datetime.datetime.fromtimestamp(final['completed_unix'],datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'raw_sha_checks_including_repeated_dependencies':checks,'unique_pinned_files':len(files),'group_means_recomputed':means,'sample_sd_ddof1_recomputed':sdcount,'six_metric_seed_absolute_and_relative_pairs':delta,'user_criteria_recomputed':criteria,'candidate_user_performance_gate':c['user_performance_gate'],'selected_candidate':None,'training_receipt_sha256':sha(b/'retrieval/final_receipt.json'),'evaluation_receipt_sha256':sha(b/'evaluation_v1/final_receipt.json'),'historical_cnn_retrained':False,'historical_cnn_reevaluated':False,'native_workers_exit0':12,'pipeline_exit0':2,'wrapper_own_exit_directly_observed':False,'new_remote_calls_in_this_check':0,'automation_recreated':False,'new_candidate_training_started':False,'continuation_scope':'User requested manual completion of current RAF6 after deleting the recurring automation. No new loop or dataset expansion performed.','verification_script_sha256':sha(out/'verify.py'),'files':files}
with (out/'RECEIPT.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'},ensure_ascii=False));print('receipt_sha256',sha(out/'RECEIPT.json'))
