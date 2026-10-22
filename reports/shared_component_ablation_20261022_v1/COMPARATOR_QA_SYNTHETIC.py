"""Independent fabricated collection integration QA; local CPU only.

This builds temporary evidence shaped like both hosts' 36 jobs, 72 prediction
files, 216 cost receipts, archive transport, raw exits, and global Validation
gates. All numbers and checkpoints are manufactured test fixtures, not research
results or GPU measurements. No remote command or real dataset is executed.
The actual comparator CLI performs check-only/create/cache checks against these
fixtures. Run from the repository with PYTHONDONTWRITEBYTECODE=1.
"""
from pathlib import Path
REPOSITORY = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPOSITORY))
from pathlib import Path
import tempfile,json,gzip,tarfile,copy,hashlib
import torch
import numpy as np
from paper.scripts import compare_shared_component_ablation as c
from paper.scripts import manage_shared_component_ablation as m
from paper.tests.test_manage_shared_component_ablation import contract as cf, make_gate
from simple_lab_test.search.common.runner import canonical_state_dict_sha256
from paper.scripts.profile_shared_component_ablation import tensor_sha

def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True,allow_nan=False))
def packed(base,contract,host):
 hd=base/'test_cost'/host;root=hd/'originals';man=root/'collection_test_cost.manifest.json'
 inv={str(p.relative_to(root)):{'sha256':c.sha(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file() and p!=man}
 manifest={'schema':'shared_component_ablation_collection_v1','host':host,'phase':'test_cost','contract_sha256':c.canonical(contract),'files':inv,'source_files':contract['source']['files'],'input_files':contract['input_files'],'legacy_files':contract.get('legacy_files',{}),'all_source_input_legacy_actual_sha_verified':True}
 write(man,manifest)
 archive=hd/'collection_test_cost.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for rel in [*inv,'collection_test_cost.manifest.json']:tar.add(root/rel,arcname=rel)
 receipt={'status':'complete','host':host,'phase':'test_cost','contract_sha256':c.canonical(contract),'archive_sha256':c.sha(archive),'archive_bytes':archive.stat().st_size,'manifest_sha256':c.sha(man),'files':len(inv)}
 write(hd/'collection_receipt.json',receipt);write(hd/'pack.stdout',receipt);(hd/'pack.stderr').write_bytes(b'')
 write(hd/'transport.json',{'actual_os_returncode':0,'timeout':False,'stdout_sha256':c.sha(hd/'pack.stdout'),'stderr_sha256':c.sha(hd/'pack.stderr'),'stdout_bytes':(hd/'pack.stdout').stat().st_size,'stderr_bytes':0,'transport_error':None})
 (hd/'transfer.stdout').write_bytes(b'');(hd/'transfer.stderr').write_bytes(b'')
 write(hd/'transfer.json',{'actual_os_returncode':0,'timeout':False,'transport_error':None,'stdout_sha256':hashlib.sha256(b'').hexdigest(),'stderr_sha256':hashlib.sha256(b'').hexdigest(),'stdout_bytes':0,'stderr_bytes':0})
def fixture(base,relative=True):
 contract=cf.__wrapped__();contract['datasets']=[{'dataset_id':d,'quantity_boundaries_all_train_rows':[10.]} for d in m.DATASETS]
 contract['legacy_files']={}
 population={'target_count':3,'target_identity_sha256':c.canonical(['part',10,11,12]),'target_quantity_sha256':hashlib.sha256(b'hard_lmm_target_quantity_v1\0'+np.asarray([1.,10.,20.],dtype='<f8').tobytes()).hexdigest()}
 for data in contract['datasets']:
  data['inherited_data_identity']={'data':{'sha256':'0'*64},'populations':{'validation':population}}
  data['full_data_identity']={'populations':{'test':population}}

 for host in contract['hosts']:contract['hosts'][host]={'root':'/remote/'+host,'gpu_uuid':'GPU-'+host,'source_root':'/remote/'+host+'/source'}
 for job in contract['jobs']:
  if job.get('reuse'):job['reuse']={'verified':True}
 state={'x':torch.tensor([1.,2.])};state_sha=canonical_state_dict_sha256(state)
 for host in contract['hosts']:
  root=base/'test_cost'/host/'originals';root.mkdir(parents=True)
  make_gate(contract,root)
  for d in m.DATASETS:
   manifest={'dataset':d,'split':'train','synthetic':False,'contract_sha256':c.canonical(contract),'preparation_seed':42,'target_ids_sha256':c.canonical([[str(x),x] for x in range(4736)]),'batches':[{'batch':i,'phase':'warmup' if i<5 else 'measured','target_split':'train','tensor_sha256':str(i).zfill(64),'targets':[[str(i*128+x),i*128+x] for x in range(128)]} for i in range(37)]}
   batches=[{'dts':torch.ones((128,2))*(i+1),'quantities':torch.ones((128,2))*i,'mask':torch.ones((128,2),dtype=torch.bool)} for i in range(37)]
   for i,b in enumerate(batches):manifest['batches'][i]['tensor_sha256']=tensor_sha(b)
   cachepath=root/'cost/batches'/f'{d}.pt';cachepath.parent.mkdir(parents=True,exist_ok=True);torch.save(batches,cachepath);manifest['file_sha256']=c.sha(cachepath);manifest['input_sha256']='0'*64
   write(root/'cost/batches'/f'{d}.json',manifest)
  entries=[];assigned=[j for j in contract['jobs'] if j['host']==host]
  for job in assigned:
   folder=root/'run'/job['id'];folder.mkdir(parents=True)
   checkpoint=folder/'checkpoint.pt';torch.save({'model_state_dict':state,'epoch':1,'selected_epoch':1,'backbone':job['arm']},checkpoint)
   selected=str(checkpoint.relative_to(root));selected=selected if relative else '/remote/'+host+'/'+selected
   train={'status':'complete','stage':'train','host':host,'job_id':job['id'],'dataset':job['dataset'],'arm':job['arm'],'seed':job['seed'],'contract_sha256':c.canonical(contract),'selected_checkpoint':selected,'checkpoint_sha256':c.sha(checkpoint),'state_sha256':state_sha,'selected_epoch':1,'training_origin':'reused_verified_fit' if job.get('reuse') else 'fresh_fit','summary':{'elapsed_seconds':20.,'completed_epochs':2,'best_epoch':1,'parameter_count':2},'new_training_epochs':0 if job.get('reuse') else 2}
   if job.get('reuse'):train['reuse']=job['reuse'];train['summary']={'status':'reused_verified_fit','best_epoch':1}
   write(folder/'train_receipt.json',train)
   for split in ('validation','test'):
    pred=folder/(split+'_predictions.jsonl.gz')
    rows=[dict(target_index=i,entity_id='part',target_seq=i+10,split=split,raw_quantity=float(v),predicted_raw_quantity=float(v)+(0. if job['arm']=='no_prototype' and job['seed']==42 else (m.ARMS.index(job['arm'])+1)*(job['seed']/10)),time_nll=float(i+1),history_length=8) for i,v in enumerate([1,10,20])]
    with gzip.open(pred,'wt') as z:
     for row in rows:z.write(json.dumps(row)+'\n')
    metrics=c.prediction_metrics(pred,10,split); metrics={'overall':metrics['overall'],'tail':metrics['large']}
    receipt={**train,'stage':split,'prediction_path':'/remote/'+host+'/'+str(pred.relative_to(root)),'prediction_rows':3,'target_identity':population,'metrics':metrics,'parameters_unchanged':True,'validation_replay_passed':True,'individual_predictions_match':True}
    write(folder/(split+'_receipt.json'),receipt)
    if split=='validation':
     train['validation_reference']={'target_identity':population,'metrics':metrics,'prediction_path':receipt['prediction_path'],'prediction_rows':3};write(folder/'train_receipt.json',train)

   for mode in ('train','eval'):
    for rep in (1,2,3):
     batch=root/'cost/batches'/f"{job['dataset']}.json";manifest=c.read(batch)
     cost={'status':'complete','host':host,'mode':mode,'timing_rep':rep,'contract_sha256':c.canonical(contract),'batch_manifest_sha256':c.sha(batch),'target_ids_sha256':manifest['target_ids_sha256'],'batch_tensor_sha256':[x['tensor_sha256'] for x in manifest['batches']],'checkpoint_binary_unchanged':True,'checkpoint_saved':False,'input_split':'train','synthetic':False,'runtime':{'gpu':{'uuid':'GPU-'+host}},'selected':{'checkpoint_sha256':train['checkpoint_sha256'],'state_sha256':state_sha,'selected_epoch':1},'job_id':job['id'],'dataset':job['dataset'],'arm':job['arm'],'training_seed':job['seed'],'parameters':2,'optimizer_steps':37 if mode=='train' else 0,'preparation_seed':42,'scientific_fit':False,'accuracy_evaluated':False,'rows':[{'batch':i,'phase':'warmup' if i<5 else 'measured','targets':128,'transfer_seconds':.01,'compute_seconds':.1,'transfer_compute_seconds':.11} for i in range(37)],'median_compute_seconds':.1,'median_transfer_compute_seconds':.11,'throughput_targets_per_second':1280.,'throughput_including_transfer_targets_per_second':128/.11,'memory_after_warmup_baseline_bytes':{'allocated':100,'reserved':200},'warmup_peak_bytes':{'allocated':150,'reserved':250},'measured_peak_bytes':{'allocated':160,'reserved':260},'total_peak_bytes':{'allocated':160,'reserved':260},'measured_incremental_peak_bytes':{'allocated':60,'reserved':60}}
     write(root/'cost/results'/job['id']/f'{mode}_rep{rep}.json',cost)
   stages=['reuse' if job.get('reuse') else 'train','validation','test']+[f'cost_{mode}_{rep}' for mode in ('train','eval') for rep in (1,2,3)]
   if job==assigned[0]:stages+=['prepare_batches']
   for stage in stages:
    stem=root/'control'/f"{job['id']}.{stage}";stem.parent.mkdir(exist_ok=True)
    for stream in ('stdout','stderr'):Path(str(stem)+'.'+stream+'.log').write_bytes(b'')
    e={'job_id':job['id'],'host':host,'stage':stage,'contract_sha256':c.canonical(contract),'actual_os_returncode':0,'timeout':False,'supervisor_error':None,'resource_incident':None,'stdout_sha256':hashlib.sha256(b'').hexdigest(),'stderr_sha256':hashlib.sha256(b'').hexdigest()}
    write(Path(str(stem)+'.exit.json'),e)
    if stage not in ('reuse','train','validation'):entries.append(e)
  write(root/'test_cost.complete.json',{'status':'complete','host':host,'phase':'test_cost','contract_sha256':c.canonical(contract),'actual_child_exits':entries})
  for stream in ('stdout','stderr'):(root/('test_cost.'+stream+'.log')).write_bytes(b'')
  write(root/'test_cost.supervisor_exit.json',{'host':host,'phase':'test_cost','contract_sha256':c.canonical(contract),'actual_os_returncode':0,'stdout_sha256':hashlib.sha256(b'').hexdigest(),'stderr_sha256':hashlib.sha256(b'').hexdigest()})
  packed(base,contract,host)
 for host in contract['hosts']:
  root=base/'test_cost'/host/'originals';gate=c.read(root/'global_validation_gate.json')
  for job in contract['jobs']:
   original=base/'test_cost'/job['host']/'originals'
   for stage in ('train','validation'):
    for kind in ('receipt','exit'):
     actualstage='reuse' if stage=='train' and job.get('reuse') else stage
     path=(original/'run'/job['id']/(stage+'_receipt.json')) if kind=='receipt' else original/'control'/(job['id']+'.'+actualstage+'.exit.json')
     item=gate['jobs'][job['id']][stage+'_'+kind];destination=root/item['path'];destination.write_bytes(path.read_bytes());item['sha256']=c.sha(destination)
  write(root/'global_validation_gate.json',gate);packed(base,contract,host)
 return contract


import sys,os,json,tempfile,time,subprocess,hashlib,tarfile,io,gzip,statistics,math
from pathlib import Path
from types import SimpleNamespace
q = SimpleNamespace(c=c, fixture=fixture, write=write, packed=packed)
checks=[]
def rejected(label,action):
 try:action()
 except (ValueError,RuntimeError,KeyError,FileNotFoundError) as e:checks.append({'name':label,'passed':True,'rejection':type(e).__name__+': '+str(e)});return
 raise AssertionError('INVALID EVIDENCE ACCEPTED: '+label)
def negative_raw_timing(value):
 value['rows'][5]['compute_seconds']=-.1;value['rows'][5]['transfer_compute_seconds']=-.09
 value['throughput_targets_per_second']=4096/sum(r['compute_seconds'] for r in value['rows'][5:])
 value['throughput_including_transfer_targets_per_second']=4096/sum(r['transfer_compute_seconds'] for r in value['rows'][5:])
def main():
 base=Path(tempfile.mkdtemp(prefix='component_comparator_final_'));ct=q.fixture(base,relative=True);q.write(base/'contract.json',ct)
 result=c.aggregate(ct,base)
 assert len(result['records'])==72 and len(result['cost_records'])==216 and len(result['paired_seed_deltas'])==432 and len(result['table'])==24
 first=next(x for x in result['table'] if x['dataset']=='raf_spare_parts' and x['split']=='test' and x['arm']=='full')
 assert math.isclose(first['overall']['qty_rmse']['mean'],statistics.mean([4.2,5.2,6.2]))
 assert math.isclose(first['overall']['qty_rmse']['sample_sd_ddof1'],statistics.stdev([4.2,5.2,6.2]))
 assert all(x['metrics']['large']['count']==1 for x in result['records'])
 assert result['adverse_seed_results']
 checks.append({'name':'actual_full_collection_and_known_mean_sample_sd_tail_boundary','passed':True,'records':72,'cost_records':216,'paired_seed_rows':432,'table_rows':24})
 root=base/'test_cost/5080/originals';job=ct['jobs'][0]
 cost=root/'cost/results'/job['id']/'train_rep1.json';test=root/'run'/job['id']/'test_receipt.json'
 original=cost.read_bytes();valid=c.read(cost);valid['warmup_peak_bytes']['reserved']=210;valid['measured_peak_bytes']['reserved']=220;valid['total_peak_bytes']['reserved']=220;valid['measured_incremental_peak_bytes']['reserved']=20;q.write(cost,valid);q.packed(base,ct,'5080');c.aggregate(ct,base);checks.append({'name':'legitimate_reserved_delta_below_allocated_delta_accepted','passed':True});cost.write_bytes(original);q.packed(base,ct,'5080')
 cases=[('eval_wrong_selected',test,lambda v:v.update(checkpoint_sha256='0'*64)),('eval_wrong_metrics',test,lambda v:v['metrics']['overall'].update(qty_rmse=999)),('cost_wrong_tensor',cost,lambda v:v.update(batch_tensor_sha256=['0'*64]*37)),('cost_negative_timing',cost,lambda v:v.update(median_compute_seconds=-1)),('cost_fabricated_positive_timing',cost,lambda v:v.update(median_compute_seconds=999)),('cost_negative_raw_timing_with_consistent_positive_summary',cost,negative_raw_timing),('cost_incomplete_rows',cost,lambda v:v['rows'].pop()),('cost_wrong_optimizer_steps',cost,lambda v:v.update(optimizer_steps=0)),('cost_negative_memory',cost,lambda v:v.update(total_peak_bytes={'allocated':-1,'reserved':-1})),('cost_wrong_total_peak',cost,lambda v:v.update(total_peak_bytes={'allocated':9999,'reserved':9999})),('cost_zero_parameters',cost,lambda v:v.update(parameters=0)),('eval_changed_frozen_population',test,lambda v:v['target_identity'].update(target_identity_sha256='0'*64))]
 for label,path,change in cases:
  original=path.read_bytes();v=c.read(path);change(v);q.write(path,v);q.packed(base,ct,'5080')
  rejected(label,lambda:c.aggregate(ct,base))
  path.write_bytes(original);q.packed(base,ct,'5080')
 pred=root/'run'/job['id']/'test_predictions.jsonl.gz';original=pred.read_bytes()
 with gzip.open(pred,'rt') as z:rows=[json.loads(x) for x in z]
 rows[1]['entity_id']=rows[0]['entity_id'];rows[1]['target_seq']=rows[0]['target_seq']
 with gzip.open(pred,'wt') as z:
  for row in rows:z.write(json.dumps(row)+'\n')
 q.packed(base,ct,'5080');rejected('duplicate_entity_sequence_with_consecutive_indices',lambda:c.aggregate(ct,base));pred.write_bytes(original);q.packed(base,ct,'5080')
 hd=base/'test_cost/5080';tr=hd/'transport.json';old=tr.read_bytes();v=c.read(tr);v['timeout']=True;q.write(tr,v);rejected('transport_timeout_rejected',lambda:c.aggregate(ct,base));tr.write_bytes(old)
 transfer=hd/'transfer.json';old=transfer.read_bytes();v=c.read(transfer);v['actual_os_returncode']=1;q.write(transfer,v);rejected('transfer_nonzero_rejected',lambda:c.aggregate(ct,base));transfer.write_bytes(old)
 archive=hd/'collection_test_cost.tar.gz'
 with tarfile.open(archive,'r:gz') as t:entries=[(m,t.extractfile(m).read()) for m in t.getmembers()]
 done=False
 with tarfile.open(archive,'w:gz') as t:
  for m,b in entries:
   if not done and m.name.endswith('checkpoint.pt'):b=b'X'*len(b);done=True
   t.addfile(m,io.BytesIO(b))
 v=c.read(hd/'collection_receipt.json');v.update(archive_sha256=c.sha(archive),archive_bytes=archive.stat().st_size);q.write(hd/'collection_receipt.json',v);q.write(hd/'pack.stdout',v);v=c.read(tr);v.update(stdout_sha256=c.sha(hd/'pack.stdout'),stdout_bytes=(hd/'pack.stdout').stat().st_size);q.write(tr,v)
 rejected('archive_same_length_payload_substitution',lambda:c.aggregate(ct,base));q.packed(base,ct,'5080')
 # Alter an entire host cache consistently; paired hosts must still share targets.
 hostroot=base/'test_cost/5090/originals';manifestpath=hostroot/'cost/batches/raf_spare_parts.json';backup={manifestpath:manifestpath.read_bytes()};manifest=c.read(manifestpath);
 for batchrow in manifest['batches']:
  batchrow['targets']=[[entity,seq+100000] for entity,seq in batchrow['targets']]
 manifest['target_ids_sha256']=c.canonical([target for batchrow in manifest['batches'] for target in batchrow['targets']]);q.write(manifestpath,manifest)
 for job5090 in ct['jobs']:
  if job5090['host']=='5090' and job5090['dataset']=='raf_spare_parts':
   for path in (hostroot/'cost/results'/job5090['id']).glob('*.json'):
    backup[path]=path.read_bytes();v=c.read(path);v.update(target_ids_sha256=manifest['target_ids_sha256'],batch_manifest_sha256=c.sha(manifestpath));q.write(path,v)
 q.packed(base,ct,'5090');rejected('cross_host_benchmark_target_drift',lambda:c.aggregate(ct,base))
 for path,content in backup.items():path.write_bytes(content)
 q.packed(base,ct,'5090')
 # Actual CLI checks in independent processes with different Python hash ordering.
 command=[sys.executable,'-m','paper.scripts.compare_shared_component_ablation','--contract',str(base/'contract.json'),'--contract-sha',c.sha(base/'contract.json'),'--retrieval',str(base),'--output',str(base/'report')]
 outputs=[]
 for index,args in enumerate((['--check-only'],[],['--check-only'])):
  env=dict(os.environ,PYTHONHASHSEED=str(index+1),PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/component-qa-mpl');r=subprocess.run(command+args,text=True,capture_output=True,env=env,cwd=str(REPOSITORY))
  outputs.append({'command':command+args,'actual_os_returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});assert r.returncode==0,outputs[-1]
 checks.append({'name':'cli_check_create_cache_across_hashseeds','passed':True,'actual_processes':outputs})
 csv=base/'report/TABLE.csv';original=csv.read_bytes();csv.write_bytes(original+b'bad\n')
 r=subprocess.run(command+['--check-only'],text=True,capture_output=True,cwd=str(REPOSITORY),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/component-qa-mpl'));assert r.returncode!=0 and 'Cached CSV differs' in r.stderr;csv.write_bytes(original)
 checks.append({'name':'cache_modified_csv_rejected','passed':True,'actual_os_returncode':r.returncode})
 return {'status':'PASS','fixture_root':str(base),'checks':checks,'remote_calls':0,'gpu_calls':0}
if __name__=='__main__':
 start=time.time()
 try:result=main()
 except Exception as e:
  result={'status':'FAIL','checks':checks,'error':type(e).__name__+': '+str(e)};q.write(Path('/tmp/shared_component_comparator_final_result.json'),result);raise
 result['elapsed_seconds']=time.time()-start;q.write(Path('/tmp/shared_component_comparator_final_result.json'),result);print(json.dumps({'status':result['status'],'checks':len(checks),'elapsed_seconds':result['elapsed_seconds']}))
