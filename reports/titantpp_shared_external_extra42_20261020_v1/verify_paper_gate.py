"""Read-only local final raw9/frozen Intermittent identity check; never contacts GPUs."""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
REPORT=Path(__file__).parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def main():
 os.environ.setdefault('MPLCONFIGDIR','/tmp/paper_research_paper_gate_matplotlib')
 from paper.scripts import manage_titantpp_shared_history_raw9_5090 as m
 from simple_lab_test.search.common.runner import canonical_state_dict_sha256,torch_load_checkpoint
 final=m.BUNDLE/'retrieval/final_receipt.json'
 if not final.exists():
  print(json.dumps({'status':'waiting_for_original_5090_final_receipt','remote_calls':0,'paper_started':False}));return
 c=m.verify_local();r=read(final)
 require(r['status']=='complete' and r['new_fit_count']==9 and r['validation_replay_count']==9
  and r['actual_scientific_worker_exit0_count']==18 and r['all_training_terminal_verified'] is True
  and r['held_out_test_evaluated'] is False,'Original raw9 final counts incomplete')
 for key in ('source_bindings','archive_bindings','report_files'):
  require(r[key], 'Empty final evidence inventory')
  for path,digest in r[key].items():require(sha(path)==digest,'Original final evidence changed: '+path)
 originals=m.BUNDLE/'retrieval/5090/originals';m.verify_collected(c,'5090',originals)
 gate=read(REPORT/'SCIENCE_SPEC.json')['paper_start_gate'];frozen=gate['snapshot_final_receipt']
 require(sha(ROOT/frozen['path'])==frozen['sha256'],'Frozen Test receipt changed')
 snapshot=read(ROOT/frozen['path'])
 for key in ('source_bindings','report_files','result_files'):
  for path,digest in snapshot.get(key,{}).items():require(sha(path)==digest,'Frozen Test evidence changed')
 collection=ROOT/Path(frozen['path']).parent/'retrieval'
 receipt=read(collection/'receipt.json')
 require(receipt['status']=='verified' and receipt['owner_absence_verified'] is True,'Frozen evaluation collection incomplete')
 require(sha(collection/'receipt.json')==snapshot['collection_receipt_sha256'],'Frozen collection receipt changed')
 require(sha(collection/'original.tar.gz')==receipt['archive_sha256'] and sha(collection/'manifest.json')==receipt['manifest_sha256'],'Frozen archive/manifest changed')
 inventory=read(collection/'manifest.json')['files']
 require(inventory and len(inventory)==receipt['file_count'],'Frozen inventory incomplete')
 for path,digest in inventory.items():require(sha(collection/'originals'/path)==digest,'Frozen evaluation original changed')
 rows=[]
 for j in c['jobs']:
  if j['dataset']!='intermittent_frozen_5000':continue
  r=read(originals/'run'/j['id']/'validation_receipt.json');f=gate['frozen_selected_states'][str(j['seed'])]
  remote=Path(c['hosts']['5090']['root']);checkpoint=originals/Path(r['selected_checkpoint']).relative_to(remote)
  require(sha(checkpoint)==r['checkpoint_sha256'],'Final selected binary differs from receipt')
  payload=torch_load_checkpoint(checkpoint,map_location='cpu');state=payload['model_state_dict']
  digest=canonical_state_dict_sha256(state)
  same=r['selected_epoch']==f['epoch'] and digest==f['state_sha256']
  rows.append({'seed':j['seed'],'final_epoch':r['selected_epoch'],'frozen_epoch':f['epoch'],
   'final_state_sha256':digest,'frozen_state_sha256':f['state_sha256'],'matches':same,
   'validation_replay_passed':r['validation_replay_passed']})
 require(len(rows)==3,'Missing Intermittent seeds')
 result={'status':'PASS' if all(x['matches'] and x['validation_replay_passed'] for x in rows) else 'MISMATCH',
  'rows':rows,'original_final_receipt_sha256':sha(final),'frozen_test_final_receipt_sha256':frozen['sha256'],
  'canonical_tensors_compared':True,'serialized_binary_equality_required':False,'remote_calls':0,
  'paper_may_start':all(x['matches'] and x['validation_replay_passed'] for x in rows)}
 target=REPORT/'PAPER_START_GATE.json'
 if target.exists():require(read(target)==result,'Existing paper gate differs; preserve original')
 else:
  with target.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
