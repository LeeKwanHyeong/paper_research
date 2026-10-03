"""Sync reviewed rows into the existing app without replacing identity or presentation."""
import json
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
s=read(P/'app/src/data.json');v=read(P/'comparison.json');reviewed=read(P/'reviewed_snapshot.json');s['queries'].update(reviewed['queries'])
labels={'yellow_trip_hourly':'Taxi','intermittent_frozen_5000':'Intermittent','insta_market_basket':'Instacart','raf_spare_parts':'RAF'}
models=['titantpp_history_mlp','titantpp_history_mlp_active_norm','rmtpp','thp','nhp','sahp','s2p2_matched_head','attnhp_matched_head']
files=list(read(P/'verification.json')['source_files'])
for qid in ['seed42','all_completed']:
 q=s['queries'][qid];q['rows']=[dict(r,dataset=labels[r['dataset']]) for r in v['rows'] if qid=='all_completed' or (r['seed']==42 and r['model'] in models)]
 q['source']['files']=files;q['source']['caveats']=['Validation only. RAF and completed 5080 CPU audits passed; 5090 final audit pending.']
for qid,q in s['queries'].items():
 q['source']['evidenceScope']='RAF completed 2026-10-01 08:55:56 KST; additional TPP observed 08:59 KST; prior completed Core/Gate records reused.'
 q['source'].pop('evidenceAsOf',None)
s['generatedAt']=datetime.now(timezone.utc).isoformat();s['buildStatus']='complete'
(P/'app/src/data.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
