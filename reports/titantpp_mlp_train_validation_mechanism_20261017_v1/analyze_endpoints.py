"""Recompute 3-seed Validation summaries from sealed selected endpoints only."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
inventory = json.loads((OUT/'SOURCE_INVENTORY.json').read_text())
rows=[]
inputs={}


def check(rel, expected):
    path=ROOT/rel
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual==expected,(rel,actual,expected)
    inputs[rel]=actual
    return path


def write_csv(name, values):
    fields=list(dict.fromkeys(k for r in values for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)


for item in inventory['rows']:
    path=check(item['endpoint_path'],item['endpoint_sha256'])
    e=json.loads(path.read_text())
    s=e['selected']
    assert s['held_out_test_evaluated'] is False and s['evaluation_scope']=='validation_only'
    assert e['best_epoch']==item['selected_epoch'] and s['state_sha256']==item['state_tensor_sha256']
    assert s['count']==item['target_count']
    cells=[('overall','all',s)]
    cells += [('quantity_bin',str(c['bin']),c) for c in s['quantity_cells']]
    cells += [('history_bin',str(c['bin']),c) for c in s['history_cells']]
    cells += [('tail','tail',s['tail'])]
    for axis,label,c in cells:
        if not c['count']:continue
        row=dict(dataset=item['dataset'],model=item['model'],seed=item['seed'],selected_epoch=item['selected_epoch'],
                 axis=axis,group=label,count=c['count'],rmse=c['qty_rmse'],mae=c['qty_mae'],time_nll=c['time_nll'],
                 sse=c['qty_rmse']**2*c['count'],sae=c['qty_mae']*c['count'],
                 quantity_boundaries=json.dumps(s['quantity_boundaries']),history_boundaries=json.dumps(s['history_boundaries']))
        assert all(math.isfinite(row[k]) for k in ['rmse','mae','time_nll','sse','sae'])
        rows.append(row)
    for cells_key in ['quantity_cells','history_cells']:
        assert sum(c['count'] for c in s[cells_key])==s['count']
        for metric,power in [('qty_rmse',2),('qty_mae',1),('time_nll',1)]:
            total=sum(c[metric]**power*c['count'] for c in s[cells_key] if c['count'])
            expected=s[metric]**power*s['count']
            assert math.isclose(total,expected,rel_tol=1e-7,abs_tol=1e-6),(item,metric,total,expected)

index={(r['dataset'],r['model'],r['seed'],r['axis'],r['group']):r for r in rows}
pairs=[('titantpp_history_mlp_width16','titantpp_history_mlp'),
       ('titantpp_all_available_history_mlp','titantpp_history_mlp'),
       ('titantpp_history_mlp','titantpp_current_only_param_matched'),
       ('titantpp_history_mlp_width16','titantpp_current_only_param_matched')]
deltas=[]
for a,b in pairs:
    for key, r in index.items():
        ds,model,seed,axis,group=key
        if model!=a:continue
        ref=index[ds,b,seed,axis,group]
        assert r['count']==ref['count'] and r['quantity_boundaries']==ref['quantity_boundaries'] and r['history_boundaries']==ref['history_boundaries']
        d=dict(dataset=ds,candidate=a,reference=b,seed=seed,axis=axis,group=group,count=r['count'])
        for metric in ['rmse','mae','time_nll','sse','sae']:
            d['delta_'+metric]=r[metric]-ref[metric]
            d['candidate_'+metric]=r[metric];d['reference_'+metric]=ref[metric]
        deltas.append(d)

summary=[]
groups=defaultdict(list)
for r in rows:groups[r['dataset'],r['model'],r['axis'],r['group']].append(r)
for key, group in groups.items():
    assert {r['seed'] for r in group}=={42,52,62}
    o=dict(zip(['dataset','model','axis','group'],key));o['count_per_seed']=group[0]['count']
    for metric in ['rmse','mae','time_nll','sse','sae']:
        v=[r[metric] for r in group];o[metric+'_mean']=st.mean(v);o[metric+'_sample_sd']=st.stdev(v)
    summary.append(o)
delta_summary=[]
groups=defaultdict(list)
for r in deltas:groups[r['dataset'],r['candidate'],r['reference'],r['axis'],r['group']].append(r)
for key,group in groups.items():
    assert {r['seed'] for r in group}=={42,52,62}
    o=dict(zip(['dataset','candidate','reference','axis','group'],key));o['count_per_seed']=group[0]['count']
    for metric in ['rmse','mae','time_nll','sse','sae']:
        v=[r['delta_'+metric] for r in group]
        o['delta_'+metric+'_mean']=st.mean(v);o['delta_'+metric+'_sample_sd']=st.stdev(v)
        o['improved_'+metric+'_seeds']=sum(x<0 for x in v)
    delta_summary.append(o)
write_csv('VALIDATION_ENDPOINTS.csv',rows)
write_csv('VALIDATION_SUMMARY.csv',summary)
write_csv('VALIDATION_SEED_DELTAS.csv',deltas)
write_csv('VALIDATION_DELTA_SUMMARY.csv',delta_summary)
(OUT/'ENDPOINT_INPUT_MANIFEST.json').write_text(json.dumps(inputs,indent=2)+'\n')
print(json.dumps({'endpoints':len(inventory['rows']),'cells':len(rows),'summary_groups':len(summary),'paired_deltas':len(deltas),'checks':'selection/state/count/quantity-history additive checks passed'}))
