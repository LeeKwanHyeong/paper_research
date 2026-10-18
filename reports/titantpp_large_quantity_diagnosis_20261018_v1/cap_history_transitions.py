"""Train/Validation-only descriptive transitions; no model or checkpoint loading."""
from pathlib import Path
import csv,hashlib,json,math,sys
import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
CONTRACT=ROOT/'search_artifacts/titantpp_mlp_cap36_20261017_v2/execution_contract.json'
SNAPSHOT=CONTRACT.parent/'input_snapshot'
INPUTS={}

def read_bound(path,expected=None):
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None: assert digest==expected,str(path)
    INPUTS[str(path.relative_to(ROOT))]=digest
    return path

def rmse_mae(q,p):
    e=p-q
    return {'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),'bias':float(np.mean(e))}

def run():
    from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    from paper.scripts.run_titantpp_mlp_cap36 import population_identity
    c=json.loads(read_bound(CONTRACT).read_text())
    rows=[]; summaries=[]; checks=[]
    for d in c['datasets']:
        binding=d['inherited_data_identity']['data']
        raw=pl.read_parquet(read_bound(SNAPSHOT/binding['path'],binding['sha256']))
        assert set(raw['chronological_split'].unique().to_list())=={'train','validation'}
        assert raw['demand_qty'].is_finite().all() and raw['demand_qty'].min()>=0 and raw['demand_qty'].max()<=d['cap_value']
        frame=prepare_count_frame(raw.sort(['oper_part_no','seq']))
        cap=float(d['cap_value']);q50=float(d['quantity_boundaries_all_train_rows'][0])
        for split in ('train','validation'):
            ds=RMTPPWeekLookbackDataset(frame,lookback_weeks=d['loader']['lookback_weeks'],max_seq_len=d['loader']['max_seq_len'],mode='all',split_col='chronological_split',target_splits={split})
            q,ident=population_identity(ds,split)
            assert ident==d['inherited_data_identity']['populations'][split]
            last=np.array([ds.val_lists[p][i] for p,i in ds.index],dtype=np.float64)
            entities=np.array([str(ds.parts[p]) for p,i in ds.index])
            assert len(last)==len(q) and np.isfinite(last).all()
            hit=q==cap;prev=last==cap
            masks={'all':np.ones(len(q),bool),'last_below_train_q50':last<q50,
              'last_q50_to_below_cap':(last>=q50)&(last<cap),'last_at_cap':prev}
            for name,mask in masks.items():
                n=int(mask.sum())
                if not n:
                    rows.append({'dataset':d['dataset_id'],'split':split,'condition':name,'n':0,'status':'N/A_empty'})
                    continue
                a,b=q[mask],last[mask]
                rows.append({'dataset':d['dataset_id'],'split':split,'condition':name,'n':n,'entities':int(len(np.unique(entities[mask]))),'cap':cap,'train_q50':q50,
                  'next_at_cap_n':int((a==cap).sum()),'next_at_cap_percent':float(100*np.mean(a==cap)),
                  'next_quantity_mean':float(a.mean()),'next_quantity_median':float(np.median(a)),
                  'next_quantity_sd':float(a.std(ddof=1)) if len(a)>1 else None,
                  **{'last_reference_'+k:v for k,v in rmse_mae(a,b).items()}})
            n_hit=int(hit.sum()); n_prev=int(prev.sum());both=int((hit&prev).sum());from_low=int((hit&~prev).sum())
            assert n_hit==both+from_low
            one={'dataset':d['dataset_id'],'split':split,'cap':cap,'target_count':len(q),'target_identity':ident,
              'at_cap_count':n_hit,'at_cap_percent':float(100*hit.mean()),
              'previous_at_cap_count':n_prev,'repeat_cap_count':both,'new_cap_after_below_count':from_low,
              'cap_targets_with_previous_below_cap_percent':100*from_low/n_hit if n_hit else None,
              'cap_targets_with_previous_below_halfcap_percent':float(100*np.mean(last[hit]<cap/2)) if n_hit else None,
              'conditional_next_cap_percent_given_previous_cap':100*both/n_prev if n_prev else None,
              'previous_cap_then_below_count':int((prev&~hit).sum()),
              'last_reference_overall':rmse_mae(q,last),
              'last_reference_at_cap':rmse_mae(q[hit],last[hit]) if n_hit else None,
              'last_reference_below_cap':rmse_mae(q[~hit],last[~hit]) if (~hit).any() else None}
            # The constant ceiling is deliberately an accounting reference, not a model recommendation.
            one['constant_cap_reference_overall']=rmse_mae(q,np.full(len(q),cap))
            summaries.append(one)
            checks.append({'dataset':d['dataset_id'],'split':split,'population_exact':True,'transition_partition_exact':True,
                'condition_partition_n':sum(r['n'] for r in rows if r['dataset']==d['dataset_id'] and r['split']==split and r['condition']!='all'),'expected':len(q)})
            assert checks[-1]['condition_partition_n']==len(q)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'CAP_HISTORY_TRANSITIONS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    r={'status':'complete','scope':'official_Q3_capped_Train_Validation_only','thresholds':'Cap Train Q3 and Train Q50 fixed before these cuts','dataset_count':3,'cohort_count':6,'summaries':summaries,'checks':checks,'inputs':INPUTS,
       'new_model_training':False,'new_model_inference':False,'remote_calls':0,'held_out_rows_opened':0,
       'interpretation_limit':'One observed last quantity is not the entire information set. Transition uncertainty does not prove irreducible Bayes error. No inferred causal attribution.'}
    (OUT/'CAP_HISTORY_TRANSITIONS.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
if __name__=='__main__':run()
