"""Freeze the approved head-only design before any real-data fitting."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P = ROOT / 'reports/phen_intermittent_full_observations_20261026_v1'

def ref(path, expected=None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)

def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def main():
    assert not (HERE/'SCIENCE_SPEC.json').exists(), 'Immutable contract already exists'
    bindings = {
        'authorization':ref(HERE/'AUTHORIZATION.json'),
        'collection':ref(P/'FINAL_COLLECTION.json','18a9f78e3a0355d502fbd7789161ea6ef1ddcb9ca1fc7410397ee6672b1b0461'),
        'collection_qa':ref(P/'POSTRUN_FINAL_INVENTORY_QA.json','f70309c7e554e6eaba12b71692d267ba8c9384e25f7a274fcdfd7e1d06e0054d'),
        'metadata':ref(P/'support/results_v1/target_metadata.parquet','dcee4a92d729a1937b1f47e64b317e49e4043adf6dfce882fa0ace8398b5edb8'),
        'support_cells':ref(P/'support/results_v1/support_cells_full_primary_exact.csv.gz','bf16af48055ebc1d628a9d80cf6c068c02203d07da0b15ecda2cd124af72677b'),
        'native_probability_helper':ref(P/'execution_v6/bundle/source42/models/TPPs/positive_integer_time.py','783113642d897aaed3da1213fc91fb993ee71af380f953a1159142a1cda8ae59'),
    }
    meta=pd.read_parquet(bindings['metadata']['path'])
    train=meta.loc[meta.split.eq('train')].sort_values('target_index').copy()
    assert len(train)==393824 and train.entity_id.nunique()==5000
    assert not train.duplicated(['entity_id','target_seq']).any()
    train['role']=''
    entity_rows=[]
    for entity,part in train.sort_values(['entity_id','target_seq','target_index']).groupby('entity_id',sort=False):
        n=len(part); assert n>=2
        cut=int(np.floor(.8*n))
        train.loc[part.index[:cut],'role']='fit'
        train.loc[part.index[cut:],'role']='dev'
        entity_rows.append(dict(entity_id=entity,events=n,fit_events=cut,dev_events=n-cut,
                                fit_last_seq=int(part.target_seq.iloc[cut-1]),dev_first_seq=int(part.target_seq.iloc[cut])))
    assert train.role.value_counts().to_dict()=={'fit':313215,'dev':80609}
    assert all(r['fit_last_seq']<r['dev_first_seq'] for r in entity_rows)
    keys=['previous_quantity','previous_gap','previous2_quantity_key','history_length']
    cells=pd.read_csv(bindings['support_cells']['path'])
    eligible=cells.loc[cells.two_route_feasibility.astype(bool),keys].drop_duplicates()
    assert len(eligible)==532
    train=train.merge(eligible.assign(matched532=True),on=keys,how='left',validate='many_to_one')
    train['matched532']=train.matched532.fillna(False).astype(bool)
    rows=[]
    for role,part in train.groupby('role'):
        filters={'all':np.ones(len(part),bool),'history128_192':part.history_length.between(128,192),
                 'matched532':part.matched532,'matched532_history128_192':part.matched532 & part.history_length.between(128,192),
                 'quantity_gt187':part.raw_quantity.gt(187)}
        filters.update({'history_'+str(g):part.history_group.eq(g) for g in part.history_group.unique()})
        for label,mask in filters.items():
            selected=part.loc[mask]
            rows.append(dict(role=role,group=label,events=len(selected),entities=selected.entity_id.nunique()))
    dest=HERE/'split';dest.mkdir(exist_ok=False)
    train[['target_index','entity_id','target_seq','role','history_length','history_group','matched532']].sort_values('target_index').to_parquet(dest/'train_roles.parquet',index=False)
    pd.DataFrame(entity_rows).to_csv(dest/'entities.csv',index=False)
    pd.DataFrame(rows).to_csv(dest/'coverage.csv',index=False)
    bindings['train_roles']=ref(dest/'train_roles.parquet')
    bindings['split_entities']=ref(dest/'entities.csv')
    bindings['split_coverage']=ref(dest/'coverage.csv')
    collection=json.loads(Path(bindings['collection']['path']).read_text())
    cps=[]
    for seed in [52,42]:
        for kind in ['best','last']:
            c=next(c for c in collection['cohorts'] if c['seed']==seed and c['checkpoint']==kind and c['split']=='train')
            cps.append(dict(seed=seed,checkpoint=kind,epoch=c['epoch'],sha256=c['checkpoint_sha256']))
    spec={
        'schema':'phen-time-head-control-v1','frozen_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Intermittent original Train and retrospective Validation; no Test; no full-model retraining',
        'bindings':bindings,'checkpoints':cps,
        'seed_order':[52,42],'replication_rule':'Run seed42 with identical settings irrespective of seed52 effect direction after technical verification; no performance gate',
        'arms':{'original':{'trainable_parameters':0},'widthonly':{'trainable_parameters':65,'trainable_keys':['time_scale_weight.weight','w_raw']},'joint':{'trainable_parameters':130,'trainable_keys':['time_scale_weight.weight','w_raw','v_t.weight','b_t']}},
        'frozen':'All original checkpoints, encoder, correction, prototypes, quantity head and cached final hidden inputs; every arm warm-starts its own original time head',
        'head':{'hidden_dim':64,'sigma_min':.001,'time_scale':3.,'distribution':'D=max(1,round(3*LogNormal(mu,sigma)))','likelihood':'native stable float64 positive-integer bin mass; no top code'},
        'split':{'source':'original Train only','sort':['entity_id','target_seq','target_index'],'rule':'first floor(0.8*n) events per entity fit; remainder dev; n<2 fails','fit_rows':313215,'dev_rows':80609,'full_train_rows':393824,'validation_rows':86285,'use_all_fit_rows':True,'limitations':'Original encoder saw all Train, including head-dev; within-entity chronological head selection is not an independent heldout generalization test.'},
        'fit_config':{'learning_rate':.001,'weight_decay':0.,'batch_size':2048,'maximum_epochs':50,'patience':8,'minimum_improvement':1e-8,'gradient_clip_norm':5.,'shuffle_seed':20261026,'evaluation_batch_size':8192},
        'optimizer':'AdamW','training_device':'cpu','dtype':'float64','torch_num_threads':1,'deterministic_algorithms':True,
        'selection':'Lowest finite event-weighted head-dev NLL over epochs including epoch0; earliest exact tie; minimum_improvement affects patience reference only, not argmin selection',
        'stopping':'8 successive epochs without >1e-8 decrease from patience reference, or epoch50; preserve last and selected separately; nonfinite objective/gradient stops and preserves failure',
        'validation_use':'Only after head-dev selection, never for fitting, hyperparameters, epoch selection, or deciding seed42 replication',
        'inference':{'hosts':{'52':'5090','42':'5080'},'reuse':'Existing best Train hidden per original FINAL_COLLECTION','missing_splits_per_seed':['best Validation','last Train','last Validation'],'qualification_events_per_cohort':2048,'full_once_after_qualification':True,'original_checkpoint_and_model_state_unchanged':True},
        'parity':{'quantity':{'atol':1e-5,'rtol':1e-5},'time_nll':{'atol':1e-4,'rtol':1e-4},'mu_sigma':'diagnostic only for rounded historical exports; use native same-hidden evaluation to check math','no_automatic_tolerance_relaxation':True},
        'report':{'primary':['whole Validation','Validation history128-192 inclusive'],'secondary':['all Train and head-fit/head-dev','fixed matched532','matched532 and history128-192','other history bins','strict quantity>187 retrospective tail'],
                  'metrics':['recorded-bin NLL','paired NLL change vs original','log-center distance outside recorded bin','sigma','95% predicted interval disjoint from observed bin','quantity RMSE/MAE unchanged'],
                  'uncertainty':'1000 paired entity-cluster bootstrap repetitions with RNG20261026; conditional on fitted heads, not training uncertainty','report_every_seed_checkpoint_arm':True,
                  'interpretation':'Readout-only improvement demonstrates compensability on fixed representations under this protocol; it does not establish head-only causality, exonerate encoder/distribution shift, or establish independent Test generalization'},
        'preservation':['Existing model/source/data/results/failures','Stopped 5080 tasks and reserved memory','Other jobs/automations/runtime/network/authentication','Existing manuscripts and Notion'],
    }
    write(HERE/'SCIENCE_SPEC.json',spec)
    write(HERE/'CONTRACT_RECEIPT.json',{'contract':ref(HERE/'SCIENCE_SPEC.json'),'status':'FROZEN_BEFORE_REAL_FIT','coverage':rows,'training_epochs_executed':0})
    print(json.dumps({'contract':ref(HERE/'SCIENCE_SPEC.json'),'coverage':rows},ensure_ascii=False))

if __name__=='__main__': main()
