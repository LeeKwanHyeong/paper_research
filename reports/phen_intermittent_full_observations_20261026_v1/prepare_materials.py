"""Copy exact frozen inputs and compact saved observations; no model execution."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def copy(src,dst,expected):
    assert sha(src)==expected,src
    dst.parent.mkdir(parents=True,exist_ok=True)
    assert not dst.exists(),dst
    shutil.copyfile(src,dst)
    assert sha(dst)==expected,dst
    return record(dst)
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    started=time.time()
    prior=ROOT/'reports/phen_intermittent_path_evolution_20261026_v1/EVALUATION_SPEC.json'
    time_spec=ROOT/'reports/phen_intermittent_time_center_width_20261026_v1/SPEC.json'
    old=json.loads(prior.read_text());ts=json.loads(time_spec.read_text())
    bundle=OUT/'bundle';bundle.mkdir()
    inv={'prior_spec':record(prior),'time_spec':record(time_spec),'copies':[],'jobs':[],'compact_references':[]}
    dataset=bundle/'inputs/data/intermittent_frozen_5000/train_validation.parquet'
    inv['copies'].append(copy(Path(ts['input']['path']),dataset,ts['input']['sha256']))
    samples={}
    for split,v in old['reference_samples'].items():
        samples[split]=copy(Path(v['path']),bundle/'references'/(split+'_sample.json'),v['sha256'])
        inv['copies'].append(samples[split])
    seen=set()
    for job in old['jobs']:
        j=dict(job);seed=j['seed'];kind=j['checkpoint_kind']
        src=Path(j['source_root']);dest=bundle/('source'+str(seed))
        if seed not in seen:
            for rel,h in j['source_files'].items():inv['copies'].append(copy(src/rel,dest/rel,h))
            seen.add(seed)
        j['source_root']=str(dest)
        for what,sub in [('contract','contracts'),('checkpoint','checkpoints'),('provenance','provenance'),('history','histories')]:
            target=bundle/sub/('seed'+str(seed))/(kind+'.json' if what in ('provenance','history') else Path(j[what+'_path']).name)
            if not target.exists():inv['copies'].append(copy(Path(j[what+'_path']),target,j[what+'_sha256']))
            else:assert sha(target)==j[what+'_sha256']
            j[what+'_path']=str(target)
        j['data_override']={'prepared_train_validation_path':str(dataset)}
        j['splits']=['train'] if kind=='best' else ['train','validation']
        tj=next(x for x in ts['jobs'] if x['id']==j['id'])
        refs={}
        for split,v in tj['native_records'].items():
            f=Path(v['path']);assert sha(f)==v['sha256']
            target=bundle/'references'/(j['id']+'__'+split+'.jsonl')
            n=0
            with f.open() as inp,target.open('x') as out:
                for line in inp:
                    x=json.loads(line)
                    keep=['target_index','entity_id','target_seq','history_length','true_quantity','raw_quantity','pred_quantity','pred_log_quantity','target_log_quantity','log_qty_mse','quantity_train_loss','time_nll','batch_index']
                    y={k:x[k] for k in keep}
                    for k in ('prototype_indices','topk_similarity','head_hidden'):y[k]=x['paths']['alpha_1'][k]
                    out.write(json.dumps(y,separators=(',',':'))+'\n');n+=1
            assert n==2048
            refs[split]=record(target)
            inv['compact_references'].append({'source':v,'output':refs[split],'rows':n,'mapping':'native scalar fields + paths.alpha_1 prototype_indices/topk_similarity/head_hidden'})
        j['reference_results']=refs
        j['reference_samples']=samples
        inv['jobs'].append(j)
    inv['reference_samples']=samples
    inv['elapsed_seconds']=time.time()-started
    write(OUT/'MATERIALS.json',inv)
    print(json.dumps({'copied_files':len(inv['copies']),'reference_rows':16384,'bytes':sum(x['bytes'] for x in inv['copies'])+sum(x['output']['bytes'] for x in inv['compact_references']),'elapsed_seconds':inv['elapsed_seconds']}))
if __name__=='__main__':main()
