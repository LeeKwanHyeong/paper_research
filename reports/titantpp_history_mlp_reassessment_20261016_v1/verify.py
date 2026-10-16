"""A separate Decimal calculation; same agent, no model replay."""
import csv
import json
import hashlib
from pathlib import Path
from decimal import Decimal as D, getcontext

getcontext().prec = 50
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = ROOT/'reports/titantpp_three_dataset_final_comparison_20261005_v1'
def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
checks = 0
def eq(a,b):
    global checks
    a,b=D(str(a)),D(str(b))
    assert abs(a-b) <= max(D('0.000001'),abs(b)*D('1e-10')), (a,b)
    checks += 1
original = {tuple(r[k] for k in ('dataset','model','split','view','seed')):r for r in rows(BASE/'metrics_per_seed.csv')}
for r in rows(OUT/'SUMMARY.csv'):
    key=tuple(r[k] for k in ('dataset','model','split','view'))
    for m in ('qty_rmse','qty_mae','time_nll'):
        vals=[D(original[key+(s,)][m]) for s in ('42','52','62')]
        mean=sum(vals)/3
        sd=(sum((v-mean)**2 for v in vals)/2).sqrt()
        eq(r[m+'_mean'],mean);eq(r[m+'_sample_sd'],sd)
for r in rows(OUT/'SEED_DELTAS.csv'):
    a=D(r['candidate_value']);b=D(r['reference_value'])
    ka=(r['dataset'],r['candidate'],r['split'],r['view'],r['seed'])
    kb=(r['dataset'],r['reference'],r['split'],r['view'],r['seed'])
    eq(a,original[ka][r['metric']]);eq(b,original[kb][r['metric']])
    eq(r['delta'],a-b);eq(r['relative_percent'],(a/b-1)*100)
    assert (r['improved']=='True') == (a<b)
for r in rows(OUT/'TAIL_NONTAIL_DECOMPOSITION.csv'):
    for label,model in [('candidate',r['candidate']),('reference',r['reference'])]:
        key=(r['dataset'],model,r['split'])
        full=original[key+('overall',r['seed'])];tail=original[key+('tail',r['seed'])]
        n,nt=D(full['count']),D(tail['count'])
        se=n*D(full['qty_rmse'])**2;ts=nt*D(tail['qty_rmse'])**2
        ae=n*D(full['qty_mae']);ta=nt*D(tail['qty_mae'])
        for k,v in [('full_sse',se),('tail_sse',ts),('non_tail_sse',se-ts),
                    ('full_sae',ae),('tail_sae',ta),('non_tail_sae',ae-ta),
                    ('non_tail_rmse',((se-ts)/(n-nt)).sqrt()),('non_tail_mae',(ae-ta)/(n-nt))]:
            eq(r[label+'_'+k],v)
    for k in ('full_sse','tail_sse','non_tail_sse','full_sae','tail_sae','non_tail_sae','non_tail_rmse','non_tail_mae'):
        eq(r['delta_'+k],D(r['candidate_'+k])-D(r['reference_'+k]))
manifest=json.loads((OUT/'INPUT_MANIFEST.json').read_text())
source_count=0
for mapping in (manifest['verified_files'],manifest['additional_analysis_inputs']):
    for p,expected in mapping.items():
        assert sha(ROOT/p)==expected,p
        source_count+=1
fallacies={
 'Simpson': 'Datasets/splits/seeds and tail/non-tail separated; no pooled winner.',
 'Ecological': 'Aggregate SSE/SAE differences do not identify individual cases or predictive history patterns.',
 'Berkson': 'Selected-checkpoint and fixed-sample conditioning retained; no causal correlation inference.',
 'Collider': 'No covariate-adjusted causal estimate; actual target quantity is a diagnostic stratum only.',
 'Base_rate': 'Counts retained: Taxi8268/8327 with tail79/83; RAF6690/5226 with tail50/39.',
 'Regression_to_mean': 'Taxi seed42 exception and all seeds shown; no durable improvement inferred from one extreme seed.',
 'Survivorship': 'All requested six MLP/control families and three seeds retained. RAF TitanB absent, not imputed. Single-seed explorations explicitly separated.',
 'Look_elsewhere': 'All widths, metrics and adverse results retained; exploratory not significant.',
 'Forking_paths': 'Baseline16 retained; Test-informed new hypothesis disclosed; no new selector.',
 'Correlation_causation': 'Width/control results include changed capacity, optimization and selected epochs. Depth insufficiency not established.',
 'Reverse_causality': 'Observed preceding inputs are causal in architecture, but dataset performance associations do not establish error causes.'}
result=dict(status='pass',verification_status='ANALYZED',decimal_numeric_checks=checks,
            source_file_rechecks=source_count,fallacy_coverage='11/11 checked',fallacy_review=fallacies,
            independent_agent_review=False,model_replay=False,new_training=0,new_inference=0,remote_calls=0)
with (OUT/'VERIFICATION.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k!='fallacy_review'},indent=2))
