"""Second arithmetic implementation using Decimal, no model execution."""
from pathlib import Path
from decimal import Decimal, getcontext
import hashlib
import json

getcontext().prec = 45
O = Path(__file__).resolve().parent
R = O.parents[1]
D = Decimal
data = json.loads((O/'RESULT.json').read_text())
receipt = json.loads((O/'RECEIPT.json').read_text())
assert hashlib.sha256((O/'RESULT.json').read_bytes()).hexdigest() == receipt['outputs']['RESULT.json']
checks = 0

def eq(a, b):
    global checks
    a, b = D(str(a)), D(str(b))
    assert abs(a-b) <= max(D('1e-7'), max(abs(a),abs(b))*D('1e-10')), (a,b)
    checks += 1

for path, h in data['source_files'].items():
    assert hashlib.sha256((R/path).read_bytes()).hexdigest() == h

# Directly recompute source-file strata using an independent decimal path.
moments = []
for ref in data['prediction_availability']:
    src = json.loads((R/ref['source']).read_text(), parse_float=D)
    cells = [src['metrics'], *src['metrics']['quantity_cells']]
    rr = [r for r in data['rows'] if (r['split'],r['arm'],r['seed']) == (ref['split'],ref['arm'],ref['seed'])]
    for c,r in zip(cells,rr):
        n,a,s,v = D(c['count']),D(c['qty_absolute_error_sum']),D(c['qty_signed_error_sum']),D(c['qty_sse'])
        for metric,value in [('qty_rmse',(v/n).sqrt()),('qty_mae',a/n),('qty_bias',s/n),
                ('overprediction_ae_mass',(a+s)/2),('underprediction_ae_mass',(a-s)/2)]:
            eq(r[metric],value)
        if r['stratum']=='overall':
            moments.append({'split':ref['split'],'arm':ref['arm'],'seed':ref['seed'],
                'mse':float(v/n),'squared_bias':float((s/n)**2),
                'residual_population_variance':float(v/n-(s/n)**2)})
    for k in ['count','qty_sse','qty_absolute_error_sum','qty_signed_error_sum','time_nll_sum']:
        eq(sum(D(c[k]) for c in cells[1:]),cells[0][k])

moment_deltas = []
for split in ['validation','test']:
    for seed in [42,52,62]:
        b = next(m for m in moments if (m['split'],m['seed'],m['arm'])==(split,seed,'CNN'))
        for arm in ['Capacity','Adaptive']:
            a = next(m for m in moments if (m['split'],m['seed'],m['arm'])==(split,seed,arm))
            delta = {k:a[k]-b[k] for k in ['mse','squared_bias','residual_population_variance']}
            eq(delta['mse'],delta['squared_bias']+delta['residual_population_variance'])
            moment_deltas.append({'split':split,'seed':seed,'arm_minus_CNN':arm,**delta})

for c in data['contrasts']:
    for d in c['strata']:
        a,b = [next(x for x in data['rows'] if (x['split'],x['seed'],x['stratum'],x['arm']) ==
                    (c['split'],c['seed'],d['stratum'],arm)) for arm in [c['candidate'],c['reference']]]
        for k,v in d.items():
            if k in ['stratum','count','relative_percent']:continue
            eq(v,D(str(a[k]))-D(str(b[k])))

for a in data['aggregates']:
    rr = [r for r in data['rows'] if (r['split'],r['arm'],r['stratum']) == (a['split'],a['arm'],a['stratum'])]
    for metric,stats in a['metrics'].items():
        values = [D(str(r[metric])) for r in rr]
        mean = sum(values)/3
        sd = (sum((v-mean)**2 for v in values)/2).sqrt()
        eq(mean,stats['mean']);eq(sd,stats['sample_sd'])

# A zero signed mean does not imply low absolute or squared error.
e = [D(10),D(-10)]
assert sum(e)==0 and sum(abs(x) for x in e)/2==10 and sum(x*x for x in e)/2==100
assert (sum(abs(x) for x in e)+sum(e))/2 == sum(max(x,D(0)) for x in e)
assert (sum(abs(x) for x in e)-sum(e))/2 == sum(max(-x,D(0)) for x in e)
out = {'status':'PASS_arithmetic_and_source_preservation','independent_agent_review':False,
    'implementation':'Second Decimal calculation in the same agent; not native evaluation',
    'source_files_verified_unchanged':len(data['source_files']),'scalar_equalities':checks,
    'synthetic_counterexample':'zero bias can coexist with large absolute and squared error',
    'moments':moments,'moment_deltas':moment_deltas,
    'verify_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (O/'VERIFICATION.json').open('x') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:out[k] for k in ['status','source_files_verified_unchanged','scalar_equalities']}))
print(json.dumps([x for x in moment_deltas if x['split']=='test' and x['arm_minus_CNN']=='Capacity'],ensure_ascii=False))
