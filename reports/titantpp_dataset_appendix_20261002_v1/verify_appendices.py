"""Independently reconcile the new paper tables and claims with saved evidence."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TEX = (ROOT / 'paper/titantpp_pakdd_2027_draft/main.tex').read_text()
DATA = dict(Taxi='yellow_trip_hourly', Intermittent='intermittent_frozen_5000',
            RAF='raf_spare_parts', Instacart='insta_market_basket')
CHECKS = {}


def rows(name):
    with (OUT / name).open() as f: return list(csv.DictReader(f))


def check(name, passed):
    CHECKS[name] = bool(passed)
    assert passed, name


def table(label):
    return next(b for b in re.findall(r'\\begin\{table\}.*?\\end\{table\}', TEX, re.S)
                if '\\label{' + label + '}' in b)


def close(x,y): return math.isclose(x,y,abs_tol=1e-7,rel_tol=1e-9)


analysis = json.loads((OUT / 'analysis.json').read_text())
sources = json.loads((OUT / 'sources.json').read_text())
check('all archived source file SHAs still match', all(
    hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected for path, expected in sources.items()))
train = {r['dataset']:r for r in rows('train_profiles.csv')}
selected = rows('selected_conditions.csv')
check('84 unique completed selected conditions', len(selected)==len({(r['dataset'],r['model'],r['seed']) for r in selected})==84)

# Freeze identities and actual boundaries, not inferred from the plotted values.
core = json.loads((ROOT / 'search_artifacts/titantpp_core_ablation_20260928_v1/frozen_execution/execution_contract.json').read_text())
raf = json.loads((ROOT / 'search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2/original/execution_contract.json').read_text())
contracts = {s['dataset_id']:s for c in [core,raf] for s in c['datasets']}
for ds in DATA.values():
    spec=contracts[ds]
    check('quantity boundaries match frozen contract: '+ds, all(
        json.loads(r['boundaries'])==spec['quantity_boundaries_all_train_rows']
        for r in rows('strata_three_seed.csv') if r['dataset']==ds and r['partition']=='quantity'))
    grouped={}
    for r in selected:
        if r['dataset']==ds and r['model']!='titantpp_history_mlp':
            grouped.setdefault(r['model'],[]).append(float(r['qty_rmse']))
    check('anchor is overall strongest external mean RMSE: '+ds,
          min(grouped, key=lambda k:statistics.mean(grouped[k]))==analysis['anchors'][ds])

profile_fields = [
    ('Train series','train_series',',.0f'),('Train event rows','train_rows',',.0f'),
    ('Train targets','train_targets',',.0f'),('Quantity: median','quantity_p50',',.0f'),
    ('Quantity: 95th percentile','quantity_p95',',.0f'),('Quantity: 99th percentile','quantity_p99',',.0f'),
    ('Quantity: maximum','quantity_max',',.0f'),('Recorded gap: median','gap_p50','.0f'),
    ('Recorded gap: 95th percentile','gap_p95','.0f'),('Usable history: median','history_p50','.0f'),
    ('Usable history: 95th percentile','history_p95','.0f'),
    ('History $\\le7$ events (\\%)','history_le7_percent','.2f'),
    ('Within-series log correlation','within_entity_log_lag1_corr','.3f'),
    ('Equal adjacent quantities (\\%)','adjacent_equal_percent','.2f')]
profile = table('tab:train-characteristics')
for label,field,fmt in profile_fields:
    line=next(x for x in profile.splitlines() if x.startswith(label+' &'))
    observed=[x.strip().rstrip('\\').strip() for x in line.split('&')[1:]]
    expected=[format(float(train[d][field]),fmt) for d in DATA.values()]
    check('train table: '+field, observed==expected)

strata=rows('anchor_strata.csv')
for partition in ['quantity','history']:
    ds=None; actual=[]
    for line in table('tab:strata-'+partition).splitlines():
        group=re.search(r'\\textit\{(\w+) vs\.',line)
        if group: ds=DATA[group[1]]
        if line.startswith('$'):
            cells=[c.strip().rstrip('\\').strip() for c in line.split('&')]
            actual.append((ds,cells))
    expected=[r for r in strata if r['partition']==partition]
    check(partition+' row count matches all datasets including empty bins',len(actual)==len(expected))
    for (ds,cells),r in zip(actual,expected):
        interval=cells[0].strip('$'); variable='q' if partition=='quantity' else 'h'
        interval=interval.replace('\\le ', '<=').replace(variable,'').replace(' ','')
        raw=r['interval']
        want=(raw if not raw.startswith('(') else raw.strip('()]').replace(',','<='))
        if raw.startswith('('): want=want.replace('<=','<<=',1)
        check(partition+' interval/count/metrics: '+ds+'/'+r['bin'],
            ds==r['dataset'] and interval==want and cells[1]==f"{int(r['count']):,}" and
            cells[2:]==[f'{float(r[k]):.4f}' if r[k] else '--' for k in
                       ['mlp_mae','mlp_rmse','anchor_mae','anchor_rmse']])

# Direct totals for the printed decomposition, independent of plotting aggregation.
for ds in DATA.values():
    anchor=analysis['anchors'][ds]
    direct=[]
    for seed in ['42','52','62']:
        a=next(r for r in selected if (r['dataset'],r['model'],r['seed'])==(ds,'titantpp_history_mlp',seed))
        b=next(r for r in selected if (r['dataset'],r['model'],r['seed'])==(ds,anchor,seed))
        direct.append((float(a['qty_sse'])-float(b['qty_sse']))/int(a['count']))
    bins=[r for r in strata if r['dataset']==ds and r['partition']=='quantity']
    check('plotted contributions sum to overall three-seed MSE difference: '+ds,
          close(sum(float(r['mse_contribution']) for r in bins),statistics.mean(direct)))

before=(OUT/'before/main.tex').read_text()
oldA=before[before.index(r'\section{Completed Subset'):before.index(r'\begin{thebibliography}')].strip()
newA=TEX[TEX.index(r'\section{Completed Subset'):TEX.index(r'\section{Data Definitions')].strip()
check('Appendix A preserved verbatim at its dated cutoff',oldA==newA)
check('RAF resolution limitation and empty strata retained','cannot resolve its short-history' in TEX and
      len([r for r in strata if r['dataset']==DATA['RAF'] and r['count']=='0'])==2)
check('descriptive association distinguished from causal evidence','remains a hypothesis rather than an identified causal effect' in TEX)
receipt=json.loads((OUT/'compilation_receipt.json').read_text())
check('native compilation passed for current source',receipt['status']=='success' and
      receipt['source_sha256']==hashlib.sha256(TEX.encode()).hexdigest())
out={'status':'passed','checked_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
     'checks':CHECKS,'source_count':len(sources),'train_statistic_cells':56,
     'stratum_rows':35,'rendered_figure_review':'see figure_review.json',
     'main_tex_sha256':hashlib.sha256(TEX.encode()).hexdigest(),
     'new_training_or_model_inference':False,'heldout_metrics_or_predictions_read':False}
(OUT/'manuscript_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'passed','checks':len(CHECKS),'source_count':len(sources)}))
