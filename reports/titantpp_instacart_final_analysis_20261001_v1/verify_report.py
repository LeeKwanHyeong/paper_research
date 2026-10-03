"""Independent arithmetic and provenance checks over the delivered analysis."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

O=Path(__file__).resolve().parent
R=O.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
x=load(O/'analysis.json'); rows=list(csv.DictReader((O/'conditions.csv').open()))
source=load(O/'source_manifest.json'); report=(O/'report.md').read_text()
for path,h in source.items(): assert sha(R/path)==h,path
assert len(rows)==42
assert len({(r['model'],r['seed'],r['endpoint']) for r in rows})==42
assert {int(r['count']) for r in rows}=={503733}
for g in x['three_seed']:
    rr=[r for r in rows if r['model']==g['model'] and r['endpoint']==g['endpoint']]
    assert sorted(int(r['seed']) for r in rr)==[42,52,62]
    for m in ('qty_mae','qty_rmse','time_nll'):
        values=[float(r[m]) for r in rr];mean=sum(values)/3
        sd=math.sqrt(sum((v-mean)**2 for v in values)/2)
        assert math.isclose(mean,g[m]['mean'],abs_tol=1e-12)
        assert math.isclose(sd,g[m]['sample_sd'],abs_tol=1e-12)
        assert f'{mean:.6f} ± {sd:.6f}' in report
assert [x['rankings'][m].index(x['representative'])+1 for m in ('qty_mae','qty_rmse','time_nll')]==[2,6,5]
for model in ('s2p2_matched_head','attnhp_matched_head'):
    p=next(p for p in x['paired'] if p['comparator']==model)
    assert p['qty_mae']['plain_wins']==2 and p['qty_rmse']['plain_wins']==0
    for b in range(6):
        base=next(r for r in x['selected_strata'] if r['model']==x['representative'] and r['partition']=='additional_history' and r['bin']==b)
        other=next(r for r in x['selected_strata'] if r['model']==model and r['partition']=='additional_history' and r['bin']==b)
        if model=='s2p2_matched_head':assert base['qty_rmse']['mean']>other['qty_rmse']['mean']
for path in re.findall(r'\]\((/[^)]+)\)',report): assert Path(path).exists(),path
manuscript=R/'paper/titantpp_history_mlp_manuscript_20261001_v1.md'
baseline=load(R/'reports/titantpp_baseline_reset_20261001_v1/baseline.json')
assert sha(manuscript)==baseline['manuscript_integration']['sha256'],'Manuscript changed outside this analysis'
v=load(O/'verification.json')
v.update(independent_mean_sample_sd_check=True,report_numbers_and_links_verified=True,
         ranking_and_seed_claims_verified=True,source_hashes_reverified_after_report=True,
         manuscript_sha256=sha(manuscript),manuscript_unchanged_from_registered_baseline=True,
         report_sha256=sha(O/'report.md'),conditions_csv_sha256=sha(O/'conditions.csv'),
         strata_csv_sha256=sha(O/'strata.csv'),report_builder_sha256=sha(O/'write_report.py'),
         verification_script_sha256=sha(__file__))
(O/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
audits={}
for host in ('5080','5090'):
    path=R/f'search_artifacts/titantpp_additional_tpp_20260930_v1/retrieved/terminal_{host}_20261001_v1/terminal_audit.json'
    a=load(path);assert a['status']=='passed'
    audits[host]={'path':str(path.relative_to(R)),'sha256':sha(path),'conditions':len(a['conditions']),
                  'checkpoint_count':a['verified_terminal_checkpoints'],'reused_prior_audit':host=='5080'}
assert sum(a['conditions'] for a in audits.values())==18
pointer={'status':'passed','scope':'additional_TPP_18_conditions_36_selected_last_checkpoints',
         'completed_kst':x['completed_analysis_kst'],'campaign_audits':audits,
         'instacart_analysis':str((O/'report.md').relative_to(R)),
         'analysis_verification':str((O/'verification.json').relative_to(R)),
         'manuscript_result_table_updated':False}
p=R/'search_artifacts/titantpp_additional_tpp_20260930_v1/retrieved/audit_completion_20261001.json'
if p.exists():assert load(p)==pointer
else:p.write_text(json.dumps(pointer,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(v,ensure_ascii=False,indent=2))
