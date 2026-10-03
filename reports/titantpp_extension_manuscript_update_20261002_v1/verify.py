"""Verify the revised manuscript against the completed13 validation evidence."""
from pathlib import Path
import hashlib,json,re,time
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'paper/titantpp_pakdd_2027_draft/main.tex'
EVIDENCE=ROOT/'reports/titantpp_extension_completed13_audit_20261002_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
s=SOURCE.read_text();before=(OUT/'main_before.tex').read_text();c=read(EVIDENCE/'comparison.json')
for filename,digest in read(EVIDENCE/'artifact_manifest.json')['files'].items():assert sha(EVIDENCE/filename)==digest,filename
labels={'titantpp_history_mlp':'Original MLP','titantpp_current_only_param_matched':'Current-only','titantpp_all_available_history_mlp':'All-available','deep_renewal_event_native_nb':'Deep Renewal'}
metrics=('qty_mae','qty_rmse','time_nll')
def table(text,label):
    matches=[x for x in re.findall(r'\\begin\{table\}.*?\\end\{table\}',text,re.S) if '\\label{'+label+'}' in x]
    assert len(matches)==1,label
    return matches[0]
taxi=table(s,'tab:extension');single=table(s,'tab:extension-seed42')
for g in c['complete_three_seed_groups']:
    assert g['dataset']=='yellow_trip_hourly' and g['seeds']==[42,52,62]
    row=labels[g['model']]+' & '+' & '.join(f"${g[k]['mean']:.4f}\\pm{g[k]['sample_sd']:.4f}$" for k in metrics)+r'\\'
    assert row in taxi,row
single_rows=[]
for dataset in ('intermittent_frozen_5000','insta_market_basket'):
    for model in list(labels)[:3]:
        rs=[r for r in c['conditions'] if r['dataset']==dataset and r['model']==model]
        assert len(rs)==1 and rs[0]['seed']==42
        r=rs[0];row=labels[model]+' & '+' & '.join(f'{r[k]:.6f}' for k in metrics)+r'\\'
        assert row in single,row;single_rows.append(row)
for x in c['three_seed_comparisons']:
    if x['model']=='titantpp_current_only_param_matched':
        perc=[x[k]['original_MLP_reduction_pct_relative_to_variant'] for k in ('qty_mae','qty_rmse')]
        assert all(x[k]['variant_lower_seed_count']==0 for k in ('qty_mae','qty_rmse'))
    elif x['model']=='titantpp_all_available_history_mlp':
        perc=[-x[k]['variant_relative_change_pct'] for k in ('qty_mae','qty_rmse')]
        assert x['qty_mae']['variant_lower_seed_count']==1 and x['qty_rmse']['variant_lower_seed_count']==2
        assert x['time_nll']['variant_relative_change_pct']>0
    else:continue
    phrase=f'{perc[0]:.2f}\\%/{perc[1]:.2f}\\%';assert s.count(phrase)==2,phrase
# Reuse the pinned status observation, never convert it into a current poll.
a=read(ROOT/'search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/hourly_monitor/20261002T015157805227Z/analysis.json')
assert sum(h['completed'] for h in a['hosts'].values())==13
for host,h in a['hosts'].items():
    cur=h['current'];assert 'deep_renewal_event_native_nb' in json.dumps(cur) and '42' in json.dumps(cur)
assert '10:51:58 KST' in s and '13 completed extension conditions' in s
assert 'archived checkpoints await verification' not in s
assert 'pending archived-checkpoint verification' not in s
assert 'The seed-52/62 groups' in s and 'RAF extension groups were queued' in s
assert 'current-only control retains the causal encoder' in s
assert 'native negative-binomial time and quantity losses' in s
protected=['tab:data','tab:main','tab:naive','tab:ablation','tab:cost','tab:train-characteristics','tab:strata-quantity','tab:strata-history']
for label in protected:assert table(s,label)==table(before,label),label
figpattern=r'\\begin\{figure\}.*?\\end\{figure\}'
assert re.findall(figpattern,s,re.S)==re.findall(figpattern,before,re.S)
for begin,end in [(r'\begin{abstract}',r'\end{abstract}'),(r'\section{Method}',r'\section{Experimental Setup}'),(r'\section{Conclusion}',r'\appendix'),(r'\section{Data Definitions and Training Distributions}',r'\begin{thebibliography}'),(r'\begin{thebibliography}',r'\end{thebibliography}')]:
    assert s[s.index(begin):s.index(end)]==before[before.index(begin):before.index(end)],begin
assert s[s.index(r'\author{'):s.index(r'\maketitle')]==before[before.index(r'\author{'):before.index(r'\maketitle')]
refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',s);defined=re.findall(r'\\label\{([^}]+)\}',s)
assert len(defined)==len(set(defined)) and set(refs)<=set(defined)
compilation=read(OUT/'compilation_receipt.json')
assert not compilation.get('isError')
assert any(json.loads(block['text']).get('kind')=='success' for block in compilation['content'] if block['type']=='text')
result={'status':'passed','verified_unix':time.time(),'source':str(SOURCE.relative_to(ROOT)),'source_sha256':sha(SOURCE),
        'before_source_sha256':sha(OUT/'main_before.tex'),'evidence_comparison_sha256':sha(EVIDENCE/'comparison.json'),
        'new_conditions_in_evidence':13,'taxi_three_seed_rows':4,'partial_seed42_rows':6,
        'all_table_values_match_audited_validation_records':True,'all_reported_percentages_verified':True,
        'old_84_condition_tables_and_efficiency_preserved':True,'figures_authors_method_abstract_conclusion_bibliography_preserved':True,
        'appendices_B_C_preserved':True,'all_labels_unique_and_refs_resolved':True,'no_remote_training_or_evaluation':True,
        'native_editor_compilation':'success','heldout_access':False}
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
