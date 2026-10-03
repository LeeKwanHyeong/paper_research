"""Compare manuscript displays against completed analysis and retained validation tables."""
from pathlib import Path
import hashlib,json,re,sys
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from render_results import validate_analysis,DATASETS,PRIMARY,MODELS,METRICS,ANCHORS,REPRESENTATIVE,BONFERRONI
A=validate_analysis(json.loads((BASE/'analysis.json').read_text()))
P=ROOT/'paper/titantpp_pakdd_2027_draft/main.tex'
s=P.read_text(); before=(BASE/'manuscript/before.tex').read_text()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(text,label):
 items=[x for x in re.findall(r'\\begin\{table\}.*?\\end\{table\}',text,re.S) if r'\label{'+label+'}' in x]
 assert len(items)==1,(label,len(items))
 return items[0]
def body(text,label):
 return re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}',table(text,label),re.S).group()
retained=['tab:main','tab:naive','tab:ablation','tab:extension','tab:cost','tab:train-characteristics','tab:strata-quantity','tab:strata-history']
for label in retained:assert body(s,label)==body(before,label),('changed pre-existing table',label)
# Values are formatted from full-precision analysis, not refitted or recomputed predictions.
name_map={n.replace(' control','').replace('All-available-history','All-available'):m for m,n in MODELS.items()}
ds_names={n:d for d,n in DATASETS.items()}; displays=0; coverage=set()
for label,expected_rows in [('tab:legacy-test-main',28),('tab:legacy-test-structural',12)]:
 d=None; count=0
 for line in table(s,label).splitlines():
  group=re.search(r'\\textit\{([^}]+)\}',line)
  if group:d=ds_names[group.group(1)];continue
  if ' & ' not in line or r'\pm' not in line:continue
  fields=line.split(' & '); m=name_map[fields[0]]; panel=A['results'][d][PRIMARY[d]]
  got=re.findall(r'\d+\.\d+',line)
  expected=[f'{panel[k][m][metric]:.4f}' for metric in METRICS for k in ('means','sample_sd')]
  assert got==expected,(label,d,m,got,expected)
  count+=1;displays+=len(got);coverage.add((d,m))
 assert count==expected_rows,(label,count)
assert len(coverage)==36
simple=table(s,'tab:legacy-test-simple')
for d,name in DATASETS.items():
 panel=A['results'][d][PRIMARY[d]]
 for model,label in [('last_observed_quantity','Last quantity'),('mean_quantity_in_same_observed_window','Window mean')]:
  row=next(l for l in simple.splitlines() if l.startswith(name+' & '+label+' & '))
  expected=[f'{panel["means"][model][metric]:.4f}' for metric in METRICS[:2]]
  assert re.findall(r'\d+\.\d+',row)==expected
  displays+=2
anchors=table(s,'tab:legacy-test-anchors')
for d,name in DATASETS.items():
 panel=A['results'][d][PRIMARY[d]]; anchor=ANCHORS[d]
 row=next(l for l in anchors.splitlines() if l.startswith(name+' & '))
 expected=[f'{panel["means"][REPRESENTATIVE]["qty_rmse"]-panel["means"][anchor]["qty_rmse"]:.4f}']
 if panel['intervals'] is not None:
  entry=panel['intervals'][anchor]['qty_rmse']
  expected += [f'{v:.4f}' for key in ('pointwise_95_percentile',BONFERRONI) for v in entry[key]]
 else:assert '--- & ---' in row
 assert re.findall(r'-?\d+\.\d+',row)==expected,(d,row,expected)
 displays+=len(expected)
labels=re.findall(r'\\label\{([^}]+)\}',s)
assert len(labels)==len(set(labels)), 'duplicate labels'
refs=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',s))
assert refs<=set(labels),('undefined refs',refs-set(labels))
bibs=set(re.findall(r'\\bibitem\{([^}]+)\}',s))
cites={item.strip() for match in re.findall(r'\\cite\{([^}]+)\}',s) for item in match.split(',')}
assert cites<=bibs,('undefined citations',cites-bibs)
result={'status':'passed','analysis_sha256':sha(BASE/'analysis.json'),'manuscript_sha256':sha(P),'before_sha256':sha(BASE/'manuscript/before.tex'),'new_displayed_numbers_checked':displays,'dataset_model_groups_covered':len(coverage),'learned_conditions_represented':len(coverage)*3,'preserved_table_bodies':retained,'undefined_references':[],'undefined_citations':[],'duplicate_labels':[],'scope':'display and preservation checks; no inference, training or resampling'}
(BASE/'manuscript/table_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
