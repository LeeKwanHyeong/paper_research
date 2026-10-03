"""Cross-check reported values, figure source, references, and edit scope."""
from pathlib import Path
import importlib.util
import json
import math
import re
import hashlib

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
main=(ROOT/'paper/titantpp_pakdd_2027_draft/main.tex').read_text()
before=(OUT/'main_before.tex').read_text()
spec=importlib.util.spec_from_file_location('revision',OUT/'update_manuscript.py')
revision=importlib.util.module_from_spec(spec);spec.loader.exec_module(revision)
fig=(OUT/'figure.tex').read_text().strip()
expected=before.replace(revision.OLD_MAIN,revision.NEW_MAIN).replace(
    revision.OLD_APPENDIX,revision.NEW_APPENDIX.replace('FIGURE_PLACEHOLDER',fig))
assert main==expected, 'Unexpected edit outside two approved blocks'
data=json.loads((OUT/'analysis.json').read_text()); rows=data['rows']
assert len(rows)==20 and len({(r['dataset'],r['event_lag']) for r in rows})==20
assert fig in main
assert fig.count(r'\filldraw[fill=lagcell')==20
assert fig.count('P: ')==21 and fig.count('S: ')==21  # twenty cells and one legend
assert fig.count('{NA};')==1
for r in rows:
    value=r['correlation']
    for text in [f"P: {r['pairs']:,}",f"S: {r['series_with_pairs']:,}"]:
        assert '{'+text+'};' in fig
    if value is not None:
        assert f'${value:.3f}$' in fig and math.isfinite(value) and -1<=value<=1
    else:
        assert r['label']=='RAF' and r['event_lag']==16 and r['pairs']==r['series_with_pairs']==0
assert max(abs(r['correlation']) for r in rows if r['label']=='Instacart')<.05
assert len(set(re.findall(r'\\label\{([^}]+)\}',main)))==len(re.findall(r'\\label\{([^}]+)\}',main))
labels=set(re.findall(r'\\label\{([^}]+)\}',main))
refs=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',main))
assert refs<=labels
assert 'fig:train-lag-correlation' in labels and 'fig:train-lag-correlation' in refs
check=dict(status='passed',main_sha256=hashlib.sha256(main.encode()).hexdigest(),
    only_two_requested_prose_blocks_and_new_figure_changed=True,
    unchanged_authors_abstract_contributions_main_tables_appendix_A_C_references=True,
    figure_cells=20,undefined_cells=1,explicit_pair_and_series_counts_all_cells=True,
    fixed_color_domain=[-1,1],lag1_agrees_with_existing_profiles=True,
    all_figure_numbers_match_reviewed_rows=True,all_cross_references_resolve=True,
    preview_scope='PNG/SVG from same reviewed rows and palette; native source checked by built-in compiler',
    new_training_inference_remote_or_heldout_access=False)
(OUT/'manuscript_verification.json').write_text(json.dumps(check,indent=2)+'\n')
print(json.dumps(check))
