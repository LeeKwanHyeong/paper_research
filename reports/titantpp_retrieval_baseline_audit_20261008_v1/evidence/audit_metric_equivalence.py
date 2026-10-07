import ast, bisect, hashlib, json, math, platform
from pathlib import Path
import numpy as np
ROOT = Path('/Users/igwanhyeong/PycharmProjects/paper_research')
OLD = ROOT/'search_artifacts/titantpp_cnn_gru_test_20261005_v1/attempt2/source/paper/scripts/evaluate_titantpp_history_width.py'
NEW = ROOT/'search_artifacts/titantpp_retrieval_test_20261008_v1/operation/evaluate.py'
BOUNDS = (7, 686, 1562, 3449)
def require(condition, message):
    if not condition: raise AssertionError(message)
def metric_class(path):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'QuantityMetrics')
    namespace = dict(np=np, math=math, bisect=bisect, require=require, BOUNDS=BOUNDS)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['QuantityMetrics'], {'path': str(path.relative_to(ROOT)), 'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'class_line': node.lineno}
old_class, old_source = metric_class(OLD)
new_class, new_source = metric_class(NEW)
rng = np.random.default_rng(42)
q = np.r_[np.asarray([0, 7, 8, 686, 687, 1562, 1563, 3449, 3450, 9999], float), rng.integers(0, 12000, 10007).astype(float)]
pred = np.maximum(0., q + rng.normal(0., 450., len(q)))
temporal = rng.normal(1., .7, len(q))
max_abs, max_rel, cases = 0., 0., []
for chunk in (1, 17, 128, len(q)):
    old, new = old_class(BOUNDS), new_class()
    for i in range(0, len(q), chunk):
        old.add(q[i:i+chunk], pred[i:i+chunk], temporal[i:i+chunk])
        new.add(q[i:i+chunk], pred[i:i+chunk], temporal[i:i+chunk])
    a,b=old.finish(),new.finish()
    scopes=[('full',a,b),('tail',a['tail'],b['tail'])]+[(f'bin{i}',x,y) for i,(x,y) in enumerate(zip(a['quantity_cells'],b['quantity_cells']))]
    for name, x,y in scopes:
        require(x['count'] == y['count'], 'Counts differ')
        require(x['represented_n'] == x['count'], 'Unexpected weighted population')
        for key in ('qty_rmse','qty_mae','qty_bias','time_nll','qty_sse','qty_absolute_error_sum','qty_signed_error_sum','time_nll_sum'):
            absolute=abs(x[key]-y[key]); relative=absolute/max(1.,abs(x[key]),abs(y[key]))
            max_abs=max(max_abs,absolute);max_rel=max(max_rel,relative)
            require(math.isclose(x[key],y[key],rel_tol=1e-12,abs_tol=1e-10),f'Metric {name}/{key} differs')
    require(a['tail']['count']==int((q>3449).sum()),'Tail boundary differs')
    cases.append({'chunk_size':chunk,'count':a['count'],'tail_count':a['tail']['count'],'all_counts_and_statistics_within_tolerance':True})
result={'schema':'retrieval_baseline_evaluation_audit_v1','scope':'synthetic metric aggregation only; no real prediction, training, GPU, or Test inference', 'python':platform.python_version(),'numpy':np.__version__,'sources':[old_source,new_source], 'test_input_count':len(q),'cases':cases,'comparison_rel_tol':1e-12,'comparison_abs_tol':1e-10,'maximum_absolute_difference_across_sufficient_statistics':max_abs,'maximum_relative_difference':max_rel,'formula_equivalence':True,'boundary_rule':'bisect/searchsorted left; tail raw quantity >3449','floating_accumulation_difference':'old NumPy float64 sums; new Python double math.fsum; formulas unchanged','verification_limit':'Synthetic aggregation equivalence does not reproduce historical native model inference or prove cause of training divergence.'}
Path('/private/tmp/retrieval_baseline_evaluation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
