"""Source identity and synthetic algebra only; no data, Torch, or inference."""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FROZEN = ROOT / 'reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluator_sources/core'
LAGS = (1, 2, 4, 8, 16, 32, 64, 128)

files = [
    'models/TPPs/CountAwareTitanCoreAblation.py',
    'models/TPPs/CountAwareTitanMultiLagDetail.py',
    'models/TPPs/CountAwareTPP.py',
    'paper/scripts/count_aware_tpp_backbone/core.py',
]
identities = {}
for rel in files:
    current, frozen = ROOT / rel, FROZEN / rel
    identities[rel] = {
        'current_sha256': hashlib.sha256(current.read_bytes()).hexdigest(),
        'frozen_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest(),
        'whole_file_identical': current.read_bytes() == frozen.read_bytes(),
    }
for rel in files[:2]:
    assert identities[rel]['whole_file_identical'], rel

for rel, names in {
    'models/TPPs/CountAwareTPP.py': ('continuous_features', 'predict_quantity', 'quantity_outputs'),
    'paper/scripts/count_aware_tpp_backbone/core.py': ('target_outputs',),
}.items():
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.FunctionDef) and node.name in names}
    current, frozen = functions(ROOT / rel), functions(FROZEN / rel)
    identities[rel]['audited_function_ast_matches'] = {name: current[name] == frozen[name] for name in names}
    assert all(identities[rel]['audited_function_ast_matches'].values()), rel

histories = (1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 128, 129, 255)
active = {str(h): sum(h > lag for lag in LAGS) for h in histories}
assert [sum(h > lag for lag in LAGS) for h in (1, 2, 3, 5, 9, 17, 33, 65, 129)] == list(range(9))
widths = {str(r): {'correction_parameters': 8 * (128*r + 64*r),
                   'total_parameters_length256_configuration_only': 96003 - 6144 + 8 * (128*r + 64*r),
                   'correction_span_upper_bound_by_K': {str(k): min(64, k*r) for k in range(9)}}
          for r in (4, 8, 16, 32, 64)}
assert widths['4']['correction_parameters'] == 6144
examples = []
for true, pred in ((200., 100.), (10100., 10000.)):
    examples.append({'true': true, 'prediction': pred,
                     'log_mse': (math.log1p(pred)-math.log1p(true))**2,
                     'raw_mse_before_fixed_scale': (pred-true)**2,
                     'raw_head_gradient_before_fixed_scale': 2*(pred-true)*pred})
head_identity_errors = []
for a in (-10., -2., 0., 2., 10.):
    q = math.expm1(math.log1p(math.exp(a)))
    head_identity_errors.append(abs(q/math.exp(a)-1))
assert max(head_identity_errors) < 1e-12

result = {
    'scope': 'source hashes and exact/synthetic mathematical checks, not model performance',
    'dataset_opened': False, 'checkpoint_loaded': False, 'training_or_inference': False,
    'source_identity': identities,
    'K_equals_sum_H_greater_than_lag': active,
    'parameter_and_span_formulas': widths,
    'same_absolute_error_examples': examples,
    'expm1_softplus_equals_exp_max_relative_error_in_examples': max(head_identity_errors),
    'prior_history_bins_constant_K': False,
    'counterexample': {'prior_bin': '(1,3]', 'H2_K': 1, 'H3_K': 2},
    'status': 'passed',
}
(OUT / 'algebra_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'output': str(OUT / 'algebra_checks.json')}, ensure_ascii=False))
