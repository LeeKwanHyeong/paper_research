"""CPU parameter-count verification only; no data, forward pass or training."""
import hashlib
import json
import platform
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def gru_count(r):
    modules = nn.ModuleList([nn.Linear(64, r, bias=False),
                             nn.GRU(r, r, num_layers=1, bias=True, batch_first=True,
                                    dropout=0, bidirectional=False),
                             nn.Linear(r, 64, bias=False)])
    n = sum(p.numel() for p in modules.parameters())
    assert n == 6*r*r + 134*r
    return n

widths = {}
for w in (4,8,12,16):
    modules = nn.ModuleList([nn.Sequential(nn.Linear(128,w,bias=False), nn.GELU(),
                                           nn.Linear(w,64,bias=False)) for _ in range(8)])
    widths[str(w)] = sum(p.numel() for p in modules.parameters())
    assert widths[str(w)] == 1536*w
cnn = nn.ParameterList([nn.Parameter(torch.zeros(3,64)) for _ in range(3)])
cnn_count = sum(p.numel() for p in cnn.parameters())
assert cnn_count == 576
grus = {str(r): gru_count(r) for r in (16,24,53,54)}
# Derived references from the sealed baseline decision, not new full-model counts.
baseline_decision = json.loads((ROOT/'reports/titantpp_width16_baseline_decision_20261004_v1/decision.json').read_text())
def locate_reference(node):
    if isinstance(node, dict):
        if 'full_model_parameters_max_len256_reference' in node:
            return node['full_model_parameters_max_len256_reference']
        for value in node.values():
            found = locate_reference(value)
            if found is not None: return found
    return None
common256 = locate_reference(baseline_decision) - widths['16']
common84 = common256 - (256 - 84)*64
arm_counts = {'baseline_mlp16': widths['16'], 'cnn_mlp16': widths['16']+cnn_count,
              'gru54': grus['54'], 'cnn_gru54': grus['54']+cnn_count}
sources = ['reports/titantpp_width16_baseline_decision_20261004_v1/decision.json',
           'models/TPPs/CountAwareTitanHistoryWidth.py',
           'models/TPPs/CountAwareTitanCoreAblation.py',
           'models/TPPs/CountAwareTitanMultiLagDetail.py',
           'models/TPPs/CountAwareTitanCausalQKV.py',
           'paper/reproducibility/current_baseline.json',
           'reports/titantpp_cnn_gru_review_20261003_v1/README.md',
           '.codex/agents/architecture-reviewer.toml']
result = {'status':'parameter_formula_verified_design_only', 'python':platform.python_version(),
          'torch':torch.__version__, 'device':'cpu', 'training_executed':False,
          'data_read':False, 'model_forward_executed':False,
          'mlp_correction_counts':widths,'gru_correction_counts':grus,'cnn_count':cnn_count,
          'selected_gru_width':54,'correction_difference_vs_mlp16':grus['54']-widths['16'],
          'correction_difference_percent':100*(grus['54']/widths['16']-1),
          'arm_correction_plus_cnn_counts':arm_counts,
          'full_model_counts_are_derived_reference_not_instantiated':True,
          'derived_full_model_reference_max_len256':{k:common256+n for k,n in arm_counts.items()},
          'derived_full_model_reference_max_len84':{k:common84+n for k,n in arm_counts.items()},
          'unexecuted_checks':['candidate full model construction', 'causality/masking',
                               'gradient', 'checkpoint rejection', 'GPU speed/memory', 'training'],
          'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'source_sha256'},ensure_ascii=False,indent=2))
