"""Replay the nonzero FULL route against its pre-hook source at 80cc675."""
import argparse
import copy
import tempfile
import importlib.util
import json
import pathlib
import subprocess
import sys

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--git-root', type=pathlib.Path, required=True)
parser.add_argument('--tested-root', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
args = parser.parse_args()
SOURCE_ROOT = args.git_root.resolve()
SAFE_ROOT = args.tested_root.resolve()
sys.path.insert(0, str(SAFE_ROOT))
temporary = tempfile.TemporaryDirectory(prefix='bounded_qk_full_parity_')
reference_path = pathlib.Path(temporary.name) / 'old_full.py'
reference_path.write_bytes(subprocess.check_output(
    ['git', 'show', '80cc675:models/TPPs/CountAwareTitanCausalQKV.py'], cwd=SOURCE_ROOT,
))
spec = importlib.util.spec_from_file_location('bounded_qk_old_full_reference', reference_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from models.TPPs.CountAwareTitanCausalQKV import CausalQKVMemoryAttention
from models.Titan.common.memory import MemoryAttention

torch.set_num_threads(1)
results = []
for training in (False, True):
    for amp in (False, True):
        torch.manual_seed(920)
        base = MemoryAttention(16, 4, 0, 16, dropout=.1)
        before = module.CausalQKVMemoryAttention(copy.deepcopy(base))
        after = CausalQKVMemoryAttention(copy.deepcopy(base))
        for name in ('causal_q_kernel', 'causal_k_kernel', 'causal_v_kernel'):
            getattr(before, name).data.normal_(0, .2)
        after.load_state_dict(before.state_dict(), strict=True)
        before.train(training)
        after.train(training)
        x = torch.randn(2, 5, 16)
        mask = torch.tensor([[True] * 5, [False, False, True, True, True]])
        values = []
        for attention in (before, after):
            input_x = x.clone().requires_grad_()
            torch.manual_seed(918)
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                output = attention(input_x, mask)
            output.float().square().sum().backward()
            values.append((output.detach(), input_x.grad))
        assert torch.equal(values[0][0], values[1][0])
        assert torch.equal(values[0][1], values[1][1])
        for name, parameter in before.named_parameters():
            actual = dict(after.named_parameters())[name]
            assert torch.equal(parameter.grad, actual.grad), name
        results.append(dict(
            training=training, cpu_bfloat16_autocast=amp,
            outputs_bitwise=True, input_gradient_bitwise=True,
            parameter_gradients_bitwise=True,
        ))
output = args.output
output.write_text(json.dumps(results, indent=2) + '\n')
print(output.read_text())
