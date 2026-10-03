"""Isolated integration with frozen TitanTPP source and generated CPU tensors only."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / 'search_artifacts/mixed_quantity_preparation_v1/snapshots/3ac65c42b35a329ede32ad95862c594579dfca9c3ae4ebf6458560cec1d0cbd8/source'
OVERLAY = ROOT / 'paper/scripts'


@pytest.mark.parametrize('flag,expected_code', [(None, 0), ('--execute', 1), ('--worker', 1)])
def test_cli_preparation_and_unapproved_execution_load_no_model_libraries(tmp_path, flag, expected_code):
    import hashlib
    contract_path = ROOT / 'paper/contracts/intermittent_mixed_cell_diagnosis_v1.json'
    contract = json.loads(contract_path.read_text())
    names = ['run_intermittent_cell_diagnosis.py', 'intermittent_cell_controls.py',
             'intermittent_cell_gradients.py', 'intermittent_cell_statistics.py',
             'run_quantity_checkpoint_diagnosis.py', 'validate_intermittent_cell_contract.py']
    binding = {'schema': 'intermittent_cell_execution_binding_v1',
        'measurement_canonical_sha256': 'ca08be3e7070aaedac7279d46af4629dc59447769350caa6b4f8dbaad4872838',
        'status': 'implemented_cpu_verified_pending_gpu_approval', 'gpu_execution_authorized': False,
        'limits': contract['limits'], 'server': '5090',
        'overlay_files': {name: hashlib.sha256((OVERLAY / name).read_bytes()).hexdigest() for name in names}}
    path = tmp_path / 'binding.json'
    path.write_text(json.dumps(binding))
    command = """import json, runpy, sys
sys.path.insert(0, sys.argv[1])
sys.argv = sys.argv[2:]
code = 0
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except ValueError as error:
    assert 'approval' in str(error).lower() or 'supervisor' in str(error).lower() or 'authority' in str(error).lower(), str(error)
    code = 1
assert not any(k in sys.modules for k in ('torch', 'numpy', 'polars', 'models', 'paper'))
raise SystemExit(code)
"""
    args = [sys.executable, '-B', '-c', command, str(OVERLAY), str(OVERLAY / names[0]),
            '--contract', str(contract_path), '--binding', str(path)]
    if flag:
        args.append(flag)
    result = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ['binding.json']


def test_frozen_titantpp_six_synthetic_states(tmp_path):
    receipt = tmp_path / 'synthetic_integration.json'
    result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                             '--synthetic-integration', str(receipt)],
                            cwd=tmp_path, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(receipt.read_text())
    assert evidence['synthetic_states_verified'] == 6
    assert evidence['paired_comparisons_verified'] == 4
    assert evidence['frozen_source_files_verified'] == 70
    assert evidence['optimizer_updates'] == 0 and evidence['real_checkpoint_loads'] == 0
    assert evidence['passed']


def integration(receipt_path):
    import copy
    import hashlib
    import random
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(OVERLAY))
    import run_intermittent_cell_diagnosis as adapter
    import intermittent_cell_controls as controls
    contract = json.loads((ROOT / 'paper/contracts/intermittent_mixed_cell_diagnosis_v1.json').read_text())
    assert controls.sha_json(contract) == controls.MEASUREMENT_SHA
    for relative, digest in contract['source']['files'].items():
        assert controls.sha_file(FROZEN / relative) == digest
    sys.path.insert(0, str(FROZEN))
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.mixed_quantity_objective import MixedQuantityObjective, mixed_joint_causal_batch_objective, _collate
    from paper.scripts.quantity_objective_comparison import QuantityStatistics
    from paper.scripts.quantity_comparison_engine import _batch_tensors
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256 as tensor_hash
    import intermittent_cell_statistics as stats
    import intermittent_cell_gradients as gradients
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    adapter.verify_frozen_imports(FROZEN, contract['source']['files'])
    import types
    sys.modules['models.synthetic_foreign_import'] = types.SimpleNamespace(__file__=str(OVERLAY / 'intermittent_cell_controls.py'))
    try:
        adapter.verify_frozen_imports(FROZEN, contract['source']['files'])
    except ValueError as error:
        assert 'Foreign frozen import' in str(error)
    else:
        raise AssertionError('A foreign model module was accepted')
    finally:
        del sys.modules['models.synthetic_foreign_import']

    # Explicit prohibitions catch accidental expansion of this CPU fixture.
    def forbidden(*args, **kwargs):
        raise AssertionError('Synthetic integration cannot load checkpoints, data files, or an optimizer')
    torch.load = torch.save = forbidden
    torch.optim.AdamW = torch.optim.Adam = torch.optim.SGD = forbidden
    import polars as pl
    pl.read_parquet = pl.scan_parquet = forbidden

    n, width = 6, 130
    truth = torch.tensor([0., 2., 31., 46., 187., 200.])
    history = torch.tensor([1, 2, 64, 65, 128, 129])
    dts, quantities = torch.zeros(n, width), torch.zeros(n, width)
    mask = torch.zeros(n, width, dtype=torch.bool)
    for row, length in enumerate(history.tolist()):
        mask[row, -(length + 1):] = True
        dts[row, -length:] = torch.linspace(.01, .2, length)
        quantities[row, -(length + 1):-1] = torch.linspace(.1, 2., length)
        quantities[row, -1] = truth[row]
    dataset = TensorDataset(dts, mask, quantities)
    identity = {'true_qty': truth.numpy(), 'series_ids': np.asarray([f'synthetic-{i}' for i in range(n)]),
                'target_position': history.numpy(), 'target_seq': (history + 2).numpy()}
    policy = copy.deepcopy(contract['train_probe'])
    policy.update(selected_sample_count=n, batches_per_state=2, batch_size=4)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(policy['selection_seed']))
    policy['expected_selected_indices_sha256'] = tensor_hash({'indices': indices})
    probe, chosen = adapter.train_loader_for_probe(dataset, policy, _collate, tensor_hash)
    assert torch.equal(chosen, indices) and not torch.equal(chosen, chosen.sort().values)
    def batch_hash(loader):
        digest = hashlib.sha256()
        for batch in loader:
            digest.update(tensor_hash(dict(zip(('dts', 'mask', 'quantities'), _batch_tensors(batch, 'cpu')))).encode('ascii'))
        return digest.hexdigest()
    policy['expected_batch_sha256'] = batch_hash(probe)
    fixture = {'train_probe': policy, 'cells': contract['cells'], 'server': {'device': 'cpu'}}
    def validation_loader():
        return DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=_collate,
                          generator=torch.Generator().manual_seed(1042))
    validation_hash = batch_hash(validation_loader())
    statistics = QuantityStatistics(mu=1.0, raw_scale=contract['train_probe']['coefficients']['raw_scale'])
    budget = controls.Budget(contract['limits'])
    reports, pairs, records_by_state = [], [], []
    original_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())

    for scope in ('primary', 'last120'):
        baseline = None
        for number, case in enumerate(adapter.CASES):
            # Generated stand-ins exercise all six adapter paths, not trained weights.
            with gradients.preserved_rng():
                torch.manual_seed(42 + number + (3 if scope == 'last120' else 0))
                model, _ = build_count_aware_model('titantpp', hidden_dim=8, train_log_mean=statistics.mu,
                                                  max_seq_len=256, time_intercept_limit=300.)
                with torch.no_grad():
                    model.quantity_head.weight.fill_(.01 * (number + 1))
                model.eval()
            before = tensor_hash(model.state_dict())
            modes = [m.training for m in model.modules()]
            obj = MixedQuantityObjective.from_dict(contract['arms'][case]['condition']['mixed_objective'])
            def forward(*args):
                return mixed_joint_causal_batch_objective(model, *args, statistics=statistics, objective=obj)
            forward.mixed_objective = obj
            with gradients.preserved_model_state(model):
                rows, val = adapter.collect_validation(model, validation_loader(), forward, _batch_tensors,
                    tensor_hash, identity, budget, device='cpu', expected_count=n, expected_batches=2,
                    expected_input_sha=validation_hash)
                assert np.array_equal(rows['history_length'], history.numpy())
                assert stats.cell_indices(rows['true_qty'], rows['history_length'], contract['cells']).tolist() == [0, 0, 3, 7, 10, 14]
                summary = stats.summarize_rows(rows, [2, 31, 46, 187], [64, 128])
                assert summary['conservation_audit']['passed']
                if baseline is None:
                    baseline = rows
                else:
                    pair = stats.compare_rows(baseline, rows, [2, 31, 46, 187], [64, 128])
                    assert pair['conservation_audit']['passed']
                    pairs.append(pair)
                stream = []
                aggregate, sample = adapter.run_train_probe(model, probe, forward, _batch_tensors,
                    tensor_hash, stats, gradients, fixture, budget, emit_record=stream.append)
                assert [r['count'] for r in stream] == [4, 2]
                assert [r['dropout_seed'] for r in stream] == [9042, 9043]
                assert all(r['forward_calls'] == 1 and r['autograd_calls'] <= 49 for r in stream)
                assert all(r['model_state_sha256'] == before for r in stream)
                assert all(r['conservation']['passed'] and r['immutability']['passed'] for r in stream)
                if case == 'B_log_original':
                    assert all(r['overall']['preclip']['all']['norms']['weighted_raw'] == 0 for r in stream)
                repeated = []
                repeat_aggregate, _ = adapter.run_train_probe(model, probe, forward, _batch_tensors,
                    tensor_hash, stats, gradients, fixture, budget, emit_record=repeated.append)
                assert aggregate == repeat_aggregate and stream == repeated
                records_by_state.append(stream)
            assert tensor_hash(model.state_dict()) == before
            assert modes == [m.training for m in model.modules()]
            assert all(p.grad is None for p in model.parameters())
            assert original_rng[0] == random.getstate()
            current_np = np.random.get_state()
            assert original_rng[1][0] == current_np[0] and np.array_equal(original_rng[1][1], current_np[1]) and original_rng[1][2:] == current_np[2:]
            assert torch.equal(original_rng[2], torch.get_rng_state())
            reports.append({'case': case, 'scope': scope, 'synthetic_tensor_sha256': before,
                            'validation': val, 'train': sample, 'state_and_rng_preserved': True})

    # Inputs/targets must match before a comparison is interpreted.
    wrong_identity = dict(identity, true_qty=np.ones(n, dtype=np.float32))
    with gradients.preserved_model_state(model):
        try:
            adapter.collect_validation(model, validation_loader(), forward, _batch_tensors, tensor_hash,
                wrong_identity, budget, device='cpu', expected_count=n, expected_batches=2, expected_input_sha=validation_hash)
        except ValueError as error:
            assert 'target order' in str(error)
        else:
            raise AssertionError('A changed target order was accepted')
    adapter.verify_frozen_imports(FROZEN, contract['source']['files'])
    for relative, digest in contract['source']['files'].items():
        assert controls.sha_file(FROZEN / relative) == digest
    assert not torch.cuda.is_initialized()
    Path(receipt_path).write_text(json.dumps({'schema': 'intermittent_cell_synthetic_integration_v1', 'passed': True,
        'synthetic_states_verified': len(reports), 'paired_comparisons_verified': len(pairs),
        'frozen_source_files_verified': len(contract['source']['files']), 'states': reports,
        'sample_count_per_state': n, 'synthetic_train_batch_sizes': [4, 2], 'deterministic_repeats_verified': 6,
        'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu'},
        'optimizer_updates': 0, 'real_checkpoint_loads': 0, 'real_data_rows': 0, 'gpu_executed': False,
        'scope': 'Synthetic adapter verification only; not historical checkpoint replay or CUDA qualification.'}, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    assert len(sys.argv) == 3 and sys.argv[1] == '--synthetic-integration'
    integration(sys.argv[2])
