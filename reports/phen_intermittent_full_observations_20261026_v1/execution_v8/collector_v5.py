#!/usr/bin/env python3
"""Single-forward, native PHEN Train/Validation observation collector.

All model/data imports come from a hash-verified frozen source in a fresh child.
This module does not launch work on import; full/GPU execution is root-controlled.
"""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from copy import deepcopy
import hashlib
import importlib.abc
import importlib.machinery
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
SCHEMA = 'phen_full_native_observations_v1'
IDENTITY = ('target_index', 'entity_id', 'target_seq', 'raw_quantity')
SCALARS = ('true_quantity', 'pred_quantity', 'target_log_quantity', 'pred_log_quantity',
           'log_qty_mse', 'quantity_train_loss', 'time_nll', 'head_preactivation')
REFERENCE_SCALARS = ('pred_quantity', 'log_qty_mse', 'time_nll')
EXPECTED = {(42, 'best'): 1, (42, 'last'): 41, (52, 'best'): 5, (52, 'last'): 45}
POPULATIONS = {'train': 393824, 'validation': 86285}
FROZEN_PACKAGES = {'models', 'utils', 'paper', 'simple_lab_test', 'data_loader'}


def require(value, message):
    if not value: raise ValueError(message)


def canonical_gpu_uuid(value):
    """Accept only the standard 128-bit hexadecimal UUID, optionally prefixed GPU-."""
    require(isinstance(value, str), 'GPU UUID must be a string')
    match = re.fullmatch(r'(?:GPU-)?([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})', value)
    require(match is not None, 'Invalid GPU UUID format: ' + repr(value))
    return match.group(1).lower()


def verify_gpu_uuid(raw, expected, output):
    record = {'raw': raw, 'expected_raw': expected, 'canonical': None,
              'expected_canonical': None, 'equal_128bit_uuid': False,
              'normalization': 'standard_8_4_4_4_12_hex_casefold_optional_exact_GPU_prefix_only'}
    try:
        record['canonical'] = canonical_gpu_uuid(raw)
        record['expected_canonical'] = canonical_gpu_uuid(expected)
        record['equal_128bit_uuid'] = record['canonical'] == record['expected_canonical']
        require(record['equal_128bit_uuid'], 'Unexpected GPU UUID: ' + json.dumps(record, sort_keys=True))
    except ValueError as error:
        record['error'] = str(error)
        write_json(Path(output) / 'GPU_UUID_FAILURE.json', record)
        raise
    return record


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def file_record(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def verify_ref(ref):
    path = Path(ref['path']).resolve()
    require(path.is_file() and sha(path) == ref['sha256'], 'Bound file changed: ' + str(path))
    if 'bytes' in ref: require(path.stat().st_size == ref['bytes'], 'Bound byte count changed')
    return path


def verify_numeric_gate_revision(spec):
    """Bind the single user-approved comparison change; this does not alter model arithmetic."""
    revision = spec['numeric_gate_revision']
    required = {'field': 'topk_similarity.atol', 'previous': 1e-7, 'approved': 1e-6, 'rtol': 1e-5}
    require(all(revision.get(k) == v for k, v in required.items()), 'Unadmitted numeric gate revision')
    authorization_path = verify_ref(revision['authorization'])
    authorization = json.loads(authorization_path.read_text())
    require(authorization.get('approved') is True
            and authorization.get('field') == required['field']
            and authorization.get('previous') == required['previous']
            and authorization.get('approved_atol') == required['approved']
            and authorization.get('rtol') == required['rtol']
            and isinstance(authorization.get('user_instruction'), str)
            and bool(authorization['user_instruction'].strip()),
            'Numeric gate authorization does not match the admitted change')
    return revision


def verify_log_mse_gate_revision(spec):
    """Bind the separately approved log-MSE comparison change; retain all model calculations."""
    revision = spec['log_mse_gate_revision']
    required = {'field': 'log_qty_mse.atol', 'previous': 1e-7, 'approved': 1e-6, 'rtol': 1e-5}
    require(all(revision.get(k) == v for k, v in required.items()), 'Unadmitted log-MSE gate revision')
    authorization_path = verify_ref(revision['authorization'])
    authorization = json.loads(authorization_path.read_text())
    require(authorization.get('approved') is True
            and authorization.get('field') == required['field']
            and authorization.get('previous') == required['previous']
            and authorization.get('approved_atol') == required['approved']
            and authorization.get('rtol') == required['rtol']
            and isinstance(authorization.get('user_instruction'), str)
            and bool(authorization['user_instruction'].strip()),
            'Log-MSE gate authorization does not match the admitted change')
    return revision


def state_sha(model):
    import torch
    digest = hashlib.sha256(b'phen_gradient_state_v1\0')
    for kind, values in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
        for name, value in sorted(values):
            tensor = value.detach().cpu().contiguous()
            digest.update(json.dumps([kind, name, str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def context_snapshot(model):
    result = {}
    for name, module in model.named_modules():
        if hasattr(module, '_ctx_mem'):
            value = module._ctx_mem
            require(value is None or value.numel() == 0, 'Dynamic context memory is not admitted')
            result[name] = None if value is None else [str(value.dtype), list(value.shape)]
    return result


def verify_source(source, manifest):
    source = Path(source).resolve()
    require(bool(manifest), 'Empty frozen source inventory')
    members = list(source.rglob('*'))
    require(not any(path.is_symlink() for path in members), 'Symlink in frozen source')
    actual = {str(path.relative_to(source)) for path in members if path.is_file()}
    require(actual == set(manifest), 'Frozen source file inventory differs: '
            + repr({'extra': sorted(actual - set(manifest)), 'missing': sorted(set(manifest) - actual)}))
    for relative, digest in manifest.items():
        path = (source / relative).resolve()
        require(path.is_relative_to(source) and path.is_file(), 'Missing or unsafe frozen source')
        require(sha(path) == digest, 'Frozen source SHA changed: ' + relative)


class FrozenSourceFinder(importlib.abc.MetaPathFinder):
    """Resolve research packages only from the bound snapshot, even after path bootstrap."""
    def __init__(self, source, manifest):
        self.source, self.manifest = Path(source).resolve(), manifest

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in FROZEN_PACKAGES: return None
        parent = self.source.joinpath(*fullname.split('.')[:-1])
        spec = importlib.machinery.PathFinder.find_spec(fullname, [str(parent)], target)
        if spec is None:
            raise ModuleNotFoundError('Module absent from bound frozen source: ' + fullname)
        if spec.origin is not None:
            origin = Path(spec.origin).resolve()
            require(origin.is_relative_to(self.source), 'Foreign module origin: ' + str(origin))
            relative = str(origin.relative_to(self.source))
            require(relative in self.manifest and sha(origin) == self.manifest[relative],
                    'Imported module is not SHA-bound: ' + fullname)
        if spec.submodule_search_locations is not None:
            expected = self.source.joinpath(*fullname.split('.')).resolve()
            locations = [Path(p).resolve() for p in spec.submodule_search_locations]
            require(locations == [expected] and expected.is_relative_to(self.source),
                    'Foreign package search path: ' + fullname)
            # A plain list prevents namespace path expansion when legacy helpers mutate sys.path.
            spec.submodule_search_locations = [str(expected)]
        return spec


def install_frozen_imports(source, manifest):
    require(not any(name.split('.')[0] in FROZEN_PACKAGES for name in sys.modules),
            'Worker process is not source-isolated')
    finder = FrozenSourceFinder(source, manifest)
    sys.meta_path.insert(0, finder)
    sys.path.insert(0, str(source))
    return finder


def verify_imports(source, manifest):
    imported = {}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in FROZEN_PACKAGES: continue
        locations = getattr(module, '__path__', None)
        if locations is not None:
            require([Path(p).resolve() for p in locations] == [source.joinpath(*name.split('.')).resolve()],
                    'Foreign package search path: ' + name)
        if getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(source), 'Foreign model/helper import: ' + str(path))
            relative = str(path.relative_to(source))
            require(relative in manifest and sha(path) == manifest[relative], 'Unbound imported source: ' + name)
            imported[name] = relative
    return imported


def bind_input_layout(spec, job, data, source):
    """Retain the frozen helper and its original relative input paths, in an explicit bundle cwd."""
    working = Path(spec['working_dir']).resolve()
    require(working.is_dir() and not working.is_relative_to(source), 'Invalid isolated working directory')
    data_root = working / 'inputs' / 'data'
    require(data_root.is_dir(), 'Missing isolated input root')
    sidecar = verify_ref(spec['split_manifest_record'])
    inherited = data['inherited_data_identity']
    require(sidecar == (working / inherited['split_manifest']['path']).resolve()
            and spec['split_manifest_record']['sha256'] == inherited['split_manifest']['sha256'],
            'Split manifest layout or original identity differs')
    require(set(job.get('data_override', {})) <= {'prepared_train_validation_path'}, 'Unadmitted data override')
    data.update(job.get('data_override', {})); data.pop('full_data_identity', None)
    input_path = Path(data['prepared_train_validation_path']).resolve()
    require(input_path == (working / inherited['data']['path']).resolve()
            and input_path.is_relative_to(data_root), 'Input layout differs from frozen relative identity')
    require(Path.cwd() == working, 'Data helper must use the explicitly bound working directory')
    return input_path, data_root, sidecar


def read_reference(ref):
    """Accept original native JSONL or a compact extraction with the same fields."""
    path = verify_ref(ref)
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line)
            if 'modes' in row:
                require('alpha_1' in row['modes'] and 'alpha_1' in row['paths'], 'Wrong native reference mode')
                native, paths = row['modes']['alpha_1'], row['paths']['alpha_1']
                row = {**{k: row[k] for k in (*IDENTITY, 'history_length', 'batch_index')},
                       **{k: native[k] for k in SCALARS},
                       'prototype_indices': paths['prototype_indices'], 'topk_similarity': paths['topk_similarity']}
            require(all(k in row for k in (*IDENTITY, 'history_length', 'batch_index', *SCALARS,
                                           'prototype_indices', 'topk_similarity') if k != 'head_preactivation'), 'Incomplete native reference')
            rows.append(row)
    require(len(rows) == 2048 and len({r['target_index'] for r in rows}) == 2048, 'Reference must contain 2048 unique targets')
    return rows


def fixed_order(population_size, reference_indices, mode, qualification_batches, batch_size):
    import numpy as np
    require(mode in ('qualification', 'full'), 'Unknown collection mode')
    reference_indices = np.asarray(reference_indices, dtype=np.int64)
    require(len(reference_indices) == 2048 and len(np.unique(reference_indices)) == 2048,
            'Invalid fixed reference indices')
    require(np.all((reference_indices >= 0) & (reference_indices < population_size)), 'Reference target outside population')
    if mode == 'qualification': return reference_indices[:qualification_batches * batch_size]
    rest = np.ones(population_size, dtype=bool); rest[reference_indices] = False
    result = np.concatenate([reference_indices, np.flatnonzero(rest)])
    require(len(result) == population_size and np.array_equal(np.sort(result), np.arange(population_size)),
            'Collection order misses or repeats targets')
    return result


class NativeObserver(AbstractContextManager):
    """Observe original method returns without changing any returned tensor."""
    def __init__(self, model):
        self.model = model
        self.values = {}
        self.handles = []
        self.calls = {'encode': 0, 'time': 0, 'quantity': 0, 'retrieve': 0, 'head': 0}

    def __enter__(self):
        import torch
        model = self.model
        for name in ('encode_task_states', '_lognormal_time_terms', 'quantity_outputs'):
            require(name not in model.__dict__, 'Existing model override: ' + name)
        require('retrieve' not in model.lmm.__dict__, 'Existing retrieval override')
        original_encode = model.encode_task_states
        original_time = model._lognormal_time_terms
        original_quantity = model.quantity_outputs
        original_retrieve = model.lmm.retrieve

        def encode(dts, quantities, mask, *, memory_write_mask=None):
            self.calls['encode'] += 1
            lengths = mask.sum(1).long()
            require(bool((lengths >= 2).all()) and mask.shape[1] == 256, 'Invalid native input shape')
            expected_mask = torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
            require(torch.equal(mask, expected_mask), 'Native valid positions must be right-padded prefixes')
            ids = torch.arange(len(lengths), device=mask.device); target = lengths - 1
            require(bool((quantities[ids, target] == 0).all()), 'Withheld target quantity leaked')
            expected_write = mask.clone(); expected_write[ids, target] = False
            require(memory_write_mask is not None and torch.equal(memory_write_mask, expected_write), 'Target write-mask changed')
            self.positions = lengths - 2
            self.values['target_gap'] = dts[ids, target].detach().clone()
            self.values['history_length'] = lengths - 1
            self.values['valid_mask_sha256'] = hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest()
            self.values['observed_mask_sha256'] = hashlib.sha256(memory_write_mask.detach().cpu().numpy().tobytes()).hexdigest()
            return original_encode(dts, quantities, mask, memory_write_mask=memory_write_mask)

        def time_terms(hidden, target):
            self.calls['time'] += 1
            result = original_time(hidden, target)
            mu, sigma, _ = result
            require(mu.dtype == sigma.dtype == torch.float64, 'Native time-head arithmetic is not float64')
            require(torch.equal(target, self.values['target_gap']), 'Time target differs from native withheld position')
            self.values['time_hidden'] = hidden.detach().clone()
            self.values['time_location_float64'] = mu.detach().clone()
            self.values['time_sigma_float64'] = sigma.detach().clone()
            return result

        def quantity(hidden, target):
            self.calls['quantity'] += 1
            result = original_quantity(hidden, target)
            self.values['quantity_hidden'] = hidden.detach().clone()
            self.values['pred_log_quantity'] = result['location'].detach().clone()
            self.values['target_log_quantity'] = torch.log1p(target.clamp_min(0)).detach().clone()
            return result

        def retrieve(*args, **kwargs):
            self.calls['retrieve'] += 1
            result = original_retrieve(*args, **kwargs)
            residual, trace = result
            ids = torch.arange(residual.shape[0], device=residual.device)
            self.values['prototype_indices'] = trace['prototype_indices'][ids, self.positions].detach().clone()
            self.values['topk_similarity'] = trace['topk_similarity'][ids, self.positions].detach().clone()
            self.values['prototype_residual'] = residual[ids, self.positions].detach().clone()
            return result

        def head(module, args, result):
            self.calls['head'] += 1
            self.values['head_hidden'] = args[0].detach().clone()
            self.values['head_preactivation'] = result.squeeze(-1).detach().clone()

        model.encode_task_states, model._lognormal_time_terms, model.quantity_outputs = encode, time_terms, quantity
        model.lmm.retrieve = retrieve
        self.handles = [model.quantity_head.register_forward_hook(head)]
        return self

    def __exit__(self, *_):
        for handle in self.handles: handle.remove()
        for name in ('encode_task_states', '_lognormal_time_terms', 'quantity_outputs'): delattr(self.model, name)
        del self.model.lmm.retrieve

    def validate(self, outputs):
        import torch
        value = self.values
        require(all(n == 1 for n in self.calls.values()), 'Expected one native forward/time/quantity/retrieval/head call')
        require(torch.equal(value['head_hidden'], value['quantity_hidden'])
                and torch.equal(value['head_hidden'], value['time_hidden']), 'Time and quantity heads did not use the same saved input')
        require(value['head_hidden'].dtype == torch.float32 and value['head_hidden'].shape[1] == 64, 'Native head input is not float32 width64')
        require(torch.equal(torch.nn.functional.softplus(value['head_preactivation']), value['pred_log_quantity']), 'Non-native quantity transform')
        require(torch.equal(torch.expm1(value['pred_log_quantity']), outputs['pred_qty']), 'Raw quantity transform changed')
        require(torch.equal((value['pred_log_quantity'] - value['target_log_quantity']).square(), outputs['log_qty_loss']), 'Native log-MSE changed')
        require(torch.equal(outputs['quantity_train_loss'], outputs['log_qty_loss']), 'Unexpected quantity loss weighting')
        require(torch.equal(value['history_length'], outputs['history_length']), 'History mask count mismatch')
        require(bool((value['target_gap'] == value['target_gap'].round()).all()) and bool((value['target_gap'] >= 1).all()), 'Invalid recorded target gap')
        indices = value['prototype_indices']
        require(indices.shape == (len(value['head_hidden']), 4) and bool(((indices >= 0) & (indices < 64)).all()), 'Invalid native top4 shape/index')
        require(all(len(set(row)) == 4 for row in indices.cpu().tolist()), 'Repeated top4 index')
        selected = self.model.lmm.mem[0][indices]
        require(torch.equal(selected.mean(dim=1), value['prototype_residual']), 'Prototype native residual is not the fixed 1/4 mean')
        for key, tensor in value.items():
            if isinstance(tensor, torch.Tensor): require(bool(torch.isfinite(tensor).all()), 'Nonfinite native observation: ' + key)
        return value


def export_batch(outputs, observed, identities):
    import numpy as np
    tensors = {'true_quantity': outputs['true_qty'], 'pred_quantity': outputs['pred_qty'],
               'log_qty_mse': outputs['log_qty_loss'], 'quantity_train_loss': outputs['quantity_train_loss'],
               'time_nll': outputs['time_loss'], 'history_length': outputs['history_length']}
    for key in ('target_log_quantity', 'pred_log_quantity', 'head_preactivation', 'target_gap',
                'time_location_float64', 'time_sigma_float64', 'prototype_indices', 'topk_similarity'):
        tensors[key] = observed[key]
    arrays = {key: value.detach().cpu().numpy() for key, value in tensors.items()}
    require(all(len(a) == len(identities) for a in arrays.values()), 'Native output shape mismatch')
    require(np.array_equal(arrays['true_quantity'], np.asarray([r['raw_quantity'] for r in identities])), 'Target raw quantity differs from native truth')
    for key in IDENTITY: arrays[key] = np.asarray([row[key] for row in identities])
    return arrays, observed['head_hidden'].detach().cpu().numpy()


def compare_reference(arrays, hidden, references, tolerances, output, label):
    """Every admitted floating field is checked; IDs/order have zero mismatch."""
    import numpy as np
    maximum, failures = {}, []
    for i, ref in enumerate(references):
        for key in (*IDENTITY, 'history_length'):
            require(arrays[key][i] == ref[key], 'Reference identity mismatch: ' + key)
        if list(map(int, arrays['prototype_indices'][i])) != ref['prototype_indices']:
            failures.append({'target_index': int(ref['target_index']), 'field': 'prototype_indices',
                             'actual': arrays['prototype_indices'][i].tolist(), 'expected': ref['prototype_indices'],
                             'policy': 'exact_order_no_tie_exception'})
        require(arrays['true_quantity'][i] == ref['true_quantity'], 'Reference target quantity mismatch')
        fields = [(key, arrays[key][i], ref[key]) for key in REFERENCE_SCALARS]
        fields.append(('topk_similarity', arrays['topk_similarity'][i], ref['topk_similarity']))
        for key, actual, expected in fields:
            actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
            require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(), 'Invalid parity field')
            error = np.abs(actual - expected)
            maximum[key] = max(maximum.get(key, 0.), float(error.max()))
            tolerance = tolerances[key]
            allowed = tolerance['atol'] + tolerance['rtol'] * np.abs(expected)
            if not bool((error <= allowed).all()):
                failures.append({'target_index': int(ref['target_index']), 'field': key,
                                 'actual': actual.tolist(), 'expected': expected.tolist(),
                                 'absolute_error': error.tolist(), 'allowed_error': allowed.tolist()})
    if failures:
        evidence = {'status': 'failed_preserved', 'batch': label, 'tolerances': tolerances,
                    'max_absolute_errors': maximum, 'failures': failures}
        write_json(output / ('PARITY_FAILURE_' + label + '.json'), evidence)
        raise ValueError('Native reference parity failed; evidence preserved: ' + label)
    return maximum


class ChunkWriter:
    def __init__(self, output, split, chunk_rows, save_hidden):
        self.output, self.split, self.chunk_rows, self.save_hidden = output, split, chunk_rows, save_hidden
        self.parts, self.hidden, self.count, self.total = [], [], 0, 0
        self.records = []

    def append(self, arrays, hidden):
        self.parts.append(arrays)
        if self.save_hidden: self.hidden.append(hidden)
        self.count += len(arrays['target_index'])
        if self.count >= self.chunk_rows: self.flush()

    def flush(self):
        if not self.count: return
        import numpy as np
        import polars as pl
        keys = self.parts[0].keys()
        arrays = {key: np.concatenate([part[key] for part in self.parts]) for key in keys}
        columns = {key: value for key, value in arrays.items() if key not in ('prototype_indices', 'topk_similarity')}
        for i in range(4):
            columns[f'prototype_index_{i}'] = arrays['prototype_indices'][:, i].astype(np.int16)
            columns[f'prototype_cosine_{i}'] = arrays['topk_similarity'][:, i].astype(np.float32)
        columns['split'] = [self.split] * self.count
        columns['time_parameter_precision'] = ['native_float64'] * self.count
        number = len(self.records)
        path = self.output / f'{self.split}_{number:05d}.parquet'
        with path.open('xb') as stream:
            pl.DataFrame(columns).write_parquet(stream, compression='zstd', row_group_size=self.chunk_rows)
        record = {'chunk': number, 'split': self.split, 'row_offset': self.total, 'rows': self.count,
                  'target_index_sha256': hashlib.sha256(arrays['target_index'].astype('<i8').tobytes()).hexdigest(),
                  'observations': file_record(path)}
        if self.save_hidden:
            hidden = np.concatenate(self.hidden)
            require(hidden.shape == (self.count, 64) and hidden.dtype == np.float32, 'Hidden export shape/dtype changed')
            hidden_path = self.output / f'{self.split}_{number:05d}.head_hidden.npz'
            with hidden_path.open('xb') as stream:
                np.savez_compressed(stream, head_hidden=hidden, target_index=arrays['target_index'].astype(np.int64))
            record['head_hidden'] = file_record(hidden_path)
        self.records.append(record); self.total += self.count
        self.parts, self.hidden, self.count = [], [], 0
        return record


def aggregate_update(acc, arrays):
    import numpy as np
    q = arrays['pred_quantity'].astype(np.float64) - arrays['true_quantity'].astype(np.float64)
    acc['count'] += len(q)
    for key, values in (('raw_squared_error_sum', q * q), ('raw_absolute_error_sum', np.abs(q)),
                        ('log_qty_mse_sum', arrays['log_qty_mse']), ('time_nll_sum', arrays['time_nll'])):
        acc[key] += math.fsum(map(float, values))


def run_worker(spec, job, output):
    import numpy as np
    runner_sha = sha(__file__)
    source = Path(job['source_root']).resolve()
    verify_source(source, job['source_files'])
    for key in ('contract', 'checkpoint'):
        require(sha(job[key + '_path']) == job[key + '_sha256'], key + ' SHA changed')
    require(not output.is_relative_to(source), 'Output cannot be inside frozen source')
    sentinel = source / 'sample_data'
    require(sentinel.is_dir() and not any(sentinel.iterdir()), 'Frozen bootstrap requires empty sample_data sentinel')
    install_frozen_imports(source, job['source_files'])
    working = Path(spec['working_dir']).resolve()
    require(working.is_dir() and not working.is_relative_to(source), 'Invalid isolated working directory')
    os.chdir(working)
    import torch
    from paper.scripts.run_multilag_detail_execution import prepare_admitted_data
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from paper.scripts.run_titantpp_mlp_cap36 import population_identity, verify_population
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    # These helpers are imported dynamically by prepare_admitted_data. Bind them before data access.
    import paper.scripts.quantity_comparison_data
    import paper.scripts.run_quantity_comparison
    from simple_lab_test.common.pathing import resolve_project_root
    require(resolve_project_root(source) == source, 'Legacy bootstrap escaped frozen source')
    imports_before_data = verify_imports(source, job['source_files'])
    torch.set_num_threads(spec['num_threads']); torch.manual_seed(job['seed'])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(spec['execution_device'])
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'Requested CUDA device unavailable')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Deterministic CUDA workspace not configured')
        torch.cuda.set_device(device)
        # One reset in this fresh child, before model transfer or any evaluation.
        # This measures this collector's allocator, not other GPU processes.
        torch.cuda.reset_peak_memory_stats(device)
        gpu_name = torch.cuda.get_device_name(device)
        require(gpu_name == spec['expected_device_name'], 'Unexpected GPU identity/name')
        properties = torch.cuda.get_device_properties(device)
        gpu_properties = {'name': properties.name, 'total_memory_bytes': properties.total_memory,
                          'major': properties.major, 'minor': properties.minor,
                          'uuid': str(getattr(properties, 'uuid', 'unavailable'))}
        gpu_properties['uuid_identity'] = verify_gpu_uuid(
            gpu_properties['uuid'], spec.get('expected_device_uuid'), output)
    else:
        require(device.type == 'cpu', 'Unsupported execution device'); gpu_name = None; gpu_properties = None
    contract = json.loads(Path(job['contract_path']).read_text())
    data = deepcopy(next(d for d in contract['datasets'] if d['dataset_id'] == job['dataset_id']))
    input_path, data_root, sidecar = bind_input_layout(spec, job, data, source)
    input_sha = data['inherited_data_identity']['data']['sha256']
    require(input_path.name == 'train_validation.parquet' and sha(input_path) == input_sha, 'Unadmitted input')
    require(data['loader']['lookback_weeks'] == 520 and data['loader']['max_seq_len'] == 256, 'Input window changed')
    frame, metadata = prepare_admitted_data(data, data_root=data_root)
    require(metadata['held_out_materialized'] is False and set(frame['chronological_split'].unique().to_list()) == {'train', 'validation'}, 'Held-out materialized')
    require(Path(metadata['resolved_inputs']['local_data_path']).resolve() == input_path
            and Path(metadata['resolved_inputs']['local_split_manifest_path']).resolve() == sidecar,
            'Frozen helper resolved different input files')
    verify_imports(source, job['source_files'])
    payload = torch.load(job['checkpoint_path'], map_location='cpu', weights_only=False)
    kind = job['checkpoint_kind']; epoch = payload['best_epoch'] if kind == 'best' else payload['epoch']
    require(epoch == job['expected_epoch'] and payload['seed'] == job['seed'], 'Checkpoint epoch/seed differs')
    require(payload['variant'] == 'count_only_log_regression' and payload['checkpoint_monitor'] == 'validation_raw_quantity_rmse', 'Loss or selection rule differs')
    state = payload['model_state_dict']; canonical = canonical_state_dict_sha256(state)
    require(canonical == payload['model_state_sha256'] == job['expected_model_state_sha256'], 'Checkpoint tensor identity differs')
    if job['builder'] == 'component_arm':
        from paper.scripts.run_shared_component_ablation import _build_model
        model, encoder = _build_model(data, 'full'); expected_backbone = 'full'
    elif job['builder'] == 'raw9_shared15':
        from paper.scripts.run_titantpp_shared_history_raw9 import build_model
        expected_backbone = 'titantpp_history_mlp_shared15_v1'
        model, encoder = build_model(data, expected_backbone)
    else: raise ValueError('Unadmitted native builder')
    require(payload['backbone'] == expected_backbone and payload['encoder_config'] == encoder, 'Native architecture metadata mismatch')
    model.load_state_dict(state, strict=True)
    require(type(model.multilag_detail).__name__ == 'SharedHistoryCorrection'
            and type(model.lmm).__name__ == 'HardLocalMemoryMatcher'
            and model.lmm.mem.shape == (1, 64, 64) and model.lmm.topk == 4, 'Wrong full PHEN architecture')
    require(model.time_head_mode == 'heteroscedastic_lognormal_duration' and model.time_scale == 3.
            and model.time_sigma_floor == .001 and model.time_observation_contract['top_code'] is None, 'Wrong time observation law')
    model.to(device=device).eval().requires_grad_(False)
    require({p.dtype for p in model.parameters()} == {torch.float32}, 'Native parameter dtype changed')
    require(not any(m.training for m in model.modules()) and all(p.grad is None for p in model.parameters()), 'Non-eval/gradient state')
    before, context_before = state_sha(model), context_snapshot(model)
    del payload, state
    result = {'schema': SCHEMA, 'mode': spec['mode'], 'job': job['id'], 'seed': job['seed'],
              'checkpoint_kind': kind, 'epoch': epoch, 'checkpoint_sha256': job['checkpoint_sha256'],
              'canonical_checkpoint_state_sha256': canonical, 'source_root': str(source),
              'runner_sha256': runner_sha, 'spec_semantic_sha256': json_sha(spec),
              'numeric_gate_revision': verify_numeric_gate_revision(spec),
              'log_mse_gate_revision': verify_log_mse_gate_revision(spec),
              'input_path': str(input_path), 'input_sha256': input_sha, 'encoder_config': encoder,
              'working_dir': str(working), 'data_root': str(data_root),
              'split_manifest_record': file_record(sidecar), 'imports_verified_before_data': imports_before_data,
              'frozen_import_policy': 'fixed_package_paths_and_sha_bound_meta_path_finder',
              'parameter_buffer_sha_before': before, 'splits': {},
              'runtime': {'python': sys.version, 'torch': torch.__version__, 'device': str(device),
                'gpu_name': gpu_name, 'gpu_properties': gpu_properties,
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                'num_threads': torch.get_num_threads(), 'eval': True, 'autograd': False,
                'optimizer_created': False, 'optimizer_steps': 0, 'tf32': False,
                'matmul_precision': torch.get_float32_matmul_precision(),
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                'historical_gpu_runtime_reproduction_claimed': False},
              'precision': {'quantity': 'native_float32', 'head_hidden': 'native_float32',
                'mu_sigma': 'native_float64_lognormal_time_terms_return', 'time_nll': 'native_float64_recorded_bin_mass'},
              'input_order': 'reference_2048_then_remaining_target_index', 'prototype_aggregation': 'fixed_arithmetic_mean_1_over_4'}
    write_json(output / 'INTENT.json', result)
    for split in job['splits']:
        ref = read_reference(job['reference_results'][split])
        sample = json.loads(verify_ref(spec['reference_samples'][split]).read_text())
        loader = make_loader(frame, target_split=split, shuffle=False, generator=None,
                             batch_size=spec['batch_size'], lookback_weeks=520, max_seq_len=256)
        truth, identity = population_identity(loader.dataset, split)
        verify_population(identity, data['inherited_data_identity']['populations'][split])
        require(len(truth) == spec['population_counts'][split] == POPULATIONS[split], 'Population count mismatch')
        require(sample['population'] == identity and sample['split'] == split
                and sample['identity_sha256'] == json_sha(sample['targets']), 'Frozen sample identity mismatch')
        require([{key: r[key] for key in IDENTITY} for r in ref] == sample['targets'], 'Native references differ from sample')
        indices = fixed_order(len(truth), [r['target_index'] for r in ref], spec['mode'], spec['qualification_batches'], spec['batch_size'])
        identities = [{'target_index': int(i), 'entity_id': str(loader.dataset.parts[loader.dataset.index[int(i)][0]]),
                       'target_seq': int(loader.dataset.seq_lists[loader.dataset.index[int(i)][0]][loader.dataset.index[int(i)][1] + 1]),
                       'raw_quantity': float(truth[int(i)])} for i in indices]
        require(identities[:min(len(indices), 2048)] == sample['targets'][:min(len(indices), 2048)], 'Native data/reference target identity mismatch')
        ordered = torch.utils.data.DataLoader(torch.utils.data.Subset(loader.dataset, indices.tolist()),
                     batch_size=spec['batch_size'], shuffle=False, num_workers=0, collate_fn=loader.collate_fn)
        writer = ChunkWriter(output, split, spec['chunk_rows'], save_hidden=(kind == 'best' and split == 'train'))
        count, batch_count, parity_count = 0, 0, 0
        maximum = {}; started = time.monotonic()
        accumulator = {'count': 0, 'raw_squared_error_sum': 0., 'raw_absolute_error_sum': 0., 'log_qty_mse_sum': 0., 'time_nll_sum': 0.}
        with (output / (split + '_progress.jsonl')).open('x') as progress:
            for bi, (_, dts, mask, _, quantities) in enumerate(ordered):
                dts, mask, quantities = dts.to(device), mask.to(device), quantities.to(device)
                batch_ids = identities[count:count + len(dts)]
                with torch.no_grad(), NativeObserver(model) as observer:
                    outputs = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
                observed = observer.validate(outputs)
                arrays, hidden = export_batch(outputs, observed, batch_ids)
                size = len(batch_ids)
                if count < 2048:
                    require(count + size <= 2048 and all(r['batch_index'] == bi for r in ref[count:count + size]), 'Reference batch structure changed')
                    parity = compare_reference(arrays, hidden, ref[count:count + size],
                                               spec['comparison_tolerances'], output, f'{split}_{bi:05d}')
                    for key, value in parity.items(): maximum[key] = max(maximum.get(key, 0.), value)
                    parity_count += size
                writer.append(arrays, hidden); aggregate_update(accumulator, arrays)
                count += size; batch_count += 1
                record = {'batch': bi, 'rows': size, 'completed_rows': count, 'planned_rows': len(indices),
                          'population_rows': len(truth), 'forward_calls_this_batch': observer.calls,
                          'valid_mask_sha256': observed['valid_mask_sha256'], 'observed_mask_sha256': observed['observed_mask_sha256'],
                          'batch_identity_sha256': json_sha(batch_ids), 'seconds': time.monotonic() - started}
                progress.write(json.dumps(record, allow_nan=False) + '\n'); progress.flush()
                if count % spec['chunk_rows'] == 0 or count == len(indices):
                    require(state_sha(model) == before and context_snapshot(model) == context_before
                            and all(p.grad is None for p in model.parameters()), 'Model state changed during native collection')
                print(json.dumps({'job': job['id'], 'split': split, **record}), flush=True)
            writer.flush()
        require(count == writer.total == len(indices) and parity_count == min(2048, len(indices)), 'Incomplete collection/parity coverage')
        n = accumulator['count']
        metrics = {'quantity_rmse': math.sqrt(accumulator['raw_squared_error_sum'] / n),
                   'quantity_mae': accumulator['raw_absolute_error_sum'] / n,
                   'log_qty_mse': accumulator['log_qty_mse_sum'] / n, 'time_nll': accumulator['time_nll_sum'] / n}
        result['splits'][split] = {'rows': count, 'population_rows': len(truth), 'population_identity': identity,
              'batch_count': batch_count, 'native_reference_rows_checked': parity_count,
              'reference_max_absolute_errors': maximum, 'metrics': metrics, 'sums': accumulator,
              'target_order_identity_sha256': json_sha(identities),
              'chunks': writer.records, 'head_hidden_saved': writer.save_hidden,
              'progress': file_record(output / (split + '_progress.jsonl')), 'seconds': time.monotonic() - started}
        write_json(output / (split + '_manifest.json'), result['splits'][split])
    result['parameter_buffer_sha_after'] = state_sha(model)
    result['runtime']['cuda_allocator_memory'] = ({
        'reset_once_before_model_transfer': True,
        'max_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
        'max_reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
        'final_allocated_bytes': int(torch.cuda.memory_allocated(device)),
        'final_reserved_bytes': int(torch.cuda.memory_reserved(device)),
        'scope': 'This fresh collector process, including model and observation hooks; not total GPU memory or a pure-model benchmark.'
    } if device.type == 'cuda' else {
        'reset_once_before_model_transfer': False,
        'max_allocated_bytes': None, 'max_reserved_bytes': None,
        'final_allocated_bytes': None, 'final_reserved_bytes': None,
        'scope': 'CPU execution; no CUDA allocator measurement.'})
    require(result['parameter_buffer_sha_after'] == before and context_snapshot(model) == context_before
            and all(p.grad is None for p in model.parameters()), 'Final model state changed')
    verify_source(source, job['source_files'])
    require(sha(input_path) == input_sha and sha(__file__) == runner_sha, 'Input/collector changed')
    for key in ('checkpoint', 'contract'):
        require(sha(job[key + '_path']) == job[key + '_sha256'], 'Original changed during collection')
    for split in job['splits']:
        verify_ref(job['reference_results'][split]); verify_ref(spec['reference_samples'][split])
    verify_ref(spec['split_manifest_record'])
    verify_numeric_gate_revision(spec)
    verify_log_mse_gate_revision(spec)
    result.update(status='complete' if spec['mode'] == 'full' else 'qualification_complete',
                  actual_imports=verify_imports(source, job['source_files']), single_forward_per_target=True,
                  held_out_materialized=False, new_training_epochs=0, optimizer_steps=0,
                  parameter_grad_fields_all_none=True, context_state_unchanged=True,
                  input_and_model_immutability_verified=True)
    return result


def validate_spec(spec):
    require(spec['schema'] == SCHEMA and spec['runner_sha256'] == sha(__file__), 'Collector/spec binding differs')
    require(spec['runner_filename'] == Path(__file__).name == 'collector_v5.py', 'Unexpected isolated collector name')
    require(Path(spec['working_dir']).is_absolute() and isinstance(spec['split_manifest_record'], dict),
            'Explicit isolated input layout is required')
    require(spec['mode'] in ('qualification', 'full'), 'Unadmitted collection scope')
    require(spec['batch_size'] == 128 and spec['chunk_rows'] == 4096
            and 1 <= spec['qualification_batches'] <= 16, 'Batch/chunk settings changed')
    require(spec['population_counts'] == POPULATIONS, 'Population contract changed')
    expected_tolerances = {'pred_quantity': {'atol': 1e-5, 'rtol': 1e-5},
                           'time_nll': {'atol': 1e-4, 'rtol': 1e-4},
                           'log_qty_mse': {'atol': 1e-6, 'rtol': 1e-5},
                           'topk_similarity': {'atol': 1e-6, 'rtol': 1e-5}}
    require(spec['comparison_tolerances'] == expected_tolerances, 'Unadmitted comparison tolerances')
    verify_numeric_gate_revision(spec)
    verify_log_mse_gate_revision(spec)
    require(spec['topk_order_policy'] == 'exact_order_no_tie_exception', 'Top4 comparison policy changed')
    require(spec['input_order'] == 'reference_2048_then_remaining_target_index', 'Collection order changed')
    require(spec['execution_device'] in ('cpu', 'cuda:0') and 1 <= spec['num_threads'] <= 8, 'Unadmitted device/thread settings')
    if spec['execution_device'].startswith('cuda'):
        require(isinstance(spec['expected_device_name'], str) and bool(spec['expected_device_name']), 'GPU identity missing')
        canonical_gpu_uuid(spec.get('expected_device_uuid'))
    require(bool(spec['jobs']) and len({j['id'] for j in spec['jobs']}) == len(spec['jobs']), 'Duplicate/empty job set')
    for job in spec['jobs']:
        require(job['dataset_id'] == 'intermittent_frozen_5000' and job['arm'] == 'full'
                and EXPECTED.get((job['seed'], job['checkpoint_kind'])) == job['expected_epoch'], 'Unadmitted checkpoint')
        require(Path(job['id']).name == job['id'] and job['id'] not in ('.', '..'), 'Unsafe job id')
        require(bool(job['splits']) and len(set(job['splits'])) == len(job['splits'])
                and set(job['splits']) <= {'train', 'validation'}, 'Unadmitted job splits')
        if spec['mode'] == 'full':
            require(job['splits'] == (['train'] if job['checkpoint_kind'] == 'best' else ['train', 'validation']),
                    'Full best/last split reuse contract differs')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job', help='Optional one-job selection; still launches an isolated child.')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    spec_path, output = args.spec.resolve(), args.output.resolve()
    require(sha(spec_path) == args.spec_sha, 'Spec SHA mismatch')
    spec = json.loads(spec_path.read_text()); validate_spec(spec)
    if args.worker:
        job = next(j for j in spec['jobs'] if j['id'] == args.worker)
        output.mkdir(parents=False, exist_ok=False)
        try:
            result = run_worker(spec, job, output)
            require(sha(spec_path) == args.spec_sha, 'Spec changed during worker')
            write_json(output / 'COMPLETE.json', result)
        except BaseException as error:
            write_json(output / 'FAILURE.json', {'status': 'failed_preserved', 'exception': type(error).__name__,
                       'message': str(error), 'traceback': traceback.format_exc(), 'retry_attempted': False})
            raise
        return
    jobs = spec['jobs'] if args.job is None else [j for j in spec['jobs'] if j['id'] == args.job]
    require(bool(jobs), 'Unknown requested job')
    output.mkdir(parents=False, exist_ok=False)
    runner_sha = sha(__file__); exits = []
    for job in jobs:
        require(sha(__file__) == runner_sha and sha(spec_path) == args.spec_sha, 'Source/spec changed between children')
        command = [sys.executable, str(Path(__file__).resolve()), '--spec', str(spec_path), '--spec-sha', args.spec_sha,
                   '--output', str(output / job['id']), '--worker', job['id']]
        started = time.monotonic()
        stdout, stderr = output / (job['id'] + '.stdout.log'), output / (job['id'] + '.stderr.log')
        environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': ''}
        if spec['execution_device'].startswith('cuda'): environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        with stdout.open('x') as out, stderr.open('x') as err:
            child = subprocess.run(command, env=environment, stdout=out, stderr=err, check=False)
        receipt = {'job': job['id'], 'argv': command, 'actual_child_exit': child.returncode,
                   'seconds': time.monotonic() - started, 'spec_sha256': args.spec_sha,
                   'collector_sha256': runner_sha, 'stdout': file_record(stdout), 'stderr': file_record(stderr)}
        path = output / (job['id'] + '.ACTUAL_EXIT.json'); write_json(path, receipt)
        exits.append(file_record(path))
        if child.returncode != 0:
            write_json(output / 'FAILURE.json', {'status': 'failed_preserved', 'job': job['id'],
                       'actual_child_exit': child.returncode, 'receipts': exits, 'retry_attempted': False})
            raise RuntimeError('Child failed; original partial output and logs preserved: ' + job['id'])
    require(sha(__file__) == runner_sha and sha(spec_path) == args.spec_sha, 'Final source/spec binding changed')
    write_json(output / 'COMPLETE.json', {'schema': SCHEMA, 'mode': spec['mode'],
        'status': 'complete' if spec['mode'] == 'full' else 'qualification_complete', 'actual_child_exits_all_zero': True,
        'receipts': exits, 'spec_sha256': args.spec_sha, 'collector_sha256': runner_sha,
        'parent_os_exit_directly_observed_by_this_process': False})


if __name__ == '__main__':
    main()
