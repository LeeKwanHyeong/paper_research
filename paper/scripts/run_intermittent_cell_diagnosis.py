#!/usr/bin/env python3
"""Six-state immutable checkpoint diagnosis. GPU work requires a separate bound approval."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from intermittent_cell_controls import (Budget, EvidenceWriter, MEASUREMENT_SHA, check_replay,
    WorkerAuthority, phase_alarm, require, sha_file, sha_json, supervise_child, verify_binding, verify_payload)

PACKAGES = {'paper', 'models', 'simple_lab_test', 'data_loader', 'utils'}
CASES = ('B_log_original', 'mixed_original', 'mixed_matched_original')
FIELDS = ('true_qty', 'pred_qty', 'history_length', 'log_qty_loss', 'raw_qty_loss',
          'quantity_train_loss', 'time_loss', 'objective_loss')


def read(path):
    return json.loads(Path(path).read_text())


def load_overlay(name):
    path = Path(__file__).resolve().with_name(name + '.py')
    spec = importlib.util.spec_from_file_location('_cell_overlay_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_frozen_imports(source, expected_files):
    source = Path(source).resolve()
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] not in PACKAGES:
            continue
        origin = getattr(module, '__file__', None)
        if origin is None:
            for location in getattr(module, '__path__', []):
                require(Path(location).resolve().is_relative_to(source), f'Foreign namespace package: {name}')
            continue
        path = Path(origin).resolve()
        require(path.is_relative_to(source), f'Foreign frozen import: {name}')
        rel = path.relative_to(source).as_posix()
        require(rel in expected_files and sha_file(path) == expected_files[rel], f'Unbound imported file: {name}')


def target_identity(dataset):
    import numpy as np
    return {'series_ids': np.asarray([str(dataset.parts[p]) for p, _ in dataset.index]),
            'target_position': np.asarray([i + 1 for _, i in dataset.index], dtype=np.int64),
            'target_seq': np.asarray([int(dataset.seq_lists[p][i + 1]) for p, i in dataset.index], dtype=np.int64),
            'true_qty': np.asarray([dataset.val_lists[p][i + 1] for p, i in dataset.index], dtype=np.float32)}


def collect_validation(model, loader, forward, batch_tensors, tensor_hash, identity, budget, *, device,
                       expected_count, expected_batches, expected_input_sha):
    import hashlib
    import numpy as np
    import torch
    require(not model.training, 'Validation requires eval mode')
    columns, digest, count, batches = {}, hashlib.sha256(), 0, 0
    with torch.no_grad():
        for batch in loader:
            budget.consume('validation_forward_batches')
            args = batch_tensors(batch, device)
            digest.update(tensor_hash(dict(zip(('dts', 'mask', 'quantities'), args))).encode('ascii'))
            outputs = forward(*args)
            n = len(args[0])
            for key in FIELDS:
                value = outputs[key]
                require(value.ndim == 1 and value.numel() == n and bool(torch.isfinite(value).all()), f'Invalid validation output: {key}')
                columns.setdefault(key, []).append(value.detach().cpu().numpy())
            count += n
            batches += 1
            budget.check()
    require(count == expected_count and batches == expected_batches, 'Validation exposure mismatch')
    require(digest.hexdigest() == expected_input_sha, 'Validation input batch identity mismatch')
    rows = {key: np.concatenate(parts) for key, parts in columns.items()}
    require(np.array_equal(identity['true_qty'], rows['true_qty']), 'Validation target order differs')
    for key in ('series_ids', 'target_position', 'target_seq'):
        require(len(identity[key]) == count, 'Incomplete validation row identity')
        rows[key] = identity[key]
    return rows, {'count': count, 'batches': batches, 'input_sha256': digest.hexdigest()}


def replay_metrics(rows):
    import numpy as np
    error = rows['pred_qty'].astype(np.float64) - rows['true_qty'].astype(np.float64)
    out = {'raw_quantity_rmse': float(np.sqrt(np.mean(error ** 2))), 'quantity_mae': float(np.mean(abs(error)))}
    for key, field in (('legacy_time_loss', 'time_loss'), ('log_quantity_mse', 'log_qty_loss'),
                       ('raw_quantity_scaled_mse', 'raw_qty_loss'), ('quantity_train_loss', 'quantity_train_loss'),
                       ('objective_loss', 'objective_loss')):
        out[key] = float(np.mean(rows[field], dtype=np.float64))
    for name, mask in (('lowest', rows['true_qty'] <= 2), ('body', rows['true_qty'] <= 46), ('tail', rows['true_qty'] > 187)):
        out[name + '_mae'] = float(np.mean(abs(error[mask]))) if bool(mask.any()) else None
    return out


def train_loader_for_probe(dataset, policy, collate, tensor_hash):
    import torch
    from torch.utils.data import DataLoader, Subset
    generator = torch.Generator(device='cpu').manual_seed(policy['selection_seed'])
    indices = torch.randperm(len(dataset), generator=generator)[:policy['selected_sample_count']]
    require(indices.numel() == policy['selected_sample_count'], 'Insufficient train probe targets')
    require(tensor_hash({'indices': indices}) == policy['expected_selected_indices_sha256'], 'Train sample selection differs from calibration')
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=policy['batch_size'], shuffle=False,
                        drop_last=False, num_workers=0, collate_fn=collate,
                        generator=torch.Generator(device='cpu').manual_seed(policy['loader_generator_seed']))
    return loader, indices


def run_train_probe(model, loader, forward, batch_tensors, tensor_hash, stats, gradients, contract, budget, *, emit_record):
    import hashlib
    import random
    import numpy as np
    import torch
    policy, obj = contract['train_probe'], forward.mixed_objective
    accumulator = gradients.GradientAccumulator()
    digest, count, batches = hashlib.sha256(), 0, 0
    model.train()
    for index, batch in enumerate(loader):
        budget.consume('train_forward_batches')
        args = batch_tensors(batch, contract['server']['device'])
        digest.update(tensor_hash(dict(zip(('dts', 'mask', 'quantities'), args))).encode('ascii'))
        with gradients.preserved_model_state(model):
            seed = policy['dropout_seed_base'] + index
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if torch.cuda.is_initialized():
                torch.cuda.manual_seed_all(seed)
            def cell_index(outputs):
                indices = stats.cell_indices(outputs['true_qty'].detach().cpu().numpy(),
                    outputs['history_length'].detach().cpu().numpy(), contract['cells'])
                return torch.as_tensor(indices, dtype=torch.int64, device=outputs['true_qty'].device)
            result = gradients.probe_batch(model, lambda: forward(*args), alpha=obj.alpha,
                quantity_scale=obj.quantity_scale, cell_indices=cell_index, cell_count=15, budget_callback=budget.check)
        require(result.summary['autograd_calls'] <= policy['max_autograd_calls_per_batch'], 'Per-batch backward cap exceeded')
        budget.consume('autograd_calls', result.summary['autograd_calls'])
        accumulator.add(result, count=result.summary['count'])
        count += result.summary['count']
        batches += 1
        emit_record({'batch_index': index, 'dropout_seed': seed, **result.summary})
        # Only one graph and detached running sums survive each iteration.
        del result, args
    require(count == policy['selected_sample_count'] and batches == policy['batches_per_state'], 'Incomplete train probe')
    require(digest.hexdigest() == policy['expected_batch_sha256'], 'Train input batch identity differs from calibration')
    return accumulator.finalize(), {'count': count, 'batches': batches, 'input_sha256': digest.hexdigest(),
                                           'indices_sha256': policy['expected_selected_indices_sha256']}


def worker(contract, binding, approval, started, *, authority=None):
    require(isinstance(authority, WorkerAuthority), 'Worker requires a live owning supervisor')
    context_sha256 = sha_json({'measurement': sha_json(contract), 'binding': sha_json(binding), 'approval': sha_json(approval)})
    authority.check_owner(started=started, limits=contract['limits'], context_sha256=context_sha256)
    verify_binding(contract, binding, Path(__file__).parent, approval=approval, execute=True)
    require(Path(sys.executable).resolve() == Path(contract['server']['python']).resolve(), 'Wrong diagnostic interpreter')
    require(all(os.environ.get(k) == v for k, v in contract['server']['runtime_expected']['environment'].items()), 'Numerical environment differs')
    source = Path(contract['source']['remote_root']).resolve()
    output = Path(contract['server']['output_dir']).resolve()
    require(not output.exists() and not output.is_relative_to(source), 'Refusing existing output/source write')
    require(not any(name.split('.')[0] in PACKAGES for name in sys.modules), 'A clean source-isolated process is required')
    budget = Budget(contract['limits'], started=started, stage_reporter=authority.set_stage)
    budget.check()
    # The lock covers this adapter's single diagnostic job, not unrelated processes.
    import fcntl
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = (output.parent / 'intermittent_cell_gpu0.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Parent owns the final terminal receipt and retains a separate byte reserve.
    writer = EvidenceWriter(output, contract['limits']['max_output_bytes'] - 16384)
    completed, reports, comparisons, attribution, receipts = [], [], [], [], []
    try:
        with phase_alarm(budget):
            for relative, digest in contract['source']['files'].items():
                path = source / relative
                require(path.resolve().is_relative_to(source) and sha_file(path) == digest, f'Frozen source changed: {relative}')
                budget.check()
            busy = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
            require(not busy, 'GPU has existing compute work; no automatic retry')
            sys.path.insert(0, str(source))
            from paper.scripts import quantity_comparison_data as data_helper
            from paper.scripts import quantity_comparison_engine as engine
            from paper.scripts import quantity_comparison_runtime as runtime
            from paper.scripts import run_quantity_comparison as frozen
            from paper.scripts import run_time_quantity_diagnostic as loader_builder
            from paper.scripts import mixed_quantity_objective as mixed
            from paper.scripts.quantity_objective_comparison import QuantityStatistics
            from simple_lab_test.search.common.runner import canonical_state_dict_sha256 as tensor_hash
            import torch
            import intermittent_cell_statistics as stats
            import intermittent_cell_gradients as gradients
            legacy = load_overlay('run_quantity_checkpoint_diagnosis')
            runtime.configure_runtime('cuda:0', 4)
            observed = runtime.runtime_identity('cuda:0')
            require(observed == contract['server']['runtime_expected'], 'Strict runtime differs; no fallback')
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, contract['limits']['max_cuda_allocated_bytes'] / total_memory), 0)
            budget.memory_bytes = lambda: torch.cuda.max_memory_allocated(0)
            verify_frozen_imports(source, contract['source']['files'])
            writer.write('execution_binding.json', binding)
            writer.write('approval_receipt.json', approval)
            writer.write('started.json', {'pid': os.getpid(), 'measurement_sha256': MEASUREMENT_SHA, 'runtime': observed})
            frame, metadata = data_helper.prepare_quantity_comparison_data(contract['dataset'])
            metadata = frozen.bind_frozen_statistics(contract['dataset'], metadata)
            require(metadata['held_out_materialized'] is False, 'Held-out materialized')
            statistics = QuantityStatistics(mu=contract['dataset']['statistics']['train_log_mean'], raw_scale=contract['dataset']['statistics']['raw_scale'])
        lookup = {(s['scope'], s['case']): s for s in contract['states']}
        for scope in ('primary', 'last120'):
            baseline_rows = baseline_metrics = baseline_state = None
            for case in CASES:
                state, arm_ref = lookup[scope, case], contract['arms'][case]
                budget.set_stage('preflight_and_finalize')
                with phase_alarm(budget), gradients.preserved_rng():
                    path = Path(state['checkpoint']['path'])
                    require(path.stat().st_size == state['checkpoint']['bytes'] and sha_file(path) == state['checkpoint']['sha256'], 'Checkpoint file differs')
                    require(sha_file(arm_ref['contract_file']['path']) == arm_ref['contract_file']['sha256'], 'Arm contract file differs')
                    arm = read(arm_ref['contract_file']['path'])
                    require(sha_json(arm) == arm_ref['contract_canonical_sha256'] and arm['identity'] == contract['training_identity'], 'Arm/source/runtime identity differs')
                    payload = legacy.load_checkpoint(path, last=scope == 'last120')
                    weights = verify_payload(payload, state, arm, tensor_hash)
                    model, train, validation, _ = loader_builder.build_arm_inputs({**contract['dataset'], 'seed': 42}, frame, metadata)
                    require(tensor_hash(model.state_dict()) == contract['initial_state_sha256'], 'Initial model differs')
                    require(engine._model_spec(model) == arm['model'], 'Model settings differ')
                    require(engine._loader_spec(validation, training=False) == arm['validation_loader']
                            and engine._loader_spec(train, training=True) == arm['train_loader'], 'Original loader identity differs')
                    require([name for name, p in model.named_parameters() if p.requires_grad] == arm_ref['active_parameter_names'], 'Parameter ordering changed')
                    model.load_state_dict(weights, strict=True)
                    model.to('cuda:0')
                    require(tensor_hash(model.state_dict()) == state['tensor_state_sha256'], 'Loaded model weights differ')
                    obj = mixed.MixedQuantityObjective.from_dict(arm_ref['condition']['mixed_objective'])
                    def forward(*args):
                        return mixed.mixed_joint_causal_batch_objective(model, *args, statistics=statistics, objective=obj)
                    forward.mixed_objective = obj
                    identity = target_identity(validation.dataset)
                    probe_loader, indices = train_loader_for_probe(train.dataset, contract['train_probe'], mixed._collate, tensor_hash)
                    del indices
                with gradients.preserved_model_state(model):
                    budget.set_stage('validation')
                    with phase_alarm(budget):
                        model.eval()
                        rows, validation_receipt = collect_validation(model, validation, forward, engine._batch_tensors,
                            tensor_hash, identity, budget, device='cuda:0', expected_count=86285, expected_batches=675,
                            expected_input_sha=state['expected_validation']['validation_batch_sha256'])
                        metrics = replay_metrics(rows)
                        reference = dict(state['expected_validation']['metrics'])
                        reference.update({k: v for k, v in state['expected_history_metrics'].items() if k in metrics})
                        replay = check_replay(metrics, reference, contract['validation_gates'])
                        summary = stats.summarize_rows(rows, contract['cells']['quantity_boundaries'], contract['cells']['history_boundaries'])
                        report = {'state_id': state['id'], 'epoch': state['epoch'], 'global_step': state['global_step'], 'split': 'validation',
                                  'metrics': metrics, 'replay': replay, 'diagnosis': summary, 'inputs': validation_receipt}
                        if case == CASES[0]:
                            baseline_rows, baseline_metrics, baseline_state = rows, metrics, state
                        else:
                            policy = contract['validation_gates']
                            paired_replay = legacy.verify_pair_replay(baseline_metrics, metrics, baseline_state['expected_validation']['metrics'],
                                state['expected_validation']['metrics'], {'paired_delta_absolute_tolerance': policy['paired_delta_absolute_tolerance'],
                                                                        'paired_delta_relative_tolerance': policy['paired_delta_relative_tolerance']})
                            comparison = {'scope': scope, 'B_state_id': baseline_state['id'], 'candidate_state_id': state['id'], 'replay': paired_replay,
                                          'diagnosis': stats.compare_rows(baseline_rows, rows, contract['cells']['quantity_boundaries'], contract['cells']['history_boundaries'])}
                            comparisons.append(comparison)
                            writer.write(state['id'] + '__paired_validation.json', comparison)
                        reports.append(report)
                        writer.write(state['id'] + '__validation.json', report)
                    budget.set_stage('train_gradient')
                    with phase_alarm(budget):
                        aggregate, sample_receipt = run_train_probe(model, probe_loader, forward, engine._batch_tensors,
                            tensor_hash, stats, gradients, contract, budget,
                            emit_record=lambda row: writer.append('train_batch_gradient_cells.jsonl', {'state_id': state['id'], **row}))
                        attribution.append({'state_id': state['id'], 'split': 'train', 'sample': sample_receipt, 'summary': aggregate})
                        writer.write(state['id'] + '__gradient_summary.json', attribution[-1])
                budget.set_stage('preflight_and_finalize')
                with phase_alarm(budget):
                    require(tensor_hash(model.state_dict()) == state['tensor_state_sha256'] and sha_file(path) == state['checkpoint']['sha256'], 'Checkpoint/model mutation')
                    receipts.append({'state_id': state['id'], 'checkpoint_sha256': state['checkpoint']['sha256'], 'tensor_sha256': state['tensor_state_sha256'],
                                     'immutability_passed': True, 'validation_replay_passed': True, 'sample': sample_receipt})
                    completed.append(state['id'])
                    writer.write(state['id'] + '__receipt.json', receipts[-1])
                    del model, train, validation, probe_loader, payload, weights, rows
                    gc.collect()
                    budget.check()
            del baseline_rows
        with phase_alarm(budget):
            require(len(completed) == 6 and len(comparisons) == 4, 'Incomplete state/pair set')
            require(budget.counts['validation_forward_batches'] == 4050 and budget.counts['train_forward_batches'] == 384, 'Incomplete total exposure')
            verify_frozen_imports(source, contract['source']['files'])
            require(runtime.runtime_identity('cuda:0') == observed, 'Runtime drifted during diagnosis')
            for relative, digest in contract['source']['files'].items():
                require(sha_file(source / relative) == digest, 'Source changed during diagnosis')
            writer.write('preflight_and_checkpoint_receipt.json', {'runtime': observed, 'checkpoints': receipts, 'source_files': 70})
            writer.write('train_sample_manifest.json', {'selection': contract['train_probe'], 'states': [r['sample'] for r in receipts]})
            writer.write('per_state_validation_summary.json', reports)
            writer.write('four_paired_validation_comparisons.json', comparisons)
            writer.write('gradient_attribution_summary.json', attribution)
            writer.write('conservation_and_immutability_audit.json', {'passed': True, 'states': completed, 'paired_comparisons': 4,
                'budget': budget.snapshot(), 'optimizer_updates': 0, 'held_out_accessed': False})
            writer.write('worker_terminal_status.json', {'status': 'complete', 'states_completed': completed, 'budget': budget.snapshot(),
                'measurement_sha256': MEASUREMENT_SHA, 'optimizer_updates': 0, 'held_out_accessed': False}, terminal=True)
    except BaseException as error:
        if not (output / 'worker_terminal_status.json').exists():
            writer.write('worker_terminal_status.json', {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)[:8000],
                'states_completed': completed, 'budget': budget.snapshot(), 'automatic_retry': False, 'optimizer_updates': 0}, terminal=True)
        raise
    finally:
        lock.close()


def supervised(args, contract, binding, approval, started):
    verify_binding(contract, binding, Path(__file__).parent, approval=approval, execute=True)
    output = Path(contract['server']['output_dir'])
    # Acquire before output inspection/spawn. A losing supervisor must never
    # publish failure into the output owned by the winning supervisor.
    import fcntl
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / 'intermittent_cell_supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not output.exists(), 'Existing output cannot be resumed or overwritten')
        context_sha256 = sha_json({'measurement': sha_json(contract), 'binding': sha_json(binding), 'approval': sha_json(approval)})
        command = [sys.executable, str(Path(__file__).resolve()), '--contract', str(args.contract.resolve()),
                   '--binding', str(args.binding.resolve()), '--approval', str(args.approval.resolve()), '--worker', '--parent-start', str(started)]
        receipt = None

        def write_terminal(payload):
            output.mkdir(exist_ok=True)
            encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
            used = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
            require(len(encoded) <= 16384 and used + len(encoded) <= contract['limits']['max_output_bytes'],
                    'Parent terminal receipt exceeds reserved output budget')
            with (output / 'terminal_status.json').open('xb') as handle:
                handle.write(encoded)

        try:
            receipt = supervise_child(command, contract['limits'], started=started, context_sha256=context_sha256)
            require(receipt['authenticated'] and receipt['exit_code'] == 0
                    and receipt['execution_context_sha256'] == context_sha256, 'Unauthenticated supervisor completion')
            worker_terminal = output / 'worker_terminal_status.json'
            require(worker_terminal.is_file() and worker_terminal.stat().st_size <= 16384, 'Worker terminal receipt missing or oversized')
            worker_status = read(worker_terminal)
            require(worker_status.get('status') == 'complete' and worker_status.get('measurement_sha256') == MEASUREMENT_SHA,
                    'Worker did not provide a complete measurement receipt')
            # Include parent finalization time, after the child has exited, in
            # the same cumulative preflight/finalization and total caps.
            final_budget = receipt['budget']
            elapsed = time.monotonic() - started
            final_budget['stage_seconds']['preflight_and_finalize'] += max(0.0, elapsed - final_budget['elapsed_seconds'])
            final_budget['elapsed_seconds'] = elapsed
            if elapsed >= contract['limits']['max_wall_seconds']:
                raise TimeoutError('Total diagnostic deadline reached during parent finalization')
            for stage, seconds in final_budget['stage_seconds'].items():
                if seconds >= contract['limits']['stage_seconds'][stage]:
                    raise TimeoutError(f'{stage} cumulative deadline reached during parent finalization')
            write_terminal({**worker_status, 'status': 'complete', 'authority': 'owning_supervisor',
                            'execution_context_sha256': context_sha256, 'parent_budget': final_budget,
                            'supervision': {'authenticated': True, 'worker_exit_code': 0,
                                            'stage_transitions': receipt['stage_transitions']}})
            return receipt
        except BaseException as error:
            # A worker-written complete receipt remains evidence only; it can
            # never override a supervisor timeout or abnormal child exit.
            if not (output / 'terminal_status.json').exists():
                write_terminal({'status': 'timeout' if isinstance(error, TimeoutError) else 'failed',
                                'authority': 'owning_supervisor', 'error_type': type(error).__name__,
                                'error': str(error)[:1000], 'measurement_sha256': MEASUREMENT_SHA,
                                'execution_context_sha256': context_sha256, 'automatic_retry': False,
                                'optimizer_updates': 0, 'held_out_accessed': False,
                                'parent_budget': getattr(error, 'supervision_budget', receipt['budget'] if receipt else None)})
            raise


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--binding', required=True, type=Path)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--parent-start', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    authority = None
    try:
        if args.worker:
            require(args.parent_start is not None, 'Worker requires a supervisor deadline')
            authority = WorkerAuthority.from_environment(started=args.parent_start)
        else:
            require(args.parent_start is None, 'Only a supervised worker may inherit a deadline')
        contract, binding = read(args.contract), read(args.binding)
        approval = read(args.approval) if args.approval else None
        execute = args.execute or args.worker
        receipt = verify_binding(contract, binding, Path(__file__).parent, approval=approval, execute=execute)
        if not execute:
            print(json.dumps(receipt | {'model_libraries_loaded': False, 'gpu_executed': False}, indent=2))
        elif args.worker:
            require(0 <= started - args.parent_start < contract['limits']['max_wall_seconds'], 'Invalid parent deadline')
            worker(contract, binding, approval, args.parent_start, authority=authority)
            authority.finish()
        else:
            supervised(args, contract, binding, approval, started)
    finally:
        if authority is not None:
            authority.close()


if __name__ == '__main__':
    main()
