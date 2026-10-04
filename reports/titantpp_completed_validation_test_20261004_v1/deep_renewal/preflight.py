"""Strict selected-checkpoint reconstruction in each original source namespace."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def check(index):
    registry = json.loads((HERE / 'evaluation_registry.json').read_text())
    row = registry['rows'][index]
    bundle = registry['bundles'][row['evaluator_source_bundle']]
    source = (ROOT / bundle['source_root']).resolve()
    for relative, expected in bundle['source_files'].items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected
    sys.dont_write_bytecode = True
    sys.path = [str(source)] + [p for p in sys.path if not p or not Path(p).resolve().is_relative_to(ROOT)]
    import torch
    factory = importlib.import_module('models.TPPs.CountAwareFactory')
    runner = importlib.import_module('simple_lab_test.search.common.runner')
    spec = next(d for d in bundle['datasets'] if d['dataset_id'] == row['dataset'])
    checkpoint = ROOT / row['checkpoint_path']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['checkpoint_file_sha256']
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert cp['best_epoch'] == row['selected_epoch'] and cp['backbone'] == row['model']
    factory.validate_checkpoint_route(cp, row['model'])
    assert runner.canonical_state_dict_sha256(cp['model_state_dict']) == row['state_tensor_sha256']
    config = {k: v for k, v in spec['model'].items() if k not in
              ('backbone', 'lambda_log_qty', 'lambda_tail', 'time_head_lr_multiplier')}
    config.update(quantity_variant='shifted_nb_quantity_nll', time_head_mode='shifted_nb_duration')
    model, _ = factory.build_count_aware_model(row['model'], **config,
        train_log_mean=spec['statistics']['train_log_mean'],
        train_log_std=spec['statistics']['train_log_std'], max_seq_len=spec['loader']['max_seq_len'])
    model.load_state_dict(cp['model_state_dict'], strict=True)
    model.eval()
    assert runner.canonical_state_dict_sha256(model.state_dict()) == row['state_tensor_sha256']
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        name = getattr(module, '__name__', '')
        if path and name.split('.')[0] in {'paper', 'models', 'data_loader', 'simple_lab_test'}:
            origin = Path(path).resolve()
            assert origin.is_relative_to(source), (name, path)
            rel = str(origin.relative_to(source))
            assert rel in bundle['source_files'], rel
    print(json.dumps({'dataset': row['dataset'], 'model': row['model'], 'seed': row['seed'],
        'selected_epoch': row['selected_epoch'], 'strict_native_reconstruction': True,
        'state_tensor_sha256': row['state_tensor_sha256'], 'device': 'local_cpu',
        'test_read': False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.index is not None:
        check(args.index)
        return
    registry = json.loads((HERE / 'evaluation_registry.json').read_text())
    results = []
    for index in range(len(registry['rows'])):
        value = subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
            '--index', str(index)], text=True, timeout=120)
        results.append(json.loads(value.splitlines()[-1]))
    (HERE / 'local_preflight.json').write_text(json.dumps({'status': 'passed',
        'conditions': len(results), 'rows': results, 'inference_started': False,
        'registry_sha256': hashlib.sha256((HERE / 'evaluation_registry.json').read_bytes()).hexdigest()},
        indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'conditions': len(results)}))


if __name__ == '__main__':
    main()
