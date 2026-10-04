"""Revalidate the four completed width16 seed42 populations without inference."""
import importlib.util
import json
import math
from pathlib import Path

from prepare import HERE, ROOT, OLD, read, sha, write


def main():
    specification = importlib.util.spec_from_file_location('prior_width16_completion', OLD / 'complete_evaluation.py')
    old = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(old)
    registry = read(HERE / 'evaluation_registry.json')
    old_contract = read(OLD / 'execution_contract.json')
    old_gate = read(OLD / 'validation_gate.json')
    assert old_gate['passed'] is True
    assert sha(OLD / 'evaluate.py') == old_contract['evaluator']['sha256']
    assert sha(OLD / 'evaluation_registry.json') == old_contract['registry_sha256']
    assert sha(OLD / 'dataset_manifest.json') == old_contract['dataset_manifest_sha256']
    rows = []
    for dataset in ('yellow_trip_hourly', 'raf_spare_parts'):
        row = next(r for r in registry['rows'] if r['dataset'] == dataset and r['seed'] == 42)
        for split in ('validation', 'test'):
            folder = OLD / 'runs' / split / dataset
            receipt, frame = old.load_predictions(folder)
            assert receipt['dataset'] == dataset and receipt['seed'] == 42 and receipt['split'] == split
            assert receipt['contract_sha256'] == sha(OLD / 'execution_contract.json')
            assert receipt['runner_sha256'] == old_contract['evaluator']['sha256']
            assert all(receipt['qualification'].values())
            for field in ('checkpoint_file_sha256', 'state_tensor_sha256', 'selected_epoch'):
                assert receipt[field] == row[field]
            result = old.metrics(frame)
            if split == 'validation':
                reference = old_contract['validation_gate']['references'][dataset]
                assert result['count'] == reference['count']
                for name in ('qty_rmse', 'qty_mae', 'time_nll'):
                    assert math.isclose(result[name], reference[name], rel_tol=1e-5, abs_tol=1e-5)
            rows.append({'dataset': dataset, 'model': row['model'], 'seed': 42, 'split': split,
                'folder': str(folder.relative_to(ROOT)), 'receipt_sha256': sha(folder / 'receipt.json'),
                'target_identity_sha256': receipt['target_identity_sha256'], 'truth_sha256': receipt['truth_sha256'],
                'checkpoint_file_sha256': receipt['checkpoint_file_sha256'],
                'state_tensor_sha256': receipt['state_tensor_sha256'], 'selected_epoch': receipt['selected_epoch'],
                'parts': receipt['parts'], 'metrics': result,
                'part_bytes_and_target_truth_digests_recomputed': True, 'new_inference': False})
    write('reused_predictions.json', {'status': 'verified', 'rows': rows})
    print(json.dumps({'reused_split_count': len(rows), 'new_inference': False}))


if __name__ == '__main__':
    main()
