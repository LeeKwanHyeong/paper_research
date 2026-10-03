"""Recompute descriptive batch summaries; no inference or training is performed."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROBE = ROOT / 'gradient_probe_v1'


def main():
    status = json.loads((PROBE / 'status.json').read_text())
    assert status['status'] == 'passed' and status['optimizer_updates'] == 0
    rows = []
    for item in status['datasets']:
        path = PROBE / (item['dataset'] + '.json')
        dataset = json.loads(path.read_text())
        for name, model in dataset['models'].items():
            records = model['records']
            assert len(records) == 16 and model['state_unchanged']
            assert all(model['pinned_joint_parity'][key] for key in ['objective_exact', 'all_parameter_gradients_exact'])
            cosine = [r['encoder_cosine'] for r in records if r['encoder_cosine_applicable']]
            ratio = [r['global_joint_clip_factor'] / r['global_quantity_clip_factor'] for r in records]
            rows.append({
                'dataset': dataset['dataset'], 'model': name, 'epoch': model['best_epoch'],
                'batch_count': len(records), 'targets': dataset['sample_count'],
                'encoder_cosine_median': statistics.median(cosine) if cosine else None,
                'encoder_cosine_min': min(cosine) if cosine else None,
                'encoder_cosine_max': max(cosine) if cosine else None,
                'negative_encoder_cosine_batches': sum(x < 0 for x in cosine),
                'applicable_encoder_cosine_batches': len(cosine),
                'joint_clipped_batches': sum(r['global_joint_clip_factor'] < 1 for r in records),
                'quantity_only_clipped_batches': sum(r['global_quantity_clip_factor'] < 1 for r in records),
                'joint_clip_factor_median': statistics.median(r['global_joint_clip_factor'] for r in records),
                'quantity_only_clip_factor_median': statistics.median(r['global_quantity_clip_factor'] for r in records),
                'joint_over_quantity_clip_factor_median': statistics.median(ratio),
                'time_head_joint_squared_norm_share_median': statistics.median(r['time_head_joint_squared_norm_share'] for r in records),
                'source': str(path.relative_to(ROOT)),
                'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    output = {
        'scope': 'train-only selected-checkpoint descriptive snapshot; not optimizer update or training trajectory',
        'aggregation': 'unweighted median across 16 equal-size batches; clipping counts use coefficient < 1',
        'uncertainty': 'overlapping histories and one seed; no independence assumption or significance claim',
        'rows': rows,
    }
    (ROOT / 'gradient_summary.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
