"""Render unsmoothed, epoch-matched Train/Validation curves from pilot records."""
from pathlib import Path
import argparse
import csv
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'phen_fixed_position_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASETS = {'raf_spare_parts': 'RAF', 'yellow_trip_hourly': 'Taxi'}
METRICS = [('qty_rmse', 'Quantity RMSE'), ('qty_mae', 'Quantity MAE'),
           ('log_qty_mse', 'Log-quantity MSE'), ('time_nll', 'Recorded time NLL'),
           ('joint_objective', 'Joint objective')]

def render(root, output, baseline_root=None):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for dataset, name in DATASETS.items():
        job = root / 'jobs' / (dataset + '__fixed_sinusoidal__seed42')
        rows = [json.loads(p.read_text()) for p in sorted((job / 'curve_epochs').glob('epoch_*.json'))]
        if not rows:
            continue
        assert all(r['dataset_id'] == dataset and r['seed'] == 42 for r in rows)
        epochs = [r['epoch'] for r in rows]
        assert epochs == list(range(1, max(epochs) + 1)), 'Missing or duplicated epoch'
        for r in rows:
            assert r['monitoring_rng_preserved'] and r['model_state_unchanged']
            records.append({'dataset': dataset, 'epoch': r['epoch'], **{
                f'{split}_{metric}': r[split][metric]
                for split in ('train_eval', 'validation') for metric, _ in METRICS}})
        baseline = []
        if baseline_root is not None:
            p = Path(baseline_root) / (dataset + '__full__seed42') / 'runs/full/count_only_log_regression/seed_42/history.json'
            baseline = json.loads(p.read_text())['history']
        fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.4), constrained_layout=True)
        for ax, (metric, label) in zip(axes.flat, METRICS):
            ax.plot(epochs, [r['train_eval'][metric] for r in rows], color='#2166ac', lw=1.7, label='Fixed position: Train (eval)')
            ax.plot(epochs, [r['validation'][metric] for r in rows], color='#c85a17', lw=1.7, label='Fixed position: Validation')
            if baseline:
                key = {'qty_rmse': 'val_qty_rmse', 'qty_mae': 'val_qty_mae', 'log_qty_mse': 'val_log_qty_mse',
                       'time_nll': 'val_time_nll', 'joint_objective': 'val_joint_objective'}[metric]
                ax.plot([r['epoch'] for r in baseline], [r[key] for r in baseline], color='#777777',
                        ls='--', lw=1.2, alpha=.85, label='Original learned position: Validation')
            ax.set_title(label, fontsize=11)
            ax.set_xlabel('Epoch')
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax = axes.flat[5]
        ax.plot(epochs, [r['train_online']['train_quantity_loss'] for r in rows], color='#6b57a5', lw=1.5, label='Train during updates')
        ax.plot(epochs, [r['train_eval']['log_qty_mse'] for r in rows], color='#2166ac', lw=1.5, label='Train at epoch end (eval)')
        ax.set_title('Train log-quantity loss: measurement modes', fontsize=11)
        ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8, loc='best')
        axes.flat[0].legend(fontsize=7.8, loc='best')
        fit_receipt = job / 'receipt.json'
        exit_receipt = root / (job.name + '.exit.json')
        complete = (fit_receipt.exists() and exit_receipt.exists()
                    and json.loads(fit_receipt.read_text())['status'] == 'complete'
                    and json.loads(exit_receipt.read_text())['actual_os_returncode'] == 0)
        state = 'completed fit' if complete else 'partial; fit not certified complete'
        fig.suptitle(f'{name} | PHEN fixed sinusoidal position | seed 42 | {state}\n'
                     'Train and Validation use the same epoch-end weights in eval mode; no smoothing', fontsize=13)
        fig.savefig(output / (name + '_learning_curves.png'), dpi=180)
        fig.savefig(output / (name + '_learning_curves.pdf'))
        plt.close(fig)
    if records:
        with (output / 'epoch_metrics.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    (output / 'PLOT_METADATA.json').write_text(json.dumps({'source_root': str(root), 'epoch_rows': len(records),
        'datasets': sorted(set(r['dataset'] for r in records)), 'smoothing': False,
        'baseline_train_eval_available': False, 'held_out_test_used': False}, indent=2) + '\n')
    return len(records)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path)
    args = parser.parse_args()
    print(json.dumps({'plotted_epoch_rows': render(args.root, args.output, args.baseline_root)}))
