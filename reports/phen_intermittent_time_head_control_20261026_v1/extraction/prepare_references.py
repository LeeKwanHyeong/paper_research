"""Create scalar-only parity references from verified saved Train/Validation outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parents[1] / 'phen_intermittent_full_observations_20261026_v1'
COLLECTION_SHA = '18a9f78e3a0355d502fbd7789161ea6ef1ddcb9ca1fc7410397ee6672b1b0461'
METADATA_SHA = 'dcee4a92d729a1937b1f47e64b317e49e4043adf6dfce882fa0ace8398b5edb8'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def verified(ref):
    path = Path(ref['path'])
    assert path.stat().st_size == ref['bytes'] and sha(path) == ref['sha256'], path
    return path


def main(output):
    output.mkdir(exist_ok=False)
    collection_path = PRIOR / 'FINAL_COLLECTION.json'
    assert sha(collection_path) == COLLECTION_SHA
    collection = json.loads(collection_path.read_text())
    assert collection['complete'] and collection['verified_rows'] == 1920436
    meta_path = PRIOR / 'support/results_v1/target_metadata.parquet'
    assert sha(meta_path) == METADATA_SHA
    meta = pd.read_parquet(meta_path, columns=['split', 'target_index', 'entity_id', 'target_seq', 'raw_quantity', 'history_length', 'target_gap'])
    manifest = {'source_collection': record(collection_path), 'source_metadata': record(meta_path), 'references': [], 'reuse_best_train': []}
    for seed in (52, 42):
        for cohort in [c for c in collection['cohorts'] if c['seed'] == seed]:
            kind, split = cohort['checkpoint'], cohort['split']
            assert cohort['actual_child_exit'] == 0 and cohort['model_state_unchanged']
            if kind == 'best' and split == 'train':
                assert cohort['hidden_rows'] == 393824 and all('head_hidden' in c for c in cohort['chunks'])
                manifest['reuse_best_train'].append(cohort)
                continue
            historical = cohort['source'] == 'historical_validation_reuse'
            refs = [cohort['parquet']] if historical else [c['observations'] for c in cohort['chunks']]
            aliases = {'target_gap': 'true_recorded_gap' if historical else 'target_gap',
                       'pred_quantity': 'predicted_raw_quantity' if historical else 'pred_quantity',
                       'time_location_float64': 'time_location' if historical else 'time_location_float64',
                       'time_sigma_float64': 'time_sigma' if historical else 'time_sigma_float64'}
            identity = ['target_index', 'entity_id', 'target_seq', 'raw_quantity', 'history_length']
            columns = identity + ['time_nll'] + list(aliases.values())
            frame = pd.concat([pq.read_table(verified(ref), columns=columns).to_pandas() for ref in refs], ignore_index=True)
            frame = frame.rename(columns={value: key for key, value in aliases.items()}).sort_values('target_index').reset_index(drop=True)
            expected = meta.loc[meta.split == split].sort_values('target_index').reset_index(drop=True)
            assert len(frame) == len(expected) == cohort['rows']
            for key in identity + ['target_gap']:
                assert np.array_equal(frame[key], expected[key]), (seed, kind, split, key)
            assert np.array_equal(frame.target_index, np.arange(len(frame)))
            arrays = {key: frame[key].to_numpy(dtype=str if key == 'entity_id' else np.int64 if key in ('target_index', 'target_seq', 'history_length', 'target_gap') else np.float64) for key in frame}
            assert all(np.isfinite(value).all() for key, value in arrays.items() if key != 'entity_id')
            path = output / f'seed{seed}_{kind}_{split}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            manifest['references'].append({'seed': seed, 'checkpoint': kind, 'split': split, 'rows': len(frame),
                'checkpoint_sha256': cohort['checkpoint_sha256'], 'mu_sigma_precision': 'exported_float32_original' if historical else 'native_float64',
                'reference': record(path), 'source_observations': refs})
            print(json.dumps({'seed': seed, 'checkpoint': kind, 'split': split, 'rows': len(frame), 'output': str(path)}), flush=True)
    assert len(manifest['references']) == 6 and len(manifest['reuse_best_train']) == 2
    manifest['no_inference'] = True
    (output / 'REFERENCES.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'references_v1')
    args = parser.parse_args()
    main(args.output.resolve())
