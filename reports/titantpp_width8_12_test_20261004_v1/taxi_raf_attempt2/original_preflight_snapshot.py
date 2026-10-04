"""Read-only byte/existence checks before a GPU worker; no project/model imports."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def checked(path, digest):
    if not path.is_file():
        raise ValueError(f'Required input path does not exist: {path}')
    if sha(path) != digest:
        raise ValueError(f'Required input SHA differs: {path}')
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def main(root, campaign):
    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    contract = read(campaign / 'execution_contract.json')
    if str(root) != contract['resources']['root']:
        raise ValueError('Preflight root differs from sealed contract')
    results = []
    for name, key in [('evaluation_registry.json', 'registry_sha256'), ('dataset_manifest.json', 'dataset_manifest_sha256'), ('selected_binding.json', 'selected_binding_sha256')]:
        results.append(checked(campaign / name, contract[key]))
    for relative, digest in read(campaign / 'code_seal.json')['files'].items():
        results.append(checked(root / relative, digest))
    registry = read(campaign / 'evaluation_registry.json')
    for bundle in registry['bundles'].values():
        for relative, digest in bundle['source_files'].items():
            source = (root / bundle['source_root']).resolve()
            path = (source / relative).resolve()
            if not path.is_relative_to(source):
                raise ValueError('Frozen source path escapes its root')
            results.append(checked(path, digest))
    for row in registry['rows']:
        results.append(checked(root / row['checkpoint_path'], row['checkpoint_file_sha256']))
    for data in read(campaign / 'dataset_manifest.json')['datasets']:
        path = root / data['path']
        results.append(checked(path, data['sha256']))
        split = data['split_manifest']
        # The approved full-file manifest is adjacent to the approved full parquet.
        # Its original relative name remains preserved in the dataset manifest.
        manifest = Path(split['path']) if Path(split['path']).is_absolute() else path.parent / Path(split['path']).name
        results.append(checked(manifest, split['sha256']))
    return {'status': 'complete', 'checks': len(results), 'files': results,
            'inference_calls': 0, 'gpu_calls': 0, 'data_values_read': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--campaign', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(main(args.root, args.campaign), allow_nan=False))
