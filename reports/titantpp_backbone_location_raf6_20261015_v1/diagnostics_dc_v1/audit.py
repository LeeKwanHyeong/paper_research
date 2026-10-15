"""Inspect only frozen tensor parameters and previously saved Validation scores.

No project module import, model class construction, model forward, or remote I/O.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3'
PINS = {
    'contract.json': 'd51ab06365a7165e00cc18849b84caba4ee2dd8af6a36aee84a5c1d62e01fab3',
    'retrieval/manifest.json': '9b7044490727a42add4b9e05538fdf315137f0a956e25d8b1422f59a37430819',
}
SEEDS = (42, 52, 62)
BOUNDARIES = (2., 30., 60., 200.)
CELL_NAMES = ('q<=2', '2<q<=30', '30<q<=60', '60<q<=200', 'q>200')
INPUTS = {}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path, expected):
    need(sha(path) == expected, 'SHA mismatch: ' + str(path))
    INPUTS[str(path.relative_to(ROOT))] = expected
    return path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def state_hash(state):
    """Byte-identical frozen runner canonical hash, without importing that runner."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        need(isinstance(tensor, torch.Tensor), 'Non-tensor checkpoint state')
        tensor = tensor.detach().cpu().contiguous()
        metadata = json.dumps({'name': name, 'dtype': str(tensor.dtype), 'shape': list(tensor.shape)},
                              sort_keys=True, separators=(',', ':')).encode()
        digest.update(len(metadata).to_bytes(8, 'big')); digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes() if tensor.numel() else b''
        digest.update(len(raw).to_bytes(8, 'big')); digest.update(raw)
    return digest.hexdigest()


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    need(values.size > 0 and np.isfinite(values).all(), 'Invalid audit vector')
    return dict(min=float(values.min()), q10=float(np.quantile(values, .1)), median=float(np.median(values)),
                q90=float(np.quantile(values, .9)), max=float(values.max()), mean=float(values.mean()))


def coefficient_statistics(kernel):
    need(kernel.shape == (3, 64) and np.isfinite(kernel).all(), 'Wrong finite kernel shape')
    signed = kernel.sum(axis=0)
    l1 = np.abs(kernel).sum(axis=0)
    l2 = np.sqrt(np.square(kernel).sum(axis=0))
    ratio = np.divide(np.abs(signed), l1, out=np.zeros_like(l1), where=l1 != 0)
    need(np.all(ratio <= 1. + 1e-14), 'DC cancellation ratio exceeds triangle inequality')
    vectors = dict(signed_sum_residual_dc=signed, abs_sum_residual_dc=np.abs(signed),
                   total_constant_gain_1_plus_sum=1. + signed, L1=l1, L2=l2)
    results = {key: summary(value) for key, value in vectors.items()}
    results['abs_sum_over_L1'] = summary(ratio[l1 != 0]) if np.any(l1 != 0) else None
    channels = []
    for i in range(64):
        channels.append(dict(channel=i, coefficients_current_lag1_lag2=kernel[:, i].tolist(),
                             **{key: float(value[i]) for key, value in vectors.items()},
                             abs_sum_over_L1=float(ratio[i]) if l1[i] != 0 else None))
    return dict(channels=channels, summary=results, L1_exact_zero_channels=int((l1 == 0).sum()),
                sum_exact_zero_channels=int((signed == 0).sum()),
                prefix_sum_for_one_available_observation=summary(kernel[0]),
                prefix_sum_for_two_available_observations=summary(kernel[:2].sum(axis=0)))


def arithmetic_checks():
    # Analytic residual arithmetic only, no torch model or forward invocation.
    kernel = np.tile(np.array([2., -3., 1.])[:, None], (1, 64))
    result = coefficient_statistics(kernel)
    need(result['sum_exact_zero_channels'] == 64 and result['summary']['abs_sum_over_L1']['max'] == 0., 'Zero DC check failed')
    need(result['prefix_sum_for_one_available_observation']['median'] == 2.
         and result['prefix_sum_for_two_available_observations']['median'] == -1., 'Boundary partial sums failed')
    zero = coefficient_statistics(np.zeros((3, 64)))
    need(zero['L1_exact_zero_channels'] == 64 and zero['summary']['abs_sum_over_L1'] is None, 'Zero denominator check failed')
    constant = np.array([7., 7., 7.])
    need(float(np.dot(kernel[:, 0], constant)) == 0., 'Constant projection arithmetic failed')
    return dict(zero_sum_constant_response=True, partial_history_boundary=True, zero_denominator=True)


def observed_validation_bias(contract, manifest):
    rows, evidence = [], []
    expected_pop = contract['datasets']['raf_spare_parts']['definition']['inherited_data_identity']['populations']['validation']
    expected_input = contract['datasets']['raf_spare_parts']['sha256']
    for job in contract['jobs']:
        if job['dataset'] != 'raf_spare_parts':
            continue
        base = BUNDLE / 'retrieval/results' / job['id'] / 'validation'
        rel = f"results/{job['id']}/validation/receipt.json"
        receipt = read(pin(base / 'receipt.json', manifest['files'][rel]))
        need(receipt['split'] == 'validation' and not receipt['held_out_test_evaluated']
             and receipt['population'] == expected_pop and receipt['baseline_validation_reproduced'], 'Baseline Validation identity differs')
        need(receipt['checkpoint_sha256'] == job['original']['checkpoint']['sha256'], 'Baseline checkpoint differs')
        paths = [pin(base / 'predictions' / part['path'], part['sha256']) for part in receipt['prediction_manifest']['parts']]
        frame = pl.scan_parquet(paths).select('entity_id', 'seq', 'seed', 'split', 'raw_quantity', 'prediction_1').collect()
        need(frame.height == 6690 and set(frame['split']) == {'validation'} and set(frame['seed']) == {job['seed']}, 'Baseline row scope differs')
        q, pred = frame['raw_quantity'].to_numpy(), frame['prediction_1'].to_numpy()
        need(np.isfinite(q).all() and np.isfinite(pred).all(), 'Nonfinite prediction')
        errors = pred - q
        bins = np.searchsorted(BOUNDARIES, q, side='left')
        for k in range(5):
            e = errors[bins == k]
            rows.append(dict(model='CNN', seed=job['seed'], bin=k, target_quantity_condition=CELL_NAMES[k],
                             count=len(e), qty_sse=float(np.square(e).sum()), qty_absolute_error_sum=float(np.abs(e).sum()),
                             qty_signed_error_sum=float(e.sum()), qty_rmse=float(np.sqrt(np.square(e).mean())),
                             qty_mae=float(np.abs(e).mean()), qty_bias=float(e.mean())))
        evidence.append(dict(model='CNN', seed=job['seed'], selected_epoch=receipt['selected_epoch'],
                             checkpoint_sha256=receipt['checkpoint_sha256'], source_closure=contract['source']['closure'],
                             runtime_host='5080', cudnn=91900, population=receipt['population'], receipt_path=str((base / 'receipt.json').relative_to(ROOT))))
    candidate_path = ROOT / 'reports/titantpp_backbone_location_raf6_20261015_v1/evaluation_v1/THREE_SEED_VALIDATION.json'
    report = read(pin(candidate_path, 'e0f628b627acef88fc5c239f387343f8241af328490b46b687af4a853f637767'))
    training_path = ROOT / 'search_artifacts/titantpp_backbone_location_raf6_20261015_v1/retrieval/final_receipt.json'
    training = read(pin(training_path, '4c3257762b90ff7edfe35b71d10d63cfabb14635c4f1720ec29b3e5a919fbbf6'))
    need(report['split'] == 'validation' and training['status'] == 'complete' and training['new_fit_count'] == 6
         and not training['held_out_test_evaluated'], 'Candidate scope/receipt differs')
    arms = {'titantpp_cnn_query_nonlinear_v1': 'Query', 'titantpp_cnn_value_nonlinear_v1': 'Value'}
    candidates = [r for r in report['seed_results'] if r['arm'] in arms]
    need({(r['arm'], r['seed']) for r in candidates} == {(a, s) for a in arms for s in SEEDS} and len(candidates) == 6, 'Candidate cohort differs')
    for candidate in candidates:
        need(candidate['split'] == 'validation' and not candidate['held_out_test_evaluated']
             and candidate['native_runtime_checked'] and candidate['status'] == 'complete'
             and candidate['source_closure'] == 'b06f495d57642d88585cbf8d85b410debad9bbc887edd7450ab6e9b4c8d08524'
             and candidate['original_population_identity'] == expected_pop and candidate['data_sha256'] == expected_input,
             'Candidate Validation proof differs')
        matches = [p for p, digest in training['files'].items() if p.endswith('best_val_qty_rmse_model.pt') and digest == candidate['checkpoint_sha256']]
        need(len(matches) == 1 and candidate['job_id'] in matches[0], 'Candidate selected binary not in training receipt')
        pin(ROOT / matches[0], candidate['checkpoint_sha256'])  # Hash only; do not load candidate models.
        need(candidate['metrics']['quantity_boundaries'] == list(BOUNDARIES), 'Target quantity boundaries differ')
        for cell in candidate['metrics']['quantity_cells']:
            n, mse, mae, bias = cell['count'], cell['qty_sse'], cell['qty_absolute_error_sum'], cell['qty_signed_error_sum']
            need(math.isclose(math.sqrt(mse / n), cell['qty_rmse'], rel_tol=1e-12)
                 and math.isclose(mae / n, cell['qty_mae'], rel_tol=1e-12)
                 and math.isclose(bias / n, cell['qty_bias'], rel_tol=1e-12), 'Candidate signed sums do not reconstruct metrics')
            rows.append(dict(model=arms[candidate['arm']], seed=candidate['seed'], bin=cell['bin'],
                             target_quantity_condition=CELL_NAMES[cell['bin']],
                             **{k: cell[k] for k in ('count', 'qty_sse', 'qty_absolute_error_sum', 'qty_signed_error_sum', 'qty_rmse', 'qty_mae', 'qty_bias')}))
        evidence.append(dict(model=arms[candidate['arm']], seed=candidate['seed'], selected_epoch=candidate['selected_epoch'],
                             checkpoint_sha256=candidate['checkpoint_sha256'], state_sha256=candidate['state_sha256'],
                             source_closure=candidate['source_closure'], runtime_host='5090', cudnn=92000,
                             population=candidate['original_population_identity']))
    aggregates = []
    for model in ('CNN', 'Query', 'Value'):
        for k in range(5):
            selected = [r for r in rows if r['model'] == model and r['bin'] == k]
            need(sorted(r['seed'] for r in selected) == list(SEEDS) and len({r['count'] for r in selected}) == 1, 'Incomplete subgroup')
            metrics = {m: dict(mean=statistics.mean(r[m] for r in selected), sample_sd=statistics.stdev(r[m] for r in selected))
                       for m in ('qty_rmse', 'qty_mae', 'qty_bias')}
            aggregates.append(dict(model=model, bin=k, target_quantity_condition=CELL_NAMES[k], count=selected[0]['count'], metrics=metrics))
    for k in range(5):
        need(len({r['count'] for r in rows if r['bin'] == k}) == 1, 'Models score different bin populations')
    return dict(split='validation', target_defined_descriptive_statistics=True, per_seed=rows, three_seed=aggregates,
                evidence=evidence, interpretation=read(OUT / 'BIAS_PLAN.json')['interpretation'])


def main():
    need(not (OUT / 'RESULT.json').exists(), 'Refuse repeated audit or overwrite')
    pin(OUT / 'PLAN.json', '1e674e091c647507c86ba6333899fc95b99810bc38fba82d9fb5b971cf6bcae1')
    pin(OUT / 'BIAS_PLAN.json', '16a9d34bb40d0682c92df00f0d03f3900d2c29d6ae28c3b50ab67d12ed751b20')
    contract = read(pin(BUNDLE / 'contract.json', PINS['contract.json']))
    manifest = read(pin(BUNDLE / 'retrieval/manifest.json', PINS['retrieval/manifest.json']))
    source = contract['source']
    need(source['closure'] == '15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53', 'Unexpected frozen source')
    need(hashlib.sha256(json.dumps(source['files'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest() == source['closure'], 'Source map closure differs')
    for rel, expected in source['files'].items():
        pin(BUNDLE / 'source' / rel, expected)
    audits = []
    for job in contract['jobs']:
        if job['dataset'] != 'raf_spare_parts':
            continue
        original = job['original']; cp = original['checkpoint']
        path = pin(BUNDLE / cp['relative_path'], cp['sha256'])
        payload = torch.load(path, map_location='cpu', weights_only=True)
        need(payload['backbone'] == 'titantpp_cnn_gru54_cnn_only' and payload['seed'] == job['seed']
             and payload['best_epoch'] == original['selected_epoch'] and not payload['held_out_test_evaluated'], 'Checkpoint model/selection differs')
        state = payload['model_state_dict']
        need(state_hash(state) == payload['model_state_sha256'] == original['checkpoint_state_sha256'], 'Tensor state hash differs')
        expected_keys = {f'encoder.layers.0.attn.causal_{branch}_kernel' for branch in 'qkv'}
        need({k for k in state if 'causal_' in k and k.endswith('_kernel')} == expected_keys, 'Unexpected CNN key set')
        for branch in 'qkv':
            key = f'encoder.layers.0.attn.causal_{branch}_kernel'
            tensor = state[key]
            need(tensor.shape == (3, 64) and tensor.dtype == torch.float32, 'Unexpected kernel representation')
            kernel = tensor.detach().cpu().numpy().astype(np.float64)
            audits.append(dict(seed=job['seed'], branch=branch.upper(), selected_epoch=original['selected_epoch'],
                               binary_path=str(path.relative_to(ROOT)), binary_sha256=cp['sha256'], state_sha256=original['checkpoint_state_sha256'],
                               tensor_key=key, tensor_shape=list(tensor.shape), tensor_dtype=str(tensor.dtype),
                               tensor_raw_sha256=hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
                               **coefficient_statistics(kernel)))
        need(sha(path) == cp['sha256'], 'Checkpoint changed during read')
    need(sorted({r['seed'] for r in audits}) == list(SEEDS) and len(audits) == 9, 'Missing audit seed/branch')
    result = dict(schema='raf_cnn_dc_parameter_audit_v1', created_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                  plan_sha256=sha(OUT / 'PLAN.json'), input_source_closure=source['closure'],
                  per_seed_branch=audits, arithmetic_checks=arithmetic_checks(),
                  no_large_gain_threshold_selected=True, local_tensor_reader_torch=str(torch.__version__),
                  native_checkpoint_runtime='5080 torch2.11.0+cu130/cuDNN91900; CPU reader2.14 performs no model arithmetic',
                  source_semantics=dict(tap_order=['current', 'previous_observed', 'second_previous_observed'],
                                        full_context_residual='(w0+w1+w2)*constant_projected_channel',
                                        full_context_total='(1+w0+w1+w2)*constant_projected_channel',
                                        insufficient_history='Unavailable taps contribute exactly zero. With1/2 observations use w0 or w0+w1.',
                                        padding='Padding is skipped; lag means previous observed event, not adjacent padded index.',
                                        withheld_valid='Valid withheld rows reset eligibility. Residual zero there; no crossing reset boundary.',
                                        target='core.target_outputs excludes final target using memory_write_mask and selects previous observed position.',
                                        memory='Persistent/context memory is concatenated after event CNN and bypasses this filter.',
                                        interpretation='Time/quantity/encoder signals are mixed in projected QKV. Constant raw quantity does not imply constant projected history.'),
                  model_class_constructions=0, model_forward_calls=0, remote_calls=0, held_out_test_read=False,
                  interpretation=read(OUT / 'PLAN.json')['scientific_scope'])
    bias = observed_validation_bias(contract, manifest)
    write(OUT / 'RESULT.json', result)
    write(OUT / 'VALIDATION_BIAS.json', bias)
    lines = ['# RAF 기존 CNN의 잔차 계수 합 검사', '',
             '선택 checkpoint3의 텐서만 읽었다. 모델 생성·forward·추론·학습·Test·원격 호출은 없다.',
             '완전한 세 관측의 동일 투영값 x에서 잔차는 (w0+w1+w2)x, 원래 값까지 더한 출력은 (1+w0+w1+w2)x다.',
             '이는 투영 채널의 상수 응답이다. 원시 수량의 과대·과소 예측 원인으로 해석할 수 없다.', '',
             '| seed | 경로 | signed sum 중앙값 | abs sum 중앙값/q90/최대 | abs sum / L1 중앙값/q90/최대 | L1 중앙값 | L2 중앙값 |',
             '|---:|---|---:|---|---|---:|---:|']
    for r in audits:
        s = r['summary']; fmt = lambda key: '/'.join(f"{s[key][q]:.6f}" for q in ('median', 'q90', 'max'))
        lines.append(f"| {r['seed']} | {r['branch']} | {s['signed_sum_residual_dc']['median']:+.6f} | {fmt('abs_sum_residual_dc')} | {fmt('abs_sum_over_L1')} | {s['L1']['median']:.6f} | {s['L2']['median']:.6f} |")
    lines += ['', 'tap 순서는 현재·이전 관측·두 번째 이전 관측이며 tensor shape는 [3,64]다. 관측이1/2개이면 없는 tap은0이므로 부분합만 적용된다.',
              'padding은 건너뛰고 valid withheld 위치는 이력 가용성을 리셋한다. persistent/context memory는 필터를 우회한다.',
              '임의의 큰 gain 임계값은 선택하지 않았다. 채널별 계수·부분합·전체 분포는 RESULT.json에 보존했다.', '',
              '## 별도 Validation 기술통계', '', '값은 3seed 평균 ± 표본SD. bias=예측−실제. 구간은 target 수량으로 정의했다.', '',
              '| target 구간/대상 수 | 모델 | RMSE | MAE | bias |', '|---|---|---:|---:|---:|']
    for r in sorted(bias['three_seed'], key=lambda r: (r['bin'], ('CNN', 'Query', 'Value').index(r['model']))):
        metrics = r['metrics']; format_metric = lambda key: f"{metrics[key]['mean']:.6f} ± {metrics[key]['sample_sd']:.6f}"
        lines.append(f"| {r['target_quantity_condition']}/{r['count']} | {r['model']} | {format_metric('qty_rmse')} | {format_metric('qty_mae')} | {format_metric('qty_bias')} |")
    lines += ['', 'DC 계수와 signed 오류는 별도 사실이다. 상관·개입·활성값 크기를 측정하지 않았으므로 DC 계수 합이 수량 bias의 원인이라고 주장하지 않는다.',
              '기존 CNN은5080/cuDNN91900, Query·Value는5090/cuDNN92000이며 해당 차이도 유지한다. 새 zero-sum 제약의 개선 여부는 확인되지 않았다.']
    with (OUT / 'RESULT.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    artifacts = {name: sha(OUT / name) for name in ('PLAN.json', 'BIAS_PLAN.json', 'audit.py', 'RESULT.json', 'VALIDATION_BIAS.json', 'RESULT.md')}
    write(OUT / 'RECEIPT.json', dict(status='complete', inputs=INPUTS, artifacts=artifacts, input_source_files_verified=len(source['files']),
                                    existing_cnn_binary_count=3, tensor_count=9, channels_per_tensor=64, model_class_constructions=0,
                                    model_forward_calls=0, new_evaluation_count=0, held_out_test_read=False, remote_calls=0))
    print(json.dumps({'status': 'complete', 'artifacts': artifacts, 'receipt_sha256': sha(OUT / 'RECEIPT.json'),
                      'summary': [{k: r[k] for k in ('seed', 'branch', 'sum_exact_zero_channels', 'summary')} for r in audits]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
