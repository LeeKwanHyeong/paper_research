"""CPU synthetic feasibility probe; neither a campaign runner nor qualification."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from models.TPPs.CountAwareFactory import build_count_aware_model
from models.Titan.common.titans_mac_optimized import apply_titantpp_mac_semantic_optimization
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.count_aware_titantpp_mac_contract import validate_titantpp_mac_primary_contract


def main():
    torch.set_num_threads(1)
    contract = json.loads((ROOT / 'search_artifacts/titantpp_core_ablation_20260928_v1/frozen_execution/execution_contract.json').read_text())
    rows = []
    for data in contract['datasets']:
        cfg = {k: v for k, v in data['model'].items()
               if k not in ('backbone', 'lambda_log_qty', 'lambda_tail', 'time_head_lr_multiplier')}
        cfg.update(train_log_mean=data['statistics']['train_log_mean'],
                   train_log_std=data['statistics']['train_log_std'], max_seq_len=data['loader']['max_seq_len'])
        torch.manual_seed(42)
        model, meta = build_count_aware_model('titantpp_titans_mac', **cfg, titans_memory_gradient_clip=1.0)
        model.eval()
        assert model.titans_mac_encoder.neural_memory.gradient_max_norm == 1.0
        assert meta['time_head']['observation_likelihood'] == cfg['time_observation_contract']
        torch.manual_seed(42)
        baseline, _ = build_count_aware_model('titantpp', **cfg)
        shared = [k for k in model.state_dict() if k in baseline.state_dict()
                  and not k.startswith(('encoder.', 'titans_mac_encoder.', 'lmm.'))]
        assert shared and all(torch.equal(model.state_dict()[k], baseline.state_dict()[k]) for k in shared)
        optimized = apply_titantpp_mac_semantic_optimization(copy.deepcopy(model)).eval()
        # Three segments; second row has padding. All values are synthetic.
        dt = torch.ones(2, 34)
        dt[0, -1] = 30 if cfg['time_observation_contract']['top_code'] else 3
        qty = (torch.arange(68).reshape(2, 34) % 9).float() + 1
        mask = torch.ones_like(dt, dtype=torch.bool)
        mask[1, 18:] = False
        with torch.no_grad():
            ref = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
            opt = target_outputs(optimized, dt, mask, qty, lambda_log_qty=1.)
            for k in ref:
                assert torch.isfinite(ref[k]).all()
                torch.testing.assert_close(ref[k], opt[k], atol=1e-6, rtol=1e-5)
            changed = qty.clone()
            changed[0, 33] = 100000
            changed[1, 17:] = 100000
            altered = target_outputs(optimized, dt, mask, changed, lambda_log_qty=1.)
            assert torch.equal(opt['pred_qty'], altered['pred_qty'])
            assert torch.equal(opt['time_loss'], altered['time_loss'])
            states = optimized.encode_task_states(dt, qty, mask)
            later = dt.clone(); later[0, 33] = 20
            changed_states = optimized.encode_task_states(later, changed, mask)
            for left, right in zip(states, changed_states):
                assert torch.equal(left[0, :33], right[0, :33])
                assert torch.equal(left[1, :17], right[1, :17])
            restored, _ = build_count_aware_model('titantpp_titans_mac', **cfg, titans_memory_gradient_clip=1.0)
            apply_titantpp_mac_semantic_optimization(restored)
            restored.load_state_dict(optimized.state_dict(), strict=True)
            restored.eval()
            replay = target_outputs(restored, dt, mask, qty, lambda_log_qty=1.)
            assert all(torch.equal(opt[k], replay[k]) for k in opt)
        for candidate in (model, optimized):
            candidate.zero_grad(set_to_none=True)
            # Eval mode disables dropout, but gradients still traverse the inner writes.
            out = target_outputs(candidate, dt, mask, qty, lambda_log_qty=1.)
            out['joint_loss'].mean().backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters())
        for (kn, p), (ko, q) in zip(model.named_parameters(), optimized.named_parameters()):
            assert kn == ko
            assert (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
        try:
            validate_titantpp_mac_primary_contract(backbones=('titantpp_titans_mac',),
                quantity_variants=('count_only_log_regression',), time_head_mode=cfg['time_head_mode'], lambda_tail=0.)
        except ValueError as exc:
            rejection = str(exc)
        else:
            raise AssertionError('Historical wrapper unexpectedly accepts the new protocol')
        rows.append({'dataset': data['dataset_id'], 'parameter_count': sum(p.numel() for p in model.parameters()),
                     'shared_head_state_keys_equal_to_B': shared,
                     'current_factory_modern_head_clip1': True, 'finite_forward_backward': True,
                     'reference_optimized_forward_gradient_agreement': True,
                     'future_target_padding_prediction_invariance': True,
                     'strict_restore_prediction_match': True, 'historical_wrapper_rejection': rejection})
    result = {'status': 'passed', 'device': 'cpu', 'torch': torch.__version__,
              'data': 'synthetic_only', 'seed': 42, 'batch_size': 2, 'sequence_length': 34,
              'cuda_qualification': False, 'real_fit_or_replay': False,
              'production_wrapper_implemented': False, 'rows': rows}
    outpath = Path(__file__).with_name('factory_probe.json')
    outpath.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
