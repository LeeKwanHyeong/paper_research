"""Train/Validation diagnostics for the frozen CNN+GRU54 and width references.

This operation adapter only expands checkpoint routing. All population,
sampling, aggregation and model code is imported from the frozen source.
"""
from pathlib import Path

ARM = 'titantpp_cnn_gru54'
WIDTH_ARMS = ('titantpp_history_mlp', 'titantpp_history_mlp_width8',
              'titantpp_history_mlp_width12', 'titantpp_history_mlp_width16')
ALLOWED_ARMS = (*WIDTH_ARMS, ARM)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_payload(payload, data, state_digest, selector):
    require(payload.get('backbone') in ALLOWED_ARMS, 'Foreign diagnostic backbone')
    require(payload.get('seed') in (42, 52, 62), 'Foreign diagnostic seed')
    require(payload.get('evaluation_scope') == 'validation_only'
            and payload.get('held_out_test_evaluated') is False, 'Held-out checkpoint forbidden')
    require(payload.get('variant') == 'count_only_log_regression'
            and payload.get('checkpoint_monitor') == 'validation_raw_quantity_rmse'
            and payload.get('checkpoint_monitor_history_key') == selector['history_key']
            and payload.get('checkpoint_selection') == selector['selection'], 'Diagnostic selector changed')
    require(payload.get('interface_meta', {}).get('time_head', {}).get('observation_likelihood')
            == data['model']['time_observation_contract'], 'Diagnostic observation likelihood changed')
    require(payload.get('model_state_sha256') == state_digest, 'Diagnostic checkpoint tensor SHA changed')
    return True


def evaluate_checkpoint(data, checkpoint_path, frame, *, engine, device='cuda:0',
                        budget_check=lambda: None, validation_replay=None,
                        train_sample_contract=None):
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from paper.scripts import evaluate_titantpp_history_width as fixed
    from paper.scripts import run_titantpp_history_width as width
    from paper.scripts.count_aware_tpp_backbone.training import checkpoint_monitor_spec
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint
    budget_check()
    payload = torch_load_checkpoint(Path(checkpoint_path), map_location='cpu')
    state = canonical_state_dict_sha256(payload['model_state_dict'])
    validate_payload(payload, data, state, checkpoint_monitor_spec('validation_raw_quantity_rmse'))
    validate_checkpoint_route(payload, payload['backbone'])
    require(validation_replay is not None, 'Use an audited full-Validation endpoint')
    loader = fixed._loader(data, frame, 'validation')
    q, population = fixed.population_identity(loader.dataset, 'validation',
        data['inherited_data_identity']['populations']['validation'])
    validation = fixed.reuse_validation_replay(validation_replay, data, population, state, q)
    del loader, q
    def builder(d, arm):
        return engine.build_model(d, arm) if arm in (engine.BASELINE, ARM) else width.build_model(d, arm)
    train = None
    if train_sample_contract is not None:
        train = fixed._train_sample(payload, state, checkpoint_path, data, frame, budget_check,
                                    device, train_sample_contract, builder)
    budget_check()
    return {'schema': 'titantpp_cnn_gru_train_validation_metrics_v1',
            'dataset': data['dataset_id'], 'model': payload['backbone'], 'seed': payload['seed'],
            'epoch': payload.get('epoch', payload.get('best_epoch')), 'validation': validation,
            'train_sample': train, 'provenance': {'checkpoint_path': str(checkpoint_path),
                'state_sha256': state, 'device': str(device),
                'checkpoint_monitor': payload['checkpoint_monitor'], 'selection_unchanged': True,
                'new_training': False, 'held_out_test_evaluated': False,
                'raw_predictions_written': False}}


def compare_evaluations(baseline, candidate):
    from paper.scripts.evaluate_titantpp_history_width import compare_evaluations as frozen_compare
    return frozen_compare(baseline, candidate)
