"""Opt-in routes and the real-data CLI boundary for state transport."""
from types import SimpleNamespace

import pytest

from paper.scripts import run_count_aware_tpp_backbone_control as shared
from paper.scripts.count_aware_tpp_backbone import constants, observed_time


@pytest.mark.parametrize("role,backbones", [
    (observed_time.STATE_ROLE, "titantpp"),
    ("experimental", "titantpp_elapsed_state_transport"),
    ("experimental", "titantpp_event_state_transport"),
])
def test_generic_cli_rejects_before_device_or_data_access(monkeypatch, role, backbones):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupervised route reached device/data access")
    monkeypatch.setattr(shared.torch.cuda, "is_available", forbidden)
    # Deliberately no data or device fields: rejection must precede all of them.
    with pytest.raises(ValueError, match="approved dedicated supervisor"):
        shared.run(SimpleNamespace(model_role=role, backbones=backbones))


@pytest.mark.parametrize("role,backbones,valid", [
    (observed_time.STATE_ROLE, observed_time.STATE_BACKBONES, True),
    (observed_time.PAIR_ROLE, observed_time.PAIR_BACKBONES, True),
    (observed_time.ROLE, ("titantpp", "nhp"), True),
    (observed_time.PAIR_ROLE, observed_time.STATE_BACKBONES, False),
    (observed_time.STATE_ROLE, observed_time.PAIR_BACKBONES, False),
    ("experimental", observed_time.STATE_BACKBONES, False),
])
def test_role_routes_remain_separate(role, backbones, valid):
    kwargs = dict(model_role=role, backbones=backbones,
                  quantity_variants=("count_only_log_regression",),
                  time_head_mode="heteroscedastic_lognormal_duration", lambda_tail=0.0)
    if valid:
        constants.validate_model_role_contract(**kwargs)
    else:
        with pytest.raises(ValueError):
            constants.validate_model_role_contract(**kwargs)
