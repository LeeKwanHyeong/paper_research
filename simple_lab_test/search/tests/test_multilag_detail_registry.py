"""Keep the new supervised route isolated from old roles and raw GPU CLI."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from paper.scripts import run_count_aware_tpp_backbone_control as shared
from paper.scripts import prepare_multilag_detail_execution as proposal
from paper.scripts.count_aware_tpp_backbone import constants, observed_time


@pytest.mark.parametrize("role,backbones", [
    (observed_time.MULTILAG_ROLE, "titantpp"),
    ("experimental", "titantpp_local_detail"),
    ("experimental", "titantpp_multilag_detail"),
])
def test_generic_cli_rejects_before_cuda_or_data(monkeypatch, role, backbones):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupervised route reached device or data")
    monkeypatch.setattr(shared.torch.cuda, "is_available", forbidden)
    with pytest.raises(ValueError, match="approved dedicated supervisor"):
        shared.run(SimpleNamespace(model_role=role, backbones=backbones))


@pytest.mark.parametrize("role,backbones,valid", [
    (observed_time.MULTILAG_ROLE, observed_time.MULTILAG_BACKBONES, True),
    (observed_time.STATE_ROLE, observed_time.STATE_BACKBONES, True),
    (observed_time.PAIR_ROLE, observed_time.PAIR_BACKBONES, True),
    (observed_time.ROLE, ("titantpp", "nhp"), True),
    (observed_time.STATE_ROLE, observed_time.MULTILAG_BACKBONES, False),
    (observed_time.MULTILAG_ROLE, observed_time.STATE_BACKBONES, False),
    ("experimental", observed_time.MULTILAG_BACKBONES, False),
])
def test_roles_cannot_relabel_new_structure(role, backbones, valid):
    kwargs = dict(model_role=role, backbones=backbones,
                  quantity_variants=("count_only_log_regression",),
                  time_head_mode="heteroscedastic_lognormal_duration", lambda_tail=0.)
    if valid:
        constants.validate_model_role_contract(**kwargs)
    else:
        with pytest.raises(ValueError):
            constants.validate_model_role_contract(**kwargs)


def test_three_dataset_scope_preserves_instacart_window_and_observation():
    specs = proposal.dataset_specs()
    assert sum(d["expected_global_steps"] * len(proposal.ARMS) for d in specs) == 6816240
    insta = next(d for d in specs if d["dataset_id"] == "insta_market_basket")
    assert insta["loader"]["max_seq_len"] == 64
    assert insta["potentially_active_lags"] == [1,2,4,8,16,32]
    assert insta["model"]["time_observation_contract"] == observed_time.observation_contract("insta_market_basket")
    assert set(x for group in proposal.ASSIGNMENTS.values() for x in group) == {d["dataset_id"] for d in specs}
    assert len([x for group in proposal.ASSIGNMENTS.values() for x in group]) == len(specs)


def test_frozen_contract_rejects_changed_scientific_scope(monkeypatch, tmp_path):
    from paper.scripts.observed_slot_parallel_common import sha_file, sha_json
    design = proposal.read(proposal.ROOT / proposal.DESIGN)
    hosts, process, libraries = {"host": "fixture"}, {}, {}
    monkeypatch.setattr(proposal, "host_specs", lambda: (hosts, process, libraries))
    contract = dict(schema=proposal.SCHEMA, model_role="observed_time_multilag_detail_v1",
        arms=list(proposal.ARMS), architecture=deepcopy(proposal.ARCHITECTURE), seed=42, epochs=120,
        limits=deepcopy(proposal.LIMITS), policy=deepcopy(proposal.POLICY), cost_gates=deepcopy(proposal.COST_GATES),
        datasets=proposal.dataset_specs(), hosts=hosts, process_environment=process, library_sha256=libraries,
        acceptance=design["prospective_comparison"]["screening_against_each_of_B_and_local_control_on_each_dataset"],
        total_optimizer_steps=6816240, total_arms=9, endpoint_replays=18, approved=False, approval_required=True,
        design_sha256=sha_file(proposal.ROOT/proposal.DESIGN), source={"files":{}, "files_sha256":sha_json({})},
        implementation_correction=deepcopy(proposal.CORRECTION),
        launch={"fixed_started_at_unix":proposal.CORRECTION["original_started_at_unix"],
                "fixed_deadline_unix":proposal.CORRECTION["original_deadline_unix"]})
    proposal.validate_contract(contract, verify_source=False)
    mutations = [lambda x:x["datasets"].pop(),
                 lambda x:x["limits"].update(total_wall_seconds=200000),
                 lambda x:x["architecture"].update(rank=8),
                 lambda x:x["datasets"][-1]["model"]["time_observation_contract"].update(top_code=None),
                 lambda x:x["policy"].update(held_out=True),
                 lambda x:x.update(endpoint_replays=12),
                 lambda x:x["launch"].update(fixed_started_at_unix=x["launch"]["fixed_started_at_unix"]+60)]
    for mutation in mutations:
        invalid=deepcopy(contract);mutation(invalid)
        with pytest.raises(ValueError):
            proposal.validate_contract(invalid, verify_source=False)
