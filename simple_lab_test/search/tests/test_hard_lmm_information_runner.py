"""Frozen contract and fail-closed stage barriers before data/probe access."""
from unittest.mock import Mock
import pytest
from paper.scripts import run_hard_lmm_information_diagnostic as run


def test_real_contract_preserves_scope_and_probe_parameters():
    c=run.load_contract()
    assert c['scope'].startswith('local_frozen_train_retrospective')
    assert c['authorization']['server_access'] is False
    assert c['sampling']['new_unused_series_claim'] is False
    assert c['evidence_gate']['paired_series_bootstrap']['repeats']==10000
    assert len(c['primary_contrasts'])==4


def test_tampered_contract_fails_before_fitting(monkeypatch,tmp_path):
    c=run.load_contract();c['decoders']['linear']['ridge']=2
    p=tmp_path/'bad.json';run.save(p,c)
    monkeypatch.setattr(run,'CONTRACT',p)
    with pytest.raises(ValueError,match='Ridge changed'):run.load_contract()


def test_existing_output_refuses_any_new_data_access(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'RAW',tmp_path/'raw');run.RAW.mkdir()
    monkeypatch.setattr(run,'RESULT',tmp_path/'result')
    forbidden=Mock(side_effect=AssertionError('No data loading allowed'))
    monkeypatch.setattr(run,'load_train_frame',forbidden)
    with pytest.raises(ValueError,match='overwrite'):run.extract_phase()
    forbidden.assert_not_called()


def test_incomplete_extraction_cannot_run_probes(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'RESULT',tmp_path)
    run.save(tmp_path/'execution_manifest.json',{'status':'failed'})
    forbidden=Mock(side_effect=AssertionError('No fitting allowed'))
    monkeypatch.setattr(run.analysis,'analyze_cache',forbidden)
    with pytest.raises(ValueError,match='Extraction incomplete'):run.analyze_phase()
    forbidden.assert_not_called()
