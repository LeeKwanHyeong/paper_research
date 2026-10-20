from copy import deepcopy
import signal
import pytest
from paper.scripts import stop_titantpp_shared_cap18_20261020 as stop


def owned(pid=123):
    return {"pid": pid, "start_ticks": 40, "pgid": pid, "cwd": stop.ROOT + "/source",
        "argv": ["/python", "-m", "paper.scripts.run_titantpp_shared_history_cap_execution",
                 "--contract", stop.ROOT + "/execution_contract.json"]}


def test_only_exact_campaign_owned():
    assert stop.campaign_owned(owned())
    for pid in stop.PROTECTED:
        assert not stop.campaign_owned(owned(pid))
    other = owned()
    other["argv"][-1] = "/another/execution_contract.json"
    assert not stop.campaign_owned(other)
    other = owned()
    other["cwd"] = "/other"
    assert not stop.campaign_owned(other)


@pytest.mark.parametrize("key,value", [("start_ticks", 41), ("argv", ["other"]), ("cwd", "/other"), ("pgid", 999)])
def test_identity_change_never_signalled(monkeypatch, key, value):
    expected, current = owned(), owned()
    current[key] = value
    monkeypatch.setattr(stop, "identity", lambda pid: current)
    monkeypatch.setattr(stop.os, "kill", lambda *a: pytest.fail("unsafe signal"))
    with pytest.raises(RuntimeError, match="identity changed"):
        stop.checked_signal(expected, signal.SIGTERM, [])


def test_protected_pid_never_signalled(monkeypatch):
    monkeypatch.setattr(stop.os, "kill", lambda *a: pytest.fail("unsafe signal"))
    with pytest.raises(RuntimeError, match="not this campaign"):
        stop.checked_signal(owned(351034), signal.SIGTERM, [])


def test_signal_only_matching_pid(monkeypatch):
    value, sent, events = owned(), [], []
    monkeypatch.setattr(stop, "identity", lambda pid: deepcopy(value))
    monkeypatch.setattr(stop.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    stop.checked_signal(value, signal.SIGTERM, events)
    assert sent == [(123, signal.SIGTERM)]
    assert events[0]["identity"] == value


def test_already_gone_no_signal(monkeypatch):
    monkeypatch.setattr(stop, "identity", lambda pid: None)
    monkeypatch.setattr(stop.os, "kill", lambda *a: pytest.fail("signal of absent process"))
    events = []
    stop.checked_signal(owned(), signal.SIGTERM, events)
    assert events == [{"pid": 123, "already_exited": True}]
