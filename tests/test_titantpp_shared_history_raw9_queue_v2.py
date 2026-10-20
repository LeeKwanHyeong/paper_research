"""Independent local-only tests for macOS bounded queue path alias correction."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch
import pytest
from paper.scripts import manage_titantpp_shared_history_raw9_queue_v2 as q


def marker(tmp_path):
    value={'pid':16494,'command':[sys.executable,*q.EXPECTED_TAIL],
           'started_at':'2026-10-19T13:24:11.684818+00:00'}
    p=tmp_path/'marker.json';p.write_text(json.dumps(value))
    return p,value


def ps_output(value, *, executable=None, shift=0, tail=None, code=0):
    started=datetime.fromisoformat(value['started_at'])+timedelta(seconds=shift)
    words=started.strftime('%a %b %d %H:%M:%S %Y')+' '+(executable or sys.executable)+' '+' '.join(tail or q.EXPECTED_TAIL)
    return SimpleNamespace(returncode=code,stdout=words+'\n',stderr='')


def test_same_current_python_executable():
    assert q.executable_matches(sys.executable,str(Path(sys.executable).resolve()))


def test_exact_mac_framework_alias():
    framework=Path('/Library/Frameworks/Python.framework/Versions/3.12')
    with patch.object(q.sys,'platform','darwin'),patch.object(q.sys,'base_prefix',str(framework)),patch.object(q.sys,'executable',str(framework/'bin/python3.12')):
        assert q.executable_matches(str(framework/'bin/python3.12'),str(framework/'Resources/Python.app/Contents/MacOS/Python'))
        assert not q.executable_matches('/usr/bin/python3',str(framework/'Resources/Python.app/Contents/MacOS/Python'))
        assert not q.executable_matches(str(framework/'bin/python3.12'),'/Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python')
        assert not q.executable_matches(str(framework/'bin/python3.12'),'/tmp/Python')


def test_nonmac_does_not_admit_framework_alias():
    with patch.object(q.sys,'platform','linux'):
        assert not q.executable_matches(sys.executable,str(Path(sys.base_prefix)/'Resources/Python.app/Contents/MacOS/Python'))


def test_exact_argv_pid_start_time_and_utc_locale(tmp_path):
    path,value=marker(tmp_path)
    with patch.object(q.subprocess,'run',return_value=ps_output(value)) as execute:
        actual=q.inspect_finisher(path)
        assert actual['alive'] and actual['pid']==16494
        args,kw=execute.call_args
        assert args[0]==['/bin/ps','-p','16494','-o','lstart=,command=']
        assert kw['env']['LC_ALL']=='C' and kw['env']['TZ']=='UTC'
        assert kw['timeout']==20


@pytest.mark.parametrize('shift',[-30,-6,6,30])
def test_pid_reuse_or_wrong_start_time_rejected(tmp_path,shift):
    path,value=marker(tmp_path)
    with patch.object(q.subprocess,'run',return_value=ps_output(value,shift=shift)):
        assert q.inspect_finisher(path)['alive'] is False


@pytest.mark.parametrize('tail',[
    ['-u','-m','paper.scripts.manage_titantpp_shared_history_cap_execution_v3','--mode','deploy'],
    ['-u','-m','paper.scripts.manage_titantpp_shared_history_raw9','--mode','finish'],
    ['-m','-u','paper.scripts.manage_titantpp_shared_history_cap_execution_v3','--mode','finish'],
])
def test_similar_or_reordered_command_rejected(tmp_path,tail):
    path,value=marker(tmp_path)
    with patch.object(q.subprocess,'run',return_value=ps_output(value,tail=tail)):
        assert q.inspect_finisher(path)['alive'] is False


def test_extra_argv_rejected(tmp_path):
    path,value=marker(tmp_path)
    with patch.object(q.subprocess,'run',return_value=ps_output(value,tail=[*q.EXPECTED_TAIL,'--extra'])):
        with pytest.raises(ValueError,match='Unrecognized'):q.inspect_finisher(path)


@pytest.mark.parametrize('exe',['/bin/sh','/tmp/python3','/usr/bin/python2'])
def test_other_executable_rejected(tmp_path,exe):
    path,value=marker(tmp_path)
    with patch.object(q.subprocess,'run',return_value=ps_output(value,executable=exe)):
        assert q.inspect_finisher(path)['alive'] is False


def test_ps_absence_and_permission_error_fail_closed(tmp_path):
    path,value=marker(tmp_path)
    with patch.object(q.subprocess,'run',return_value=ps_output(value,code=1)):
        assert q.inspect_finisher(path)['alive'] is False
    with patch.object(q.subprocess,'run',side_effect=PermissionError('sandbox denies ps')):
        with pytest.raises(PermissionError):q.inspect_finisher(path)


@pytest.mark.parametrize('mutation',['pidbool','pidzero','wrongmarker','missingstarted'])
def test_invalid_marker_rejected(tmp_path,mutation):
    path,value=marker(tmp_path)
    if mutation=='pidbool':value['pid']=True
    elif mutation=='pidzero':value['pid']=0
    elif mutation=='wrongmarker':value['command'][-1]='status'
    elif mutation=='missingstarted':value.pop('started_at')
    path.write_text(json.dumps(value))
    with patch.object(q.subprocess,'run',return_value=ps_output({'started_at':'2026-10-19T13:24:11.684818+00:00'})):
        with pytest.raises((ValueError,KeyError)):q.inspect_finisher(path)


def queue_fixture(tmp_path,monkeypatch):
    base=tmp_path/'base'; report=base/'queue_v2';bundle=tmp_path/'bundle';base.mkdir();bundle.mkdir()
    (bundle/'execution_contract.json').write_text('{}')
    (base/'QUEUE_FAILURE.json').write_text(json.dumps({'error':'Cap18 finite finisher stopped before verified completion'}))
    contract={'predecessor':{'receipt_path':str(tmp_path/'cap_final.json'),'finisher_receipt_path':str(tmp_path/'finisher.json')}}
    monkeypatch.setattr(q,'BASE_REPORT',base);monkeypatch.setattr(q,'REPORT',report)
    monkeypatch.setattr(q._manager,'BUNDLE',bundle)
    monkeypatch.setattr(q._manager,'verify_local',lambda **kw:contract)
    monkeypatch.setattr(q,'inspect_finisher',lambda _: {'alive':True,'pid':16494})
    return base,report,bundle,contract


@pytest.mark.parametrize('existing',['attempt','retrieval','old_launch','new_queue','new_wait','wrongfailure'])
def test_previous_remote_or_queue_attempt_never_spawns(tmp_path,monkeypatch,existing):
    base,report,bundle,contract=queue_fixture(tmp_path,monkeypatch)
    paths={'attempt':bundle/'attempts/5080.json','retrieval':bundle/'retrieval/5080',
           'old_launch':base/'LAUNCHED_5080.json','new_queue':report/'QUEUE_STARTED.json','new_wait':report/'WAIT_STARTED.json'}
    if existing=='wrongfailure':(base/'QUEUE_FAILURE.json').write_text('{"error":"different failure"}')
    else:
        path=paths[existing];path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    with patch.object(q.subprocess,'Popen',side_effect=AssertionError('No child may spawn')):
        with pytest.raises(ValueError):q.start_queue()


def test_new_queue_uses_separate_markers_same_bundle_and_one_spawn(tmp_path,monkeypatch):
    base,report,bundle,contract=queue_fixture(tmp_path,monkeypatch)
    old_failure=(base/'QUEUE_FAILURE.json').read_bytes()
    with patch.object(q.subprocess,'Popen',return_value=SimpleNamespace(pid=987654)) as spawn:
        out=q.start_queue()
        assert spawn.call_count==1
        assert out['shared_exclusive_scientific_bundle']==str(bundle)
        assert out['recurring_automation'] is False
        assert out['command'][-1]=='wait-predecessor'
        assert (report/'QUEUE_STARTED.json').exists() and (report/'PREDECESSOR_IDENTITY.json').exists()
        assert (base/'QUEUE_FAILURE.json').read_bytes()==old_failure
        with pytest.raises(ValueError):q.start_queue()
        assert spawn.call_count==1


def test_local_adapter_pins_old_manager_and_reuses_frozen_remote_bundle():
    assert hashlib.sha256(q._PATH.read_bytes()).hexdigest()==q._SHA
    assert q._manager.REPORT==q.REPORT==q.BASE_REPORT/'queue_v2'
    assert q._manager._impl.REPORT==q.REPORT
    assert q._manager._finisher_alive is q._finisher_alive
    assert q._manager.BUNDLE.name=='titantpp_history_mlp_shared_raw9_20261020_v1'
