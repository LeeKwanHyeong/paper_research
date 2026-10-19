"""Isolated export/bootstrap regression; no SSH, inference, training or old writes."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from paper.scripts import manage_titantpp_late_path18_5090_recovery_v2 as manager

FROZEN = {
    'paper/scripts/manage_titantpp_late_path18_5090_recovery.py': 'be1de7b185a1f2c49cc01ee2e163ae6fea115f52e4e73230f77929e4bae1b9ce',
    'paper/scripts/titantpp_late_path18_5090_recovery_runtime.py': '4edc46490bf67c4bb47f8b7fc9b217347afc0802c98aa4f1a6066010078de377',
    'paper/scripts/run_titantpp_late_path18_recovery.py': '912403adc0d2de7c4d304b3edd459aee66da59f8aa39bc5019924b10e3a9e27d',
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def bootstrap(source):
    # Execute the exact frozen resolver from the exported source in a fresh
    # interpreter, outside the repository which concealed the missing marker.
    code = "import runpy,sys;from pathlib import Path;m=runpy.run_path(sys.argv[1]);p=m['resolve_project_root'](sys.argv[2]);print(p)"
    return subprocess.run([sys.executable,'-B','-c',code,str(source/'simple_lab_test/common/pathing.py'),
        str(source/'simple_lab_test/search/common/experiment_utils.py')],capture_output=True,text=True,cwd=source)


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    tmp=tmp_path_factory.mktemp('late-recovery-v2-export')
    original=manager.ROOT/'search_artifacts'/manager.FAILED_NAME/'recovery_contract.json'
    c=deepcopy(manager.read(original))
    c['campaign']=manager.NAME
    c['root']=str(Path(c['root']).parent/(manager.NAME+'_5090'));c['source_root']=c['root']+'/source'
    c['environment']={**c['environment'],'PYTHONPATH':c['source_root'],
        'MPLCONFIGDIR':c['root']+'/cache/matplotlib','XDG_CACHE_HOME':c['root']+'/cache'}
    c['operation_files']={name:manager.sha(manager.ROOT/name) for name in manager.OPERATIONS}
    c['operation_files_sha256']=manager.canonical(c['operation_files'])
    c['failed_predecessor']=manager.failed_predecessor_evidence()
    c['deployment_layout']={'required_empty_directories':['source/sample_data'],'scientific_source_changed':False}
    cp=tmp/'recovery_contract.json';cp.write_text(json.dumps(c))
    payload=manager.package_deployment(c,cp)
    return c,payload,tmp


def test_preserved_v1_failure_is_package_root_bootstrap_before_training():
    value=manager.failed_predecessor_evidence()
    assert value['qualification_actual_os_returncode']==1
    assert value['real_data_training_started'] is False and value['scientific_source_changed'] is False
    assert value['diagnosis_sha256']==manager.FAILURE_DIAGNOSIS_SHA256
    assert manager.ORIGIN==manager.ROOT/'search_artifacts'/manager.FAILED_NAME/'origin_archive'
    assert manager.NAME.endswith('_v2')
    assert manager.OPERATIONS==('paper/scripts/manage_titantpp_late_path18_5090_recovery_v2.py',
        'paper/scripts/titantpp_late_path18_5090_recovery_runtime.py','paper/scripts/run_titantpp_late_path18_recovery.py')


def test_original_deployment_reproduces_missing_sentinel_in_exported_environment(tmp_path):
    prior=manager.ROOT/'search_artifacts'/manager.FAILED_NAME/'deployment.tar.gz'
    with tarfile.open(prior) as archive:
        assert not any(m.name.rstrip('/')=='source/sample_data' for m in archive.getmembers())
        archive.extractall(tmp_path,filter='data')
    result=bootstrap(tmp_path/'source')
    assert result.returncode!=0 and 'Could not locate the paper_research project root' in result.stderr


def test_actual_export_install_and_frozen_resolver_succeed_with_only_empty_directory(exported):
    from paper.scripts import titantpp_late_path18_5090_recovery_runtime as runtime
    c,payload,tmp=exported
    science=manager.read(manager.ORIGIN/'originals/execution_contract.json')
    assert len(runtime.validate_contract(c,science=science))==4
    target=tmp/'installed'
    result=subprocess.run([sys.executable,'-B','-c',manager.installation_code(target,digest(payload))],input=payload,capture_output=True)
    assert result.returncode==0,result.stderr.decode()
    found=bootstrap(target/'source')
    assert found.returncode==0,found.stderr
    assert Path(found.stdout.strip())==target/'source'
    assert (target/'source/sample_data').is_dir() and not list((target/'source/sample_data').iterdir())
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        assert [m.name for m in archive if m.isdir()]==['source/sample_data']
    for name,sha in {**c['source']['files'],**c['input_files'],**c['operation_files']}.items():
        assert manager.sha(target/'source'/name)==sha
    assert manager.read(target/'recovery_contract.json')==c
    # Prior attempt, scientific code and workers remain byte-identical.
    assert all(manager.sha(manager.ROOT/name)==sha for name,sha in FROZEN.items())


def malicious_archive(kind):
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as archive:
        if kind!='missing_marker':
            marker=tarfile.TarInfo('source/sample_data');marker.type=tarfile.DIRTYPE;archive.addfile(marker)
        name={'parent':'../outside','absolute':'/outside','directory':'source/other','duplicate':'source/sample_data'}.get(kind,'source/item')
        member=tarfile.TarInfo(name)
        if kind in ('directory','duplicate'):member.type=tarfile.DIRTYPE
        elif kind in ('symlink','hardlink'):
            member.type=tarfile.SYMTYPE if kind=='symlink' else tarfile.LNKTYPE;member.linkname='/outside'
        archive.addfile(member)
    return buf.getvalue()


@pytest.mark.parametrize('kind',('parent','absolute','directory','duplicate','symlink','hardlink','missing_marker'))
def test_installer_rejects_nonexplicit_directory_links_and_path_escapes_before_root_creation(tmp_path,kind):
    payload=malicious_archive(kind);target=tmp_path/'new'
    result=subprocess.run([sys.executable,'-B','-c',manager.installation_code(target,digest(payload))],input=payload,capture_output=True)
    assert result.returncode!=0 and not target.exists()


def test_installer_rejects_existing_root_without_overwrite(exported):
    c,payload,tmp=exported;target=tmp/'existing';target.mkdir()
    sentinel=target/'preserved';sentinel.write_bytes(b'unchanged')
    result=subprocess.run([sys.executable,'-B','-c',manager.installation_code(target,digest(payload))],input=payload,capture_output=True)
    assert result.returncode!=0 and sentinel.read_bytes()==b'unchanged'


def test_packager_rejects_changed_contract_and_source_before_any_deployment(exported,tmp_path):
    c,payload,_=exported;cp=tmp_path/'contract.json';cp.write_text('{}')
    with pytest.raises(ValueError,match='contract differs'):manager.package_deployment(c,cp)
    c=deepcopy(c);name=next(iter(c['source']['files']));c['source']['files'][name]='0'*64;cp.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='source bytes changed'):manager.package_deployment(c,cp)


def test_diagnosis_tamper_is_rejected_without_changing_original(tmp_path,monkeypatch):
    original=manager.ROOT/'reports'/manager.FAILED_NAME/'qualification_failure_diagnosis.json'
    before=original.read_bytes()
    bad=tmp_path/'reports'/manager.FAILED_NAME/'qualification_failure_diagnosis.json'
    bad.parent.mkdir(parents=True);bad.write_bytes(before+b' ')
    monkeypatch.setattr(manager,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='diagnosis changed'):manager.failed_predecessor_evidence()
    assert original.read_bytes()==before


def test_remote_collection_import_and_tmux_session_are_v2_only():
    source=Path(manager.__file__).read_text()
    assert 'session = "late_path18_recovery_5090_20261019_v2"' in source
    assert 'from paper.scripts.manage_titantpp_late_path18_5090_recovery_v2 import verify_terminal_evidence' in source
    assert 'from paper.scripts.manage_titantpp_late_path18_5090_recovery import verify_terminal_evidence' not in source
