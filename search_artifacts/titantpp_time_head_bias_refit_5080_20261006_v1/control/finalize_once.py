"""Seal and retrieve six verified head-stage originals once; no inference or retry."""
from pathlib import Path
import datetime
import fcntl
import importlib.util
import json
import subprocess
import tarfile

BUNDLE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('time_head_refit_monitor', Path(__file__).with_name('monitor_once.py'))
monitor = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(monitor)
read, digest, sha, write, require = monitor.read, monitor.digest, monitor.sha, monitor.write, monitor.require


SEAL_CODE = r'''
import shutil,tarfile,time
c=read(root/'execution_contract.json')
snapshot=validate_snapshot(c,collect(c,root,EXPECTED),terminal=True)
dest=root/'transfer';dest.mkdir(exist_ok=True)
receipt=dest/'receipt.json';archive=dest/'originals.tar.gz'
require(not receipt.exists() and not archive.exists(),'Archive already exists; manual recovery only')
with (dest/'seal.claim').open('x') as f:json.dump({'started_unix':time.time(),'contract_sha256':EXPECTED},f)
sources=dest/'source_checkpoints';sources.mkdir(exist_ok=False)
for j in c['jobs']:
 p=Path(j['checkpoint']['path']);require(sha(p)==j['checkpoint']['sha256'],'Original checkpoint changed')
 d=sources/j['id'];d.mkdir();shutil.copyfile(p,d/'source_selected.pt')
 require(sha(d/'source_selected.pt')==j['checkpoint']['sha256'],'Copied original checkpoint changed')
names=['execution_contract.json','approval.json','start_permit.json','training_permit.json','qualification','operation','run','logs','status.json','supervisor.json','supervisor_process_exit.json']
inputs=[(root/name,name) for name in names]+[(sources,'source_checkpoints')]
files={}
for path,arcname in inputs:
 require(path.exists() and not path.is_symlink(),'Missing/linked archive input')
 for p in ([path] if path.is_file() else sorted(path.rglob('*'))):
  require(not p.is_symlink() and (p.is_dir() or p.is_file()),'Unsafe archive input')
  if p.is_file():files[arcname if p==path else arcname+'/'+str(p.relative_to(path))]=sha(p)
with tarfile.open(archive,'w:gz') as tf:
 for path,arcname in inputs:tf.add(path,arcname=arcname)
for path,arcname in inputs:
 for p in ([path] if path.is_file() else path.rglob('*')):
  if p.is_file():require(sha(p)==files[arcname if p==path else arcname+'/'+str(p.relative_to(path))],'Input changed while sealing')
r={'status':'sealed','contract_sha256':EXPECTED,'archive_path':str(archive),'archive_sha256':sha(archive),
 'archive_files':files,'actual_seal_observed_utc':snapshot['actual_observed_utc'],'time':time.time(),'held_out_test_evaluated':False}
receipt.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps(r))
'''


def validate_archive_members(members):
    names = set()
    for member in members:
        p = Path(member.name)
        require(not p.is_absolute() and '..' not in p.parts and member.name not in names
            and (member.isfile() or member.isdir()) and not member.issym() and not member.islnk(), 'Unsafe/duplicate archive member')
        names.add(member.name)


def verify_retrieved(c, originals, sealed):
    originals = Path(originals)
    require(sealed['status'] == 'sealed' and sealed['contract_sha256'] == digest(c)
        and sealed['held_out_test_evaluated'] is False and monitor.valid_sha(sealed['archive_sha256']), 'Archive receipt binding changed')
    actual_files = {str(p.relative_to(originals)) for p in originals.rglob('*') if p.is_file()}
    require(actual_files == set(sealed['archive_files']), 'Archive membership changed')
    monitor.verify_files(originals, sealed['archive_files'])
    require(read(originals/'execution_contract.json') == c, 'Archived contract changed')
    monitor.verify_files(originals,c['operation']['files'])
    require(read(originals/'status.json')['status'] == 'complete'
        and read(originals/'status.json')['complete'] == [j['id'] for j in c['jobs']], 'Archived supervisor status changed')
    exit_receipt = read(originals/'supervisor_process_exit.json')
    require(exit_receipt['returncode'] == 0 and exit_receipt['contract_sha256'] == digest(c), 'Archived supervisor exit changed')
    rows = []
    for job in c['jobs']:
        d = originals/'run'/job['id']
        verified = monitor.verify_job(c,job,d,verify_original=False)
        require(sha(originals/'source_checkpoints'/job['id']/'source_selected.pt') == job['checkpoint']['sha256'], 'Retrieved original checkpoint changed')
        rows.append({'job':job,'result':verified['result'],'all_owned_originals_SHA_verified':True,
            'selected_checkpoint_sha256':verified['selected_checkpoint_sha256'],
            'last_checkpoint_sha256':verified['last_checkpoint_sha256'],
            'worker_terminal_sha256':verified['worker_terminal_sha256']})
    return rows


def main():
    c = read(BUNDLE/'execution_contract.json'); pointer = read(BUNDLE/'current.json')
    require(pointer['canonical_sha256'] == digest(c) and pointer['root'] == c['root'], 'Current pointer mismatch')
    out = BUNDLE/'analysis'; out.mkdir(exist_ok=True)
    if (out/'completion_receipt.json').exists():
        receipt = read(out/'completion_receipt.json')
        require(receipt['contract_sha256'] == digest(c), 'Completion contract mismatch')
        require(sha(BUNDLE/'retrieved/originals.tar.gz') == receipt['archive_sha256'], 'Retrieved archive changed')
        require(sha(out/'Validation_refit_comparison.json') == receipt['comparison_sha256'], 'Comparison changed')
        print('already_finalized'); return
    if (out/'failure.json').exists() or (out/'retrieval.claim').exists():
        print('manual_action_required_prior_attempt'); return
    cache = BUNDLE/'hourly_monitor/terminal_cache.json'
    if not cache.exists():
        write(out/'pending.json',{'status':'pending_verified_six_terminals','remote_retrieval_started':False,
            'held_out_test_evaluated':False,'time':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        print('pending_verified_six_terminals'); return
    snapshot = monitor.validate_snapshot(c,read(cache),terminal=True)
    with (out/'finalize.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (out/'retrieval.claim').open('x') as stream:
            json.dump({'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'contract_sha256':digest(c)},stream)
        program = monitor.core_code()+'\nroot=Path('+repr(c['root'])+')\nEXPECTED='+repr(digest(c))+'\n'+SEAL_CODE
        response = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5080','python3 -'],
            input=program,text=True,capture_output=True,check=True,timeout=300)
        sealed = json.loads(response.stdout)
        require(sealed['status'] == 'sealed' and sealed['contract_sha256'] == digest(c)
            and sealed['archive_path'] == c['root']+'/transfer/originals.tar.gz'
            and sealed['held_out_test_evaluated'] is False and monitor.valid_sha(sealed['archive_sha256']), 'Wrong remote archive receipt/path')
        dest = BUNDLE/'retrieved'; dest.mkdir(exist_ok=True)
        require(not (dest/'originals.tar.gz').exists(), 'Refusing to overwrite retrieved archive')
        write(dest/'remote_archive_receipt.json',sealed)
        subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',
            '5080:'+sealed['archive_path'],str(dest/'originals.tar.gz')],check=True,timeout=600)
        require(sha(dest/'originals.tar.gz') == sealed['archive_sha256'],'Transferred archive SHA mismatch')
        originals = dest/'original'; originals.mkdir(exist_ok=False)
        with tarfile.open(dest/'originals.tar.gz') as archive:
            validate_archive_members(archive.getmembers())
            archive.extractall(originals,filter='data')
        rows = verify_retrieved(c,originals,sealed)
        write(out/'Validation_refit_comparison.json',{'scope':'Validation_development_selection',
            'rows':rows,'new_head_fits':6,'trainable_scalars_per_fit':2,'held_out_test_evaluated':False,
            'independent_calibration':False,'S2P2_matching_refit':'not_performed','CPU_binary_reinference_audit':'not_performed'})
        write(out/'completion_receipt.json',{**sealed,'status':'six_head_refits_originals_SHA_retrieved',
            'actual_terminal_observed_utc':snapshot['actual_observed_utc'],'local_originals_SHA_verified':True,
            'comparison_sha256':sha(out/'Validation_refit_comparison.json'),'CPU_binary_reinference_audit':'not_performed'})
        print('six_head_refits_complete_originals_SHA_retrieved')


if __name__ == '__main__':
    try: main()
    except BlockingIOError: print('another_finalizer_owns_lock')
    except Exception as exc:
        write(BUNDLE/'analysis/failure.json',{'error_type':type(exc).__name__,'error':str(exc),
            'automatic_retry':False,'training_failure':False,'held_out_test_evaluated':False})
        raise
