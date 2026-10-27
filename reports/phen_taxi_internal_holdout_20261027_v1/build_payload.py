"""Freeze the reviewed implementation and safe inputs for one isolated run."""
import ast, hashlib, json, os, shutil, subprocess, tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ART=ROOT/'search_artifacts/phen_taxi_internal_holdout_20261027_v1'
PAYLOAD=ART/'payload'
REMOTE='/home/leekwanhyeong/workspace/paper_research/search_artifacts/phen_taxi_internal_holdout_20261027_v1_5090'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def cp(p,dest):
    target=PAYLOAD/dest;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    shutil.copyfile(p,target)
    assert sha(p)==sha(target)
    return target

def main():
    qa=json.loads((HERE/'LOCAL_QA.json').read_text());assert qa['status']=='PASS'
    for path,digest in qa['reviewed_sources'].items():assert sha(path)==digest,path
    originals=json.loads((HERE/'ORIGINALS.json').read_text())
    for name,entry in originals.items():
        assert Path(name).stat().st_size==entry['bytes'] and sha(name)==entry['sha256'],name
    parent=json.loads((PAYLOAD/'materials/execution_contract.json').read_text())
    for name,digest in parent['source']['files'].items():assert sha(PAYLOAD/'source'/name)==digest,name
    runner=ROOT/'paper/scripts/run_taxi_internal_holdout_diagnostic.py'
    for path in [runner,HERE/'supervise.py',HERE/'native_smoke.py']:ast.parse(path.read_text())
    cp(runner,'source/paper/scripts/run_taxi_internal_holdout_diagnostic.py')
    for n in ['supervise.py','native_smoke.py','SCIENCE_SPEC.json','AUTHORIZATION.json']:
        cp(HERE/n,n)
    host=parent['hosts']['5090'];probe=json.loads((HERE/'preflight/5090/stdout.log').read_text())
    assert probe['source_mismatches']==[] and probe['gpu_exit']==probe['apps_exit']==0
    assert not probe['apps'].strip() and probe['processes']==[]
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    runner_relative=str(runner.relative_to(ROOT))
    subprocess.run(['git','ls-files','--error-unmatch',runner_relative],check=True,capture_output=True)
    committed=subprocess.check_output(['git','show','HEAD:'+runner_relative])
    assert hashlib.sha256(committed).hexdigest()==sha(runner), 'Runner differs from committed HEAD blob'
    assert subprocess.run(['git','diff','--quiet','HEAD','--',str(runner.relative_to(ROOT))]).returncode==0
    env=dict(host['environment']);env.update(PYTHONPATH=REMOTE+'/source',SOURCE_REVISION=revision,
        MPLCONFIGDIR=REMOTE+'/cache/matplotlib',XDG_CACHE_HOME=REMOTE+'/cache',TORCHINDUCTOR_CACHE_DIR=REMOTE+'/cache/torchinductor')
    py=host['python']
    common=['--root',REMOTE+'/run','--source-contract',REMOTE+'/materials/execution_contract.json',
      '--input-frame',REMOTE+'/materials/train_validation.parquet','--train-metadata',REMOTE+'/materials/train_metadata.parquet',
      '--validation-metadata',REMOTE+'/materials/validation_metadata.parquet','--science-spec',REMOTE+'/SCIENCE_SPEC.json','--device','cuda:0']
    prefix=[py,'-u','-m','paper.scripts.run_taxi_internal_holdout_diagnostic']
    config={'schema':'taxi_internal_holdout_execution_v1','host':'5090','test_access':False,'fresh_fit_count':1,
      'remote_root':REMOTE,'source_root':REMOTE+'/source','python':py,'boot_id':probe['boot_id'],'gpu_uuid':host['gpu_uuid'],
      'environment':env,'source_commit':revision,'base_source_commit':parent['source']['base_git_revision'],
      'science_sha256':sha(PAYLOAD/'SCIENCE_SPEC.json'),'timeout_seconds':86400,'prelaunch_timeout_seconds':180,
      'output_relative_path':'run','complete_relative_path':'run/fit/COMPLETE.json',
      'command':prefix+['--mode','fit']+common,
      'prelaunch_commands':[prefix+['--mode','synthetic'],prefix+['--mode','prepare']+common,
        [py,'-u',REMOTE+'/native_smoke.py','--prepared',REMOTE+'/run/prepared','--device','cuda:0','--output',REMOTE+'/NATIVE_SMOKE.json']],
      'input_files':{str(p.relative_to(PAYLOAD)):sha(p) for p in sorted(PAYLOAD.rglob('*')) if p.is_file()}}
    put(PAYLOAD/'EXECUTION.json',config)
    members={str(p.relative_to(PAYLOAD)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(PAYLOAD.rglob('*')) if p.is_file()}
    with tarfile.open(ART/'payload.tar.gz','x:gz') as tar:
        for n in members:tar.add(PAYLOAD/n,arcname=n,recursive=False)
    package={'archive_sha256':sha(ART/'payload.tar.gz'),'archive_bytes':(ART/'payload.tar.gz').stat().st_size,'members':members}
    put(HERE/'PAYLOAD.json',package)
    put(HERE/'PREFLIGHT_QA.json',{'status':'PASS','local_qa_sha256':sha(HERE/'LOCAL_QA.json'),
      'archive_sha256':package['archive_sha256'],'execution_sha256':sha(PAYLOAD/'EXECUTION.json'),
      'reviewed_sources':qa['reviewed_sources'],'Test_access':False,'source_commit':revision})
    print(json.dumps({'members':len(members),'archive_bytes':package['archive_bytes'],'archive_sha256':package['archive_sha256'],'remote_root':REMOTE,'source_commit':revision}))

if __name__=='__main__':main()
