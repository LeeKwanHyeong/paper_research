"""Prepare bounded GPU process observation, preserving failed v4.

This is a new one-attempt operational root with unchanged science131 and the
original5080 absolute lease. No model fit or qualification is retried in v4.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tarfile

from paper.scripts.prepare_titantpp_cnn_gru_components import canonical,sha,read,safe,require,write_json
from paper.scripts.launch_titantpp_cnn_gru_components_v4 import check_bundle as check_parent

PROJECT=Path(__file__).resolve().parents[2]
SCHEMA="titantpp_cnn_gru_components_20261012_v5"
PARENT=PROJECT/'search_artifacts/titantpp_cnn_gru_components_20261012_v4'
PARENT_SHA='aa3bc1583a72251043ad3df09bca23a33205419e9e25d3dc64c9681550c745a1'
DESIGN='d3a579750f60c037e45b881ee03ad27a211bdbcc61f2677a4fa6ca128b0336a5'
ORIGINAL=PROJECT/'search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/execution_contract.json'
OPS={'operation/campaign.py':PROJECT/'paper/scripts/run_titantpp_cnn_gru_component_campaign_v5.py',
     'operation/pipeline.py':PROJECT/'paper/scripts/run_titantpp_cnn_gru_component_phase_v5.py',
     'operation/frozen_runtime.py':PARENT/'operation/frozen_runtime.py',
     'operation/qualification.py':PARENT/'operation/qualification.py'}
NEW_TOOLS=('paper/scripts/prepare_titantpp_cnn_gru_components_v5.py',
    'paper/scripts/run_titantpp_cnn_gru_component_phase_v5.py',
    'paper/scripts/run_titantpp_cnn_gru_component_campaign_v5.py',
    'paper/scripts/launch_titantpp_cnn_gru_components_v5.py',
    'paper/tests/test_titantpp_component_preparation_v5.py',
    'paper/tests/test_titantpp_component_campaign_v5.py',
    'paper/tests/test_titantpp_component_launcher_v5.py')


def contract(parent):
    require(canonical(parent)==PARENT_SHA,'Parent v4 changed')
    original=read(ORIGINAL)
    require(canonical(original)=='518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b'
        and original['design_sha256']==DESIGN,'Original frozen design provenance changed')
    deployment=read(PARENT/'control/deployment/5080.json')
    require(deployment['contract_sha256']==PARENT_SHA and deployment['status']=='deployed','Parent deployment changed')
    start=deployment['start_permit']
    require(start['host']=='5080' and start['contract_sha256']==PARENT_SHA
        and start['started_at_unix']==1791794258.15949 and start['deadline_unix']==1792399058.15949,
        'Original absolute lease changed')
    c=deepcopy(parent);c['schema']=SCHEMA;c['design_sha256']=DESIGN
    c['operational_replacement']={'parent_contract_sha256':PARENT_SHA,'parent_scientific_fits':0,
        'reason':'An owned capacity process disappeared from proc while NVML still listed its PID through two immediate rechecks; before scientific fit',
        'original_design_contract_raw_sha256':sha(ORIGINAL),
        'parent_deployment_receipt_raw_sha256':sha(PARENT/'control/deployment/5080.json'),
        'failed_parent_root_preserved':True,'native_parent_only_passed':True,'failed_qualification_is_not_training_failure':True,'host_leases':{'5080':{
            'started_at_unix':start['started_at_unix'],'deadline_unix':start['deadline_unix']}}}
    for host,spec in c['hosts'].items():
        root=str(Path(spec['root']).parent/(SCHEMA+'_'+host))
        spec.update(root=root,source_root=root+'/source',operation_root=root+'/operation',
            output_dir=root+'/run',tmux=SCHEMA+'_'+host)
        spec['environment'].update(PYTHONPATH=root+'/source',MPLCONFIGDIR=root+'/cache/matplotlib',XDG_CACHE_HOME=root+'/cache')
    op={name:sha(origin) for name,origin in OPS.items()}
    c['operation']={'files':op,'files_sha256':canonical(op)}
    for field in ('source','input_files','datasets','dataset_sha256','training','diagnostic','architecture',
        'analysis','parallelism','jobs','assignments','arms','seeds','policy','launch_policy','design_sha256'):
        require(c[field]==parent[field],'Scientific drift: '+field)
    return c


def prepare(destination):
    parent,proof=check_parent(PARENT)
    require(proof['contract_sha256']==PARENT_SHA,'Parent pointer changed')
    require(not (PARENT/'control/intents/train_5080.json').exists()
        and not (PARENT/'control/training_permits/5080.json').exists(),'Scientific fit could have started')
    require(not destination.exists(),'Existing replacement root preserved')
    c=contract(parent)
    tools=deepcopy(read(PARENT/'local_tools_integrity.json')['files'])
    tools.update({name:sha(PROJECT/name) for name in NEW_TOOLS})
    destination.mkdir(parents=True,exist_ok=False)
    for name,digest in {**c['source']['files'],**c['input_files']}.items():
        target=safe(destination/'source',name);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(safe(PARENT/'source',name),target)
        require(sha(target)==digest,'Scientific/input copy changed')
    for name,origin in OPS.items():
        target=safe(destination,name);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(origin,target);require(sha(target)==c['operation']['files'][name],'Operation copy changed')
    write_json(destination/'execution_contract.json',c)
    write_json(destination/'current.json',{'schema':SCHEMA,'contract_sha256':canonical(c),'bundle':str(destination.resolve()),'status':'prepared_independent_hosts'})
    write_json(destination/'launch_hold.json',{'hold':True,'contract_sha256':canonical(c),'reason':'Host-specific explicit approval and strict native qualification required'})
    members=['execution_contract.json',*c['operation']['files'],*('source/'+n for n in {**c['source']['files'],**c['input_files']})]
    with tarfile.open(destination/'deployment.tar.gz','w:gz') as archive:
        for name in members:archive.add(safe(destination,name),arcname=name,recursive=False)
    write_json(destination/'local_tools_integrity.json',{'files':tools,'files_sha256':canonical(tools)})
    write_json(destination/'preparation_receipt.json',{'status':'prepared_independent_hosts','contract_sha256':canonical(c),
        'contract_raw_sha256':sha(destination/'execution_contract.json'),'payload_sha256':sha(destination/'deployment.tar.gz'),
        'local_tools_integrity_raw_sha256':sha(destination/'local_tools_integrity.json'),'source_files':131,'input_files':7,'operation_files':4,
        'remote_calls':0,'native_qualification':'not_run','fresh_training_fits':0,'parent_failed_preserved':True,'lease_extended':False})
    return c


def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,default=PROJECT/'search_artifacts'/SCHEMA)
    a=p.parse_args();c=prepare(a.destination)
    print(json.dumps({'status':'prepared_independent_hosts','contract_sha256':canonical(c),'fresh_fits':0,'lease_extended':False}))


if __name__=='__main__':main()
