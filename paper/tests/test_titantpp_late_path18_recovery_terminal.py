"""Native lifecycle formats with fabricated children; no SSH, GPU or research rows."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper.scripts import manage_titantpp_late_path18_5090_recovery as manager
from paper.scripts import titantpp_late_path18_5090_recovery_runtime as runtime
from paper.tests.test_titantpp_late_path18_5090_recovery_runtime import contract
from paper.tests.test_titantpp_quantity_loss_mlp18_operations import proof


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


@pytest.fixture
def terminal(contract, monkeypatch):
    """Use production qualify, launch_dispatch, dispatch and run_stage writers."""
    c = contract
    root, old = Path(c['root']), Path(c['immutable_originals']['root'])
    science = runtime.read(c['scientific_contract']['path'])
    endpoints = {}
    for job in c['jobs']:
        if job['recovery_mode'] == 'reference':
            endpoint = {'schema': 'titantpp_late_path18_stage_v1', 'status': 'complete', 'stage': 'train',
                'job_id': job['id'], 'contract_sha256': runtime.sha_json(science),
                'selected_checkpoint': job['origin_selected_checkpoint']['path'], 'selected_epoch': 9,
                'checkpoint_sha256': job['origin_selected_checkpoint']['sha256'], 'state_sha256': 'a'*64,
                'target_identity': {'split': 'validation', 'row_count': 2},
                'validation_reference': {'target_identity': {'split': 'validation', 'row_count': 2},
                                         'metrics': {'overall': {'qty_rmse': 2., 'qty_mae': 1.}}}}
            dump(job['origin_train_receipt']['path'], endpoint)
            job['origin_train_receipt']['sha256'] = runtime.sha_file(job['origin_train_receipt']['path'])
            endpoints[job['id']] = endpoint
    c['immutable_originals']['files'] = {str(p.relative_to(old)): runtime.sha_file(p) for p in old.rglob('*') if p.is_file()}
    c['immutable_originals']['files_sha256'] = runtime.sha_json(c['immutable_originals']['files'])
    cp = root/'recovery_contract.json'; dump(cp,c)

    def copied(binding, folder, name):
        target = folder/'origin'/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(binding['path']).read_bytes())
        return {'original_path':binding['path'],'original_sha256':binding['sha256'],
                'copy_path':str(target),'copy_sha256':runtime.sha_file(target)}

    def scientific_receipt(job, stage):
        folder = root/'run'/job['id']; folder.mkdir(parents=True,exist_ok=True)
        base = {'schema':runtime.STAGE_SCHEMA,'status':'complete','stage':stage,'job_id':job['id'],
            'contract_sha256':runtime.sha_json(c),'scientific_contract_sha256':runtime.sha_json(science),
            'scientific_contract_raw_sha256':c['scientific_contract']['raw_sha256'],
            **{k:job[k] for k in ('host','dataset','arm','seed','time_alpha','loss_mode','time_input')},
            'recovery_mode':job['recovery_mode'],'candidate_name':job['arm'],'late_path_parameter_count':1024,
            'late_path_mode':'observation' if job['arm']==runtime.ARMS[0] else 'hidden',
            'time_input_transform':'log1p_nonnegative_float32','quantity_input_transform':'log1p_nonnegative_float32',
            'held_out_test_evaluated':False,'quantity_only':True,'time_head_trained':False,
            'quantity_loss_normalization':'none','original_files_modified':False,'scientific_worker_source_unchanged':True,
            'time_head_no_training_verification':proof()}
        if stage=='train':
            checkpoint=folder/'selected.pt';checkpoint.write_bytes(b'fabricated selected state')
            endpoint={'selected_checkpoint':str(checkpoint),'selected_epoch':20,'checkpoint_sha256':runtime.sha_file(checkpoint),
                'state_sha256':'b'*64,'target_identity':{'split':'validation','row_count':2}}
            metrics={'overall':{'qty_rmse':3.,'qty_mae':2.}}
            nested={**base,**endpoint,'schema':'titantpp_late_path18_stage_v1','contract_sha256':runtime.sha_json(science),
                'validation_reference':{'metrics':metrics,'target_identity':endpoint['target_identity']},
                'files':{str(checkpoint.relative_to(root)):runtime.sha_file(checkpoint)}}
            path=folder/'train_scientific_receipt.json';dump(path,nested)
            if job['recovery_mode']=='resume':
                lineage={'resume_checkpoint':copied(job['origin_resume_checkpoint'],folder,'last_epoch_state.pt'),
                    'resumed_from_epoch':job['resume_epoch'],'next_epoch':job['resume_epoch']+1,
                    'original_started_at':job['original_started_at'],
                    'resume_validation':{'validated':True,'prefix_optimizer_steps':100}}
            else:lineage={'fresh_from_original_seed':True,'resumed_from_epoch':None}
            value={**base,**endpoint,'metrics':metrics,'lineage':lineage,
                   'scientific_receipt':{'path':str(path),'sha256':runtime.sha_file(path)}}
        elif job['recovery_mode']=='reference':
            original=endpoints[job['id']]
            traincopy=copied(job['origin_train_receipt'],folder,'train_receipt.json')
            checkpoint=copied(job['origin_selected_checkpoint'],folder,'selected.pt')
            value={**base,**{k:original[k] for k in ('selected_epoch','checkpoint_sha256','state_sha256','target_identity')},
                'selected_checkpoint':checkpoint['copy_path'],'metrics':original['validation_reference']['metrics'],
                'lineage':{'reference_train_receipt':traincopy,'reference_checkpoint':checkpoint,
                    'reference_retrained':False,'original_selected_checkpoint':original['selected_checkpoint']},
                'validation_replay_passed':True}
        else:
            train=runtime.read(folder/'train_receipt.json')
            value={**base,**{k:train[k] for k in ('selected_checkpoint','selected_epoch','checkpoint_sha256','state_sha256','target_identity','metrics')},
                'lineage':{'recovery_train_receipt':{'path':str(folder/'train_receipt.json'),'sha256':runtime.sha_file(folder/'train_receipt.json')},
                    'recovery_scientific_receipt':train['scientific_receipt']},'validation_replay_passed':True}
        receipt=folder/(stage+'_receipt.json')
        value['files']={str(p.relative_to(root)):runtime.sha_file(p) for p in folder.rglob('*') if p.is_file() and p!=receipt}
        dump(receipt,value)
        return value

    processes={};context=[None];nextpid=[3000]
    class Child:
        def __init__(self,command,**kwargs):
            nextpid[0]+=1;self.pid=nextpid[0];self.returncode=None;self.command=command;self.kwargs=kwargs
            processes[self.pid]=self
            if '--stage' in command:
                stage=command[command.index('--stage')+1]
                job=next(j for j in c['jobs'] if j['id']==command[command.index('--job')+1])
                value=({'status':'passed','real_data_loaded':False,'candidate_name':job['arm'],
                    'cuda_memory_peaks':{job['arm']:{'max_memory_reserved_bytes':100}}}
                    if stage=='qualify' else scientific_receipt(job,stage))
                kwargs['stdout'].write(json.dumps(value).encode());kwargs['stdout'].flush()
        def poll(self):
            if '--stage' in self.command:self.returncode=0
            return self.returncode
        def wait(self,timeout=None):
            if '--mode' in self.command:
                prior=context[0];context[0]=self.pid
                try:
                    mode=self.command[self.command.index('--mode')+1]
                    value=runtime.qualify(c,cp) if mode=='qualify' else runtime.dispatch(c,cp)
                    if mode=='qualify':
                        Path(self.kwargs['stdout'].name).write_bytes(json.dumps(value).encode())
                finally:context[0]=prior
            self.returncode=0
            return 0
    def identity(pid):
        p=processes.get(pid)
        if p is None or p.returncode is not None:return None
        return {'pid':pid,'pgid':pid,'start_ticks':pid*10,'argv':p.command,'cwd':c['source_root']}
    monkeypatch.setattr(runtime.subprocess,'Popen',Child)
    monkeypatch.setattr(runtime,'proc_identity',identity)
    monkeypatch.setattr(runtime.os,'getpid',lambda:context[0])
    monkeypatch.setattr(runtime,'runtime_check',lambda c:{'gpu':{'total_memory_bytes':100000}})
    monkeypatch.setattr(runtime,'gpu_snapshot',lambda uuid:{'compute_pids':[]})
    monkeypatch.setattr(runtime.time,'sleep',lambda seconds:None)
    monkeypatch.chdir(c['source_root'])
    assert runtime.launch_dispatch(c,cp)==0
    return c,root,old


def reseal(root):
    final=runtime.read(root/'final_receipt.json')
    for stage in ('train','validation'):
        sp=root/(stage+'_stage_receipt.json');stage_value=runtime.read(sp)
        for key in stage_value['receipts']:
            stage_value['receipts'][key]['sha256']=runtime.sha_file(root/stage_value['receipts'][key]['path'])
        dump(sp,stage_value);final['stages'][stage]=stage_value
    (root/'artifact_manifest.json').unlink()
    final['artifact_manifest']=runtime.artifact_manifest(root)
    dump(root/'final_receipt.json',final)


def test_actual_runtime_lifecycle_formats_verify_and_archive_remap(terminal,tmp_path):
    import shutil
    c,root,old=terminal
    result=manager.verify_terminal_evidence(c,check_live=True)
    assert result['actual_scientific_worker_exit0_count']==14
    assert result['actual_qualification_child_exit0_count']==6
    assert result['actual_dispatcher_exit0_count']==1 and result['fresh_process_absence_checked']
    assert not result['wrapper_own_os_exit_directly_observed']
    copied=tmp_path/'retrieved';origins=tmp_path/'preserved'
    shutil.copytree(root,copied);shutil.copytree(old,origins)
    shutil.rmtree(root);shutil.rmtree(old)
    result=manager.verify_terminal_evidence(c,execution_root=copied,origin_root=origins)
    assert result['status']=='verified' and not result['fresh_process_absence_checked']
    with pytest.raises(ValueError,match='native roots'):
        manager.verify_terminal_evidence(c,execution_root=copied,origin_root=origins,check_live=True)


@pytest.mark.parametrize('mutation',['count','timeout','launch','receipt_sha','empty_manifest','omit_manifest','dispatcher','qualification',
    'nested','selection','target','empty_files','resume_lineage','original','live_pid','live_gpu'])
def test_terminal_forgery_missing_evidence_and_live_ownership_rejected(terminal,mutation,monkeypatch):
    c,root,old=terminal;key=c['training_jobs'][0]
    if mutation=='count':
        p=root/'final_receipt.json';v=runtime.read(p);v['validation_replay_count']=9;dump(p,v)
    elif mutation in ('timeout','launch'):
        p=root/'run'/key/('train_exit.json' if mutation=='timeout' else 'train_launch.json');v=runtime.read(p)
        v['timeout' if mutation=='timeout' else 'start_ticks']=True if mutation=='timeout' else 1;dump(p,v)
        if mutation=='timeout':
            sp=root/'train_stage_receipt.json';stage=runtime.read(sp);stage['exits'][key]=v;dump(sp,stage)
        reseal(root)
    elif mutation=='receipt_sha':
        p=root/'run'/key/'train_receipt.json';v=runtime.read(p);v['extra']='changed';dump(p,v)
    elif mutation in ('empty_manifest','omit_manifest'):
        p=root/'artifact_manifest.json';v=runtime.read(p)
        if mutation=='empty_manifest':v['files']={}
        else:v['files'].pop('run/'+key+'/train_launch.json')
        dump(p,v);fp=root/'final_receipt.json';f=runtime.read(fp)
        f['artifact_manifest'].update(sha256=runtime.sha_file(p),file_count=len(v['files']));dump(fp,f)
    elif mutation=='dispatcher':
        p=root/'dispatcher_exit.json';v=runtime.read(p);v['pid']+=1;dump(p,v)
    elif mutation=='qualification':
        p=root/'qualification.json';v=runtime.read(p);v['per_arm'][runtime.ARMS[0]].pop();dump(p,v);reseal(root)
    elif mutation in ('selection','target','empty_files','resume_lineage'):
        p=root/'run'/key/'validation_receipt.json' if mutation in ('selection','target') else root/'run'/key/'train_receipt.json'
        v=runtime.read(p)
        if mutation=='selection':v['selected_epoch']+=1
        if mutation=='target':v['target_identity']={'row_count':1}
        if mutation=='empty_files':v['files']={}
        if mutation=='resume_lineage':v['lineage']['next_epoch']+=1
        dump(p,v);reseal(root)
    elif mutation=='nested':
        p=root/'run'/key/'train_scientific_receipt.json';v=runtime.read(p);v['target_identity']={'row_count':1};dump(p,v)
        tr=root/'run'/key/'train_receipt.json';v=runtime.read(tr)
        v['scientific_receipt']['sha256']=runtime.sha_file(p);v['files'][str(p.relative_to(root))]=runtime.sha_file(p);dump(tr,v);reseal(root)
    elif mutation=='original':(old/'execution_contract.json').write_text('{}')
    elif mutation=='live_pid':monkeypatch.setattr(runtime,'proc_identity',lambda pid:{'pid':pid})
    elif mutation=='live_gpu':monkeypatch.setattr(runtime,'gpu_state',lambda c:{'compute_pids':[999]})
    with pytest.raises((ValueError,KeyError)):
        manager.verify_terminal_evidence(c,check_live=mutation.startswith('live_'))
