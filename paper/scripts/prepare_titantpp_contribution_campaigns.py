"""Prepare three approved, ordered campaigns; local immutable work only."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import time
import sys
_PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(_PROJECT_ROOT))
from paper.scripts import titantpp_contribution_runtime as op

PROJECT=Path(__file__).resolve().parents[2]
REPORT=PROJECT/'reports/titantpp_contribution_execution_20261008_v1'
PARENT=PROJECT/'search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1'
CNN=PROJECT/'search_artifacts/titantpp_cnn_gru54_3seed_dual_20261005_v1'
WIDTH=PROJECT/'search_artifacts/titantpp_history_width16_dual_20261003_v1'
REPAIR=PROJECT/'search_artifacts/titantpp_contextual_retrieval_pilot_repair_20261007_v1'
USER='Instacart의 목적과 실행 조건을 먼저 확정한 이후 5080/5090에서 학습 진행. 그 이후 대표 구조의 이력 보정 추가 효과 및 고정 비중 검색의 추가 효과 분리까지 진행. 위 작업들에 대해서는 승인한다.'

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def copy_verified(original,target,digest):
    assert op.sha_file(original)==digest,'Original changed '+str(original)
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
    assert op.sha_file(target)==digest

def prepare():
    parent=op.read(PARENT/'execution_contract.json');cnn=op.read(CNN/'execution_contract.json')
    width=op.read(WIDTH/'execution_contract.json');repair=op.read(REPAIR/'execution_contract.json')
    assert parent['source']['files_sha256']==op.PARENT_CLOSURE
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
    additions={
      'instacart':[],
      'history':['models/TPPs/CountAwareTitanCNNHistoryControls.py','paper/scripts/run_titantpp_cnn_history_controls.py','paper/tests/test_titantpp_cnn_history_controls.py'],
      'retrieval':['models/TPPs/CountAwareTitanFixedRetrievalControls.py','paper/scripts/run_titantpp_fixed_retrieval_controls.py','paper/tests/test_titantpp_fixed_retrieval_controls.py']}
    stages={};previous=None
    for stage,arms in op.STAGES.items():
        name='titantpp_contribution_'+stage+'_20261008_v1';bundle=PROJECT/'search_artifacts'/name
        assert not bundle.exists(),'Preserved existing stage; no replacement'
        bundle.mkdir();source=bundle/'source';source.mkdir();files=deepcopy(parent['source']['files'])
        for p,v in files.items():copy_verified(PARENT/'source'/p,source/p,v)
        for p in additions[stage]:
            committed=subprocess.check_output(['git','show',revision+':'+p],cwd=PROJECT)
            assert __import__('hashlib').sha256(committed).hexdigest()==op.sha_file(PROJECT/p),'Uncommitted science '+p
            copy_verified(PROJECT/p,source/p,op.sha_file(PROJECT/p));files[p]=op.sha_file(source/p)
        if stage=='instacart':
            datasets=[deepcopy(next(d for d in width['datasets'] if d['dataset_id']=='insta_market_basket'))]
            data=datasets[0];data['prepared_train_validation_path']='inputs/data/insta_market_basket/instacart_marked_target_with_split.parquet'
            for kind,name0 in [('data','instacart_marked_target_with_split.parquet'),('split_manifest','instacart_marked_target_split_manifest.json')]:
                data['inherited_data_identity'][kind]['path']='inputs/data/insta_market_basket/'+name0
            original_inputs={p:PROJECT/'sample_data/insta_market_basket'/Path(p).name for p in [data['inherited_data_identity']['data']['path'],data['inherited_data_identity']['split_manifest']['path']]}
            groups={'5080':[['insta_market_basket',42],['insta_market_basket',62]],'5090':[['insta_market_basket',52]]}
            runners={a:'paper.scripts.run_titantpp_history_width' if 'width' in a else 'paper.scripts.run_titantpp_cnn_gru' for a in arms}
            design={'question':'Complete width8/12 and establish CNN+GRU54 on Instacart; supplementary development comparison',
                'existing_width16':'immutable historical reference only; no paired same-GPU efficiency or new untouched Test claim',
                'time_unit':'day','tail_boundary':35,'max_seq_len':64,'lookback_weeks':52,
                'Q1_Q2_contribution_claim':'Instacart baseline alone does not establish these effects'}
        else:
            datasets=deepcopy(cnn['datasets']);original_inputs={r['path']:CNN/'source'/r['path'] for d in datasets for r in [d['inherited_data_identity']['data'],d['inherited_data_identity']['split_manifest']]}
            groups={'5080':[[d,s] for d in ('yellow_trip_hourly','raf_spare_parts') for s in op.SEEDS],
                '5090':[['intermittent_frozen_5000',s] for s in op.SEEDS]}
            module='paper.scripts.run_titantpp_cnn_history_controls' if stage=='history' else 'paper.scripts.run_titantpp_fixed_retrieval_controls'
            runners={a:module for a in arms}
            if stage=='history':
                design={'question':'Explicit previous-state correction beyond encoded prefix history',
                    'correction_parameters_full_current_only':25308,'disabled_correction_trainable':0,
                    'current_only':'same current encoded Q/K/V projections, pointwise x/tanh/GELU CNN bases, two repeated-current GRU steps with state reset per position; no previous-row/state direct correction input',
                    'limitations':['Current encoded hidden still contains prefix history','Control activation/depth/cost differ; equal active parameter budget is not identical functional capacity'],
                    'full_mean_alias':'new full CNN+GRU54 fits reused as Q2 mean; no duplicate mean fit'}
            else:
                runners['titantpp_cnn_gru54']='paper.scripts.run_titantpp_cnn_gru'
                design={'question':'Fixed split-query retrieval against weighted and exact-active-budget general retrieval',
                    'additional_parameters':dict(zip(arms,[0,512,512])),'normalization':'cosine','temperature':1.0,
                    'memory_slots':64,'topk':4,'history_summary':32,'generic512':'8x64 query matrix, no learned temperature parameters',
                    'limits':['Fixed split query normalization differs from generic joint query normalization','Fixed two-view score has concatenation algebraic equivalence; novelty not established by two-view labels'],
                    'mean_baseline':'same dataset/seed/host new history-stage full CNN+GRU54 terminal; same source126 common tensors and batch order',
                    'diagnostics':'same selected top4 and score mean versus softmax aggregation, algebraic equivalence; no extra fits'}
        inputs={}
        for d in datasets:
            for r in [d['inherited_data_identity']['data'],d['inherited_data_identity']['split_manifest']]:
                p=r['path'];copy_verified(original_inputs[p],source/p,r['sha256']);inputs[p]=r['sha256']
        sentinel=source/'sample_data/.frozen_root_sentinel';sentinel.parent.mkdir(exist_ok=True);sentinel.write_bytes(b'');inputs[str(sentinel.relative_to(source))]=op.sha_file(sentinel)
        hosts={}
        for h in groups:
            spec=deepcopy(repair['hosts'][h]);root='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'+name+'_'+h
            spec.update(root=root,source_root=root+'/source',operation_root=root+'/operation',output_dir=root+'/run',tmux=name+'_'+h,
                assigned_datasets=sorted({d for d,s in groups[h]}),old_owner_roots=[])
            spec['environment'].update(SOURCE_REVISION=parent['source']['git_revision'] if stage=='instacart' else revision,
                PYTHONPATH=root+'/source',MPLCONFIGDIR=root+'/cache/matplotlib',XDG_CACHE_HOME=root+'/cache')
            hosts[h]=spec
        operation=bundle/'operation';operation.mkdir();shutil.copyfile(PROJECT/'paper/scripts/titantpp_contribution_runtime.py',operation/'campaign.py')
        ops={'operation/campaign.py':op.sha_file(operation/'campaign.py')}
        c={'schema':name,'stage':stage,'arms':list(arms),'seeds':list(op.SEEDS),'assignments':groups,'hosts':hosts,'runner_modules':runners,
            'datasets':datasets,'dataset_sha256':{d['dataset_id']:op.sha_json(d) for d in datasets},
            'source':{'git_revision':hosts['5080']['environment']['SOURCE_REVISION'],'base_git_revision':hosts['5080']['environment']['SOURCE_REVISION'],
                'parent_science_revision':parent['source']['git_revision'],'parent_files':parent['source']['files'],
                'files':files,'files_sha256':op.sha_json(files),'operational_git_revision':revision},
            'operation':{'files':ops,'files_sha256':op.sha_json(ops)},'input_files':inputs,'training':deepcopy(op.TRAINING),
            'architecture':design,'design_sha256':op.sha_json(design),'diagnostic':deepcopy(cnn['diagnostic']),
            'limits':{'total_wall_seconds':168*3600,'per_condition_seconds':36*3600,'qualification_seconds':5400,
                'workers_per_host':1,'automatic_retry':False,'root_bytes':8*1024**3,'min_free_bytes':5*1024**3,'cpu_rss_bytes':16*1024**3,'gpu_fraction_max':.8},
            'measurement_arms':list(arms)+(['titantpp_cnn_gru54'] if stage=='retrieval' else []),
            'cost_limits':{'inference_median_ratio_max':1.20,'training_step_median_ratio_max':1.30,'peak_allocated_ratio_max':1.20,'warmup_steps':5,'measure_steps':20,'candidate_eligibility_not_training_gate':True},
            'evaluation_scope':'validation_only','held_out_test_evaluated':False,'prior_stage':previous,
            'claim_contract_v2_rawSHA':op.sha_file(PROJECT/'paper/contracts/titantpp_claims_validation_20261008_v2.json'),
            'policy':{'new_Test_access':False,'automatic_retry':False,'resume':False,'shared_runtime_changes':False,'originals_preserved':True}}
        c['jobs']=op.expected_jobs(c);op.validate(c,verify_files=False);digest=op.sha_json(c)
        approval={'approved':True,'contract_sha256':digest,'hosts':list(hosts),'new_fits':len(c['jobs']),'user_instruction':USER,'recorded_unix':time.time(),'Test_access':False}
        write(bundle/'execution_contract.json',c);write(bundle/'approval.json',approval);write(bundle/'design_contract.json',design)
        write(bundle/'current.json',{'schema':name,'canonical_sha256':digest,'hosts':{h:s['root'] for h,s in hosts.items()},'source_closure_sha256':c['source']['files_sha256'],'operation_closure_sha256':c['operation']['files_sha256'],'status':'prepared','prior_stage':previous})
        control_files=['paper/scripts/'+n+'.py' for n in ('titantpp_contribution_runtime','prepare_titantpp_contribution_campaigns','deploy_titantpp_contribution_campaigns','observe_titantpp_contribution_campaigns','finalize_titantpp_contribution_campaigns','advance_titantpp_contribution_campaigns')]
        seal={'contract_sha256':digest,'project_files':{p:op.sha_file(PROJECT/p) for p in control_files}}
        write(bundle/'control/integrity_receipt.json',seal)
        current=op.read(bundle/'current.json');current['control_integrity_raw_sha256']=op.sha_file(bundle/'control/integrity_receipt.json')
        (bundle/'current.json').write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
        stages[stage]={'bundle':str(bundle.relative_to(PROJECT)),'contract_sha256':digest,'jobs':len(c['jobs']),'status':'prepared'}
        previous={'stage':stage,'bundle':str(bundle.relative_to(PROJECT)),'contract_sha256':digest,'jobs':len(c['jobs'])}
    historical=[]
    for seed,epoch,digest in [(42,30,'41376370ce0d25f20e3aa1a08896f9a5662ba9a39f6da4012a750dcdf954a47d'),(52,15,'e18554ec784f7e27c10f7fa580a1f7812ddcfb4247e42265ee20e0128282e538'),(62,35,'808c9240845d4eaabe33f0659ea235724fb825aee280e128574cd06f19c241af')]:
        name=f'reports/titantpp_completed_validation_test_20261004_v1/width16/original/insta_market_basket__{seed}__titantpp_history_mlp_width16/runs/titantpp_history_mlp_width16/count_only_log_regression/seed_{seed}/best_val_qty_rmse_model.pt'
        assert op.sha_file(PROJECT/name)==digest,'Historical original changed'
        historical.append({'seed':seed,'selected_epoch':epoch,'path':name,'binary_raw_sha256':digest,'historical_reference_only':True,'performance_read_now':False})
    write(REPORT/'historical_instacart_registry.json',{'conditions':historical,'fresh_same_GPU_comparator':False})
    write(REPORT/'execution_plan.json',{'approval':USER,'stages':stages,'total_fresh_fits':63,'baseline_aliases_reused':9,'stage_order':['instacart','history','retrieval'],
        'Test_evaluation':False,'same_dataset_seed_same_host':True,'created_unix':time.time()})
    print(json.dumps({'status':'prepared','stages':stages,'fresh_fits':63}))

if __name__=='__main__':prepare()
