"""Prepare the authorized history27 observation-only diagnostic bundle."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True
PROJECT = Path('/Users/igwanhyeong/PycharmProjects/paper_research')
PARENT = PROJECT/'search_artifacts/titantpp_history_instacart_test_20261012_v1'
BUNDLE = PROJECT/'search_artifacts/titantpp_history_diagnostics_20261012_v1'
REPORT = PROJECT/'reports/titantpp_history_diagnostics_20261012_v1'
os.environ['MPLCONFIGDIR'] = str(REPORT/'runtime_cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(REPORT/'runtime_cache')
PARENT_SHA = 'f7fe042869e239a369c4ca77aa0c519bad1cde40d10688f6538e51ef803c183c'
PARENT_RAW = '158df041c2b8d5ac8ba69f0626bd78753e0c31f6aa207c4fe14ba166840d7e47'
SOURCE_CLOSURE = 'c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            h.update(part)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def copy_frozen(relative, expected):
    src, dest = PARENT/relative, BUNDLE/relative
    assert sha(src) == expected, relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    assert sha(dest) == expected, relative

def boundaries(values):
    import numpy as np
    return sorted(set(float(np.quantile(values, q)) for q in (.25, .5, .75, .9, .99)))

def derive_strata(parent):
    import numpy as np
    import polars as pl
    source = PARENT/parent['sources']['history']['root']
    sys.path.insert(0, str(source))
    from paper.scripts.run_taxi_quantity_interface_ablation import RMTPPWeekLookbackDataset
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    pop_path = PROJECT/'reports/titantpp_test_reversal_diagnosis_20261012_v1/population_diagnosis.json'
    population = json.loads(pop_path.read_bytes())
    result = {}
    for ds, spec in parent['datasets'].items():
        if ds == 'insta_market_basket':
            continue
        definition = spec['definition']
        frame = prepare_count_frame(pl.read_parquet(PARENT/spec['training_data_path'])
            .filter(pl.col('chronological_split').is_in(['train', 'validation']))
            .sort(['oper_part_no', 'seq']))
        dataset = RMTPPWeekLookbackDataset(frame,
            lookback_weeks=definition['loader']['lookback_weeks'],
            max_seq_len=definition['loader']['max_seq_len'], mode='all',
            target_splits={'train'}, split_col='chronological_split')
        index = np.asarray(dataset.index, dtype=np.int64).reshape(-1, 2)
        last_gaps = np.asarray([dataset.dt_lists[int(p)][int(end)] for p, end in index], dtype=np.float64)
        assert np.isfinite(last_gaps).all() and len(last_gaps) == definition['time_statistics']['expected_train_targets']
        s = population['datasets'][ds]['splits']['train']
        hist_q = s['observed_history_length']['quantiles']
        gap_q = s['recorded_gap']['quantiles']
        result[ds] = {
            'input_history_boundaries': sorted(set(float(hist_q[str(q)]) for q in (.25,.5,.75,.9,.99))),
            'input_last_gap_boundaries': boundaries(last_gaps),
            'target_gap_boundaries': sorted(set(float(gap_q[str(q)]) for q in (.25,.5,.75,.9,.99))),
            'train_history_max': s['observed_history_length']['maximum'],
            'input_last_gap_train_sha256': hashlib.sha256(b'diagnostic_train_input_last_gap_v1\0'+last_gaps.astype('<f8').tobytes()).hexdigest(),
            'boundary_source': 'eligible Train targets only, quantiles .25/.5/.75/.9/.99 deduplicated',
            'target_gap_scope': 'recorded next-event duration; descriptive outcome, not input-known',
            'target_quantity_scope': 'original fixed Train-derived quantity bins, descriptive outcome',
            'calendar_scope': 'last observed calendar is input-known; target calendar descriptive only',
            'population_diagnosis_raw_sha256': sha(pop_path),
        }
    return result

def main():
    assert sha(PARENT/'contract.json') == PARENT_RAW
    parent = json.loads((PARENT/'contract.json').read_bytes())
    assert parent['contract_sha256'] == PARENT_SHA
    assert canonical({k:v for k,v in parent.items() if k!='contract_sha256'}) == PARENT_SHA
    assert parent['sources']['history']['closure'] == SOURCE_CLOSURE
    assert not BUNDLE.exists(), 'No bundle preparation retry/overwrite'
    inputs = json.loads((PARENT/'input_integrity.json').read_bytes())['files']
    jobs = [j for j in parent['jobs'] if j['study'] == 'history']
    assert len(jobs)==27 and len({j['id'] for j in jobs})==27
    strata = derive_strata(parent)
    job_ids = {j['id'] for j in jobs}
    selected_files = {}
    for rel, digest in inputs.items():
        parts = Path(rel).parts
        include = rel.startswith('sources/history/') or rel.startswith('provenance/history_')
        include |= rel.startswith('data/full/') and Path(rel).stem in strata
        include |= rel.startswith('checkpoints/') and Path(rel).stem in job_ids
        include |= rel.startswith('provenance/checkpoint_metadata/') and Path(rel).stem in job_ids
        include |= len(parts)>2 and parts[:2]==('provenance','jobs') and parts[2] in job_ids
        if include:
            copy_frozen(rel, digest)
            selected_files[rel] = digest
    references = {}
    for job in jobs:
        references[job['id']] = {}
        for split, name in [('validation','validation_replay.json'),('test','test_metrics.json')]:
            src=PARENT/'results'/job['id']/name
            rel=f'reference/results/{job["id"]}/{name}'
            dest=BUNDLE/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dest)
            digest=sha(src)
            assert sha(dest)==digest
            references[job['id']][split]={'relative_path':rel,'sha256':digest}
            selected_files[rel]=digest
    c=copy.deepcopy(parent)
    c.pop('contract_sha256')
    c.update(schema='titantpp_history_diagnostics_20261012_v1',created_unix=time.time(),
        evaluation_scope='existing_Test_exploratory_error_diagnostics',parent_contract_sha256=PARENT_SHA,
        parent_contract_raw_sha256=PARENT_RAW)
    c['sources']={'history':c['sources']['history']}
    c['datasets']={ds:spec for ds,spec in c['datasets'].items() if ds in strata}
    c['jobs']=jobs
    for ds in strata:c['datasets'][ds]['diagnostics']=strata[ds]
    c['authorization']={'human_request':'진행해보자',
        'approved_task':'기존 선택 checkpoint27개의 Validation/Test54 split에서 series·이력·시간 head·CNN/GRU 출력을 오류와 연결하는 진단 계측',
        'existing_Test_access_and_inference_authorized':True,'training':False,
        'checkpoint_or_hyperparameter_reselection':False,'new_dataset':False,'new_retrieval_training':False,
        'shared_Runtime_or_auth_changes':False,'manuscript_edit':False,'route_bypass_interventions':False}
    c['execution']['remote_root']='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_diagnostics_20261012_v1_5080'
    c['execution']['gate']='all27_native_diagnostic_validation_before_any_Test'
    c['diagnostics']={'reference_receipts':references,'same_native_tolerance':{'absolute':1e-5,'relative':1e-5},
        'aggregate_only':True,'raw_predictions_written':False,'full_residual_telemetry_only':True,
        'parameter_updates':0,'encoder_path_interventions':0,'expected_split_evaluations':54,
        'time_metric':'recorded_positive_integer_time_nll',
        'quantity_scale_is_uncertainty':False,'observational_norm_is_signed_quantity_effect':False,
        'stored_Test_is_independent':False}
    c['contract_sha256']=canonical(c)
    write(BUNDLE/'contract.json',c)
    selected_files['contract.json']=sha(BUNDLE/'contract.json')
    write(BUNDLE/'input_integrity.json',{'schema':'diagnostic_frozen_inputs_v1',
        'contract_sha256':c['contract_sha256'],'files':selected_files})
    write(REPORT/'APPROVAL.json',{'human_request':'진행해보자','authorized':True,
        'authorized_scope':c['authorization'],'contract_sha256':c['contract_sha256'],
        'observed_unix':time.time(),'prior_plan':'reports/titantpp_test_reversal_diagnosis_20261012_v1/MINIMUM_EXPERIMENT_PLAN.md'})
    print(json.dumps({'status':'prepared','jobs':27,'files':len(selected_files),
        'contract_sha256':c['contract_sha256'],'strata':strata},ensure_ascii=False))

if __name__=='__main__':main()
