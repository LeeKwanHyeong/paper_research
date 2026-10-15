#!/usr/bin/env python3
"""Fixed-checkpoint Key-only available-gain-preserving content on/off rows; no training, Test, or baseline inference.

Operation-only adapter. Scientific imports are isolated to the sealed 145-file
source; helper/row utilities are separately pinned operational dependencies.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

SCHEMA = 'titantpp_key_content_probe_v1'
PERMIT_SCHEMA = 'titantpp_key_content_probe_permit_v1'
ARM = 'titantpp_cnn_key_only_v1'
COHORT = 'qk_location_raf6_v1'
SEEDS = (42, 52, 62)
EPOCHS = {42: 9, 52: 5, 62: 26}
COUNTS = {'train': 25779, 'validation': 6690}
SOURCE_CLOSURE = 'fc0668016b8eb89752e03d0751e317e77b06d8135c0b82f8b43625e345abf9a3'
DATA_SHA = 'd805cad0242078fedfbb87eed18bde9b73d37bfc9b93c6b75aac49087244efa2'
SPEC_SHA = 'f8ce822618bc0dcbacda3512215a9e368afb054a26466616cca1089b12d18678'
ACCEPTANCE_SHA = '6520d428c1cd3c4ac4f211c2c4e89a6b5b570d4efe22cf4c64ff73a948771d2e'
STAGES = ('qualification', 'on_validation', 'off_train', 'off_validation')
VALIDATION_TOLERANCE = {'rtol': 1e-5, 'atol': 1e-5}
GPU_UUID = 'GPU-c9adc246-ef66-7906-bc07-6e152fa9952f'
ABSOLUTE_DEADLINE = 1792679647.915156
REMOTE_ROOT = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_key_content_probe_20261016_v1_5090'
INTERVENTION = 'replace_available_past_key_content_with_current_preserve_gain'
PARENT_RECEIPT_SHA = '42951c5ffb69639a438a002f0dd9e70eee1f3d8c234ad87b2ad42861c5041ff7'
ROW_TOLERANCE = {'relative':1e-5,'absolute':1e-5}
TRUTH_FIELDS = ('target_id', 'entity_id', 'seq', 'split', 'recorded_date', 'raw_quantity', 'recorded_gap',
                'history_length', 'last_quantity', 'mean_quantity')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def canonical(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, relative):
    root = Path(root).resolve()
    need(isinstance(relative, str) and not Path(relative).is_absolute(), 'Artifact path must be relative')
    p = (root / relative).resolve()
    need(p.is_relative_to(root), 'Artifact path escapes root')
    return p


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, 'Operational dependency cannot load')
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def verify_contract(root, expected_rawsha):
    """Fail before model/data imports; usable by the stdlib orchestration layer."""
    root = Path(root).resolve()
    need(sha(root / 'contract.json') == expected_rawsha, 'Contract raw SHA mismatch')
    c = read(root / 'contract.json')
    need(c.get('contract_sha256') == canonical({k: v for k, v in c.items() if k != 'contract_sha256'}), 'Contract canonical mismatch')
    need(c.get('schema') == SCHEMA and c.get('allowed_splits') == ['train', 'validation']
         and c.get('new_training') is False and c.get('frozen_selection') is True
         and c.get('max_parallel_per_host') == 3 and c.get('host') == '5090'
         and c.get('remote_root') == REMOTE_ROOT and c.get('stages') == list(STAGES)
         and c.get('conditions') == ['on', 'off'] and c.get('validation_tolerance') == VALIDATION_TOLERANCE, 'Diagnostic scope differs')
    need(c.get('intervention') == INTERVENTION and c.get('parent_native_receipt_sha256') == PARENT_RECEIPT_SHA
         and canonical(c.get('content_multipliers'))==canonical([1,0])
         and type(c.get('new_on_train_count')) is int and c['new_on_train_count']==0
         and type(c.get('reused_on_train_count')) is int and c['reused_on_train_count']==3,
         'Only available-gain-preserving current-content substitution is permitted')
    need(type(c.get('deadline_unix')) in (float, int) and math.isfinite(c['deadline_unix'])
         and time.time() < c['deadline_unix'] <= ABSOLUTE_DEADLINE, 'Inherited deadline invalid or expired')
    jobs = c.get('jobs', [])
    need(len(jobs) == 3 and {j.get('seed') for j in jobs} == set(SEEDS), 'Only three current selected candidates permitted')
    for j in jobs:
        need(j.get('arm') == ARM and j.get('cohort') == COHORT and j.get('dataset') == 'raf_spare_parts'
             and j.get('evaluation_host') == '5090' and j.get('selected_epoch') == EPOCHS[j['seed']]
             and j.get('id') == f"raf_spare_parts__{j['seed']}__{ARM}", 'Foreign model/seed/selection/host')
    source = c['source']
    need(len(source['files']) == 145 and source['closure'] == SOURCE_CLOSURE
         and canonical(source['files']) == SOURCE_CLOSURE, 'Frozen scientific source lineage differs')
    need(set(c['datasets']) == {'raf_spare_parts'}, 'Foreign dataset')
    data = c['datasets']['raf_spare_parts']
    need(data['sha256'] == DATA_SHA and data['definition']['data_sha256'] == DATA_SHA, 'Frozen Train/Validation input differs')
    populations = data['definition']['inherited_data_identity']['populations']
    need(set(populations) == set(COUNTS) and all(populations[s]['target_count'] == n for s, n in COUNTS.items()), 'Target population differs')
    need(data['definition']['loader'] == {'batch_size':128, 'drop_last':False, 'lookback_weeks':84,
         'max_seq_len':84, 'num_workers':0, 'train_shuffle':True, 'validation_shuffle':False}, 'Frozen loader differs')
    expected = c['runtimes']['5090']
    need(expected['host'] == '5090' and expected['gpu_uuid'] == GPU_UUID and expected['python_version'] == '3.12.13'
         and expected['torch'] == '2.11.0+cu130' and expected['cudnn'] == 92000 and expected['torch_num_threads'] == 4,
         'Approved native Runtime differs')
    operations, inputs = c['operation_files'], c['input_files']
    need(not set(operations) & set(inputs), 'Operation/input path overlap')
    for name in ('operation/evaluate.py', 'operation/helper.py', 'operation/row_utils.py', 'operation/lineage.json'):
        need(name in operations, 'Missing operational dependency '+name)
    need(sha(__file__) == operations['operation/evaluate.py'], 'Worker code differs')
    for n, h in {**operations, **inputs}.items():
        need(sha(checked(root, n)) == h, 'Frozen operation/input changed: '+n)
    for n, h in source['files'].items():
        need(sha(checked(checked(root, source['root']), n)) == h, 'Frozen scientific source changed: '+n)
    need(inputs.get(data['path']) == DATA_SHA, 'Dataset missing from pinned inputs')
    for j in jobs:
        for pathkey, hashkey in [('checkpoint_metadata_path','checkpoint_metadata_sha256'),
                                 ('selection_history_path','selection_history_sha256')]:
            need(inputs.get(j[pathkey]) == j[hashkey], 'Selection metadata pin missing')
        need(inputs.get(j['checkpoint']['relative_path']) == j['checkpoint']['sha256'], 'Checkpoint pin missing')
    for name in ('spec', 'acceptance', 'approval'):
        need(inputs.get(c[name+'_path']) == c[name+'_sha256'], name+' pin missing')
    need(c['spec_sha256'] == SPEC_SHA and c['acceptance_sha256'] == ACCEPTANCE_SHA, 'Current diagnostic specification differs')
    for pathkey,hashkey in [('parent_native_receipt_path','parent_native_receipt_sha256'),('parent_contract_path','parent_contract_raw_sha256')]:
        need(inputs.get(c[pathkey])==c[hashkey], 'Parent provenance pin missing')
    parent=read(checked(root,c['parent_native_receipt_path']))
    parent_contract=read(checked(root,c['parent_contract_path']))
    need(parent.get('status')=='complete' and parent.get('contract_raw_sha256')==c['parent_contract_raw_sha256']
         and parent_contract.get('contract_sha256')==c['parent_contract_sha256']
         and canonical({k:v for k,v in parent_contract.items() if k!='contract_sha256'})==c['parent_contract_sha256'],
         'Completed parent provenance differs')
    need(set(c.get('reference_rows',{})) == {j['id'] for j in jobs}, 'All stored on reference jobs required')
    for j in jobs:
        need(set(c['reference_rows'][j['id']]) == set(COUNTS), 'Both stored on splits required')
        for split in COUNTS: validate_stored_reference(root,c,j,split)
    approval = read(checked(root, c['approval_path']))
    need(approval.get('authorized') is True and approval.get('new_training') is False
         and approval.get('frozen_selection') is True and approval.get('allowed_splits') == ['train','validation']
         and approval.get('historical_CNN_inference') is False and approval.get('held_out_test_evaluated') is False,
         'Diagnostic approval missing')
    return c



def validate_six_metrics(actual, reference):
    for group in (None, 'tail'):
        a = actual if group is None else actual[group]
        b = reference if group is None else reference[group]
        need(type(a.get('count')) is int and a['count'] == b['count'] >= 0, 'Validation metric population differs')
        for metric in ('qty_rmse', 'qty_mae', 'time_nll'):
            if a['count'] == 0:
                need(a.get(metric) is None and b.get(metric) is None, 'Empty metric must be unavailable')
            else:
                need(type(a.get(metric)) in (float, int) and type(b.get(metric)) in (float, int)
                     and math.isfinite(a[metric]) and math.isfinite(b[metric])
                     and math.isclose(a[metric], b[metric], rel_tol=1e-5, abs_tol=1e-5), 'Validation six-metric replay differs')


def predictive_state(model, runner):
    """Include persistent state, nonpersistent buffers, and the inherited context cache."""
    contexts = {}
    for name, submodule in model.named_modules():
        if hasattr(submodule, '_ctx_mem'):
            value = submodule._ctx_mem
            contexts[name] = None if value is None else runner.canonical_state_dict_sha256({'context': value})
    return {'state_sha256': runner.canonical_state_dict_sha256(model.state_dict()),
            'all_named_buffers_sha256': runner.canonical_state_dict_sha256(dict(model.named_buffers())),
            'context_cache_sha256': canonical(contexts)}


def require_fixed_state(ctx):
    current = predictive_state(ctx['model'], ctx['runner'])
    need(current == ctx['predictive_state'] and current['state_sha256'] == ctx['state'],
         'Diagnostic mutated model parameters, buffers, or contextual memory')
    need(not ctx['model'].training and not any(p.requires_grad for p in ctx['model'].parameters()),
         'Diagnostic model must remain frozen and in evaluation mode')
    return current


def validate_predictive_snapshot(receipt, state_sha):
    before=receipt.get('predictive_state_before');after=receipt.get('predictive_state_after')
    keys={'state_sha256','all_named_buffers_sha256','context_cache_sha256'}
    need(isinstance(before,dict) and isinstance(after,dict) and set(before)==set(after)==keys,
         'Predictive snapshot requires exact state/buffer/context keys')
    need(all(type(value) is str and len(value)==64 and all(ch in '0123456789abcdef' for ch in value)
             for value in before.values()) and before==after and before['state_sha256']==state_sha,
         'Predictive state/buffer/context snapshot differs')
    return before


def stage_identity(stage):
    need(stage in STAGES, 'Only qualification and current-candidate on/off Train/Validation are permitted')
    return ('synthetic_both', 'qualification') if stage == 'qualification' else tuple(stage.split('_', 1))


def validate_stored_reference(root, c, job, split):
    """Pure stdlib provenance check of immutable parent on rows; never materialize parquet."""
    root=Path(root);need(split in COUNTS,'Only stored Train/Validation on references permitted')
    ref=c['reference_rows'][job['id']][split]
    for kind in ('receipt','manifest'):
        need(c['input_files'].get(ref[kind+'_path'])==ref[kind+'_sha256']
             and sha(checked(root,ref[kind+'_path']))==ref[kind+'_sha256'],'Stored reference pin differs')
    need(sha(checked(root,c['spec_path']))==c['spec_sha256'],'Reference specification SHA differs')
    spec=read(checked(root,c['spec_path']))
    pin=spec['parent_prediction_references'][job['id']+'/on_'+split]
    need(ref['receipt_sha256']==pin['receipt_sha256'] and ref['manifest_sha256']==pin['manifest_sha256'],
         'Stored reference differs from approved original')
    r=read(checked(root,ref['receipt_path']));mpath=checked(root,ref['manifest_path']);m=read(mpath)
    need(r.get('schema')=='titantpp_key_history_probe_v1' and r.get('status')=='complete'
         and r.get('job_id')==job['id'] and r.get('split')==split and r.get('stage')=='on_'+split
         and r.get('condition')=='on' and r.get('selected_epoch')==job['selected_epoch']
         and r.get('checkpoint_sha256')==job['checkpoint']['sha256']
         and r.get('source_closure')==SOURCE_CLOSURE and r.get('data_sha256')==DATA_SHA
         and r.get('state_including_buffers_unchanged') is True and r.get('parameters_unchanged') is True
         and r.get('state_sha256_before')==r.get('state_sha256_after')==job['checkpoint_state_sha256']
         and r.get('held_out_test_evaluated') is False and r.get('no_training') is True
         and r.get('native_runtime_checked') is True and r.get('branch_function_restored') is True
         and r.get('intervention')=='key_history_taps_only' and r.get('current_tap_preserved') is True
         and r.get('history_taps_disabled') is False and r.get('original_full_function_checks_preserved') is True,
         'Stored on scientific receipt identity differs')
    state=validate_predictive_snapshot(r,job['checkpoint_state_sha256'])
    need(state==pin['predictive_state'] and r.get('runtime')==r.get('runtime_after')==pin['native_runtime'],
         'Stored on state/native Runtime differs')
    need(r.get('population')==pin['population']==c['datasets']['raf_spare_parts']['definition']['inherited_data_identity']['populations'][split]
         and r.get('original_population_match') is True,'Stored on population differs')
    for key in ('target_identity_sha256','truth_sha256','recorded_gap_sha256'):
        need(r.get(key)==pin[key],'Stored on target identity differs')
    need(r['metrics']==pin['metrics'],'Stored original metrics differ')
    validate_six_metrics(r['metrics'],r['metrics'])
    need(r['manifest_sha256']==ref['manifest_sha256'] and r['prediction_rows']==m['prediction_rows']==COUNTS[split]
         and m['job_id']==job['id'] and m['split']==split and m.get('condition')=='on'
         and m.get('stage')=='on_'+split and m.get('contract_sha256')==r['contract_sha256']
         and m.get('input_sha256')==DATA_SHA and m.get('source_closure')==SOURCE_CLOSURE
         and m.get('checkpoint_sha256')==job['checkpoint']['sha256'] and m.get('runtime')==r['runtime'],
         'Stored on manifest scientific identity differs')
    need(m['parts'] and sum(x['rows'] for x in m['parts'])==COUNTS[split]
         and len({x['path'] for x in m['parts']})==len(m['parts'])==len(pin['parts']), 'Stored shard coverage differs')
    for part,old in zip(m['parts'],pin['parts']):
        path=checked(mpath.parent,part['path'])
        need(type(part['rows']) is int and part['rows']>0 and part['rows']==old['rows']
             and part['sha256']==old['sha256'] and c['input_files'].get(str(path.relative_to(root)))==part['sha256']
             and sha(path)==part['sha256'],'Stored on shard not pinned or changed')
    if split=='validation':validate_six_metrics(r['metrics'],job['validation_reference'])
    return {'receipt_sha256':ref['receipt_sha256'],'manifest_sha256':ref['manifest_sha256'],
            'prediction_rows':COUNTS[split],'predictive_state':state,'reference':ref,'receipt':r,'manifest':m}


def verify_stored_on_train_gate(root,c,expected_sha,*,reference_root=None):
    need(type(expected_sha) is str,'Stored on Train gate SHA required')
    root=Path(root);path=root/'stored_on_train_gate.json'
    need(sha(path)==expected_sha,'Stored on Train gate SHA differs');g=read(path)
    ids={j['id'] for j in c['jobs']}
    need(g.get('schema')=='titantpp_key_content_probe_stored_on_train_gate_v1'
         and g.get('contract_sha256')==c['contract_sha256'] and g.get('parent_native_receipt_sha256')==PARENT_RECEIPT_SHA
         and g.get('all_stored_references_verified') is True and set(g.get('job_ids',[]))==ids
         and len(g['job_ids'])==3 and set(g.get('references',{}))==ids,'Stored on Train gate incomplete')
    for j in c['jobs']:
        need(g['references'][j['id']]==validate_stored_reference(root if reference_root is None else reference_root,c,j,'train'),'Stored on Train gate reference differs')
    return g


def verify_reference(root,c,job,split,gate_sha):
    if split=='train':
        verify_stored_on_train_gate(root,c,gate_sha)
        v=validate_stored_reference(root,c,job,split);ref=v['reference']
        return v['receipt'],v['manifest'],{'path':ref['receipt_path'],'sha256':ref['receipt_sha256'],
                                         'manifest_path':ref['manifest_path']}
    gate=verify_gate(root,c,'on_validation',gate_sha);ref=gate['receipts'][job['id']]
    receipt=read(checked(root,ref['path']))
    return receipt,read(checked(root,receipt['manifest_path'])),{**ref,'manifest_path':receipt['manifest_path']}


def validate_rowwise(actual,reference,expected_count):
    """Replay parent on Validation: exact roles/truth/history, tolerant prediction/time loss."""
    fields=set(TRUTH_FIELDS)|{'dataset','seed','history_log_mean','history_log_std','last_log_deviation',
        'last_observed_log_change','previous_observation_available','observed_history_sha256','condition','stage'}
    need(len(actual)==len(reference)==expected_count,'Rowwise replay coverage differs')
    a={r['target_id']:r for r in actual};b={r['target_id']:r for r in reference}
    need(len(a)==len(b)==expected_count and set(a)==set(b),'Rowwise replay duplicate/foreign targets')
    maxima={'prediction':0.,'time_nll':0.}
    for key,x in a.items():
        y=b[key]
        need(x.get('condition')==y.get('condition')=='on' and x.get('stage')==y.get('stage')=='on_validation'
             and all(k in x and k in y and x[k]==y[k] for k in fields),'Rowwise replay role/truth/history differs')
        for name in maxima:
            need(type(x.get(name)) in (int,float) and type(y.get(name)) in (int,float)
                 and math.isfinite(x[name]) and math.isfinite(y[name])
                 and math.isclose(x[name],y[name],rel_tol=ROW_TOLERANCE['relative'],abs_tol=ROW_TOLERANCE['absolute']),
                 'Rowwise replay '+name+' differs')
            maxima[name]=max(maxima[name],abs(x[name]-y[name]))
    return {'status':'PASS','target_count':expected_count,'exact_targets':True,'exact_truth_and_history':True,
            'exact_condition_and_stage':True,'prediction_match':True,'time_loss_match':True,
            'tolerance':dict(ROW_TOLERANCE),'max_absolute_error':maxima}


def validate_replay_receipt(replay):
    need(isinstance(replay,dict) and replay.get('status')=='PASS' and replay.get('target_count')==COUNTS['validation']
         and replay.get('tolerance')==ROW_TOLERANCE and all(replay.get(k) is True for k in
         ('exact_targets','exact_truth_and_history','exact_condition_and_stage','prediction_match','time_loss_match'))
         and set(replay.get('max_absolute_error',{}))=={'prediction','time_nll'}
         and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in replay['max_absolute_error'].values()),
         'Per-target on Validation prediction/time replay missing')


def validate_paired_rows(actual, reference, expected_count):
    """Exact on/off target/truth/history pairing; predictions and losses may differ."""
    need(len(actual)==len(reference)==expected_count,'Paired row coverage differs')
    fields=set(TRUTH_FIELDS)|{'dataset','seed','history_log_mean','history_log_std',
        'last_log_deviation','last_observed_log_change','previous_observation_available','observed_history_sha256'}
    def indexed(rows):
        result={}
        for row in rows:
            ident=row['target_id'];need(ident not in result,'Duplicate target in paired rows');result[ident]=row
        return result
    for condition,rows in [('off',actual),('on',reference)]:
        for row in rows:
            need(row.get('split') in COUNTS and row.get('condition')==condition
                 and row.get('stage')==condition+'_'+row['split'], 'Paired row condition/stage differs')
    a,b=indexed(actual),indexed(reference)
    need(set(a)==set(b),'Paired target identity differs')
    for ident in a:
        x,y=a[ident],b[ident]
        need(all(k in x and k in y and x[k]==y[k] for k in fields),'Paired truth/history differs')
        for row in (x,y):
            for key in ('prediction','time_nll'):
                need(type(row.get(key)) in (float,int) and math.isfinite(row[key]),'Nonfinite paired prediction/time loss')
    return {'status':'PASS','target_count':expected_count,'exact_targets':True,'exact_truth_and_history':True,
            'prediction_equality_required':False,'time_loss_equality_required':False}

def read_manifest_rows(root, manifest_path, pl):
    path = checked(root, manifest_path); m = read(path)
    pieces = []
    for part in m['parts']:
        p = checked(path.parent, part['path']); need(sha(p) == part['sha256'], 'Row shard changed')
        frame = pl.read_parquet(p); need(frame.height == part['rows'], 'Row shard count differs')
        pieces.extend(frame.to_dicts())
    need(len(pieces) == m['prediction_rows'], 'Manifest row coverage differs')
    return pieces


def verify_gate(root, c, stage, expected_sha,*,reference_root=None):
    root=Path(root)
    need(stage in ('qualification','on_validation'),'Unexpected gate stage')
    name=stage+'_gate.json';need(sha(root/name)==expected_sha,'Gate raw SHA differs')
    g=read(root/name);ids={j['id'] for j in c['jobs']}
    need(g.get('schema')==f'titantpp_key_content_probe_{stage}_gate_v1'
         and g.get('contract_sha256')==c['contract_sha256'] and g.get('all_native_exit0') is True
         and set(g.get('job_ids',[]))==ids and len(g['job_ids'])==3
         and set(g.get('receipts',{}))==set(g.get('worker_exits',{}))==ids,'Incomplete global gate')
    for job in c['jobs']:
        ident=job['id'];items=[]
        for key in ('receipts','worker_exits'):
            ref=g[key][ident];path=checked(root,ref['path'])
            need(sha(path)==ref['sha256'],'Gate evidence changed');items.append(read(path))
        receipt,actual=items;condition,split=stage_identity(stage)
        need(receipt.get('status')=='complete' and receipt.get('job_id')==ident
             and receipt.get('stage')==stage and receipt.get('split')==split and receipt.get('condition')==condition
             and receipt.get('contract_sha256')==c['contract_sha256']
             and receipt.get('parameters_unchanged') is True and receipt.get('state_including_buffers_unchanged') is True
             and receipt.get('state_sha256_before')==receipt.get('state_sha256_after')==job['checkpoint_state_sha256']
             and receipt.get('branch_function_restored') is True and receipt.get('native_runtime_checked') is True
             and receipt.get('no_training') is True and receipt.get('held_out_test_evaluated') is False,
             'Gate scientific receipt identity differs')
        validate_intervention_receipt(receipt)
        validate_predictive_snapshot(receipt,job['checkpoint_state_sha256'])
        need(actual.get('job_id')==ident and actual.get('stage')==stage
             and actual.get('contract_sha256')==c['contract_sha256'] and actual.get('actual_child_exit_observed') is True
             and type(actual.get('exit_code')) is int and actual['exit_code']==0,'Gate actual worker exit0 missing')
        if stage=='qualification':
            need(receipt.get('qualification_passed') is True and receipt.get('real_data_loaded') is False,'Native synthetic qualification missing')
            continue
        need(receipt.get('selected_epoch')==job['selected_epoch'] and receipt.get('checkpoint_sha256')==job['checkpoint']['sha256']
             and receipt.get('source_closure')==SOURCE_CLOSURE and receipt.get('data_sha256')==DATA_SHA
             and receipt.get('population')==c['datasets']['raf_spare_parts']['definition']['inherited_data_identity']['populations'][split]
             and receipt.get('original_population_match') is True and receipt.get('prediction_rows')==COUNTS[split],
             'Full on target/source/checkpoint identity missing')
        need(receipt.get('stored_prediction_rows_available') is True,'Stored on rows must be replayed')
        validate_replay_receipt(receipt.get('rowwise_replay'))
        parent=validate_stored_reference(root if reference_root is None else reference_root,c,job,'validation')
        need(receipt['predictive_state_before']==parent['predictive_state'] and receipt.get('runtime')==parent['receipt']['runtime'],
             'New on state/Runtime differs from stored on')
        validate_six_metrics(receipt['metrics'],receipt['metrics'])
        need(receipt['metrics']['count']==COUNTS[split],'On full metric count differs')
        if stage=='on_validation':
            need(receipt.get('validation_match')=='PASS'
                 and receipt.get('validation_replay_check')=={'six_metrics':True,'relative_tolerance':1e-5,'absolute_tolerance':1e-5},
                 'Full on Validation replay missing')
            validate_six_metrics(receipt['metrics'],job['validation_reference'])
        else:
            need(receipt.get('validation_match') is None,'Train must not claim stored Validation replay')
        manifest=checked(root,receipt['manifest_path']);need(sha(manifest)==receipt['manifest_sha256'],'Gate row manifest changed')
        m=read(manifest)
        need(m['prediction_rows']==COUNTS[split] and m['job_id']==ident and m.get('stage')==stage
             and m['contract_sha256']==c['contract_sha256'] and m['split']==split and m.get('condition')=='on'
             and m.get('input_sha256')==DATA_SHA and m.get('source_closure')==SOURCE_CLOSURE
             and m.get('checkpoint_sha256')==job['checkpoint']['sha256'],'Gate row manifest identity differs')
        need(m['parts'] and sum(part['rows'] for part in m['parts'])==COUNTS[split]
             and len({part['path'] for part in m['parts']})==len(m['parts']),'Gate row shard coverage differs')
        for part in m['parts']:
            need(type(part['rows']) is int and part['rows']>0 and sha(checked(manifest.parent,part['path']))==part['sha256'],'Gate row shard changed')
    return g

def verify_permit(root, c, rawsha, job, stage, permit_path, permit_sha):
    condition, split = stage_identity(stage)
    need(isinstance(permit_path,str) and isinstance(permit_sha,str), 'Explicit operational permit required')
    path = checked(root,permit_path); need(sha(path) == permit_sha,'Permit bytes changed')
    p = read(path); now = time.time(); ids={j['id'] for j in c['jobs']}
    need(p.get('schema') == PERMIT_SCHEMA and p.get('authorized') is True and p.get('stage') == stage
         and p.get('host') == '5090' and p.get('contract_sha256') == c['contract_sha256']
         and p.get('contract_raw_sha256') == rawsha and p.get('evaluator_sha256') == sha(__file__)
         and p.get('new_training') is False and p.get('checkpoint_reselection') is False
         and p.get('automatic_retry') is False and set(p.get('job_ids',[])) == ids and len(p['job_ids']) == 3
         and job['id'] in ids, 'Unapproved diagnostic permit')
    need(type(p.get('not_before_unix')) in (float,int) and type(p.get('deadline_unix')) in (float,int)
         and math.isfinite(p['not_before_unix']) and math.isfinite(p['deadline_unix'])
         and p['not_before_unix'] <= now < p['deadline_unix'] <= c['deadline_unix'], 'Permit expired or extends deadline')
    if stage == 'on_validation': verify_gate(root,c,'qualification',p.get('qualification_gate_sha256'))
    elif condition == 'off':
        verify_gate(root,c,'on_validation',p.get('on_validation_gate_sha256'))
        verify_stored_on_train_gate(root,c,p.get('stored_on_train_gate_sha256'))
    return p


def observed_features(quantities, mask, gaps, np):
    """Use observed positions only; withhold the final target before all transforms."""
    q=quantities.detach().cpu().double().numpy(); d=gaps.detach().cpu().double().numpy()
    m=mask.detach().cpu().numpy().astype(bool).copy(); need(q.ndim==2 and q.shape==d.shape==m.shape and q.shape[1]>=2,'Invalid history shape')
    m[:,-1]=False; length=m.sum(1); need((length>=1).all(),'Empty observed context')
    need(np.isfinite(q[m]).all() and (q[m]>=0).all() and np.isfinite(d[m]).all(),'Invalid observed history')
    clean=np.where(m,q,0.); logq=np.log1p(clean); mean=logq.sum(1)/length
    var=np.where(m,(logq-mean[:,None])**2,0.).sum(1)/length
    last=np.asarray([q[i,np.flatnonzero(m[i])[-1]] for i in range(len(q))])
    changes=np.asarray([logq[i,np.flatnonzero(m[i])[-1]]-logq[i,np.flatnonzero(m[i])[-2]] if length[i]>1 else 0. for i in range(len(q))])
    identities=[hashlib.sha256(b'observed_loader_float32_order_v1\0'+q[i,m[i]].astype('<f8').tobytes()+d[i,m[i]].astype('<f8').tobytes()).hexdigest() for i in range(len(q))]
    return dict(history_length=length,last_quantity=last,mean_quantity=clean.sum(1)/length,
                history_log_mean=mean,history_log_std=np.sqrt(var),last_log_deviation=np.log1p(last)-mean,
                last_observed_log_change=changes,previous_observation_available=length>1),identities


def load_observed_frame(path, split, pl):
    need(split in COUNTS,'Test or another split is not authorized')
    allowed=['train'] if split=='train' else ['train','validation']
    # Reject even an accidentally misnamed mixed export before materializing its rows.
    labels=pl.scan_parquet(path).select(pl.col('chronological_split').unique()).collect()['chronological_split'].to_list()
    need(set(labels)<=set(COUNTS),'Input contains an unauthorized split')
    frame=pl.scan_parquet(path).filter(pl.col('chronological_split').is_in(allowed)).collect()
    need(frame.height>0 and set(frame['chronological_split'].unique())<=set(allowed),'Unexpected materialized split')
    return frame.sort(['oper_part_no','seq'])


def intervention_fields(condition):
    need(condition in ('on','off','synthetic_both'),'Unknown intervention condition')
    return {'intervention':INTERVENTION,'original_kernel_unchanged':True,'original_availability_unchanged':True,
            'available_tap_gain_preserved':True,'past_content_replaced':condition=='off',
            'original_current_parameter_unchanged':True,'effective_current_coefficient_preserved':condition=='on',
            'original_full_function_checks_preserved':True}



def validate_intervention_receipt(receipt):
    need(all(receipt.get(k)==v and (type(receipt.get(k)) is bool if type(v) is bool else True)
             for k,v in intervention_fields(receipt.get('condition')).items()),
         'Current-tap/past-tap intervention proof differs')


@contextmanager
def branch_scope(branch,scale,torch):
    """Replace only available past contents, preserving kernel, availability and lag order."""
    need(type(scale) in (int,float) and scale in (0,1),'Only numeric zero or one content multipliers are allowed')
    original=branch.causal_depthwise_residual;base=importlib.import_module('models.TPPs.CountAwareTitanCNNGRU')
    base_original=base.causal_depthwise_residual
    need(not getattr(original,'_key_content_probe_adapter',False),'Nested branch intervention forbidden')
    need(original is base_original,'Candidate original residual binding differs')
    audit={'condition':'on' if scale==1 else 'off','past_content_multiplier':scale,'calls':0,
           'original_guards_preserved':True,'restored':False,**intervention_fields('on' if scale else 'off')}
    def adapter(values,kernel,sources,available):
        # The unaltered computation must run first: invalid past values/indices must not be hidden.
        residual=original(values,kernel,sources,available)
        need(bool(torch.isfinite(residual).all()),'Nonfinite original residual must not be hidden by intervention')
        audit['calls']+=1
        if scale==1:return residual
        temporary=sources.clone()
        for lag in (1,2):
            temporary[...,lag]=torch.where(available[...,lag],sources[...,0],sources[...,lag])
        substituted=original(values,kernel,temporary,available)
        need(bool(torch.isfinite(substituted).all()),'Nonfinite content-substituted residual')
        return substituted
    adapter._key_content_probe_adapter=True
    try:
        branch.causal_depthwise_residual=adapter
        yield audit
    finally:
        unchanged=branch.causal_depthwise_residual is adapter
        branch.causal_depthwise_residual=original;audit['restored']=branch.causal_depthwise_residual is original
        need(unchanged and audit['restored'] and base.causal_depthwise_residual is base_original,
             'Branch function identity changed inside intervention')


def branch_module():
    return importlib.import_module('models.TPPs.CountAwareTitanQKLocationCNN')


def prepare_model(root,c,job,output):
    """Load only pinned source and selected checkpoint; no research rows."""
    helper=module(root/'operation/helper.py','_key_content_probe_frozen_helper')
    utils=module(root/'operation/row_utils.py','_key_content_probe_rows_utils')
    source=c['source'];source_root=helper.verify_source(root,source)
    checkpoint=checked(root,job['checkpoint']['relative_path']);metadata=checked(root,job['checkpoint_metadata_path'])
    history=helper.read_selection_history(checked(root,job['selection_history_path']))
    need(helper.strict_selected_epoch(history)==(job['selected_epoch'],job['selected_metric_value']),'First strict finite selected checkpoint differs')
    helper.isolate_source(source_root);sys.dont_write_bytecode=True
    os.environ['MPLCONFIGDIR']=str(output/'cache/matplotlib');os.environ['XDG_CACHE_HOME']=str(output/'cache')
    import numpy as np
    import polars as pl
    import torch
    numerical=importlib.import_module('paper.scripts.quantity_comparison_runtime')
    expected=c['runtimes']['5090'];numerical.configure_runtime('cuda:0',expected['torch_num_threads'])
    runtime=helper.record_executable(numerical.runtime_identity('cuda:0'));helper.check_runtime(runtime,expected,'cuda')
    engine=importlib.import_module(helper.ROUTE);engine.install_hooks()
    factory=importlib.import_module('models.TPPs.CountAwareFactory')
    core=importlib.import_module('paper.scripts.count_aware_tpp_backbone.core')
    loaders=importlib.import_module('paper.scripts.run_taxi_quantity_interface_ablation')
    runner=importlib.import_module('simple_lab_test.search.common.runner')
    definition=c['datasets']['raf_spare_parts']['definition'];stats=definition['statistics']
    payload=runner.torch_load_checkpoint(checkpoint,map_location='cpu')
    helper.validate_payload(payload,job,source,definition,read(metadata));factory.validate_checkpoint_route(payload,ARM)
    state=job['checkpoint_state_sha256'];need(runner.canonical_state_dict_sha256(payload['model_state_dict'])==state,'Selected tensor SHA differs')
    config={k:v for k,v in definition['model'].items() if k not in ('backbone','lambda_log_qty','lambda_tail','time_head_lr_multiplier')}
    model,_=factory.build_count_aware_model(ARM,**config,train_log_mean=stats['train_log_mean'],train_log_std=stats['train_log_std'],max_seq_len=84)
    model.load_state_dict(payload['model_state_dict'],strict=True)
    counts={'registered':sum(p.numel() for p in model.parameters()),'active_before_freeze':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    need(counts=={'registered':103775,'active_before_freeze':79043},'Current candidate parameter count differs')
    structure=helper.verify_qk_local_structure(model)
    need(getattr(model,'qk_location_arm',None)==ARM and structure['retained_projection']=='k'
         and structure['active_query_cnn_parameters']==0 and structure['active_key_cnn_parameters']==192,
         'Only the current Key-only candidate can be probed')
    model.to('cuda:0').eval().requires_grad_(False)
    need(runner.canonical_state_dict_sha256(model.state_dict())==state,'Loaded selected state differs')
    helper.verify_imports(source_root,source)
    return dict(helper=helper,utils=utils,source_root=source_root,np=np,pl=pl,torch=torch,numerical=numerical,runtime=runtime,
                model=model,core=core,loaders=loaders,runner=runner,definition=definition,counts=counts,state=state,
                predictive_state=predictive_state(model,runner))


def synthetic_residual_checks(ctx):
    """Device-native current/previous/two-previous taps and all inherited boundaries."""
    torch=ctx['torch'];branch=branch_module();device=next(ctx['model'].parameters()).device
    need(getattr(ctx['model'],'qk_location_arm',None)==ARM
         and ctx['model'].encoder.layers[0].attn.retained_projection=='k', 'Only Key-only synthetic candidate permitted')
    base=importlib.import_module('models.TPPs.CountAwareTitanCNNGRU')
    values=torch.tensor([[[2.,5.],[99.,99.],[4.,3.],[7.,8.],[99.,99.],[6.,2.],[9.,5.]]],device=device)
    valid=torch.tensor([[True,False,True,True,True,True,True]],device=device)
    writable=valid.clone();writable[:,4]=False
    _,sources,available=base.observed_context(valid,memory_write_mask=writable)
    kernel=torch.tensor([[.3,-.2],[.1,.4],[-.2,.5]],device=device)
    original=branch.causal_depthwise_residual;base_original=base.causal_depthwise_residual
    expected=original(values,kernel,sources,available)
    need(torch.equal(expected[:,[1,4]],torch.zeros_like(expected[:,[1,4]])),'Padding/withheld residual differs')
    need(torch.equal(expected[:,0],values[:,0]*kernel[0]) and torch.equal(expected[:,5],values[:,5]*kernel[0]),'First/reset observed tap differs')
    need(torch.equal(expected[:,2],values[:,2]*kernel[0]+values[:,0]*kernel[1]),'Two-observation residual differs')
    need(torch.equal(expected[:,3],values[:,3]*kernel[0]+values[:,2]*kernel[1]+values[:,0]*kernel[2]),'Three-observation residual differs')
    temporary=sources.clone()
    for lag in (1,2):temporary[...,lag]=torch.where(available[...,lag],sources[...,0],sources[...,lag])
    expected_off=original(values,kernel,temporary,available)
    closed=sum(values*kernel[lag]*available[...,lag,None] for lag in range(3))
    need(torch.equal(expected_off,closed),'Available-tap gain was not preserved')
    constant=torch.full_like(values,2.)
    need(torch.equal(original(constant,kernel,sources,available),original(constant,kernel,temporary,available)),'Latent constant response changed')
    need(torch.equal(expected[:,[0,5]],expected_off[:,[0,5]]),'One-observation history masking must be identity')
    kernel_before=kernel.clone();sources_before=sources.clone();available_before=available.clone();values_before=values.clone()
    for scale in (1,0):
        with branch_scope(branch,scale,torch) as audit:
            need(torch.equal(branch.causal_depthwise_residual(values,kernel,sources,available),expected if scale else expected_off), 'Binary residual differs')
            sanitized=values.clone();sanitized[:,[1,4]]=float('nan')
            need(torch.isfinite(branch.causal_depthwise_residual(sanitized,kernel,sources,available)).all(),'Padded nonfinite sanitization differs')
            invalid=values.clone();invalid[:,0]=float('nan');bad_kernel=kernel.clone();bad_kernel[1,0]=float('inf')
            bad_second=kernel.clone();bad_second[2,0]=float('nan')
            errors=[(invalid,kernel,sources,available),(values,kernel[:1],sources,available),
                    (values,kernel,sources.float(),available),(values,kernel,sources,available.float()),(values,bad_kernel,sources,available),(values,bad_second,sources,available)]
            for args in errors:
                rejected=False
                try:branch.causal_depthwise_residual(*args)
                except (ValueError,TypeError):rejected=True
                need(rejected,'Original residual shape/mask/nonfinite guard bypassed')
        need(audit['restored'] and branch.causal_depthwise_residual is original and base.causal_depthwise_residual is base_original,'Native residual function not restored')
    need(torch.equal(kernel,kernel_before) and torch.equal(sources,sources_before) and torch.equal(available,available_before)
         and torch.equal(values,values_before),'Intervention mutated original tensor arguments')
    require_fixed_state(ctx)
    return {'on_exact_original':True,'off_available_past_contents_replaced':True,'available_tap_gain_preserved':True,'constant_response_preserved':True,'one_observation_on_off_identity':True,'original_kernel_unchanged':True,'one_two_three_observations':True,
            'padding_skip_withheld_reset':True,'original_guards_preserved':True,'base_cnn_function_unchanged':True,'branch_function_restored':True}

def synthetic_checks(ctx, condition='both'):
    """Fabricated inputs only; qualification exercises both adapters without state mutation."""
    need(condition in ('on', 'off', 'both'), 'Unknown synthetic condition')
    torch,model,core=ctx['torch'],ctx['model'],ctx['core']; branch=branch_module()
    before=require_fixed_state(ctx);device=next(model.parameters()).device
    q=torch.tensor([[0.,0.,1.,3.,2.,8.,5.,7.],[1.,0.,2.,2.,6.,4.,3.,1.]],device=device)
    d=torch.tensor([[0.,0.,1.,2.,1.,3.,2.,4.],[0.,1.,2.,1.,3.,1.,2.,3.]],device=device)
    mask=torch.ones_like(q,dtype=torch.bool);mask[0,:2]=False
    boundary=synthetic_residual_checks(ctx)
    checks={}; original=branch.causal_depthwise_residual
    for label in ('on','off') if condition=='both' else (condition,):
        with branch_scope(branch,1 if label=='on' else 0,torch) as audit, torch.inference_mode():
            value=core.target_outputs(model,d,mask,q,lambda_log_qty=1.)
            checks[label]=ctx['helper'].qualify(torch,model,core.target_outputs,d,mask,q,value,ctx['definition']['model']['time_observation_contract']['top_code'])
            need(all(bool(torch.isfinite(value[k]).all()) for k in ('pred_qty','time_loss')),'Nonfinite synthetic output')
            if label=='off': need(audit['calls']>0,'Synthetic off branch never used')
        need(audit['restored'] and branch.causal_depthwise_residual is original,'Synthetic branch function not restored')
        need(require_fixed_state(ctx)==before,'Synthetic qualification mutated predictive state')
    return {'conditions': checks,'residual_boundary_checks':boundary,'parameters_unchanged':True,'state_including_buffers_unchanged':True,
            'finite_outputs':True,'original_guards_preserved':True,'branch_function_restored':True}


def final_integrity(root,c,rawsha,job,stage,p,permit_path,permit_sha,ctx):
    require_fixed_state(ctx)
    need(verify_contract(root,rawsha)==c,'Contract/files changed during diagnostic')
    need(verify_permit(root,c,rawsha,job,stage,permit_path,permit_sha)==p,'Operational permit changed')
    imports=ctx['helper'].verify_imports(ctx['source_root'],c['source'])
    after=ctx['helper'].record_executable(ctx['numerical'].runtime_identity('cuda:0'))
    ctx['helper'].check_runtime(after,c['runtimes']['5090'],'cuda')
    need(after==ctx['runtime'],'Native Runtime changed during diagnostic')
    return imports,after


def worker(root,contract_sha,job_id,stage,*,permit_path=None,permit_sha=None):
    condition, split = stage_identity(stage)
    mode = 'qualify' if stage == 'qualification' else 'worker'
    root=Path(root).resolve();need(root==Path(REMOTE_ROOT).resolve(),'Worker requires the approved native diagnostic root')
    c=verify_contract(root,contract_sha)
    matching=[j for j in c['jobs'] if j['id']==job_id];need(len(matching)==1,'Unknown current candidate')
    job=matching[0];p=verify_permit(root,c,contract_sha,job,stage,permit_path,permit_sha)
    output=root/'results'/job_id;name=stage+'_metrics'
    need(not any((output/(name+suffix)).exists() for suffix in ('.json','_claim.json','_failure.json')),'Existing attempt prohibits automatic retry')
    started=time.time();write(output/(name+'_claim.json'),{'job_id':job_id,'stage':stage,'split':split,'condition':condition,'pid':os.getpid(),'ppid':os.getppid(),'argv':sys.argv,'contract_sha256':c['contract_sha256'],'started_unix':started})
    try:
        ctx=prepare_model(root,c,job,output)
        parent_state=validate_stored_reference(root,c,job,'validation')
        need(ctx['predictive_state']==parent_state['predictive_state'] and ctx['runtime']==parent_state['receipt']['runtime'],
             'Current fresh state/Runtime differs from stored on reference')
        qualification=synthetic_checks(ctx, 'both' if mode=='qualify' else condition)
        common={'schema':SCHEMA,'status':'complete','job_id':job_id,'stage':stage,'split':split,'condition':condition,'host':'5090','contract_sha256':c['contract_sha256'],
            'contract_raw_sha256':contract_sha,'checkpoint_sha256':job['checkpoint']['sha256'],'state_sha256':ctx['state'],
            'state_sha256_before':ctx['state'],'state_sha256_after':ctx['state'],
            'predictive_state_before':ctx['predictive_state'],'predictive_state_after':ctx['predictive_state'],
            'state_including_buffers_unchanged':True,
            'selected_epoch':job['selected_epoch'],'source_closure':c['source']['closure'],'source_revision':c['source']['revision'],
            'permit_sha256':permit_sha,'parameters_unchanged':True,'no_training':True,'held_out_test_evaluated':False,
            'historical_CNN_inference':False,'historical_QueryOnly_inference':False,**intervention_fields(condition),'native_runtime_checked':True,'branch_function_restored':True,'qualification':qualification,'physical_parameter_counts':ctx['counts'],
            'runtime':ctx['runtime'],'pid':os.getpid(),'ppid':os.getppid(),'argv':sys.argv,'started_unix':started}
        if mode=='qualify':
            imports,after=final_integrity(root,c,contract_sha,job,stage,p,permit_path,permit_sha,ctx)
            result={**common,'qualification_passed':True,'real_data_loaded':False,'runtime_after':after,'imported_frozen_modules':imports,'finished_unix':time.time()}
            write(output/(name+'.json'),result);return result
        np,pl,torch=ctx['np'],ctx['pl'],ctx['torch'];helper,utils=ctx['helper'],ctx['utils'];definition=ctx['definition'];model=ctx['model']
        data=c['datasets']['raf_spare_parts'];frame=load_observed_frame(checked(root,data['path']),split,pl)
        helper.validate_frame(frame,definition,'validation',np,pl)
        train_q=frame.filter(pl.col('chronological_split')=='train')['demand_qty'].to_numpy().astype(np.float64);stats=definition['statistics']
        need(hashlib.sha256(b'quantity_comparison_all_train_row_quantity_v1\0'+train_q.astype('<f8').tobytes()).hexdigest()==stats['all_train_quantity_sha256'],'Frozen Train quantity identity differs')
        for key,v in [('train_log_mean',np.log1p(train_q).mean()),('train_log_std',np.log1p(train_q).std())]:
            need(math.isclose(float(v),stats[key],rel_tol=0,abs_tol=1e-12),'Train-only statistics differ')
        frame=ctx['core'].prepare_count_frame(frame)
        loader=ctx['loaders'].make_loader(frame,target_split=split,batch_size=128,shuffle=False,generator=None,lookback_weeks=84,max_seq_len=84)
        dataset=loader.dataset;population=helper.population_identity(dataset,split,np)
        need(population==definition['inherited_data_identity']['populations'][split] and len(dataset)==COUNTS[split],'Frozen target population differs')
        metadata=frame.select(['oper_part_no','seq','chronological_split','demand_dt']);offsets=[];total=0
        for seq in dataset.seq_lists:offsets.append(total);total+=len(seq)
        need(total==frame.height,'Loader frame mapping differs');del frame
        metrics=utils.RawMetrics(definition['quantity_boundaries_all_train_rows'],np)
        rowpath=root/'rows'/job_id/stage;writer=utils.PredictionWriter(rowpath,pl)
        target_digest,truth_digest=hashlib.sha256(),hashlib.sha256()
        time_digest=hashlib.sha256(b'titantpp_components_recorded_target_time_v1\0'+split.encode()+b'\0')
        processed=0;inference_seconds=0.;seen=set()
        branch=branch_module(); original=branch.causal_depthwise_residual
        with branch_scope(branch,1 if condition=='on' else 0,torch) as intervention, torch.inference_mode():
            for batch_index,(_,dts,mask,parts,qty) in enumerate(loader):
                need(time.time()<p['deadline_unix'],'Fixed diagnostic deadline expired')
                dts,mask,qty=(x.to('cuda:0') for x in (dts,mask,qty));features,hist_ids=observed_features(qty,mask,dts,np)
                torch.cuda.synchronize();tick=time.monotonic();v=ctx['core'].target_outputs(model,dts,mask,qty,lambda_log_qty=1.);torch.cuda.synchronize();inference_seconds+=time.monotonic()-tick
                if batch_index==0:qualification=helper.qualify(torch,model,ctx['core'].target_outputs,dts,mask,qty,v,definition['model']['time_observation_contract']['top_code'])
                q,pred,t=(v[k].cpu().numpy().astype(np.float64) for k in ('true_qty','pred_qty','time_loss'))
                need(np.array_equal(v['history_length'].cpu().numpy(),features['history_length']),'Observed history count differs')
                metrics.add(q,pred,t);gaps=dts[:,-1].cpu().numpy().astype(np.float64);time_digest.update(gaps.astype('<f8').tobytes());rows=[]
                for i in range(len(q)):
                    part,end=dataset.index[processed+i];raw=metadata.row(offsets[part]+end+1,named=True)
                    entity,seq=str(dataset.parts[part]),int(dataset.seq_lists[part][end+1])
                    need(int(parts[i])==part and raw['chronological_split']==split and str(raw['oper_part_no'])==entity and int(raw['seq'])==seq and raw['demand_dt'] is not None,'Loader row mapping differs')
                    need(float(q[i])==float(dataset.val_lists[part][end+1]),'Target raw quantity differs')
                    key=(split,entity,seq);need(key not in seen,'Duplicate diagnostic target');seen.add(key)
                    date=str(raw['demand_dt']);target_id=canonical([job['dataset'],data['sha256'],entity,seq,split,date])
                    row={'condition':condition,'stage':stage,'target_id':target_id,'entity_id':entity,'seq':seq,'split':split,'recorded_date':date,
                         'dataset':job['dataset'],'seed':job['seed'],'raw_quantity':float(q[i]),'recorded_gap':float(gaps[i]),
                         **{k:x[i].item() for k,x in features.items()},'prediction':float(pred[i]),'time_nll':float(t[i]),
                         'signed_error':float(pred[i]-q[i]),'absolute_error':float(abs(pred[i]-q[i])),'squared_error':float((pred[i]-q[i])**2),
                         'observed_history_sha256':hist_ids[i]}
                    target_digest.update((target_id+'\n').encode());truth_digest.update(json_bytes([row[k] for k in TRUTH_FIELDS])+b'\n');rows.append(row)
                writer.add(rows);processed+=len(q)
                if batch_index%50==0:print(json.dumps({'job_id':job_id,'split':split,'processed':processed,'expected':COUNTS[split]}),flush=True)
        need(intervention['restored'] and branch.causal_depthwise_residual is original, 'Branch function not restored after data')
        if condition=='off': need(intervention['calls']>0, 'Off intervention was never invoked')
        need(processed==COUNTS[split]==len(seen),'Incomplete target coverage')
        result_metrics={'split':split,**metrics.finish()}
        if stage=='on_validation': validate_six_metrics(result_metrics,job['validation_reference'])
        manifest=writer.finish();need(manifest['prediction_rows']==processed,'Shard count differs')
        write(rowpath/'manifest.json',{**manifest,'schema':'current_candidate_branch_probe_shards_v1','job_id':job_id,'split':split,'stage':stage,'condition':condition,
            'contract_sha256':c['contract_sha256'],'input_sha256':data['sha256'],'checkpoint_sha256':job['checkpoint']['sha256'],
            'source_closure':c['source']['closure'],'runtime':ctx['runtime'],
            'target_identity_domain':'sha256 canonical [dataset,pureTrainValidationInputSHA,entity,seq,split,date]',
            'history_identity_domain':'observed_loader_float32_order_v1; observed raw quantities then gaps as float64 bytes',
            'comparison_scope':'off minus on of same KeyOnly checkpoint; off replaces available past content with current K while preserving available gain; never original CNN'})
        paired=None;paired_reference=None;rowwise_replay=None
        if stage=='on_validation':
            parent=validate_stored_reference(root,c,job,'validation');old=parent['receipt'];ref=parent['reference']
            need(target_digest.hexdigest()==old['target_identity_sha256'] and truth_digest.hexdigest()==old['truth_sha256']
                 and time_digest.hexdigest()==old['recorded_gap_sha256'],'Replayed on target/truth/gap differs')
            validate_six_metrics(result_metrics,old['metrics'])
            rowwise_replay=validate_rowwise(read_manifest_rows(root,str((rowpath/'manifest.json').relative_to(root)),pl),
                                           read_manifest_rows(root,ref['manifest_path'],pl),COUNTS['validation'])
        if condition=='off':
            reference_receipt,_,reference_pin=verify_reference(root,c,job,split,p['stored_on_train_gate_sha256' if split=='train' else 'on_validation_gate_sha256'])
            need(target_digest.hexdigest()==reference_receipt['target_identity_sha256'] and truth_digest.hexdigest()==reference_receipt['truth_sha256']
                 and time_digest.hexdigest()==reference_receipt['recorded_gap_sha256'],'On/off target/truth/gap identity differs')
            paired=validate_paired_rows(read_manifest_rows(root,str((rowpath/'manifest.json').relative_to(root)),pl),
                                       read_manifest_rows(root,reference_pin['manifest_path'],pl),COUNTS[split])
            paired_reference={'receipt_path':reference_pin['path'],'receipt_sha256':reference_pin['sha256'],
                              'manifest_path':reference_pin['manifest_path'],'manifest_sha256':reference_receipt['manifest_sha256']}
        imports,after=final_integrity(root,c,contract_sha,job,stage,p,permit_path,permit_sha,ctx)
        result={**common,'real_data_loaded':True,'population':population,'metrics':result_metrics,'prediction_rows':processed,
            'original_population_identity':population,'prediction_manifest':manifest,'manifest_path':str((rowpath/'manifest.json').relative_to(root)),'manifest_sha256':sha(rowpath/'manifest.json'),
            'target_identity_sha256':target_digest.hexdigest(),'truth_sha256':truth_digest.hexdigest(),'recorded_gap_sha256':time_digest.hexdigest(),
            'target_identity_domain':'current diagnostics pureTrainValidationInputSHA; not assumed equal to prior evaluation full-input identity',
            'data_sha256':data['sha256'],'qualification':{**qualification,'parameters_unchanged':True,'state_including_buffers_unchanged':True,'finite_outputs':True},
            'rowwise_replay':rowwise_replay,'stored_prediction_rows_available':True,'original_population_match':True,
            'paired_on_reference':paired_reference,'paired_target_check':paired,'paired_targets_match':paired is not None,'branch_intervention':intervention,
            'validation_match':'PASS' if stage=='on_validation' else None,'validation_replay_check':{'six_metrics':stage=='on_validation','relative_tolerance':1e-5,'absolute_tolerance':1e-5},
            'runtime_after':after,'imported_frozen_modules':imports,'finished_unix':time.time(),'inference_seconds':inference_seconds}
        write(output/(name+'.json'),result);return result
    except BaseException as error:
        write(output/(name+'_failure.json'),{'status':'failed','job_id':job_id,'split':stage,'contract_sha256':c['contract_sha256'],
            'stage':stage,'condition':condition,'error':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'automatic_retry':False,'finished_unix':time.time()})
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--contract-sha',required=True)
    p.add_argument('--job-id',required=True);p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--permit-path',required=True);p.add_argument('--permit-sha',required=True);a=p.parse_args()
    result=worker(a.root,a.contract_sha,a.job_id,a.stage,permit_path=a.permit_path,permit_sha=a.permit_sha)
    print(json.dumps({'status':result['status'],'job_id':a.job_id,'stage':a.stage,'split':result['split']}),flush=True)

if __name__=='__main__':main()
