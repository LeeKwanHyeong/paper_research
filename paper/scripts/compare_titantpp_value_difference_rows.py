"""Local saved-row comparison only; no neural imports, inference, Test or transport.

All current and historical byte pins are checked before reading row shards. The
prior reconstruction entry point is imported but its main function is never run.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[2]
NAME = 'titantpp_backbone_value_difference_rows_20261016_v1'
BUNDLE = PROJECT / 'search_artifacts' / NAME
REPORT = PROJECT / 'reports/titantpp_backbone_value_difference_raf3_20261016_v1'
PRIOR = REPORT / 'diagnostics_paired_regret_v1'
OUTPUT = REPORT / 'diagnostics_current_candidate_rows_v1'
CNN_BUNDLE = PROJECT / 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3'
BASE_RECEIPT_SHA = '0c96e3be22deeae3f8d8ab8dde87640ab546acfd7be98f11b553c68b187cc7b9'
PLAN_SHA = '0708d991be70c3792af95e6266005de2c9cea14d6851fc5aaa731351727ed8b3'
PRIOR_RESULT_SHA = '1e986148042e1aa80446352960bde17a764c7329cc7237521e470054a84ba78e'
RECONSTRUCTION_SHA = '7deb615a622c7abde1e98547917291f8e8cef99fb91f940036e25af29b217e22'
ANALYSIS_SHA = 'c689918ad4f64913cf1b15894a3e4e177667711675a59e9056e72bfa6400cb33'
BASE_CONTRACT_SHA = 'd51ab06365a7165e00cc18849b84caba4ee2dd8af6a36aee84a5c1d62e01fab3'
BASE_SOURCE = '15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53'
SOURCE = '4fce860e41ee716592abf168af8c3985859b74ea6d080357690124b19a88d361'
INPUT_SHA = 'd805cad0242078fedfbb87eed18bde9b73d37bfc9b93c6b75aac49087244efa2'
ARM = 'titantpp_cnn_value_difference_v1'
CNN_ARM = 'titantpp_cnn_gru54_cnn_only'
SEEDS = (42, 52, 62)
SPLITS = ('train', 'validation')
COUNTS = {'train': 25779, 'validation': 6690}
EPOCHS = {42: 9, 52: 5, 62: 22}
CNN_EPOCHS = {42: 9, 52: 5, 62: 8}
BOUNDS = (2., 30., 60., 200.)
COMMON = ('raw_quantity','history_length','last_quantity','mean_quantity','history_log_mean',
          'history_log_std','last_log_deviation','last_observed_log_change','previous_observation_available')
TRUTH_FIELDS = ('target_id','entity_id','seq','split','recorded_date','raw_quantity','recorded_gap','history_length','last_quantity','mean_quantity')


def need(ok, message):
    if not ok: raise ValueError(message)


def json_bytes(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def canonical(v): return hashlib.sha256(json_bytes(v)).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def safe(root, relative):
    root=Path(root).resolve();rel=Path(relative)
    need(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe artifact path')
    p=root/rel;need(p.resolve().is_relative_to(root) and not p.is_symlink(), 'Artifact escapes root')
    return p


def write(path, value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


class Pins:
    def __init__(self, project):self.project=Path(project).resolve();self.files={}
    def add(self, path, expected):
        path=Path(path).resolve();need(path.is_relative_to(self.project), 'Input outside project')
        actual=sha(path);need(actual==expected,'Input SHA differs: '+str(path))
        rel=str(path.relative_to(self.project));need(rel not in self.files or self.files[rel]==actual,'Conflicting byte pin')
        self.files[rel]=actual;return path
    def again(self):
        for rel,digest in self.files.items():need(sha(safe(self.project,rel))==digest,'Input changed during comparison: '+rel)


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def selected_epoch(history):
    history=history['history'] if isinstance(history,dict) else history
    best=None;last=0
    for row in history:
        epoch=row['epoch'];need(type(epoch) is int and epoch>last,'Selection history not strictly ordered');last=epoch
        value=row.get('val_qty_rmse')
        if type(value) in (float,int) and math.isfinite(value) and (best is None or value<best[1]):best=(epoch,float(value))
    need(best is not None,'No finite selected checkpoint');return best


def verify_selection(bundle, job, pins, source, expected_epoch):
    binary=job['checkpoint'];pins.add(safe(bundle,binary['relative_path']),binary['sha256'])
    history=read(pins.add(safe(bundle,job['selection_history_path']),job['selection_history_sha256']))
    meta=read(pins.add(safe(bundle,job['checkpoint_metadata_path']),job['checkpoint_metadata_sha256']))
    chosen=selected_epoch(history)
    need(chosen[0]==expected_epoch==job['selected_epoch']==meta['best_epoch'],'Frozen earliest selected epoch differs')
    need(meta['model_state_sha256']==job['checkpoint_state_sha256'],'Selected tensor metadata differs')
    need(meta['resume_identity']['interface_meta']['source_files_sha256']==source
         and meta['resume_identity']['arguments']['data_sha256']==INPUT_SHA,'Selected source/input differs')
    if 'selected_metric_value' in job:need(chosen[1]==job['selected_metric_value'],'Frozen selected metric differs')
    return {'epoch':chosen[0],'checkpoint_sha256':binary['sha256'],'state_sha256':job['checkpoint_state_sha256']}


def verify_source_subset(candidate, baseline):
    need(len(baseline['files'])==131 and canonical(baseline['files'])==baseline['closure']==BASE_SOURCE,'CNN source131 differs')
    need(len(candidate['files'])==141 and canonical(candidate['files'])==candidate['closure']==SOURCE,'Current source141 differs')
    need(all(candidate['files'].get(name)==value for name,value in baseline['files'].items()),'Inherited CNN source bytes differ')
    return 131


def verify_final_scope(r):
    expected={'status':'complete','candidate_count':3,'seed_count':3,'data_worker_count':6,'qualification_count':3,
              'actual_native_worker_exit0_count':9,'actual_data_worker_exit0_count':6,'actual_pipeline_exit0_count':1,
              'actual_supervisor_child_exit0_count':1,'validation_replay_count':3,'all_owned_absent':True,
              'new_training':False,'historical_CNN_inference':False,'held_out_test_evaluated':False,'automatic_retry':False}
    need(all(r.get(k)==v and type(r.get(k)) is type(v) for k,v in expected.items()),'Current native completion proof incomplete')
    need(r.get('stages')==['qualification','validation','train'] and r.get('source_closure')==SOURCE
         and r.get('input_sha256')==INPUT_SHA,'Current native scientific scope differs')


def verify_predictive_receipt(r, job, split, c):
    need(split in SPLITS and r.get('split')==split and r.get('job_id')==job['id'] and r.get('status')=='complete', 'Wrong data receipt')
    need(r.get('parameters_unchanged') is True and r.get('state_including_buffers_unchanged') is True
         and r.get('no_training') is True and r.get('held_out_test_evaluated') is False
         and r.get('native_runtime_checked') is True and r.get('real_data_loaded') is True,'Frozen native state proof missing')
    state=job['checkpoint_state_sha256'];before=r['predictive_state_before'];after=r['predictive_state_after']
    need(before==after and set(before)=={'state_sha256','all_named_buffers_sha256','context_cache_sha256'}
         and before['state_sha256']==state and all(isinstance(v,str) and len(v)==64 for v in before.values())
         and r.get('state_sha256_before')==r.get('state_sha256_after')==r.get('state_sha256')==state,'Parameter/buffer/context state changed')
    need(r.get('checkpoint_sha256')==job['checkpoint']['sha256'] and r.get('selected_epoch')==job['selected_epoch']
         and r.get('source_closure')==c['source']['closure'] and r.get('source_revision')==c['source']['revision']
         and r.get('data_sha256')==INPUT_SHA and r.get('contract_sha256')==c['contract_sha256'],'Prediction science binding differs')
    need(r.get('population')==c['datasets']['raf_spare_parts']['populations'][split]
         and r.get('prediction_rows')==COUNTS[split],'Prediction target population differs')
    need(r['runtime']==r['runtime_after'] and r['host']=='5090','Native runtime changed')
    need(split!='validation' or r.get('validation_match')=='PASS','Native Validation replay failed')


def verify_candidate(receipt_sha, pins):
    final=read(pins.add(BUNDLE/'retrieval/final_receipt.json',receipt_sha));verify_final_scope(final)
    for rel,digest in final['files'].items():pins.add(safe(PROJECT,rel),digest)
    boot=read(pins.add(BUNDLE/'bootstrap_integrity.json',final['bootstrap_integrity_sha256']))
    need(boot['contract_raw_sha256']==final['contract_raw_sha256'],'Native contract raw SHA differs')
    for rel,digest in boot['files'].items():pins.add(safe(BUNDLE,rel),digest)
    for path,digest in boot['external_dependencies'].items():pins.add(path,digest)
    c=read(pins.add(BUNDLE/'contract.json',final['contract_raw_sha256']))
    need(c['contract_sha256']==final['contract_sha256']==canonical({k:v for k,v in c.items() if k!='contract_sha256'}),'Native contract canonical differs')
    need(c['allowed_splits']==list(SPLITS) and c['source']['closure']==SOURCE and len(c['jobs'])==3
         and {j['seed'] for j in c['jobs']}==set(SEEDS) and all(j['arm']==ARM and j['dataset']=='raf_spare_parts' for j in c['jobs']),'Native candidate cohort differs')
    originals=safe(PROJECT,final['originals_path']);need(originals==BUNDLE/'retrieval/originals','Foreign original archive root')
    pins.add(BUNDLE/'retrieval/originals.tar.gz',final['archive_sha256'])
    manifest=read(pins.add(originals/'ARCHIVE_MANIFEST.json',final['manifest_sha256']))
    for rel,item in manifest['files'].items():
        path=pins.add(safe(originals,rel),item['sha256']);need(path.stat().st_size==item['bytes'],'Archived original size differs')
    # Invoke only the existing collector's read-only evidence checker, never advance/deploy/transport.
    guard_path=PROJECT/'paper/scripts/advance_titantpp_value_difference_rows.py'
    pins.add(guard_path,boot['external_dependencies'][str(guard_path)])
    guard=load_module(guard_path,'_value_rows_completed_collector');guard.BUNDLE=BUNDLE;guard.PROJECT=PROJECT
    data_receipts,terminal=guard.check_receipts(originals,c,final['contract_raw_sha256'])
    need(data_receipts==final['data_receipts'] and terminal['all_owned_absent'] is True,'Completion proof changed')
    selections={}
    for j in c['jobs']:
        selections[j['id']]=verify_selection(BUNDLE,j,pins,SOURCE,EPOCHS[j['seed']])
        expected={'seed':j['seed'],'epoch':j['selected_epoch'],'checkpoint_sha256':j['checkpoint']['sha256'],
                  'tensor_sha256':j['checkpoint_state_sha256'],'metadata_sha256':j['checkpoint_metadata_sha256'],'history_sha256':j['selection_history_sha256']}
        need(final['selected_models'].get(j['id'])==expected,'Final receipt selected model differs')
    return c,final,originals,selections


def verify_baseline(candidate, pins):
    r=read(pins.add(PRIOR/'RECEIPT.json',BASE_RECEIPT_SHA))
    need(r['status']=='complete' and r['allowed_splits']==list(SPLITS) and len(r['inputs'])==175
         and r['source_files_verified']==131 and r['inference_count']==r['remote_count']==0 and r['held_out_test_read'] is False,'Prior CNN diagnostic differs')
    for rel,digest in r['inputs'].items():pins.add(safe(PROJECT,rel),digest)
    for rel,digest in r['output_files'].items():pins.add(safe(PRIOR,rel),digest)
    plan=read(pins.add(PRIOR/'PLAN.json',PLAN_SHA));previous=read(pins.add(PRIOR/'RESULT.json',PRIOR_RESULT_SHA))
    pins.add(PRIOR/'analyze.py',RECONSTRUCTION_SHA)
    need(plan['splits']==list(SPLITS) and plan['seeds']==list(SEEDS) and plan['expected_target_count']==COUNTS
         and plan['input_sha256']==INPUT_SHA and previous['plan_sha256']==PLAN_SHA,'Frozen comparison plan differs')
    cuts=(previous['cuts']['q25'],previous['cuts']['q75'])
    need(len(cuts)==2 and all(math.isfinite(v) for v in cuts) and cuts[0]<=cuts[1]
         and previous['cuts']['uses_target_quantity'] is False,'Frozen Train cuts invalid')
    base=read(pins.add(CNN_BUNDLE/'contract.json',BASE_CONTRACT_SHA));verify_source_subset(candidate['source'],base['source'])
    need(base['datasets']['raf_spare_parts']['sha256']==INPUT_SHA
         and base['datasets']['raf_spare_parts']['definition']['inherited_data_identity']['populations']==candidate['datasets']['raf_spare_parts']['populations'],
         'CNN/current pure input populations differ')
    return base,previous,cuts,plan


def aggregate_rows(rows, prediction_key, time_key):
    def cell(subset):
        need(bool(subset),'Empty non-tail target population')
        q=[float(r['raw_quantity']) for r in subset];p=[float(r[prediction_key]) for r in subset];t=[float(r[time_key]) for r in subset]
        need(all(math.isfinite(v) for v in q+p+t) and min(q+p)>=0,'Invalid saved prediction')
        errors=[a-b for a,b in zip(p,q)];n=len(errors);sse=math.fsum(e*e for e in errors);ae=math.fsum(abs(e) for e in errors)
        return {'count':n,'qty_rmse':math.sqrt(sse/n),'qty_mae':ae/n,'time_nll':math.fsum(t)/n,
                'qty_sse':sse,'qty_absolute_error_sum':ae,'qty_signed_error_sum':math.fsum(errors)}
    whole=cell(rows);tail=[r for r in rows if r['raw_quantity']>200.]
    whole['tail']=cell(tail) if tail else {'count':0,'qty_rmse':None,'qty_mae':None,'time_nll':None,'qty_sse':0.,'qty_absolute_error_sum':0.,'qty_signed_error_sum':0.}
    return whole


def check_metrics(actual, expected, *, rtol=1e-10, atol=1e-8):
    for label in (None,'tail'):
        a=actual if label is None else actual[label];e=expected if label is None else expected[label]
        need(type(a['count']) is int and a['count']==e['count'],'Stored metric target count differs')
        for metric in ('qty_rmse','qty_mae','time_nll'):
            if a['count']==0:need(a[metric] is None and e[metric] is None,'Empty metric must be unavailable')
            else:need(math.isfinite(a[metric]) and math.isfinite(e[metric]) and math.isclose(a[metric],e[metric],rel_tol=rtol,abs_tol=atol),'Stored six metrics differ: '+metric)


def read_shards(directory, manifest, pins, columns, pl):
    need(manifest['prediction_rows']>0 and manifest['parts'],'Empty prediction manifest')
    rows=[];seen=set()
    for part in manifest['parts']:
        need(part['path'] not in seen,'Duplicate shard');seen.add(part['path'])
        p=pins.add(safe(directory,part['path']),part['sha256']);need(p.stat().st_size==part['bytes'],'Shard size differs')
        frame=pl.read_parquet(p,columns=columns);need(frame.height==part['rows'],'Shard target count differs');rows.extend(frame.to_dicts())
    need(len(rows)==manifest['prediction_rows']==sum(p['rows'] for p in manifest['parts']),'Manifest total differs')
    return rows


def reconstruct_contexts(frame, histories, np):
    """Fingerprint the already reconstructed observed interval, preserving loader float32/clip semantics."""
    groups={str(g['oper_part_no'][0]):g for g in frame.partition_by('oper_part_no',maintain_order=True)}
    result={}
    for split in SPLITS:
        result[split]={}
        for h in histories[split]:
            g=groups[h['entity_id']];seq=g['seq'].to_numpy();index=int(np.searchsorted(seq,h['seq']))
            need(index<len(seq) and int(seq[index])==h['seq'],'Missing raw target')
            idx=np.flatnonzero((seq>=h['first_observed_seq']) & (seq<=h['last_observed_seq']))
            need(len(idx)==h['history_length'] and int(idx.max())<index,'Observed context overlaps target')
            q=g['demand_qty'].to_numpy().astype(np.float32).astype(np.float64)
            d=np.maximum(g['delta_t'].to_numpy().astype(np.int32),1).astype(np.float32).astype(np.float64)
            history_sha=hashlib.sha256(b'observed_loader_float32_order_v1\0'+q[idx].astype('<f8').tobytes()+d[idx].astype('<f8').tobytes()).hexdigest()
            result[split][h['entity_id'],h['seq']]={'recorded_date':str(g['demand_dt'][index]),'recorded_gap':float(d[index]),'history_sha256':history_sha}
    return result


def verify_candidate_row_identity(rows, receipt, contexts, job, split):
    need(split in SPLITS and len(rows)==COUNTS[split],'Wrong candidate row scope')
    target,truth=hashlib.sha256(),hashlib.sha256();times=hashlib.sha256(b'titantpp_components_recorded_target_time_v1\0'+split.encode()+b'\0');seen=set()
    for r in rows:
        need(r['dataset']=='raf_spare_parts' and r['seed']==job['seed'] and r['split']==split,'Mixed candidate rows')
        k=str(r['entity_id']),int(r['seq']);need(k not in seen and k in contexts,'Duplicate/foreign candidate target');seen.add(k)
        context=contexts[k]
        need(r['recorded_date']==context['recorded_date'] and r['recorded_gap']==context['recorded_gap']
             and r['observed_history_sha256']==context['history_sha256'],'Raw target date/gap/history differs')
        identity=canonical(['raf_spare_parts',INPUT_SHA,k[0],k[1],split,r['recorded_date']]);need(r['target_id']==identity,'Candidate target hash domain differs')
        target.update((identity+'\n').encode());truth.update(json_bytes([r[f] for f in TRUTH_FIELDS])+b'\n')
        import struct
        times.update(struct.pack('<d',float(r['recorded_gap'])))
    need(seen==set(contexts),'Candidate target coverage differs')
    need(target.hexdigest()==receipt['target_identity_sha256'] and truth.hexdigest()==receipt['truth_sha256']
         and times.hexdigest()==receipt['recorded_gap_sha256'],'Candidate row identity digest differs')
    return {'target_identity_sha256':target.hexdigest(),'truth_sha256':truth.hexdigest(),'recorded_gap_sha256':times.hexdigest(),
            'target_hash_domain':'current pure Train/Validation input SHA; not compared to absent CNN row target_id'}


def markdown(result):
    s=result['summary'];fmt=lambda x:'N/A' if x is None else f'{x:+.6f}'
    lines=['# RAF 현재 ValueDifference−기존 CNN 행별 오류 분석','',
      '회수한 현재 후보와 기존 CNN의 저장된 Train·Validation 예측을 같은 대상별로 비교했다. 새 추론·학습·원격 호출·Test 열람은 없다.',
      '양수 Δ는 현재 후보의 악화, 음수는 개선이다. 평균과 표본 SD는 세 seed의 요약이며 독립 대상을 세 배로 세지 않았다.','',
      f"고정 Train 변화 폭 경계: Q25={result['cuts'][0]:.12g}, Q75={result['cuts'][1]:.12g}. 기존 진단의 15개 조건과 공통 개체 가중을 그대로 적용했다.",'',
      '| split | seed | CNN RMSE | 후보 RMSE | ΔRMSE | CNN MAE | 후보 MAE | ΔMAE | ΔSSE | ΔAE | Δ편향합 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in s['per_seed']:
        if row['family']!='whole' or row['weighting']!='row':continue
        m=row['metrics'];lines.append('| '+' | '.join([row['split'],str(row['seed']),*[fmt(x) for x in
          (m['cnn']['rmse'],m['candidate']['rmse'],m['difference']['rmse'],m['cnn']['mae'],m['candidate']['mae'],m['difference']['mae'],m['difference']['se_sum'],m['difference']['ae_sum'],m['difference']['signed_sum'])]])+' |')
    lines+=['','## 세 seed 평균과 표본 SD','', '| split | 가중 | CNN RMSE | 후보 RMSE | ΔRMSE | CNN MAE | 후보 MAE | ΔMAE |','|---|---|---:|---:|---:|---:|---:|---:|']
    stat=lambda x:f"{x['mean']:.6f} ± {x['sample_sd']:.6f}"
    for row in s['three_seed']:
        if row['cell']!='whole|None|None':continue
        m=row['metrics'];lines.append('| '+' | '.join([row['split'],row['weighting'],*[stat(m[k][v]) for k,v in [('cnn','rmse'),('candidate','rmse'),('difference','rmse'),('cnn','mae'),('candidate','mae'),('difference','mae')]]])+' |')
    lines+=['','## 사전 고정 조건별 기여','', '| 조건 | split | 가중 | 평균 ΔRMSE | 평균 ΔMAE | 평균 ΔSSE | 평균 ΔAE | 평균 Δ편향합 |','|---|---|---|---:|---:|---:|---:|---:|']
    for row in s['three_seed']:
        m=row['metrics']['difference'];lines.append('| '+' | '.join([row['cell'].replace('|',' / '),row['split'],row['weighting'],*[fmt(m[k]['mean']) for k in ('rmse','mae','se_sum','ae_sum','signed_sum')]])+' |')
    lines+=['','모든 seed별 수량·시간 지표, 공통 개체 조건에서 남은 행/제외 행과 가산 합계 검증은 RESULT.json에 보존했다.',
      '기존 CNN 행에는 target_id·날짜·시간간격이 없다. 같은 순수 입력 SHA, split·개체·순번, 정답 수량·관측 이력을 결속했다. 후보의 별도 target/truth hash는 자체 재계산했으며 서로 다른 hash domain을 같은 것으로 취급하지 않았다.',
      '기존 CNN은 5080/cuDNN91900, 현재 후보는 5090/cuDNN92000이다. 같은 대상의 오차 차이는 확인할 수 있지만 Runtime·최적화·V 필터 자유도 192→128(전체 576→512)·경계/매개화 차이의 원인을 분리하지는 못한다.',
      '15개 조건의 공통 개체 집합은 조건마다 달라 가중 지표를 조건 간 더하지 않는다. 행 가중 ΔAE·ΔSSE·부호 있는 오차 합은 분할별 합계가 보존되는지 검증했다.',
      '기존 Test 결과를 참고한 탐색적 후속 진단이다. Test 개선·독립 일반화·통계 유의성·원인 확정을 주장하지 않는다.']
    return '\n'.join(lines)+'\n'


def run(receipt_sha, analysis_sha):
    need(not OUTPUT.exists(),'Existing analysis output: no overwrite or automatic retry')
    need(analysis_sha==ANALYSIS_SHA,'Unapproved pure analysis version')
    pins=Pins(PROJECT);pins.add(Path(__file__),sha(__file__))
    analysis_path=pins.add(PROJECT/'paper/scripts/analyze_titantpp_value_difference_rows.py',analysis_sha)
    c,final,originals,selections=verify_candidate(receipt_sha,pins)
    base,previous,cuts,plan=verify_baseline(c,pins)
    reconstruction=load_module(PRIOR/'analyze.py','_frozen_current_row_reconstruction')
    analysis=load_module(analysis_path,'_current_candidate_row_analysis')
    import numpy as np
    import polars as pl
    data=c['datasets']['raf_spare_parts'];path=pins.add(safe(BUNDLE,data['path']),INPUT_SHA)
    labels=pl.scan_parquet(path).select(pl.col('chronological_split').unique()).collect()['chronological_split'].to_list()
    need(set(labels)==set(SPLITS),'Input contains foreign split')
    raw=pl.read_parquet(path).sort(['oper_part_no','seq']);need(raw.height==37469,'Pure input row count differs')
    histories,audits={},{}
    for split in SPLITS:histories[split],audits[split]=reconstruction.build_rows(raw,split,data['populations'][split])
    eligible=[max(abs(r['d1']),abs(r['d2'])) for r in histories['train'] if r['history_length']>=3]
    calculated=tuple(float(v) for v in np.quantile(eligible,[.25,.75],method='linear'))
    need(calculated==cuts and len(eligible)==previous['cuts']['eligible_train_targets'],'Previously frozen Train cuts changed')
    contexts=reconstruct_contexts(raw,histories,np);del raw
    basejobs={j['seed']:j for j in base['jobs'] if j['dataset']=='raf_spare_parts'}
    need(set(basejobs)==set(SEEDS) and all(j['arm']==CNN_ARM for j in basejobs.values()),'Stored CNN seed cohort differs')
    baseline_host=read(CNN_BUNDLE/'retrieval/runtime/5080/receipt.json');paired={};evidence=[]
    cols=['dataset','seed','split','entity_id','seq',*COMMON]
    candidate_cols=cols+['prediction','time_nll','recorded_date','recorded_gap','target_id','observed_history_sha256','signed_error','absolute_error','squared_error']
    for j in sorted(c['jobs'],key=lambda j:j['seed']):
        seed=j['seed'];bj=basejobs[seed];original=bj['original'];baseline_selection=verify_selection(CNN_BUNDLE,original,pins,BASE_SOURCE,CNN_EPOCHS[seed])
        for split in SPLITS:
            rel=f"results/{j['id']}/{split}_metrics.json";entry=final['data_receipts'][j['id']+'/'+split]
            need(entry['path']==rel,'Unexpected candidate receipt path');receipt=read(pins.add(originals/rel,entry['sha256']))
            verify_predictive_receipt(receipt,j,split,c)
            manifest_path=pins.add(safe(originals,receipt['manifest_path']),receipt['manifest_sha256']);manifest=read(manifest_path)
            candidate_rows=read_shards(manifest_path.parent,manifest,pins,candidate_cols,pl)
            row_identity=verify_candidate_row_identity(candidate_rows,receipt,contexts[split],j,split)
            candidate_metrics=aggregate_rows(candidate_rows,'prediction','time_nll');check_metrics(candidate_metrics,receipt['metrics'])
            if split=='validation':check_metrics(candidate_metrics,j['validation_reference'],rtol=1e-5,atol=1e-5)
            bp=CNN_BUNDLE/f"retrieval/results/{bj['id']}/{split}/receipt.json";br=read(bp)
            worker=next(x for x in baseline_host['workers'] if x['job_id']==bj['id'] and x['split']==split)
            need(worker['actual_worker_exit_code']==0 and worker['receipt']['sha256']==sha(bp),'CNN actual worker exit binding differs')
            need(br['status']=='complete' and br['contract_sha256']==BASE_CONTRACT_SHA and br['split']==split and br['host']=='5080'
                 and br['parameters_unchanged'] is True and br['no_training'] is True and br['held_out_test_evaluated'] is False
                 and br['checkpoint_sha256']==original['checkpoint']['sha256'] and br['state_sha256']==original['checkpoint_state_sha256']
                 and br['selected_epoch']==CNN_EPOCHS[seed] and br['population']==receipt['population'],'CNN stored provenance differs')
            need(split!='validation' or br['baseline_validation_reproduced'] is True,'CNN original Validation replay failed')
            baseline_rows=read_shards(bp.parent/'predictions',br['prediction_manifest'],pins,cols+['prediction_1','time_nll_1'],pl)
            baseline_metrics=aggregate_rows(baseline_rows,'prediction_1','time_nll_1');check_metrics(baseline_metrics,br['metrics']['1.0'])
            # Old CNN time hash is bare float64 gap bytes; the candidate uses an explicit domain prefix.
            old_gaps=np.asarray([contexts[split][str(r['entity_id']),int(r['seq'])]['recorded_gap'] for r in baseline_rows],dtype='<f8')
            need(hashlib.sha256(old_gaps.tobytes()).hexdigest()==br['recorded_gap_sha256'],'CNN stored time-target identity differs')
            paired[seed,split]=analysis.paired_rows(candidate_rows,baseline_rows,histories[split],seed=seed,split=split,expected_count=COUNTS[split])
            evidence.append({'seed':seed,'split':split,'count':COUNTS[split],'candidate_receipt_sha256':entry['sha256'],'baseline_receipt_sha256':sha(bp),
                             'candidate_selection':selections[j['id']],'baseline_selection':baseline_selection,'row_identity':row_identity,
                             'candidate_metrics':candidate_metrics,'baseline_metrics':baseline_metrics,'candidate_runtime':receipt['runtime'],'baseline_runtime':br['runtime']})
    summary=analysis.summarize(paired,cuts)
    result={'schema':'current_value_difference_cnn_paired_rows_v1','created_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
            'allowed_splits':list(SPLITS),'candidate_model_count':3,'stored_baseline_model_count':3,'candidate_receipt_sha256':receipt_sha,
            'baseline_diagnostic_receipt_sha256':BASE_RECEIPT_SHA,'frozen_plan_sha256':PLAN_SHA,'analysis_module_sha256':analysis_sha,
            'cuts':list(cuts),'reconstruction':audits,'evidence':evidence,'summary':summary,'inference_count':0,'remote_count':0,
            'held_out_test_read':False,'source131_inherited_identically_in_source141':True,'candidate_history_decomposition_available':True,
            'scope':'Descriptive paired error contributions; no causal claim or new method selection.'}
    pins.again();OUTPUT.mkdir(parents=True,exist_ok=False)
    try:
        write(OUTPUT/'RESULT.json',result)
        with (OUTPUT/'RESULT.md').open('x') as f:f.write(markdown(result))
        pins.again()
        receipt={'schema':'current_value_difference_paired_rows_receipt_v1','status':'complete','allowed_splits':list(SPLITS),
                 'candidate_model_count':3,'stored_baseline_model_count':3,'paired_split_count':6,'native_data_exit0_count_verified':6,
                 'inference_count':0,'remote_count':0,'held_out_test_read':False,'all_inputs_unchanged_after':True,
                 'inputs':pins.files,'output_files':{n:sha(OUTPUT/n) for n in ('RESULT.json','RESULT.md')},
                 'candidate_receipt_sha256':receipt_sha,'baseline_diagnostic_receipt_sha256':BASE_RECEIPT_SHA,
                 'analysis_module_sha256':analysis_sha,'additive_closure_count':len(summary['additive_closure'])}
        write(OUTPUT/'RECEIPT.json',receipt)
    except BaseException as error:
        write(OUTPUT/'FAILURE.json',{'error':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'automatic_retry':False});raise
    return {'status':'complete','output':str(OUTPUT),'receipt_sha256':sha(OUTPUT/'RECEIPT.json'),'artifacts':receipt['output_files']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt-sha',required=True);p.add_argument('--analysis-sha',required=True)
    a=p.parse_args();print(json.dumps(run(a.receipt_sha,a.analysis_sha),ensure_ascii=False))

if __name__=='__main__':main()
