#!/usr/bin/env python3
"""Retrospective train-only frozen representation extraction and fixed OOF probes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset, collate_week_lookback
from models.TPPs.CountAwareFactory import validate_checkpoint_route
from models.Titan.common.key_value_memory import KEY_VALUE_BACKBONE
from paper.scripts.analyze_count_aware_b0_retrieval import checkpoint_path, restore_b0
from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame, target_outputs
from paper.scripts.diagnose_hard_lmm_weighted_static import restore
from paper.scripts.run_hard_lmm_query_diagnostic import read, save, digest, check, fold_for_series, load_train_frame
from paper.scripts.hard_lmm_information_features import extract, HISTORY_FEATURE_NAMES
from paper.scripts import hard_lmm_information_analysis as analysis
from simple_lab_test.search.common.runner import canonical_state_dict_sha256

CONTRACT = ROOT / 'paper/contracts/hard_lmm_information_access_v1.json'
RAW = ROOT / 'search_artifacts/hard_lmm_information_access_20260905'
RESULT = ROOT / 'paper/results/hard_lmm_information_access_20260905'
MODELS = ('original', 'separate_key')
DATASETS = ('yellow_trip_hourly', 'insta_market_basket')


def now():
    return datetime.now(timezone.utc).isoformat()


def sources():
    names = subprocess.check_output(['git', 'ls-files', 'models', 'data_loader', 'utils', 'paper/scripts',
                                    'simple_lab_test/common', 'simple_lab_test/search/common'], cwd=ROOT, text=True).splitlines()
    names += [str(p.relative_to(ROOT)) for pattern in ('*hard_lmm_information*.py',)
              for folder in ('paper/scripts', 'simple_lab_test/search/tests') for p in (ROOT/folder).glob(pattern)]
    return {p: digest(ROOT/p) for p in sorted(set(names)) if (ROOT/p).is_file()}


def verify(hashes):
    for name, expected in hashes.items():
        check(digest(ROOT/name) == expected, f'Changed input/source: {name}')


def load_contract():
    c = read(CONTRACT)
    verify(c['frozen_documents'])
    check(c['datasets'] == list(DATASETS) and c['models'] == list(MODELS), 'Expanded scope')
    check(analysis.RIDGE == c['decoders']['linear']['ridge'] == 1., 'Ridge changed')
    check(analysis.RANDOM_WIDTH == c['decoders']['nonlinear']['features'] == 128, 'Decoder changed')
    check(analysis.FAMILY_QUANTILE == c['evidence_gate']['paired_series_bootstrap']['lower_quantile'], 'Gate changed')
    check(set(analysis.CONTRASTS) == set(c['primary_contrasts']), 'Comparison set changed')
    check(list(HISTORY_FEATURE_NAMES) == c['history_features']['names'], 'Feature order changed')
    check(analysis.BOOTSTRAP_REPEATS == c['evidence_gate']['paired_series_bootstrap']['repeats'] == 10000, 'Bootstrap changed')
    check(analysis.SEED == c['decoders']['nonlinear']['seed'], 'Probe seed changed')
    check(analysis.SHAM_SEED == c['controls']['sham_seed'], 'Control seed changed')
    check(analysis.MINIMUM_GAIN == c['evidence_gate']['relative_pooled_residual_mse_improvement_minimum'], 'Evidence threshold changed')
    for name, (candidate, *refs) in analysis.CONTRASTS.items():
        check(candidate == c['primary_contrasts'][name]['candidate'] and refs == c['primary_contrasts'][name]['references'], 'Comparison changed')
    return c


def extract_one(model, dataset, old, label):
    model.requires_grad_(False).eval()
    before = canonical_state_dict_sha256(model.state_dict())
    indices = old['target_index']
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=64, shuffle=False,
                        collate_fn=collate_week_lookback, num_workers=0)
    arrays, offset, started, parity = {}, 0, time.monotonic(), {}
    for number, (_, dts, mask, parts, qty) in enumerate(loader):
        batch = extract(model, dts, mask, qty)
        if 'base_logpred' not in batch:
            batch['base_logpred'] = batch.pop('base_log_prediction')
        n = len(parts)
        for key in ('h', 'quantity', 'history_length', 'z', 'prediction'):
            torch.testing.assert_close(batch[key], old[key][offset:offset+n], rtol=1e-5, atol=1e-6)
            parity[key] = max(parity.get(key, 0.), float((batch[key].double()-old[key][offset:offset+n].double()).abs().max()))
        torch.testing.assert_close(batch['base_logpred'], old['quantity'][offset:offset+n].log1p()-old['log_residual'][offset:offset+n], rtol=1e-5, atol=1e-6)
        check(torch.equal(parts.cpu(), old['series_index'][offset:offset+n]), 'Series order differs')
        ends = torch.tensor([dataset.index[int(i)][1] for i in indices[offset:offset+n]])
        check(torch.equal(ends, old['context_end'][offset:offset+n]), 'Context endpoint differs')
        if number == 0:
            with torch.no_grad():
                official = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
            torch.testing.assert_close(batch['prediction'], official['pred_qty'].cpu(), rtol=1e-5, atol=1e-6)
        for key, value in batch.items():
            check(torch.isfinite(value).all().item(), f'Nonfinite stage {key}')
            arrays.setdefault(key, []).append(value.detach().cpu())
        offset += n
        if number % 32 == 0 or offset == len(indices):
            print({'stage':'extract', 'model':label, 'completed':offset, 'total':len(indices), 'seconds':round(time.monotonic()-started,2)}, flush=True)
    check(offset == 8192, 'Partial diagnostic cohort')
    check(before == canonical_state_dict_sha256(model.state_dict()), 'Frozen parameters changed')
    result = {key: torch.cat(value) for key, value in arrays.items()}
    result.update(target_index=indices.clone(), series_id=old['series_index'].clone(),
                  fold=old['fold'].clone(), context_end=old['context_end'].clone())
    check(torch.equal(result['fold'], torch.tensor([fold_for_series(x) for x in result['series_id']])), 'Fold changed')
    check(torch.equal(result['h'], result['layer2_last']), 'Unexpected additional encoder transform')
    return result, {'state_sha256_unchanged':before, 'cache_parity_max_abs':parity,
                    'official_prediction_parity_first_batch':True, 'layer2_equals_h':True}


def extract_phase():
    c = load_contract()
    check(not RAW.exists() and not RESULT.exists(), 'Refusing to overwrite extraction artifacts')
    prior = read(ROOT/c['prior_manifest'])
    registry = read(ROOT/'paper/contracts/count_aware_hard_lmm_frozen_probe_v1.json')
    old_contract = read(ROOT/'paper/contracts/hard_lmm_query_diagnostic_v1.json')
    kv_audit = read(ROOT/old_contract['separate_key_audit'])
    torch.set_num_threads(4)
    torch.manual_seed(42)
    RAW.mkdir(parents=True); RESULT.mkdir(parents=True)
    manifest = {'status':'extracting','started_at':now(),'source_context_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'contract_sha256':digest(CONTRACT),'source_hashes':sources(),'frozen_documents':c['frozen_documents'],
                'runtime':{'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,'polars':pl.__version__,'platform':platform.platform(),'device':'cpu','torch_threads':4},
                'datasets':{},'validation_rows_materialized':False,'held_out_rows_materialized':False,'backbone_parameters_updated':False,'server_accessed':False,
                'retrospective_cohort_reuse':True}
    save(RESULT/'execution_manifest.json', manifest)
    save(RESULT/'execution_contract.json', c)
    try:
        for name in DATASETS:
            row = next(r for r in registry['datasets'] if r['dataset']==name)
            inputs = prior['datasets'][name]['input_hashes']
            verify(inputs)
            for rec in c['prior_caches'][name].values():
                check(digest(ROOT/rec['path']) == rec['sha256'], 'Prior cache changed')
            old = {key:torch.load(ROOT/c['prior_caches'][name][key]['path'],map_location='cpu',weights_only=False) for key in MODELS}
            for key in ('target_index','series_index','fold','context_end','quantity','history_length'):
                check(torch.equal(old['original'][key],old['separate_key'][key]), f'Paired prior identity: {key}')
            check(len(old['original']['target_index'])==8192, 'Wrong cohort size')
            for fold in (0,1):
                check(old['original']['series_index'][old['original']['fold']==fold].unique().numel()>=10, 'Insufficient fold series')
            frame=load_train_frame(ROOT/row['data_path'])
            dataset=RMTPPWeekLookbackDataset(prepare_count_frame(frame),lookback_weeks=row['lookback'],max_seq_len=row['max_seq_len'],mode='all',split_col='chronological_split',target_splits={'train'})
            check(len(dataset)==prior['datasets'][name]['train_targets'], 'Population changed')
            launch=read(ROOT/row['artifact_dir']/'launch_contract.json')
            quantiles=dict(zip(launch['quantity_contract']['quantiles'],launch['quantity_contract']['boundaries']))
            meta={'input_hashes':inputs,'sample_targets':8192,'series_count':old['original']['series_index'].unique().numel(),
                  'fold_counts':[int((old['original']['fold']==f).sum()) for f in (0,1)],'models':{}}
            extracted={}
            for label in MODELS:
                if label=='original':
                    path=checkpoint_path(ROOT/row['artifact_dir'],42)
                    model,audit=restore_b0(path,launch,'cpu')
                    check(audit['model_state_sha256']==row['checkpoint_state_sha256'],'Original state mismatch')
                else:
                    path=ROOT/old_contract['separate_key_artifact']/name/'runs'/KEY_VALUE_BACKBONE/'count_only_log_regression/seed_42/best_val_joint_objective_model.pt'
                    payload=torch.load(path,map_location='cpu',weights_only=False)
                    validate_checkpoint_route(payload,KEY_VALUE_BACKBONE)
                    check(canonical_state_dict_sha256(payload['model_state_dict'])==kv_audit['runs'][name]['checkpoint_state_sha256'],'KV state mismatch')
                    model,_=restore(path,KEY_VALUE_BACKBONE)
                cache,audit=extract_one(model,dataset,old[label],f'{name}/{label}')
                cache.update(body_threshold=float(quantiles[.95]),tail_threshold=float(quantiles[.99]))
                dest=RAW/name/f'{label}_cache.pt';dest.parent.mkdir(parents=True,exist_ok=True);torch.save(cache,dest)
                meta['models'][label]={'cache_path':str(dest.relative_to(ROOT)),'cache_sha256':digest(dest),**audit}
                extracted[label]=cache
                del model
            check(torch.equal(extracted['original']['history_features'],extracted['separate_key']['history_features']),'History features not paired')
            manifest['datasets'][name]=meta
            save(RESULT/'execution_manifest.json',manifest)
            del frame,dataset,old,extracted
        check(digest(CONTRACT)==manifest['contract_sha256'],'Contract changed during extraction')
        verify(manifest['source_hashes']);verify(c['frozen_documents'])
        for m in manifest['datasets'].values(): verify(m['input_hashes'])
        manifest.update(status='extracted',extraction_completed_at=now())
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        save(RESULT/'execution_manifest.json',manifest)
        raise
    save(RESULT/'execution_manifest.json',manifest)


def analyze_phase():
    c=load_contract(); manifest=read(RESULT/'execution_manifest.json')
    check(manifest['status']=='extracted','Extraction incomplete or analysis already completed')
    check(digest(CONTRACT)==manifest['contract_sha256'],'Contract changed')
    verify(manifest['source_hashes']);verify(c['frozen_documents'])
    torch.set_num_threads(4)
    manifest.update(status='analyzing', analysis_started_at=now())
    save(RESULT/'execution_manifest.json',manifest)
    try:
        results={}; hashes={}
        for name in DATASETS:
            results[name]={}
            for label in MODELS:
                rec=manifest['datasets'][name]['models'][label];path=ROOT/rec['cache_path']
                check(digest(path)==rec['cache_sha256'],'Stage cache changed')
                cache=torch.load(path,map_location='cpu',weights_only=False)
                print({'stage':'analyze','dataset':name,'model':label,'started_at':now()},flush=True)
                result,predictions=analysis.analyze_cache(cache,bootstrap_repeats=c['evidence_gate']['paired_series_bootstrap']['repeats'])
                dest=RAW/name/f'{label}_oof.npz'
                check(not dest.exists(),'Refusing OOF overwrite')
                np.savez_compressed(dest,**predictions)
                hashes[str(dest.relative_to(ROOT))]=digest(dest)
                results[name][label]=result
                save(RESULT/'analysis.json',results)
                print({'stage':'analyzed','dataset':name,'model':label,'completed_at':now()},flush=True)
        decision=analysis.aggregate_common_evidence(results)
        save(RESULT/'evidence_decision.json',decision)
        manifest.update(status='complete',completed_at=now(),oof_hashes=hashes,
                        analysis_sha256=digest(RESULT/'analysis.json'),decision_sha256=digest(RESULT/'evidence_decision.json'))
        check(digest(CONTRACT)==manifest['contract_sha256'],'Contract changed during analysis')
        verify(manifest['source_hashes']);verify(c['frozen_documents'])
        save(RESULT/'execution_manifest.json',manifest)
    except BaseException as exc:
        manifest.update(status='failed_analysis',error=f'{type(exc).__name__}: {exc}',failed_at=now())
        save(RESULT/'execution_manifest.json',manifest)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('extract','analyze'))
    args=parser.parse_args()
    (extract_phase if args.phase=='extract' else analyze_phase)()
