"""Fabricated saved rows and receipts only; no actual comparison or remote work."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts import compare_titantpp_value_difference_rows as c
from paper.scripts import analyze_titantpp_value_difference_rows as a


def complete():
    return {'status':'complete','candidate_count':3,'seed_count':3,'data_worker_count':6,'qualification_count':3,
            'actual_native_worker_exit0_count':9,'actual_data_worker_exit0_count':6,'actual_pipeline_exit0_count':1,
            'actual_supervisor_child_exit0_count':1,'validation_replay_count':3,'all_owned_absent':True,
            'new_training':False,'historical_CNN_inference':False,'held_out_test_evaluated':False,'automatic_retry':False,
            'stages':['qualification','validation','train'],'source_closure':c.SOURCE,'input_sha256':c.INPUT_SHA}


def row_fixture(split='validation',seed=42):
    h={'entity_id':'x','seq':4,'split':split,'raw_quantity':5.,'history_length':3,'last_quantity':4.,'mean_quantity':3.,
       'history_log_mean':1.,'history_log_std':.2,'last_log_deviation':.3,'last_observed_log_change':.2,
       'previous_observation_available':True,'d1':.2,'d2':-.1}
    row={**h,'dataset':'raf_spare_parts','seed':seed,'prediction':7.,'time_nll':3.,'signed_error':2.,'absolute_error':2.,'squared_error':4.,
         'recorded_date':'20200401','recorded_gap':1.,'observed_history_sha256':'h'*64}
    row['target_id']=c.canonical(['raf_spare_parts',c.INPUT_SHA,'x',4,split,'20200401'])
    ctx={('x',4):{'recorded_date':'20200401','recorded_gap':1.,'history_sha256':'h'*64}}
    receipt={'target_identity_sha256':hashlib.sha256((row['target_id']+'\n').encode()).hexdigest(),
             'truth_sha256':hashlib.sha256(c.json_bytes([row[f] for f in c.TRUTH_FIELDS])+b'\n').hexdigest(),
             'recorded_gap_sha256':hashlib.sha256(b'titantpp_components_recorded_target_time_v1\0'+split.encode()+b'\0'+struct.pack('<d',1.)).hexdigest()}
    baseline={**h,'dataset':'raf_spare_parts','seed':seed,'prediction_1':4.,'time_nll_1':2.}
    return row,baseline,h,ctx,receipt


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/private/tmp',prefix='value-row-comparison-test-');self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_completion_requires_exact_data_exits_pipeline_and_no_test(self):
        c.verify_final_scope(complete())
        for key,value in [('actual_data_worker_exit0_count',5),('actual_native_worker_exit0_count',6),('actual_pipeline_exit0_count',0),
                          ('historical_CNN_inference',True),('held_out_test_evaluated',True),('all_owned_absent',False),('candidate_count',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):c.verify_final_scope({**complete(),key:value})

    def test_pins_pre_and_post_checks_and_escape(self):
        p=self.root/'original';p.write_bytes(b'fixed');pins=c.Pins(self.root);pins.add(p,c.sha(p));pins.again()
        with self.assertRaises(ValueError):pins.add(p,'0'*64)
        p.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed during'):pins.again()
        for rel in ('../outside','/private/tmp/outside'):
            with self.assertRaises(ValueError):c.safe(self.root,rel)

    def test_source131_must_be_exact_byte_subset_of_source141(self):
        source={f'source{i}':f'{i:064x}' for i in range(141)};base={k:v for k,v in list(source.items())[:131]}
        candidate={'files':source,'closure':c.canonical(source)};baseline={'files':base,'closure':c.canonical(base)}
        with patch.object(c,'SOURCE',candidate['closure']),patch.object(c,'BASE_SOURCE',baseline['closure']):
            self.assertEqual(c.verify_source_subset(candidate,baseline),131)
            modified=deepcopy(candidate);modified['files']['source0']='x'*64
            with self.assertRaises(ValueError):c.verify_source_subset(modified,baseline)
            candidate['files']['source0']='x'*64;candidate['closure']=c.canonical(candidate['files'])
            with patch.object(c,'SOURCE',candidate['closure']),self.assertRaisesRegex(ValueError,'Inherited'):c.verify_source_subset(candidate,baseline)

    def test_first_strict_finite_minimum_no_reselection(self):
        h={'history':[{'epoch':1,'val_qty_rmse':None},{'epoch':2,'val_qty_rmse':3.},{'epoch':3,'val_qty_rmse':2.},{'epoch':4,'val_qty_rmse':2.}]}
        self.assertEqual(c.selected_epoch(h),(3,2.))
        with self.assertRaises(ValueError):c.selected_epoch(h['history'][::-1])
        with self.assertRaises(ValueError):c.selected_epoch([{'epoch':1,'val_qty_rmse':None}])

    def test_predictive_buffers_context_and_selected_state_bound(self):
        state='a'*64;job={'id':'j','selected_epoch':9,'checkpoint_state_sha256':state,'checkpoint':{'sha256':'b'*64}}
        contract={'source':{'closure':c.SOURCE,'revision':'fixed'},'contract_sha256':'c'*64,
                  'datasets':{'raf_spare_parts':{'populations':{'validation':{'target_count':6690}}}}}
        audit={'state_sha256':state,'all_named_buffers_sha256':'d'*64,'context_cache_sha256':'e'*64}
        r={'status':'complete','job_id':'j','split':'validation','parameters_unchanged':True,'state_including_buffers_unchanged':True,
           'no_training':True,'held_out_test_evaluated':False,'native_runtime_checked':True,'real_data_loaded':True,
           'predictive_state_before':audit,'predictive_state_after':deepcopy(audit),'state_sha256_before':state,'state_sha256_after':state,'state_sha256':state,
           'checkpoint_sha256':'b'*64,'selected_epoch':9,'source_closure':c.SOURCE,'source_revision':'fixed','data_sha256':c.INPUT_SHA,
           'contract_sha256':'c'*64,'population':{'target_count':6690},'prediction_rows':6690,'runtime':{},'runtime_after':{},'host':'5090','validation_match':'PASS'}
        c.verify_predictive_receipt(r,job,'validation',contract)
        for field in audit:
            changed=deepcopy(r);changed['predictive_state_after'][field]='f'*64
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'state changed'):c.verify_predictive_receipt(changed,job,'validation',contract)
        changed=deepcopy(r);changed['selected_epoch']=10
        with self.assertRaises(ValueError):c.verify_predictive_receipt(changed,job,'validation',contract)

    def test_incomplete_native_receipt_stops_before_row_read_or_collector(self):
        bundle=self.root/'bundle';(bundle/'retrieval').mkdir(parents=True)
        p=bundle/'retrieval/final_receipt.json';p.write_text(json.dumps({**complete(),'actual_data_worker_exit0_count':5}))
        with patch.object(c,'BUNDLE',bundle),patch.object(c,'PROJECT',self.root),patch.object(c,'load_module') as modules:
            with self.assertRaises(ValueError):c.verify_candidate(c.sha(p),c.Pins(self.root))
            modules.assert_not_called()

    def test_existing_result_directory_never_overwritten(self):
        out=self.root/'out';out.mkdir();(out/'RESULT.json').write_bytes(b'old')
        with patch.object(c,'OUTPUT',out),patch.object(c,'verify_candidate') as native,self.assertRaisesRegex(ValueError,'no overwrite'):
            c.run('x'*64,c.ANALYSIS_SHA)
        native.assert_not_called();self.assertEqual((out/'RESULT.json').read_bytes(),b'old')

    def test_unapproved_analysis_version_blocks_before_receipt(self):
        with patch.object(c,'OUTPUT',self.root/'new'),patch.object(c,'verify_candidate') as native,self.assertRaisesRegex(ValueError,'analysis version'):
            c.run('x'*64,'0'*64)
        native.assert_not_called()

    def test_expected_selections_metadata_and_input_pins_required(self):
        binary=self.root/'selected.pt';binary.write_bytes(b'not a serialized model; hashing only')
        hist=self.root/'history.json';hist.write_text(json.dumps({'history':[{'epoch':1,'val_qty_rmse':3.},{'epoch':2,'val_qty_rmse':2.},{'epoch':3,'val_qty_rmse':2.}]}))
        metadata={'best_epoch':2,'model_state_sha256':'a'*64,'resume_identity':{'interface_meta':{'source_files_sha256':'s'},'arguments':{'data_sha256':c.INPUT_SHA}}}
        meta=self.root/'metadata.json';meta.write_text(json.dumps(metadata))
        job={'selected_epoch':2,'selected_metric_value':2.,'checkpoint_state_sha256':'a'*64,
             'checkpoint':{'relative_path':'selected.pt','sha256':c.sha(binary)},'selection_history_path':'history.json','selection_history_sha256':c.sha(hist),
             'checkpoint_metadata_path':'metadata.json','checkpoint_metadata_sha256':c.sha(meta)}
        result=c.verify_selection(self.root,job,c.Pins(self.root),'s',2);self.assertEqual(result['epoch'],2)
        with self.assertRaises(ValueError):c.verify_selection(self.root,job,c.Pins(self.root),'s',3)
        metadata['resume_identity']['arguments']['data_sha256']='other';meta.write_text(json.dumps(metadata));job['checkpoint_metadata_sha256']=c.sha(meta)
        with self.assertRaisesRegex(ValueError,'source/input'):c.verify_selection(self.root,job,c.Pins(self.root),'s',2)


class RowArithmeticTests(unittest.TestCase):
    def test_candidate_target_truth_gap_domains_recomputed(self):
        row,_,_,contexts,receipt=row_fixture()
        with patch.object(c,'COUNTS',{'train':1,'validation':1}):
            result=c.verify_candidate_row_identity([row],receipt,contexts,{'seed':42},'validation')
            self.assertEqual(result['truth_sha256'],receipt['truth_sha256'])
            bare=hashlib.sha256(struct.pack('<d',1.)).hexdigest();self.assertNotEqual(bare,receipt['recorded_gap_sha256'])
            for field,value in [('target_id','foreign'),('recorded_date','20200501'),('recorded_gap',2.),('observed_history_sha256','z'*64),('seed',52),('split','test')]:
                wrong={**row,field:value}
                with self.subTest(field=field),self.assertRaises(ValueError):c.verify_candidate_row_identity([wrong],receipt,contexts,{'seed':42},'validation')
            wrong=dict(receipt,truth_sha256='x'*64)
            with self.assertRaises(ValueError):c.verify_candidate_row_identity([row],wrong,contexts,{'seed':42},'validation')

    def test_row_shards_count_hash_duplicate_and_column_allowlist(self):
        import polars as pl
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temp:
            root=Path(temp);p=root/'part.parquet';pl.DataFrame({'raw_quantity':[1.,2.],'prediction':[2.,1.],'prediction_other':[999.,999.]}).write_parquet(p)
            manifest={'prediction_rows':2,'parts':[{'path':'part.parquet','rows':2,'bytes':p.stat().st_size,'sha256':c.sha(p)}]}
            result=c.read_shards(root,manifest,c.Pins(root),['raw_quantity','prediction'],pl)
            self.assertEqual(result,[{'raw_quantity':1.,'prediction':2.},{'raw_quantity':2.,'prediction':1.}])
            duplicate=deepcopy(manifest);duplicate['parts']*=2
            with self.assertRaisesRegex(ValueError,'Duplicate'):c.read_shards(root,duplicate,c.Pins(root),['raw_quantity'],pl)
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA differs'):c.read_shards(root,manifest,c.Pins(root),['raw_quantity'],pl)

    def test_whole_and_tail_six_metrics_hand_calculated(self):
        rows=[{'raw_quantity':q,'p':p,'t':t} for q,p,t in [(1.,2.,.5),(200.,198.,1.),(201.,197.,1.5)]]
        result=c.aggregate_rows(rows,'p','t')
        self.assertAlmostEqual(result['qty_rmse'],7.**.5);self.assertAlmostEqual(result['qty_mae'],7./3.);self.assertEqual(result['time_nll'],1.)
        self.assertEqual(result['tail']['count'],1);self.assertEqual(result['tail']['qty_rmse'],4.);self.assertEqual(result['tail']['qty_mae'],4.);self.assertEqual(result['tail']['time_nll'],1.5)
        c.check_metrics(result,deepcopy(result))
        for group in (None,'tail'):
            for field in ('qty_rmse','qty_mae','time_nll'):
                changed=deepcopy(result);cell=changed if group is None else changed[group];cell[field]+=.01
                with self.subTest(group=group,field=field),self.assertRaises(ValueError):c.check_metrics(result,changed)

    def test_reconstruct_history_uses_target_excluded_loader_float32_and_gap_clip(self):
        import numpy as np
        import polars as pl
        raw=pl.DataFrame({'oper_part_no':['x']*4,'seq':[1,2,3,4],'chronological_split':['train','train','train','validation'],
                          'demand_qty':[2.,4.,8.,99.],'delta_t':[0.,1.,1.,1.],'demand_dt':[20200101,20200201,20200301,20200401]})
        h={'entity_id':'x','seq':4,'first_observed_seq':1,'last_observed_seq':3,'history_length':3}
        history={'train':[], 'validation':[h]};actual=c.reconstruct_contexts(raw,history,np)['validation']['x',4]
        expected=hashlib.sha256(b'observed_loader_float32_order_v1\0'+np.array([2.,4.,8.],dtype='<f8').tobytes()+np.array([1.,1.,1.],dtype='<f8').tobytes()).hexdigest()
        self.assertEqual(actual,{'recorded_date':'20200401','recorded_gap':1.,'history_sha256':expected})
        changed=raw.with_columns(pl.when(pl.col('seq')==4).then(500.).otherwise(pl.col('demand_qty')).alias('demand_qty'))
        self.assertEqual(actual,c.reconstruct_contexts(changed,history,np)['validation']['x',4])
        history['validation'][0]['last_observed_seq']=4;history['validation'][0]['history_length']=4
        with self.assertRaisesRegex(ValueError,'overlaps target'):c.reconstruct_contexts(raw,history,np)

    def test_pure_pairing_preserves_adverse_seed_and_additive_cells(self):
        cohort={}
        for seed in c.SEEDS:
            for split in c.SPLITS:
                row,baseline,h,_,_=row_fixture(split,seed)
                cohort[seed,split]=a.paired_rows([row],[baseline],[h],seed=seed,split=split,expected_count=1)
        summary=a.summarize(cohort,(.2,1.))
        self.assertEqual(len(summary['per_seed']),180);self.assertEqual(len(summary['support']),90)
        self.assertEqual(len(summary['additive_closure']),12)
        whole=next(x['metrics'] for x in summary['per_seed'] if x['family']=='whole')
        self.assertEqual(whole['difference']['mae'],1.);self.assertEqual(whole['difference']['se_sum'],3.)
        text=c.markdown({'summary':summary,'cuts':[.2,1.]})
        self.assertIn('+1.000000',text);self.assertIn('Runtime',text);self.assertIn('Test 개선',text)

    def test_nonfinite_or_negative_saved_prediction_rejected(self):
        for value in (float('nan'),float('inf'),-1.):
            with self.subTest(value=value),self.assertRaises(ValueError):c.aggregate_rows([{'raw_quantity':1.,'p':value,'t':1.}],'p','t')

    def test_pure_analysis_and_reconstruction_import_do_not_call_main(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temp:
            p=Path(temp)/'safe.py';p.write_text('def main():\n raise RuntimeError("entry forbidden")\ndef build_rows(): return 1\nif __name__=="__main__":main()\n')
            m=c.load_module(p,'_synthetic_never_main');self.assertEqual(m.build_rows(),1)

if __name__=='__main__':unittest.main()
