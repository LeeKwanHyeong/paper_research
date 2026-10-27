"""Independent CPU synthetic QA; no real fitting, inference, remote or Test data."""
import copy
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import polars as pl
import torch

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
from paper.scripts import run_taxi_internal_holdout_diagnostic as runner


def row(index, origin, label, entity=None):
    return dict(split='train', target_index=index, entity_id=entity or str(index),
                target_seq=label, raw_quantity=10., target_gap=label-origin,
                history_length=1, origin_time=origin, label_time=label)


def dataset(seqs=(1, 2, 4, 7, 12), window=168, maximum=256):
    frame=pl.DataFrame({'oper_part_no':['a']*len(seqs),'seq':list(seqs),
                        'delta_t':[1]+[b-a for a,b in zip(seqs,seqs[1:])],
                        'mark':[0]*len(seqs),'scale_residual':[float(10+i) for i in range(len(seqs))],
                        'chronological_split':['train']*len(seqs)})
    return RMTPPWeekLookbackDataset(frame,lookback_weeks=window,max_seq_len=maximum,
                                   mode='all',target_splits={'train'})


def metadata(ds):
    result=[]
    for idx,(part,end) in enumerate(ds.index):
        sample=ds[idx]
        result.append(dict(split='train',target_index=idx,entity_id=str(ds.parts[part]),
                           target_seq=ds.seq_lists[part][end+1],raw_quantity=float(sample['values'][-1]),
                           target_gap=float(sample['dts'][-1]),history_length=int(sample['mask'].sum())-1,
                           origin_time=ds.seq_lists[part][end],label_time=ds.seq_lists[part][end+1]))
    return result


class PartitionTests(unittest.TestCase):
    def test_boundaries_and_crossing_queries_are_purged(self):
        rows=[row(0,309,310),row(1,310,311),row(2,311,411),row(3,411,412),
              row(4,412,413),row(5,300,500)]
        groups=runner.partition_rows(rows)
        self.assertEqual({k:[r['target_index'] for r in v] for k,v in groups.items()},
                         {'fit':[0],'dev':[2],'audit':[4],'purged':[1,3,5]})
        self.assertEqual(sum(map(len,groups.values())),len(rows))

    def test_full_contract_counts_with_synthetic_identities(self):
        rows=[]
        for count,(origin,label) in [(23108,(309,310)),(7536,(311,411)),(7487,(412,413)),(262,(310,311))]:
            rows.extend(row(i,origin,label) for i in range(len(rows),len(rows)+count))
        groups=runner.partition_rows(rows)
        self.assertEqual({k:len(v) for k,v in groups.items()},dict(fit=23108,dev=7536,audit=7487,purged=262))

    def test_duplicate_missing_target_and_invalid_chronology_fail(self):
        good=[row(0,309,310),row(1,311,312)]
        cases=[[],good+[copy.deepcopy(good[0])],[good[1]]]
        for change in [dict(origin_time=310),dict(target_gap=2),dict(label_time=float('nan')),dict(split='test')]:
            altered=copy.deepcopy(good);altered[0].update(change);cases.append(altered)
        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):runner.partition_rows(rows)


class LoaderExposureTests(unittest.TestCase):
    def test_context_shadow_only_changes_validator_copy_not_loader_targets(self):
        from paper.scripts.count_aware_tpp_backbone.observed_time import validate_recorded_grid
        frame=pl.DataFrame({'oper_part_no':['a']*4,'seq':[309,310,311,312],'delta_t':[1,1,1,1],
                            'mark':[0]*4,'scale_residual':[1.,2.,3.,4.],
                            'chronological_split':['train','train','context_only','validation']})
        before=frame.clone()
        def native(shadow,name):
            self.assertEqual(shadow['chronological_split'].to_list(),['train','train','validation','validation'])
            for key in ['oper_part_no','seq','delta_t','scale_residual']:self.assertEqual(shadow[key].to_list(),before[key].to_list())
            validate_recorded_grid(shadow,name)
        runner.validate_context_only_grid(frame,'yellow_trip_hourly',{('a',311)},native)
        self.assertTrue(frame.equals(before))
        dev=RMTPPWeekLookbackDataset(frame,mode='all',target_splits={'validation'})
        fit=RMTPPWeekLookbackDataset(frame,mode='all',target_splits={'train'})
        self.assertEqual([dev.seq_lists[p][i+1] for p,i in dev.index],[312])
        self.assertEqual([fit.seq_lists[p][i+1] for p,i in fit.index],[310])
        for bad,keys in [(frame,{('a',999)}),(frame.with_columns(pl.when(pl.col('seq')==311).then(pl.lit('test')).otherwise(pl.col('chronological_split')).alias('chronological_split')),{('a',311)})]:
            with self.assertRaises(ValueError):runner.validate_context_only_grid(bad,'yellow_trip_hourly',keys,native)

    def test_both_grid_validator_bindings_restore_after_failure(self):
        from paper.scripts.count_aware_tpp_backbone import observed_time,training
        before1,before2=observed_time.validate_recorded_grid,training.validate_recorded_grid
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            with runner.internal_grid_guard(set()):
                self.assertIs(observed_time.validate_recorded_grid,training.validate_recorded_grid)
                self.assertIsNot(observed_time.validate_recorded_grid,before1)
                raise RuntimeError('synthetic')
        self.assertIs(observed_time.validate_recorded_grid,before1);self.assertIs(training.validate_recorded_grid,before2)

    def test_metadata_matches_actual_native_context_truncation(self):
        for ds in [dataset(),dataset(window=3),dataset(maximum=3)]:
            proof=runner.validate_metadata(ds,metadata(ds),'train')
            self.assertEqual(proof['targets'],len(ds))

    def test_metadata_order_target_quantity_and_history_mismatch_fail(self):
        ds=dataset();base=metadata(ds)
        for field,value in [('target_index',999),('entity_id','other'),('target_seq',999),
                            ('raw_quantity',99.),('history_length',99),('origin_time',-1)]:
            records=copy.deepcopy(base);records[0][field]=value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):runner.validate_metadata(ds,records,'train')

    def test_context_and_target_exposure_are_both_blocked(self):
        ds=dataset()
        for heldout in [{('a',12)},{('a',7)}]:
            with self.assertRaisesRegex(ValueError,'Heldout'):runner.leakage_audit(ds,[3],heldout)
        proof=runner.leakage_audit(ds,[0],{('a',7),('a',12)})
        self.assertEqual(proof['heldout_context_overlap'],0)

    def test_truncated_out_old_context_is_not_false_leakage(self):
        ds=dataset(maximum=3)
        self.assertEqual(runner.leakage_audit(ds,[3],{('a',1),('a',2)})['heldout_context_overlap'],0)
        ds=dataset(window=3)
        self.assertEqual(runner.leakage_audit(ds,[3],{('a',1),('a',2),('a',4)})['heldout_context_overlap'],0)

    def test_subset_order_and_seeded_shuffle_preserve_target_population(self):
        ds=dataset();indices=[3,0,2]
        def values(loader):return [float(x) for batch in loader for x in batch[-1][:,-1]]
        self.assertEqual(values(runner._subset_loader(ds,indices)),[14.,11.,13.])
        a=runner._subset_loader(ds,indices,shuffle=True,generator=torch.Generator().manual_seed(42))
        b=runner._subset_loader(ds,indices,shuffle=True,generator=torch.Generator().manual_seed(42))
        av,bv=values(a),values(b)
        self.assertEqual(av,bv);self.assertCountEqual(av,[14.,11.,13.]);self.assertFalse(a.drop_last)


class RngTests(unittest.TestCase):
    def test_evaluation_wrapper_preserves_rng_and_rejects_nonfinite_time(self):
        model=torch.nn.Dropout(.5);model.train()
        generator=torch.Generator().manual_seed(8)
        loader=runner._subset_loader(dataset(),[0,1],generator=generator)
        good={'val_joint_objective':2.,'val_time_nll':1.,'val_log_qty_mse':1.,'qty_rmse':3.,'qty_mae':2.,'evaluated_count':2}
        before=torch.get_rng_state().clone();gb=generator.get_state().clone()
        def evaluate(**kwargs):
            kwargs['model'].eval();list(kwargs['loader']);torch.rand(2)
            return dict(good)
        self.assertEqual(runner.evaluate_preserving_rng(evaluate,model=model,loader=loader),good)
        self.assertTrue(model.training);self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(torch.equal(gb,generator.get_state()))
        for key,value in [('val_time_nll',float('nan')),('val_joint_objective',float('inf')),('evaluated_count',0)]:
            bad={**good,key:value}
            with self.subTest(key=key):
                with self.assertRaises(ValueError):runner.evaluate_preserving_rng(lambda **kw:bad,model=model,loader=loader)
                self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_all_CPU_random_streams_loader_generator_and_mixed_modes_restore(self):
        model=torch.nn.Sequential(torch.nn.Dropout(.5),torch.nn.Linear(1,1))
        model.train();model[1].eval()
        generator=torch.Generator().manual_seed(42)
        py=random.getstate();ns=np.random.get_state();ts=torch.get_rng_state().clone();gs=generator.get_state().clone()
        with runner.preserved_rng(model,[generator]):
            random.random();np.random.rand();torch.rand(4);torch.rand(4,generator=generator);model.eval()
            list(runner._subset_loader(dataset(),[0,1]))
        self.assertEqual(py,random.getstate());self.assertEqual(ns[0],np.random.get_state()[0]);np.testing.assert_array_equal(ns[1],np.random.get_state()[1])
        self.assertEqual(ns[2:],np.random.get_state()[2:]);self.assertTrue(torch.equal(ts,torch.get_rng_state()));self.assertTrue(torch.equal(gs,generator.get_state()))
        self.assertTrue(model.training);self.assertTrue(model[0].training);self.assertFalse(model[1].training)

    def test_exception_and_initialized_CUDA_state_are_restored(self):
        before=torch.get_rng_state().clone();sentinel=[torch.tensor([3],dtype=torch.uint8)]
        with mock.patch.object(torch.cuda,'is_initialized',return_value=True),mock.patch.object(torch.cuda,'get_rng_state_all',return_value=sentinel),mock.patch.object(torch.cuda,'set_rng_state_all') as setter:
            with self.assertRaisesRegex(RuntimeError,'synthetic'):
                with runner.preserved_rng():torch.rand(3);raise RuntimeError('synthetic')
            setter.assert_called_once_with(sentinel)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


class EvaluationPolicyTests(unittest.TestCase):
    def test_actual_model_factory_consumes_explicit_fit_statistics(self):
        import importlib
        module=importlib.import_module('models.TPPs.CountAwareHistoryMLPControl')
        factory=importlib.import_module('models.TPPs.CountAwareFactory')
        data={'model':{'hidden_dim':64,'time_initial_location':.123,'time_initial_scale':.456},
              'statistics':{'train_log_mean':2.2,'train_log_std':.7},'mu_all_train_rows':999.,'loader':{'max_seq_len':256}}
        fake=torch.nn.Identity()
        with mock.patch.object(module,'CountAwareHistoryMLPControl',return_value=fake) as constructor,mock.patch.object(factory,'with_time_metadata',side_effect=lambda model,meta:(model,meta)):
            module.build_model(data,'single_mlp')
        self.assertEqual(constructor.call_args.kwargs['train_log_mean'],2.2)
        self.assertEqual(constructor.call_args.kwargs['train_log_std'],.7)
        self.assertEqual(constructor.call_args.kwargs['time_initial_location'],.123)
        self.assertEqual(constructor.call_args.kwargs['time_initial_scale'],.456)

    def test_original_early_stopping_minimum_patience_and_earliest_ties(self):
        from paper.scripts.count_aware_tpp_backbone.training import early_stopping_exhausted,earliest_strict_minimum
        history=[{'epoch':e,'dev':1. if e in [1,2] else 2.} for e in range(1,42)]
        self.assertEqual(earliest_strict_minimum(history,metric_key='dev')['epoch'],1)
        self.assertFalse(early_stopping_exhausted(history[:39],min_epochs=40,patience=40,metric_key='dev'))
        self.assertFalse(early_stopping_exhausted(history[:40],min_epochs=40,patience=40,metric_key='dev'))
        self.assertTrue(early_stopping_exhausted(history,min_epochs=40,patience=40,metric_key='dev'))
        with self.assertRaises(ValueError):earliest_strict_minimum([{'epoch':1,'dev':float('nan')}],metric_key='dev')

    def test_native_metric_accumulation_matches_independent_event_arithmetic(self):
        from paper.scripts.count_aware_tpp_backbone import core
        model=torch.nn.Identity()
        output={k:torch.zeros(2) for k in ['joint_loss','time_loss','quantity_train_loss','log_qty_loss','quantity_distribution_nll','quantity_location_huber','tail_aux_loss','tail_indicator','quantity_scale','true_qty','pred_qty']}
        output.update(time_loss=torch.tensor([2.,4.]),log_qty_loss=torch.tensor([.25,.75]),quantity_train_loss=torch.tensor([.25,.75]),joint_loss=torch.tensor([2.25,4.75]),true_qty=torch.tensor([1.,5.]),pred_qty=torch.tensor([4.,1.]),history_length=torch.tensor([1,1]))
        batch=(None,torch.ones((2,2)),torch.ones((2,2),dtype=torch.bool),None,torch.ones((2,2)))
        with mock.patch.object(core,'target_outputs',return_value=output):
            measured=runner.evaluate_preserving_rng(core.evaluate,model=model,loader=[batch],quantity_contract={'strata':[]},device='cpu',lambda_log_qty=1.,max_batches=None,include_breakdowns=False)
        self.assertEqual(measured['val_time_nll'],3.);self.assertEqual(measured['val_log_qty_mse'],.5)
        self.assertEqual(measured['val_joint_objective'],3.5);self.assertEqual(measured['qty_mae'],3.5);self.assertAlmostEqual(measured['qty_rmse'],math.sqrt(12.5))

    def test_actual_input_reader_rejects_synthetic_Test_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'synthetic.parquet'
            pl.DataFrame({'chronological_split':['train','validation','test'],'demand_qty':[1.,2.,3.],'oper_part_no':['a']*3,'seq':[1,2,3]}).write_parquet(path)
            with self.assertRaisesRegex(ValueError,'Test is locked'):runner.safe_load_input(SimpleNamespace(input_frame=path))

    def test_grid_and_endpoint_only_validation(self):
        self.assertEqual(runner.audit_epochs(45,27),[1,5,10,20,27,40,45])
        p=runner.authorized_evaluation_pairs(45,27)
        self.assertEqual(p['validation'],[27,45]);self.assertEqual(p['audit'],[1,5,10,20,27,40,45]);self.assertNotIn('test',p)
        self.assertEqual(runner.authorized_evaluation_pairs(40,40)['validation'],[40])

    def test_invalid_epoch_contract_fails_closed(self):
        for last,best in [(0,0),(301,1),(40,41),(40,0),(True,1),(1,True)]:
            with self.subTest(last=last,best=best):
                with self.assertRaises(ValueError):runner.audit_epochs(last,best)

    def test_aggregate_is_event_weighted_and_objectives_separate(self):
        rows=[dict(true_qty=1.,pred_qty=4.,log_qty_mse=.2,time_nll=2.,joint_objective=2.2),
              dict(true_qty=5.,pred_qty=1.,log_qty_mse=.8,time_nll=4.,joint_objective=4.8),
              dict(true_qty=9.,pred_qty=9.,log_qty_mse=0.,time_nll=9.,joint_objective=9.)]
        result=runner.aggregate_events(rows)
        self.assertAlmostEqual(result['qty_rmse'],math.sqrt(25/3));self.assertAlmostEqual(result['qty_mae'],7/3)
        self.assertEqual(result['time_nll'],5.);self.assertAlmostEqual(result['joint_objective'],16/3)
        with self.assertRaises(ValueError):runner.aggregate_events([])
        rows[0]['time_nll']=float('nan')
        with self.assertRaises(ValueError):runner.aggregate_events(rows)

    def test_exclusive_writes_never_repair_failed_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'attempt.json';runner.write_json(p,{'status':'failure'})
            with self.assertRaises(FileExistsError):runner.write_json(p,{'status':'success'})
            self.assertEqual(json.loads(p.read_text()),{'status':'failure'})


class SupportTests(unittest.TestCase):
    def test_small_and_empty_report_groups_retain_counts_but_no_metrics(self):
        rows=[dict(entity_id=str(i%10),condition_cell='a',fit_supported_cell=True,within_fit_position_range=True,true_qty=1.,pred_qty=2.,time_nll=1.,log_qty_mse=.1,joint_objective=1.1) for i in range(30)]
        rows.append({**rows[0],'condition_cell':'rare','fit_supported_cell':False,'within_fit_position_range':False})
        groups={r['group']:r for r in runner.group_event_metrics(rows)}
        self.assertTrue(groups['common_supported']['report_eligible']);self.assertEqual(groups['outside_common_support']['count'],1)
        self.assertIsNone(groups['outside_common_support']['time_nll']);self.assertFalse(groups['cell:rare']['report_eligible'])
        empty={r['group']:r for r in runner.group_event_metrics(rows[:-1])}['outside_common_support']
        self.assertEqual(empty['count'],0);self.assertIsNone(empty['qty_rmse'])

    def test_fit_only_cells_exact_support_boundary_and_missing_distinct(self):
        def features(i,missing=False):return dict(entity_id=str(i%10),history_length=2,log_previous_quantity=1.,log_previous_gap=0.,log_quantity_change=None if missing else 0.)
        rows=[features(i) for i in range(30)]+[features(i,True) for i in range(29)]
        contract=runner._support_contract(rows)
        normal=runner._cell(features(0),contract);missing=runner._cell(features(0,True),contract)
        self.assertNotEqual(normal,missing);self.assertTrue(contract['fit_cells'][normal]['supported']);self.assertFalse(contract['fit_cells'][missing]['supported'])
        fewer_entities=[{**features(i),'entity_id':str(i%9)} for i in range(30)]
        self.assertFalse(next(iter(runner._support_contract(fewer_entities)['fit_cells'].values()))['supported'])

    def test_past_feature_change_is_log_difference_and_no_future_quantity(self):
        ds=dataset();m=metadata(ds);before=runner._features(ds,m)
        self.assertIsNone(before[0]['log_quantity_change'])
        self.assertAlmostEqual(before[1]['log_quantity_change'],math.log1p(11)-math.log1p(10))
        ds.val_lists[0][-1]=99999.
        after=runner._features(ds,m)
        for a,b in zip(before,after):
            for k in ['log_previous_quantity','log_previous_gap','log_quantity_change']:self.assertEqual(a[k],b[k])


if __name__=='__main__':unittest.main()
