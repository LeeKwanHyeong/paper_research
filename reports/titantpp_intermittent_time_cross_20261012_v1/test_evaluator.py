"""CPU fabricated regressions: no selected checkpoint or real data inference."""
from pathlib import Path
from unittest import mock
import contextlib
import copy
import importlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT/'search_artifacts/titantpp_history_train_fit_20261012_v1/sources/history'

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
E=load(ROOT/'paper/scripts/evaluate_titantpp_intermittent_time_cross.py','cross_evaluator_fixture')

class AggregateTests(unittest.TestCase):
    def contract(self):
        c=json.loads((ROOT/'search_artifacts/titantpp_history_train_fit_20261012_v1/contract.json').read_text())
        c.update(schema=E.KIND,parent_contract_sha256=E.PARENT)
        c['jobs']=[j for j in c['jobs'] if j['dataset']==E.DATASET]
        c['datasets']={E.DATASET:c['datasets'][E.DATASET]}
        c['time_cross']={'evaluation_splits':['train','validation'],'new_fits':0,'test_inference':0,'model_split_passes':18,'joint_worker_count':6,'concurrent_joint_workers':1,'resident_models_max':3,'reference_receipts':{j['id']:{'train':{},'validation':{}} for j in c['jobs']}}
        return c

    def test_only_selected_nine_train_validation_matrix_and_native_parent(self):
        c=self.contract()
        with tempfile.TemporaryDirectory(prefix='time_cross_contract_fixture_') as tmp:
            root=Path(tmp)
            def write(contract):
                contract['contract_sha256']=E.canonical({k:v for k,v in contract.items() if k!='contract_sha256'})
                (root/'contract.json').write_bytes(E.json_bytes(contract))
            write(c);self.assertEqual(len(E.verify_contract(root)['jobs']),9)
            wrong=copy.deepcopy(c);wrong['time_cross']['evaluation_splits'].append('test');write(wrong)
            with self.assertRaisesRegex(ValueError,'Train/Validation-only'):E.verify_contract(root)
            wrong=copy.deepcopy(c);wrong['time_cross']['resident_models_max']=4;write(wrong)
            with self.assertRaisesRegex(ValueError,'Train/Validation-only'):E.verify_contract(root)
            wrong=copy.deepcopy(c);wrong['jobs']=wrong['jobs'][:-1];write(wrong)
            with self.assertRaisesRegex(ValueError,'Intermittent9'):E.verify_contract(root)
            job=c['jobs'][0]
            original=json.loads((ROOT/'search_artifacts/titantpp_history_train_fit_20261012_v1/results'/job['id']/'train_metrics.json').read_text())
            path=root/'native_train.json';path.write_bytes(E.json_bytes(original))
            c['time_cross']['reference_receipts'][job['id']]['train']={'relative_path':'native_train.json','sha256':E.sha(path),'contract_sha256':E.PARENT}
            self.assertEqual(E.reference_receipt(root,c,job,'train')['split'],'train')
            original['contract_sha256']=E.VALIDATION_PARENT;path.write_bytes(E.json_bytes(original))
            c['time_cross']['reference_receipts'][job['id']]['train']['sha256']=E.sha(path)
            with self.assertRaisesRegex(ValueError,'parent contract'):E.reference_receipt(root,c,job,'train')

    def test_paired_effect_algebra_and_squared_moments(self):
        a=E.CrossAccumulator([1.,3.,5.,7.])
        a.add([1.,2.,8.],{'ff':[1.,3.,9.],'fc':[2.,2.,4.],'cf':[4.,5.,6.],'cc':[5.,3.,3.]})
        cells=a.finish();whole=next(x for x in cells if x['group']=='all')
        self.assertEqual(whole['count'],3)
        self.assertEqual(whole['sums']['scale_effect'],-5.)
        self.assertEqual(whole['squares']['scale_effect'],27.)
        self.assertEqual(whole['sums']['location_effect'],2.)
        self.assertEqual(whole['sums']['interaction'],1.)
        self.assertEqual(whole['sums']['total_effect'],-2.)
        self.assertAlmostEqual(whole['means']['scale_effect']+whole['means']['location_effect']+whole['means']['interaction'],whole['means']['total_effect'])
        self.assertEqual({x['key'] for x in cells if x['group']=='target_gap_bin'},{'0','1','4'})
        with self.assertRaisesRegex(ValueError,'aligned'):a.add([1.],{'ff':[],'fc':[1.],'cf':[1.],'cc':[1.]})
        with self.assertRaisesRegex(ValueError,'Nonfinite'):a.add([1.],{'ff':[math.nan],'fc':[1.],'cf':[1.],'cc':[1.]})

    def test_cli_test_and_foreign_worker_forbidden(self):
        with self.assertRaises(SystemExit):E.main(['--root','/tmp/unused','--seed','42','--split','test','--device','cpu'])
        with self.assertRaisesRegex(ValueError,'Joint cross worker'):E.main(['--root','/tmp/unused','--seed','42','--split','train','--device','cpu','--arm',E.FULL])

class FrozenScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(FROZEN))
        import torch
        cls.torch=torch
        engine=importlib.import_module('paper.scripts.run_titantpp_cnn_history_controls');engine.install_hooks()
        cls.factory=importlib.import_module('models.TPPs.CountAwareFactory')
        cls.core=importlib.import_module('paper.scripts.count_aware_tpp_backbone.core')
        cls.runner=importlib.import_module('simple_lab_test.search.common.runner')
        cls.mass=staticmethod(importlib.import_module('models.TPPs.positive_integer_time').positive_integer_log_mass)

    def test_frozen_integer_cdf_cross_cells_and_alignment_negative(self):
        t=self.torch
        gap=t.tensor([1.,2.,3.,7.]);q=t.tensor([2.,3.,4.,5.]);h=t.tensor([3,4,5,6])
        f=(t.tensor([-.2,-.1,.1,.5],dtype=t.float64),t.tensor([.6,.8,.7,.9],dtype=t.float64),q,h)
        c=(t.tensor([.3,.1,-.2,.2],dtype=t.float64),t.tensor([1.,.5,.8,.6],dtype=t.float64),q.clone(),h.clone())
        original_f=-self.mass(gap,f[0],f[1],time_scale=3.,top_code=None)
        original_c=-self.mass(gap,c[0],c[1],time_scale=3.,top_code=None)
        scores=E.paired_scores(t,self.mass,gap,f,c,original_f,original_c)
        def cdf(x):return .5*(1.+math.erf(x/math.sqrt(2.)))
        for name,mu,sigma in (('ff',f[0],f[1]),('fc',f[0],c[1]),('cf',c[0],f[1]),('cc',c[0],c[1])):
            expected=[]
            for d,m,s in zip(gap.tolist(),mu.tolist(),sigma.tolist()):
                high=cdf((math.log((d+.5)/3)-m)/s)
                low=0. if d==1 else cdf((math.log((d-.5)/3)-m)/s)
                expected.append(-math.log(high-low))
            t.testing.assert_close(scores[name],t.tensor(expected,dtype=t.float64),rtol=1e-12,atol=1e-12)
        wrong=(*c[:2],q.flip(0),h)
        with self.assertRaisesRegex(ValueError,'alignment'):E.paired_scores(t,self.mass,gap,f,wrong,original_f,original_c)
        with self.assertRaisesRegex(ValueError,'exactly reproduced'):E.paired_scores(t,self.mass,gap,f,c,original_f+.01,original_c)

    def test_real_frozen_hook_all_arms_bitwise_original_state_and_actual_mass(self):
        t=self.torch
        for arm in E.HISTORY_ARMS:
            with self.subTest(arm=arm),t.random.fork_rng(devices=[]):
                t.manual_seed(42)
                model,_=self.factory.build_count_aware_model(arm,hidden_dim=64,quantity_variant='count_only_log_regression',lambda_tail=0.,time_head_mode='heteroscedastic_lognormal_duration',time_scale=3.,time_sigma_floor=.001,time_initial_location=-.2,time_initial_scale=.6,time_observation_contract={'mode':'positive_integer_round_clamp_v1','top_code':None,'unit':'week'},train_log_mean=1.2,train_log_std=.8,max_seq_len=8)
                model.eval().requires_grad_(False)
                before=self.runner.canonical_state_dict_sha256(model.state_dict())
                dts=t.tensor([[0.,0.,1.,2.,1.,3.,1.,2.],[1.,2.,1.,3.,1.,2.,1.,3.]])
                q=t.tensor([[0.,0.,2.,4.,1.,3.,2.,5.],[1.,2.,3.,4.,3.,2.,1.,6.]])
                mask=t.ones_like(dts,dtype=t.bool);mask[0,:2]=False
                with t.inference_mode():
                    baseline=self.core.target_outputs(model,dts,mask,q,lambda_log_qty=1.)
                    observer=E.TimeObserver(model,t);observer.begin()
                    value=self.core.target_outputs(model,dts,mask,q,lambda_log_qty=1.)
                    mu,sigma=observer.parameters(dts[:,-1]);observer.close()
                    self.assertTrue(all(t.equal(baseline[k],value[k]) for k in baseline))
                    self.assertTrue(t.equal(-self.mass(dts[:,-1],mu,sigma,time_scale=3.,top_code=None),value['time_loss']))
                    altered=q.clone();altered[:,-1]+=1000
                    changed=self.core.target_outputs(model,dts,mask,altered,lambda_log_qty=1.)
                    self.assertTrue(t.equal(changed['pred_qty'],value['pred_qty']))
                    self.assertTrue(t.equal(changed['time_loss'],value['time_loss']))
                self.assertEqual(before,self.runner.canonical_state_dict_sha256(model.state_dict()))

    def test_full_joint_worker_train_validation_actual_loader_models_and_receipts(self):
        """Full worker path with real frozen loaders/models but fabricated data."""
        import hashlib
        import numpy as np
        import polars as pl
        t=self.torch
        loader_module=importlib.import_module('paper.scripts.run_taxi_quantity_interface_ablation')
        metrics_module=importlib.import_module('paper.scripts.evaluate_titantpp_history_width')
        numerical=importlib.import_module('paper.scripts.quantity_comparison_runtime')
        rows=[('A',0,'train',1.),('A',1,'train',7.),('A',2,'train',205.),('A',3,'validation',250.),
              ('C',0,'train',5.),('C',1,'train',200.),('C',2,'validation',270.)]
        frame=pl.DataFrame({'oper_part_no':[r[0] for r in rows],'seq':[r[1] for r in rows],
            'chronological_split':[r[2] for r in rows],'demand_qty':[r[3] for r in rows],
            'delta_t':[1]*len(rows),'demand_dt':[20010101+r[1] for r in rows]}).sort(['oper_part_no','seq'])
        config=dict(hidden_dim=64,quantity_variant='count_only_log_regression',lambda_tail=0.,
            time_head_mode='heteroscedastic_lognormal_duration',time_scale=3.,time_sigma_floor=.001,
            time_initial_location=-.2,time_initial_scale=.6,
            time_observation_contract={'mode':'positive_integer_round_clamp_v1','top_code':None,'unit':'week'})
        train_q=frame.filter(pl.col('chronological_split')=='train')['demand_qty'].to_numpy().astype(np.float64)
        stats={'all_train_quantity_sha256':hashlib.sha256(b'quantity_comparison_all_train_row_quantity_v1\0'+train_q.astype('<f8').tobytes()).hexdigest(),
            'train_log_mean':float(np.log1p(train_q).mean()),'train_log_std':float(np.log1p(train_q).std())}
        definition={'loader':{'batch_size':128,'num_workers':0,'drop_last':False,'lookback_weeks':520,'max_seq_len':8},
                    'statistics':stats,'model':config,'quantity_boundaries_all_train_rows':[2,31,46,187],
                    'inherited_data_identity':{'populations':{}}}
        with tempfile.TemporaryDirectory(prefix='time_cross_whole_worker_fixture_') as tmp:
            root=Path(tmp);frame.write_parquet(root/'fixture.parquet');(root/'operation_integrity.json').write_text('{}')
            source={'closure':E.SOURCE,'revision':'fabricated-only'}
            spec={'definition':definition,'data_sha256':E.sha(root/'fixture.parquet'),'training_data_path':'fixture.parquet',
                  'training_data_sha256':E.sha(root/'fixture.parquet'),'diagnostics':{'target_gap_boundaries':[1.,3.,5.,7.]}}
            jobs,models,states=[],{},{}
            for arm in E.HISTORY_ARMS:
                model,_=self.factory.build_count_aware_model(arm,**config,train_log_mean=stats['train_log_mean'],train_log_std=stats['train_log_std'],max_seq_len=8)
                model.eval().requires_grad_(False)
                model_path=root/(arm+'.pt');metadata_path=root/(arm+'.json');metadata_path.write_text('{}')
                t.save({'model_state_dict':model.state_dict()},model_path)
                state_sha=self.runner.canonical_state_dict_sha256(model.state_dict())
                j={'id':'fixture42__'+arm,'study':'history','dataset':E.DATASET,'arm':arm,'seed':42,
                   'selected_epoch':1,'initial_state_sha256':'0'*64,'source_revision_history':['fabricated-only'],
                   'checkpoint':{'relative_path':model_path.name,'sha256':E.sha(model_path)},
                   'checkpoint_metadata_path':metadata_path.name,'checkpoint_metadata_sha256':E.sha(metadata_path),'checkpoint_state_sha256':state_sha}
                jobs.append(j);models[arm]=model;states[arm]=model.state_dict()
            c={'schema':E.KIND,'contract_sha256':'fabricated whole worker only','jobs':jobs,'sources':{'history':source},
               'datasets':{E.DATASET:spec},'runtime':{'torch_num_threads':4},'time_cross':{'reference_receipts':{j['id']:{} for j in jobs}}}
            # Reference construction is independent direct frozen forward plus the
            # actual chronological loader, target mapping and old metric class.
            references={}
            for split in E.SPLITS:
                loader,dataset,identity,metadata,offsets=E.prepare_population(frame,self.core,loader_module,definition,split,np)
                definition['inherited_data_identity']['populations'][split]=identity
                target_digest,truth_digest=hashlib.sha256(),hashlib.sha256();processed=0
                metrics={arm:metrics_module.QuantityMetrics([2,31,46,187]) for arm in E.HISTORY_ARMS}
                with t.inference_mode():
                    for _,dts,mask,parts,qty in loader:
                        values={arm:self.core.target_outputs(m,dts,mask,qty,lambda_log_qty=1.) for arm,m in models.items()}
                        for arm,value in values.items():metrics[arm].add(*(value[k].numpy().astype(np.float64) for k in ('true_qty','pred_qty','time_loss')))
                        hm=mask.clone();hm[:,-1]=False;lengths=hm.sum(1)
                        last=qty[:,-2].double().numpy();mean=((qty.double()*hm).sum(1)/lengths).numpy()
                        q=values[E.FULL]['true_qty'].double().numpy()
                        for index in range(len(q)):
                            part,end=dataset.index[processed+index];raw=metadata.row(offsets[part]+end+1,named=True)
                            entity,seq=str(dataset.parts[part]),int(dataset.seq_lists[part][end+1]);date=str(raw['demand_dt'])
                            tid=hashlib.sha256(E.json_bytes([E.DATASET,spec['data_sha256'],entity,seq,split,date])).hexdigest()
                            row={'target_id':tid,'entity_id':entity,'seq':seq,'split':split,'recorded_date':date,'raw_quantity':float(q[index]),
                                 'recorded_gap':float(dts[index,-1]),'history_length':int(lengths[index]),'last_quantity':float(last[index]),'mean_quantity':float(mean[index])}
                            target_digest.update((tid+'\n').encode());truth_digest.update(E.json_bytes([row[k] for k in E.TRUTH_FIELDS])+b'\n')
                        processed+=len(q)
                for job in jobs:
                    arm=job['arm'];metric={'split':split,**metrics[arm].finish()}
                    if split=='validation':job['validation_reference']=metric
                    references[(job['id'],split)]={'metrics':metric,'original_population_identity':identity,'target_identity_sha256':target_digest.hexdigest(),'truth_sha256':truth_digest.hexdigest()}
                    c['time_cross']['reference_receipts'][job['id']][split]={'relative_path':'fixture_reference','sha256':'fixture'}
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(sys,'path',sys.path.copy()))
                for name,value in {'verify_contract':lambda *a:c,'verify_operations':lambda *a:{},'reference_receipt':lambda r,c,j,s:references[(j['id'],s)],'PROJECT_PREFIXES':set()}.items():stack.enter_context(mock.patch.object(E,name,value))
                for name,value in {'verify_source':lambda *a:FROZEN,'verify_imports':lambda *a:{},'check_runtime':lambda *a:None,'validate_payload':lambda *a:None,'validate_frame':lambda *a:None}.items():stack.enter_context(mock.patch.object(E.original,name,value))
                stack.enter_context(mock.patch.object(self.factory,'build_count_aware_model',lambda arm,**kw:(models[arm],{})))
                stack.enter_context(mock.patch.object(self.factory,'validate_checkpoint_route',lambda *a:None))
                stack.enter_context(mock.patch.object(numerical,'configure_runtime',lambda *a:None))
                stack.enter_context(mock.patch.object(numerical,'runtime_identity',lambda *a:{}))
                for split in E.SPLITS:
                    self.assertEqual(E.worker(root,42,split,'cpu'),0)
                    joint=E.verify_joint_receipt(root,c,42,split)
                    self.assertEqual(joint['actual_model_split_passes'],3)
                    for job in jobs:
                        r=E.read(root/'results'/job['id']/(split+'_metrics.json'))
                        self.assertEqual(r['pid'],joint['pid']);self.assertTrue(r['shared_pid_model_split_pass'])
                        self.assertEqual(r['state_sha256'],r['state_after_sha256']);self.assertEqual(r['native_six_metrics_match'],'PASS')
                        self.assertEqual(r['held_out_test_evaluated'],False)
                        self.assertEqual(r['prediction_rows'],3 if split=='train' else 2)
                with self.assertRaisesRegex(ValueError,'prohibits repeat'):E.worker(root,42,'train','cpu')

if __name__=='__main__':unittest.main()
