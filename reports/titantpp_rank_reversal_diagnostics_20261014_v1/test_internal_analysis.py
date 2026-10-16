"""Conservation/near-zero guards for saved-sum diagnostics, no ML runtime."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('internal_analysis', Path(__file__).with_name('internal_analysis.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SavedSumGuards(unittest.TestCase):
    def test_zero_and_cancellation_denominators(self):
        self.assertIsNone(module.signed_share(4, 0))
        self.assertIsNone(module.signed_share(1e8, 1e-4))
        self.assertEqual(module.signed_share(8, 4), 2)
        self.assertEqual(module.signed_share(-8, 4), -2)

    def test_metric_conservation_rejects_lost_cell(self):
        cells=[]
        for i in range(5):
            c={'bin':i,'count':1,'qty_sse':4.,'qty_absolute_error_sum':2.,
               'qty_signed_error_sum':-2.,'time_nll_sum':1.,'qty_rmse':2.,
               'qty_mae':2.,'qty_bias':-2.}
            cells.append(c)
        m={'count':5,'qty_sse':20.,'qty_absolute_error_sum':10.,'qty_signed_error_sum':-10.,
           'time_nll_sum':5.,'qty_rmse':2.,'qty_mae':2.,'qty_bias':-2.,
           'quantity_cells':cells,'tail':{k:v for k,v in cells[-1].items() if k!='bin'}}
        module.check_metrics(m)
        changed=copy.deepcopy(m);changed['quantity_cells'][0]['qty_sse']=5.
        with self.assertRaisesRegex(ValueError, 'Bin sum mismatch'):
            module.check_metrics(changed)
        changed=copy.deepcopy(m);changed['quantity_cells'][0]['count']=2
        with self.assertRaisesRegex(ValueError, 'Bin count'):
            module.check_metrics(changed)

    def test_symmetric_reversal_formula_with_cancellation(self):
        pv,pt,dv,dt=.2,.1,-2.,5.
        mix=(pt-pv)*(dt+dv)/2
        within=(pt+pv)*(dt-dv)/2
        self.assertTrue(module.near(mix+within,pt*dt-pv*dv))
        self.assertLess(mix,0)
        self.assertGreater(within,mix+within)


if __name__=='__main__':
    unittest.main()
