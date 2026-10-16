"""Synthetic coverage/deduplication guards; no real results or datasets."""
import importlib.util,unittest
from pathlib import Path
p=Path('reports/titantpp_components_inter62_test_20261014_v1/merge_three_seed_results.py');spec=importlib.util.spec_from_file_location('inter62_merge',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class MergeTests(unittest.TestCase):
 def rows(self):return [{'split':split,'seed':seed,'arm':arm,'metrics':{name:1.0+seed/100 for name,path in m.METRICS}} for split in ('validation','test') for seed in (42,52,62) for arm in ('NONE','CNN','GRU','FULL')]
 def test_missing_seed62_cannot_be_reported_as_three_seed(self):
  with self.assertRaisesRegex(ValueError,'Missing'):m.summarize([r for r in self.rows() if r['seed']!=62])
 def test_duplicate_reused_test_id_is_rejected(self):
  rows=self.rows();rows.append(dict(rows[-1]))
  with self.assertRaisesRegex(ValueError,'Duplicate'):m.summarize(rows)
 def test_complete_cross_seed_rows_report_sample_sd_n3(self):
  summary,paired=m.summarize(self.rows());self.assertEqual(summary['test']['FULL']['전체 RMSE']['n'],3);self.assertAlmostEqual(summary['test']['FULL']['전체 RMSE']['sample_sd'],0.1);self.assertEqual(len(paired),18)
if __name__=='__main__':unittest.main()
