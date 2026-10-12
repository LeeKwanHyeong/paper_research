from copy import deepcopy
import unittest
from paper.scripts import prepare_titantpp_cnn_gru_components_v5 as p


class ReplacementPreparation(unittest.TestCase):
    def test_design_restored_science_and_original_absolute_lease_preserved(self):
        parent=p.read(p.PARENT/'execution_contract.json');c=p.contract(parent)
        self.assertEqual(c['design_sha256'],p.DESIGN)
        for field in ('source','input_files','datasets','dataset_sha256','training','diagnostic','architecture',
            'analysis','parallelism','jobs','assignments','arms','seeds','policy','launch_policy'):
            self.assertEqual(c[field],parent[field])
        self.assertEqual(c['operational_replacement']['host_leases'],{'5080':{
            'started_at_unix':1791794258.15949,'deadline_unix':1792399058.15949}})
        self.assertNotIn('5090',c['operational_replacement']['host_leases'])
        self.assertEqual(c['operation']['files']['operation/qualification.py'],parent['operation']['files']['operation/qualification.py'])

    def test_parent_mutation_rejected(self):
        parent=p.read(p.PARENT/'execution_contract.json');parent['training']['warm_start']=True
        with self.assertRaises(ValueError):p.contract(parent)


if __name__=='__main__':unittest.main()
