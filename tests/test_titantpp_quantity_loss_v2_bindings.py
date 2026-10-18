"""V2 result paths change without changing the preserved aggregation logic."""
import sys
import unittest
from unittest.mock import patch

from paper import scripts as package
from paper.scripts import compare_titantpp_quantity_loss_mlp18_v2 as comparison
from paper.scripts import finish_titantpp_quantity_loss_mlp18_v2 as completion
from paper.scripts import manage_titantpp_quantity_loss_mlp18 as original


class BindingTests(unittest.TestCase):
    def test_manager_binding_targets_v2_and_restores_original_on_error(self):
        original_attribute = package.manage_titantpp_quantity_loss_mlp18
        original_module = sys.modules['paper.scripts.manage_titantpp_quantity_loss_mlp18']
        with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            with comparison.manager_binding() as manager:
                self.assertEqual(manager.NAME, 'titantpp_quantity_loss_mlp18_20261019_v2')
                self.assertIs(package.manage_titantpp_quantity_loss_mlp18, manager)
                raise RuntimeError('synthetic failure')
        self.assertIs(package.manage_titantpp_quantity_loss_mlp18, original_attribute)
        self.assertIs(sys.modules['paper.scripts.manage_titantpp_quantity_loss_mlp18'], original_module)
        self.assertEqual(original.NAME, 'titantpp_quantity_loss_mlp18_20261019_v1')

    def test_same_comparator_is_called_with_new_manager_only(self):
        def check_binding():
            self.assertEqual(package.manage_titantpp_quantity_loss_mlp18.NAME, 'titantpp_quantity_loss_mlp18_20261019_v2')
            return 'unchanged aggregation'
        with patch.object(comparison.previous, 'main', side_effect=check_binding) as invoke:
            self.assertEqual(comparison.main(), 'unchanged aggregation')
        invoke.assert_called_once()
        self.assertIs(package.manage_titantpp_quantity_loss_mlp18, original)

    def test_completion_wait_imports_v2_without_changing_v1_function(self):
        prior = completion.previous.wait_program
        contract = {'hosts': {'5080': {'root': '/isolated/v2'}}}
        code = completion.wait_program(contract, '5080', 'canonical', {'pid': 1})
        compile(code, 'v2_wait', 'exec')
        self.assertIn('import titantpp_quantity_loss_mlp18_runtime_v2 as r', code)
        self.assertNotIn('import titantpp_quantity_loss_mlp18_runtime as r', code)
        self.assertIs(completion.previous.wait_program, prior)


if __name__ == '__main__':
    unittest.main()
