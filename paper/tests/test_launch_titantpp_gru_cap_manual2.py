"""Minimal manual2 routing and fixed-science boundaries; no remote calls."""
from copy import deepcopy
import unittest
from paper.scripts import launch_titantpp_gru_cap_manual2 as manual
from paper.tests.test_launch_titantpp_gru_cap_validation import OperationalBoundaries


class ManualRequalificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OperationalBoundaries.setUpClass()
        cls.prepared = OperationalBoundaries.prepared
        cls.parent = OperationalBoundaries.parent

    def overlay(self):
        c = OperationalBoundaries().overlay()
        for host, spec in c['hosts'].items():
            oldroot = spec['root']; newroot = oldroot + '_manual2'
            for key in ('root', 'source_root', 'operation_root', 'output_dir'):
                spec[key] = spec[key].replace(oldroot, newroot)
            spec['tmux'] += '_manual2'
            spec['environment'] = {k: v.replace(oldroot, newroot) for k, v in spec['environment'].items()}
        c.update(design_sha256=self.parent['design_sha256'], architecture=deepcopy(self.parent['architecture']),
            manual_requalification=manual.manual_bridge(), operational_replacement={'host_leases': deepcopy(manual.LEASES)})
        return c

    def validate(self, c):
        return manual.validate_contract(c, self.prepared, self.parent)

    def test_manual2_accepts_only_fixed_science_and_original_deadlines(self):
        self.validate(self.overlay())
        for lease in manual.LEASES.values():
            self.assertEqual(lease['deadline_unix'] - lease['started_at_unix'], 604800)

    def test_legacy_missing_keys_and_modified_science_fail(self):
        for mutate in (lambda c: c.pop('design_sha256'), lambda c: c.pop('architecture'),
                lambda c: c['training'].update(maximum_epochs=301), lambda c: c['candidate'].update(rho=.5)):
            c = self.overlay(); mutate(c)
            with self.assertRaises((KeyError, ValueError)):
                self.validate(c)

    def test_failed_root_manual_approval_or_deadline_drift_blocked(self):
        for mutate in (lambda c: c['hosts']['5080'].update(root=c['hosts']['5080']['root'].removesuffix('_manual2')),
                lambda c: c['manual_requalification'].update(approved=False),
                lambda c: c['operational_replacement']['host_leases']['5080'].update(deadline_unix=1792568742.8913815)):
            c = self.overlay(); mutate(c)
            with self.assertRaises(ValueError): self.validate(c)

    def test_remote_script_changes_only_owned_root_guard(self):
        from paper.scripts import launch_titantpp_cnn_gru_components_v5 as base
        launcher = manual.launcher_for(manual.BUNDLE)
        expected = base.REMOTE_SCRIPT.replace("(c['schema']+'_'+host),'foreign root')",
            "(c['schema']+'_'+host+'_manual2'),'foreign manual2 root')")
        self.assertEqual(launcher.REMOTE_SCRIPT, expected)
        self.assertIn("(c['schema']+'_'+host),'foreign root')", base.REMOTE_SCRIPT)

    def test_foreign_bundle_blocked_before_remote(self):
        with self.assertRaises(ValueError): manual.check_bundle('/tmp/unapproved')


if __name__ == '__main__':
    unittest.main()
