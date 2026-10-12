from types import SimpleNamespace
import unittest
from unittest.mock import patch
from paper.scripts import launch_titantpp_components_5080_health_compat_v2 as bridge


class ObserverCompatibility(unittest.TestCase):
    def classify(self, *, argv=None, comm='unattended-upgr', shutdown='b false', changed=False):
        expected=['/usr/bin/python3','/usr/share/unattended-upgrades/unattended-upgrade-shutdown','--wait-for-signal']
        row='S 1 '+'0 '*17+'469'
        pid=SimpleNamespace(name='1514')
        reads={'stat':SimpleNamespace(read_text=lambda: '1514 (x) '+row),
            'cmdline':SimpleNamespace(read_bytes=lambda: ('\x00'.join(expected if argv is None else argv)+'\x00').encode()),
            'status':SimpleNamespace(read_text=lambda:'Uid:\t0\t0\t0\t0')}
        class Proc:
            name='1514'
            def __truediv__(self,name): return reads[name]
        ns={'p':Proc(),'comm':comm,'updates':[],'idle_shutdown_observers':[],
            'sha':lambda path:'changed' if changed else '8fbf4e278c4ad457e82cc05586631b1a0550dd9d10892a70487ef74c623066e0',
            'capture':lambda args:shutdown,'subprocess':bridge.sealed.subprocess}
        exec('if True:\n if True:\n'+bridge.UPDATES_NEW,ns)
        return ns['updates'],ns['idle_shutdown_observers']

    def test_exact_waiter_source_and_nonshutdown_are_classified(self):
        updates,idle=self.classify()
        self.assertEqual(updates,[]);self.assertEqual(len(idle),1)

    def test_updater_unknown_args_shutdown_or_changed_source_still_block(self):
        for kwargs in ({'argv':['/usr/bin/python3','/usr/bin/unattended-upgrade']},
            {'comm':'apt-get'},{'shutdown':'b true'},{'changed':True}):
            updates,idle=self.classify(**kwargs)
            self.assertEqual(len(updates),1);self.assertEqual(idle,[])

    def test_unreadable_logind_is_not_assumed_idle(self):
        self.assertEqual(len(self.classify(shutdown='unresolved')[0]),1)

    def test_remote_script_compiles_and_only_health_is_extended(self):
        text=bridge.health_script()
        self.assertIn('PreparingForShutdown',text)
        self.assertIn(bridge.previous.NEW,text)


if __name__=='__main__':unittest.main()
