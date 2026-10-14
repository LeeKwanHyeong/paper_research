import base64
import json
import hashlib
import unittest
from unittest.mock import patch
from paper.scripts import advance_titantpp_gru_cap_parallel3_v3 as a
from paper.scripts import titantpp_components_status_transport_v2 as t


class ProbeTests(unittest.TestCase):
    def test_parallel_owner_uses_real_reader_signature(self):
        ns=dict(vars(t));exec(a.EXTRA_PROBES,ns)
        c={'hosts':{'5080':{'python':'python','root':'/owned'}}}
        argv=['python','/owned/parallel3_v2/runtime.py','--contract','/owned/execution_contract.json','--host','5080','--mode','dispatch']
        raw=json.dumps({'pid':11,'ppid':10,'argv':argv}).encode()
        files={'parallel3_v2/launch.json':{'base64':base64.b64encode(raw).decode(),'raw_sha256':hashlib.sha256(raw).hexdigest()}}
        def reader(pid,proc_root):
            self.assertEqual(proc_root,'/proc')
            return {'pid':pid,'ppid':10 if pid==11 else 1,'pgrp':pid,
                'argv':argv if pid==11 else argv[:-1]+['pipeline'],'identity_stable_during_query':True}
        ns['collect_process_observations']=lambda *args:({},{},[])
        ns['status_proc_read']=reader
        obs,_,owners=ns['parallel_process_observations']({},files,[],c,'5080')
        self.assertEqual(owners,[10,11]);self.assertEqual(obs['11']['state'],'live')
        ns['status_proc_read']=lambda *args:(_ for _ in ()).throw(OSError('disappeared during read'))
        obs,_,_=ns['parallel_process_observations']({},files,[],c,'5080')
        self.assertEqual(obs['11']['state'],'query_error')
        ns['status_proc_read']=lambda *args:(_ for _ in ()).throw(FileNotFoundError())
        obs,_,_=ns['parallel_process_observations']({},files,[],c,'5080')
        self.assertEqual(obs['11']['state'],'absent')


if __name__=='__main__':unittest.main()
