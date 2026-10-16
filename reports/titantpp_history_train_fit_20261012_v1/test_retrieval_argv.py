"""Guard the real Python argv representation without accepting other workers."""
import copy
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent

def terminal_matches(row):
    return (row['worker_pid']==row['launch_pid']==row['exit_pid']
            and row['worker_ppid']==row['launch_ppid']==row['exit_ppid']
            and row['exit_argv']==row['launch_argv']
            and row['worker_argv']==row['launch_argv'][1:]
            and row['exit_code']==0 and row['actual_child_exit_observed'] is True
            and row['timed_out'] is False and row['supervisor_aborted'] is False)

class PythonArgvContract(unittest.TestCase):
    def setUp(self):
        self.data=json.loads((ROOT/'RETRIEVAL_DIAGNOSIS_OBSERVATION.json').read_text())

    def test_actual_all27_match_corrected_convention(self):
        self.assertEqual(len(self.data['rows']),27)
        self.assertTrue(all(terminal_matches(r) for r in self.data['rows']))

    def test_original_comparison_rejects_legitimate_sys_argv(self):
        self.assertTrue(all(r['worker_argv']!=r['launch_argv'] for r in self.data['rows']))

    def test_wrong_process_parent_or_command_is_rejected(self):
        for key in ('worker_pid','worker_ppid'):
            row=copy.deepcopy(self.data['rows'][0]);row[key]+=1
            self.assertFalse(terminal_matches(row))
        for key in ('worker_argv','exit_argv'):
            row=copy.deepcopy(self.data['rows'][0]);row[key][-3]='foreign_job'
            self.assertFalse(terminal_matches(row))

    def test_failure_or_timeout_still_rejected(self):
        for key,value in [('exit_code',1),('timed_out',True),('supervisor_aborted',True),('actual_child_exit_observed',False)]:
            row=copy.deepcopy(self.data['rows'][0]);row[key]=value
            self.assertFalse(terminal_matches(row))

if __name__=='__main__':unittest.main()
