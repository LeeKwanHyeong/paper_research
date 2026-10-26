"""Local synthetic artifact failures; never calls SSH, GPU or a model."""
import json
from pathlib import Path
import tempfile
import unittest

import remote_control as control


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def fixture(root):
    out = root / "output_full"
    job = out / "fake_job"
    job.mkdir(parents=True)
    spec = root / "fake_spec.json"
    write_json(spec, {"mode":"full", "runner_sha256":"runner", "population_counts":{"train":2},
                      "jobs":[{"id":"fake_job", "checkpoint_sha256":"checkpoint", "splits":["train"]}]})
    chunk = job / "train_00000.parquet"
    chunk.write_bytes(b"opaque synthetic observation bytes")
    hidden = job / "train_00000.head_hidden.npz"
    hidden.write_bytes(b"opaque synthetic hidden bytes")
    progress = job / "train_progress.jsonl"
    progress.write_text('{"rows":2}\n')
    v = {"rows":2,"chunks":[{"rows":2,"observations":control.rec(chunk),"head_hidden":control.rec(hidden)}],
         "head_hidden_saved":True,"progress":control.rec(progress)}
    write_json(job / "train_manifest.json",v)
    write_json(job / "INTENT.json",{"synthetic":True})
    write_json(job / "COMPLETE.json",{"status":"complete", "checkpoint_sha256":"checkpoint", "splits":{"train":v}})
    stdout = out / "fake_job.stdout"; stdout.write_bytes(b"done\n")
    stderr = out / "fake_job.stderr"; stderr.write_bytes(b"")
    exit_path = out / "fake_job.ACTUAL_EXIT.json"
    exit_record = {"actual_child_exit":0,"spec_sha256":control.sha(spec),"collector_sha256":"runner",
                   "stdout":control.rec(stdout),"stderr":control.rec(stderr)}
    write_json(exit_path,exit_record)
    write_json(out / "COMPLETE.json",{"spec_sha256":control.sha(spec),"collector_sha256":"runner",
                                      "actual_child_exits_all_zero":True,"receipts":[control.rec(exit_path)]})
    return out,spec,job,chunk,hidden,exit_path


class OutputInventoryTests(unittest.TestCase):
    def test_completed_job_cannot_hide_missing_tampered_or_extra_chunk(self):
        for mutation in ("missing_observation","missing_hidden","tampered_observation","extra_member"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                out,spec,job,chunk,hidden,_ = fixture(Path(td))
                control.verify_output(out,spec)
                if mutation == "missing_observation": chunk.unlink()
                elif mutation == "missing_hidden": hidden.unlink()
                elif mutation == "tampered_observation": chunk.write_bytes(b"changed")
                else: (job / "undeclared.bin").write_bytes(b"extra")
                with self.assertRaises(AssertionError): control.verify_output(out,spec)

    def test_self_consistent_receipt_or_split_edit_cannot_hide_failed_exit_or_wrong_rows(self):
        for mutation in ("failed_exit","wrong_rows","wrong_checkpoint"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                out,spec,job,_,_,exit_path = fixture(Path(td))
                if mutation == "failed_exit":
                    r=json.loads(exit_path.read_text());r['actual_child_exit']=1;write_json(exit_path,r)
                    c=json.loads((out/'COMPLETE.json').read_text());c['receipts']=[control.rec(exit_path)];write_json(out/'COMPLETE.json',c)
                else:
                    c=json.loads((job/'COMPLETE.json').read_text())
                    if mutation == "wrong_checkpoint":c['checkpoint_sha256']='different'
                    else:
                        c['splits']['train']['rows']=3;c['splits']['train']['chunks'][0]['rows']=3
                        write_json(job/'train_manifest.json',c['splits']['train'])
                    write_json(job/'COMPLETE.json',c)
                with self.assertRaises(AssertionError):control.verify_output(out,spec)


if __name__ == '__main__': unittest.main()
