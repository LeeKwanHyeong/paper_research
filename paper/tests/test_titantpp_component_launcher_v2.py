"""Host-independent v2 launch tests; fake transport only, SSH/GPU/training0."""
from __future__ import annotations

from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import tarfile
import unittest
from unittest.mock import patch

from paper.scripts import launch_titantpp_cnn_gru_components_v2 as launcher
from paper.tests import test_titantpp_component_launcher as fixtures


class IndependentComponentLauncherTests(unittest.TestCase):
    # Reuse the small opaque stdlib fixture without changing the frozen v1
    # fixture source or running its dual-host test cases.
    bytes = fixtures.ComponentLauncherTests.bytes
    json = fixtures.ComponentLauncherTests.json
    deployment = fixtures.ComponentLauncherTests.deployment
    snapshot = fixtures.ComponentLauncherTests.snapshot
    set_cache = fixtures.ComponentLauncherTests.set_cache
    add_record = fixtures.ComponentLauncherTests.add_record
    live_snapshot = fixtures.ComponentLauncherTests.live_snapshot
    completed_snapshot = fixtures.ComponentLauncherTests.completed_snapshot

    def setUp(self):
        patch.object(fixtures, "launcher", launcher).start()
        fixtures.ComponentLauncherTests.setUp(self)
        self.c["launch_policy"] = {"host_independent_start": True,
            "all_hosts_qualification_before_any_fit": False, "fresh_user_start_instruction_required": True,
            "automatic_reassignment": False, "continuous_remote_polling": False}
        self.json("execution_contract.json", self.c)
        self.json("current.json", {"schema": launcher.SCHEMA, "contract_sha256": launcher.canonical(self.c),
            "bundle": str(self.bundle)})
        self.json("launch_hold.json", {"hold": True, "contract_sha256": launcher.canonical(self.c)})
        tool = self.project / "paper/scripts/launch_titantpp_cnn_gru_components_v2.py"
        tool.write_text("sealed_fake_v2_tool")
        files = {"paper/scripts/launch_titantpp_cnn_gru_components_v2.py": launcher.sha(tool)}
        self.json("local_tools_integrity.json", {"files": files, "files_sha256": launcher.canonical(files)})
        members = launcher.expected_members(self.c, launcher.sha(self.bundle / "execution_contract.json"))
        with tarfile.open(self.bundle / "deployment.tar.gz", "w:gz") as archive:
            for name in members:
                archive.add(self.bundle / name, arcname=name, recursive=False)
        self.json("preparation_receipt.json", {"contract_sha256": launcher.canonical(self.c),
            "contract_raw_sha256": launcher.sha(self.bundle / "execution_contract.json"),
            "local_tools_integrity_raw_sha256": launcher.sha(self.bundle / "local_tools_integrity.json"),
            "payload_sha256": launcher.sha(self.bundle / "deployment.tar.gz")})

    def release(self, host="5080", *, hosts=None):
        approval = {"approved": True, "explicit_new_user_instruction": True,
            "user_instruction": "5080 Taxi와 RAF만 먼저 진행하고 5090은 나중에 별도 지시한다",
            "hosts": hosts if hosts is not None else [host], "contract_sha256": launcher.canonical(self.c)}
        self.json("launch_authorization_" + host + ".json", approval)
        self.json("launch_release_" + host + ".json", {"released": True,
            "contract_sha256": launcher.canonical(self.c),
            "original_hold_raw_sha256": launcher.sha(self.bundle / "launch_hold.json"),
            "authorization_raw_sha256": launcher.sha(self.bundle / ("launch_authorization_" + host + ".json"))})
        return approval

    def test_check_local_only_preserves_36_scientific_jobs_and_24_12_assignment(self):
        remote = fixtures.FakeRemote(self.c)
        report = launcher.run_mode(self.bundle, "check", "5080", remote=remote)
        self.assertEqual(report["requested_host"], "5080")
        self.assertEqual(report["source_files"], 131)
        self.assertFalse(remote.calls)
        self.assertEqual(len(self.c["jobs"]), 36)
        self.assertEqual(sum(j["host"] == "5080" for j in self.c["jobs"]), 24)
        self.assertEqual(sum(j["host"] == "5090" for j in self.c["jobs"]), 12)

    def test_unapproved_5090_all_remote_modes_fail_before_ssh(self):
        self.release("5080")
        for mode in ("deploy", "qualify", "train", "status"):
            remote = fixtures.FakeRemote(self.c)
            with self.assertRaises(ValueError):
                launcher.run_mode(self.bundle, mode, "5090", remote=remote)
            self.assertFalse(remote.calls)
            self.assertFalse(remote.uploads)

    def test_dual_host_approval_is_not_accepted_for_single_host_mode(self):
        self.release("5080", hosts=["5080", "5090"])
        remote = fixtures.FakeRemote(self.c)
        with self.assertRaises(ValueError):
            launcher.run_mode(self.bundle, "deploy", "5080", remote=remote)
        self.assertFalse(remote.calls)

    def test_deploy_only_5080_ignores_5090_busy_and_never_calls_it(self):
        self.release("5080")
        remote = fixtures.FakeRemote(self.c, busy="5090")
        report = launcher.run_mode(self.bundle, "deploy", "5080", remote=remote)
        self.assertEqual(set(report["hosts"]), {"5080"})
        self.assertEqual([(host, action) for host, action, _ in remote.calls],
            [("5080", "health"), ("5080", "reserve"), ("5080", "deploy")])
        self.assertEqual([host for host, local, path in remote.uploads], ["5080"])
        self.assertFalse((self.bundle / "control/deployment/5090.json").exists())
        self.assertFalse((self.bundle / "launch_authorization_5090.json").exists())

    def test_target_busy_health_can_be_observed_again_but_mutation_not_retried(self):
        self.release("5080")
        remote = fixtures.FakeRemote(self.c, busy="5080")
        first = launcher.run_mode(self.bundle, "deploy", "5080", remote=remote)
        second = launcher.run_mode(self.bundle, "deploy", "5080", remote=remote)
        self.assertEqual(first["status"], second["status"])
        self.assertEqual([(h, a) for h, a, p in remote.calls], [("5080", "health"), ("5080", "health")])
        self.assertFalse(remote.uploads)
        remote.busy = None
        launcher.run_mode(self.bundle, "deploy", "5080", remote=remote)
        count = len(remote.calls)
        with self.assertRaises(ValueError):
            launcher.run_mode(self.bundle, "deploy", "5080", remote=remote)
        self.assertEqual(len(remote.calls), count)

    def test_qualify_only_target_with_own_deployment_and_no_other_authorization(self):
        approval = self.release("5080")
        self.json("control/deployment/5080.json", self.deployment("5080", approval))
        remote = fixtures.FakeRemote(self.c)
        launcher.run_mode(self.bundle, "qualify", "5080", remote=remote)
        self.assertEqual([(h, a) for h, a, p in remote.calls], [("5080", "health"), ("5080", "phase")])
        self.assertEqual(remote.calls[-1][2]["phase"], "qualify")
        self.assertFalse((self.bundle / "control/deployment/5090.json").exists())

    def test_own_qualification_is_strict_but_other_host_is_not_required(self):
        approval = self.release("5080")
        self.json("control/deployment/5080.json", self.deployment("5080", approval))
        remote = fixtures.FakeRemote(self.c)
        with self.assertRaises(ValueError):
            launcher.run_mode(self.bundle, "train", "5080", remote=remote)
        self.assertFalse(remote.calls)
        good = self.snapshot("5080")
        bad = deepcopy(good)
        name = "phases/qualify_exit.json"
        value = __import__("json").loads(base64.b64decode(bad["files"][name]["base64"]))
        value["os_exit_code"] = 1
        self.add_record(bad, name, value)
        self.set_cache("5080", bad)
        with self.assertRaises(ValueError):
            launcher.run_mode(self.bundle, "train", "5080", remote=remote)
        self.assertFalse(remote.calls)
        self.set_cache("5080", good)
        launcher.run_mode(self.bundle, "train", "5080", remote=remote)
        self.assertEqual([(h, a) for h, a, p in remote.calls], [("5080", "health"), ("5080", "phase")])
        permit = remote.calls[-1][2]["training_permit"]
        self.assertEqual(set(permit["qualifications"]), {"5080"})
        self.assertEqual(set(remote.calls[-1][2]["qualification_raw"]), {"5080"})
        content = base64.b64decode(remote.calls[-1][2]["qualification_raw"]["5080"])
        self.assertEqual(hashlib.sha256(content).hexdigest(), permit["qualifications"]["5080"]["raw_sha256"])
        self.assertFalse((self.bundle / "control/qualification_cache/5090").exists())
        self.assertFalse((self.bundle / "control/training_permits/5090.json").exists())

    def test_status_reads_only_target_and_keeps_observation_distinct_from_training(self):
        self.release("5080")
        remote = fixtures.FakeRemote(self.c, status={"5080": self.snapshot("5080")})
        report = launcher.run_mode(self.bundle, "status", "5080", remote=remote)
        self.assertEqual(set(report["hosts"]), {"5080"})
        self.assertEqual([(h, a) for h, a, p in remote.calls], [("5080", "status")])
        self.assertEqual(report["hosts"]["5080"]["status"], "qualification_passed")
        launcher.run_mode(self.bundle, "status", "5080", remote=remote)
        self.assertEqual(len(remote.calls), 1)

    def test_wrong_owner_root_runtime_rawsha_rejected(self):
        snapshot = self.snapshot("5080")
        for key, value in (("host", "5090"), ("root", "/foreign"), ("gpu_uuid", launcher.GPU_UUIDS["5090"])):
            bad = deepcopy(snapshot)
            bad[key] = value
            with self.assertRaises(ValueError):
                launcher.qualification_binding(bad, self.c, "5080")
        bad = deepcopy(snapshot)
        bad["files"]["qualification/receipt.json"]["raw_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            launcher.qualification_binding(bad, self.c, "5080")

    def test_v1_bytes_unchanged_and_v2_remote_helper_syntax_valid(self):
        project = Path(__file__).resolve().parents[2]
        self.assertEqual(launcher.sha(project / "paper/scripts/launch_titantpp_cnn_gru_components.py"),
            "85c11cc3f603b8bd456f5fcb62285e3fad5f6919209b79a419b2c1c58982147d")
        compile(launcher.REMOTE_SCRIPT, "v2_remote_helper_not_executed", "exec")

    def test_large_payload_uses_stdin_without_shell_interpolation_or_argv_limit(self):
        instruction = "literal $(touch /foreign); `whoami`; ' quoted"
        payload = {"host": "5080", "contract": self.c,
            "approval": {"user_instruction": instruction}, "qualification_raw": "x" * (200 * 1024)}
        response = type("Response", (), {"returncode": 0, "stdout": json.dumps({"status": "fake"})})()
        with patch.object(launcher.subprocess, "run", return_value=response) as process:
            launcher.SSHRemote().call("5080", "phase", payload)
        args, kwargs = process.call_args.args[0], process.call_args.kwargs
        self.assertEqual(args[5], "5080")
        self.assertNotIn(instruction, args[-1])
        self.assertLess(len(args[-1].encode()), 128 * 1024)
        self.assertEqual(json.loads(kwargs["input"]), payload)


if __name__ == "__main__":
    unittest.main()
