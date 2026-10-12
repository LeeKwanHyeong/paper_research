"""Host-independent v3 launch/health tests; fake transport only, SSH/GPU/training0."""
from __future__ import annotations

from copy import deepcopy
import ast
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paper.scripts import launch_titantpp_cnn_gru_components_v3 as launcher
from paper.scripts import launch_titantpp_cnn_gru_components_v2 as original
from paper.scripts import launch_titantpp_components_5080_health_compat as lock_bridge
from paper.scripts import launch_titantpp_components_5080_health_compat_v2 as observer_bridge
from paper.tests import test_titantpp_component_launcher as fixtures


class IndependentComponentLauncherV3Tests(unittest.TestCase):
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
        self.c["design_sha256"] = launcher.FROZEN_DESIGN_SHA256
        self.json("execution_contract.json", self.c)
        self.json("current.json", {"schema": launcher.SCHEMA, "contract_sha256": launcher.canonical(self.c),
            "bundle": str(self.bundle)})
        self.json("launch_hold.json", {"hold": True, "contract_sha256": launcher.canonical(self.c)})
        tool = self.project / "paper/scripts/launch_titantpp_cnn_gru_components_v3.py"
        tool.write_text("sealed_fake_v3_tool")
        files = {"paper/scripts/launch_titantpp_cnn_gru_components_v3.py": launcher.sha(tool)}
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

    def test_v1_v2_bytes_unchanged_and_v3_remote_helper_syntax_valid(self):
        project = Path(__file__).resolve().parents[2]
        self.assertEqual(launcher.sha(project / "paper/scripts/launch_titantpp_cnn_gru_components.py"),
            "85c11cc3f603b8bd456f5fcb62285e3fad5f6919209b79a419b2c1c58982147d")
        self.assertEqual(launcher.sha(project / "paper/scripts/launch_titantpp_cnn_gru_components_v2.py"),
            "2d1a6a11511f7f9df41b60402b9f6c124428d127c0ec67d21065b9fe95cbb78c")
        compile(launcher.REMOTE_SCRIPT, "v3_remote_helper_not_executed", "exec")

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

    def remote_health(self):
        return next(node for node in ast.parse(launcher.REMOTE_SCRIPT).body if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "mode" and any(isinstance(value, ast.Constant) and value.value == "health"
                for value in node.test.comparators))

    def lock_block(self, text, *, unreadable=False):
        statements = self.remote_health().body
        first = next(i for i, node in enumerate(statements) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "locks" for target in node.targets))
        last = next(i for i, node in enumerate(statements) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "probe_code" for target in node.targets))
        def path(name):
            if name == "/proc/locks":
                def read_text():
                    if unreadable:
                        raise PermissionError("read-only kernel lock inventory unavailable")
                    return text
                return SimpleNamespace(read_text=read_text)
            if name == "/var/lib/dpkg/lock":
                return SimpleNamespace(exists=lambda: True,
                    stat=lambda: SimpleNamespace(st_dev=os.makedev(8, 2), st_ino=123456))
            return SimpleNamespace(exists=lambda: False)
        namespace = {"pathlib": SimpleNamespace(Path=path), "os": os}
        code = compile(ast.Module(body=statements[first:last], type_ignores=[]), "v3_lock_read_only", "exec")
        with patch.object(os, "open", side_effect=AssertionError("Root package lock must never be opened")):
            exec(code, namespace)
        return namespace["locks"]

    def observer(self, *, argv=None, comm="unattended-upgr", uid="0\t0\t0\t0", state="S", ppid=1,
            changed_ticks=False, helper_changed=False, shutdown="b false", query_failed=False):
        expected = ["/usr/bin/python3", "/usr/share/unattended-upgrades/unattended-upgrade-shutdown", "--wait-for-signal"]
        ticks = [469, 470 if changed_ticks else 469]
        class Proc:
            name = "1514"
            def __truediv__(self, name):
                if name == "stat":
                    return SimpleNamespace(read_text=lambda: "1514 (x) " + state + " " + str(ppid) + " "
                        + "0 " * 17 + str(ticks.pop(0)))
                if name == "cmdline":
                    return SimpleNamespace(read_bytes=lambda: ("\x00".join(expected if argv is None else argv) + "\x00").encode())
                return SimpleNamespace(read_text=lambda: "Uid:\t" + uid)
        block = next(node for node in ast.walk(self.remote_health()) if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "comm")
        def capture(command):
            if query_failed:
                raise OSError("Read-only logind query failed")
            return shutdown
        namespace = {"p": Proc(), "comm": comm, "updates": [], "idle_shutdown_observers": [],
            "sha": lambda path: "changed" if helper_changed else "8fbf4e278c4ad457e82cc05586631b1a0550dd9d10892a70487ef74c623066e0",
            "capture": capture, "subprocess": launcher.subprocess}
        exec(compile(ast.Module(body=[block], type_ignores=[]), "v3_observer_read_only", "exec"), namespace)
        return namespace["updates"], namespace["idle_shutdown_observers"]

    def test_proc_inode_lock_check_is_read_only_and_fails_closed(self):
        self.assertEqual(self.lock_block("12: POSIX ADVISORY WRITE 123 08:02:123456 0 EOF"), ["/var/lib/dpkg/lock"])
        self.assertEqual(self.lock_block("12: FLOCK ADVISORY WRITE 123 08:03:123456 0 EOF"), [])
        self.assertEqual(self.lock_block("12: OFDLCK ADVISORY WRITE -1 08:02:123457 0 EOF"), [])
        with self.assertRaises(PermissionError):
            self.lock_block("", unreadable=True)

    def test_shutdown_observer_only_exact_identity_is_nonblocking(self):
        updates, idle = self.observer()
        self.assertEqual(updates, [])
        self.assertEqual(len(idle), 1)
        self.assertEqual(idle[0]["helper_source_sha256"], "8fbf4e278c4ad457e82cc05586631b1a0550dd9d10892a70487ef74c623066e0")
        for kwargs in ({"argv": ["/usr/bin/python3", "/usr/bin/unattended-upgrade"]}, {"comm": "apt-get"},
            {"uid": "1000\t1000\t1000\t1000"}, {"state": "R"}, {"ppid": 2}, {"changed_ticks": True},
            {"helper_changed": True}, {"shutdown": "b true"}, {"shutdown": "unresolved"}, {"query_failed": True}):
            updates, idle = self.observer(**kwargs)
            self.assertEqual(len(updates), 1, kwargs)
            self.assertEqual(idle, [], kwargs)

    def test_health_ast_matches_verified_compatibility_and_science_routes_unchanged(self):
        expected_health = next(node for node in ast.parse(observer_bridge.health_script()).body if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "mode" and any(isinstance(value, ast.Constant) and value.value == "health"
                for value in node.test.comparators))
        # The top-level health If also owns the other action branches in its
        # orelse. Compare health body only; deployment's inherited lease is a
        # deliberate separate administrative change tested below.
        self.assertEqual(ast.dump(ast.Module(body=self.remote_health().body, type_ignores=[]), include_attributes=False),
            ast.dump(ast.Module(body=expected_health.body, type_ignores=[]), include_attributes=False))
        old = {node.name: node for node in ast.parse(Path(original.__file__).read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        new = {node.name: node for node in ast.parse(Path(launcher.__file__).read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for name in ("authorization", "qualification_binding", "host_training_gate", "training_status", "run_mode",
            "SSHRemote", "validate_health", "decode_snapshot", "intent", "one_call", "expected_members", "inspect_archive"):
            self.assertEqual(ast.dump(old[name], include_attributes=False), ast.dump(new[name], include_attributes=False), name)

    def test_missing_or_altered_frozen_design_binding_blocks_before_ssh(self):
        self.release("5080")
        remote = fixtures.FakeRemote(self.c)
        for value in (None, "0" * 64):
            bad = deepcopy(self.c)
            if value is None:
                bad.pop("design_sha256")
            else:
                bad["design_sha256"] = value
            with self.assertRaises(ValueError):
                launcher.validate_contract(bad)
        self.assertFalse(remote.calls)

    def start_lease(self, *, host="5080", previous=None, now=1791794358.15949):
        c = deepcopy(self.c)
        if previous is not None:
            c["operational_replacement"] = {"host_leases": {host: previous}}
        approval = {"approved": True}
        namespace = {"c": c, "host": host, "digest": launcher.canonical(c), "approval": approval,
            "time": SimpleNamespace(time=lambda: now), "math": math, "canon": launcher.canonical, "need": launcher.require}
        node = next(node for node in ast.parse(launcher.REMOTE_SCRIPT).body if isinstance(node, ast.FunctionDef) and node.name == "start_lease")
        exec(compile(ast.Module(body=[node], type_ignores=[]), "v3_inherited_lease", "exec"), namespace)
        return namespace["start_lease"]()

    def test_old_5080_lease_is_preserved_and_not_extended_by_new_root(self):
        previous = {"started_at_unix": 1791794258.15949, "deadline_unix": 1792399058.15949,
            "source_start_permit_raw_sha256": "a" * 64}
        start = self.start_lease(previous=previous)
        self.assertEqual(start["started_at_unix"], previous["started_at_unix"])
        self.assertEqual(start["deadline_unix"], previous["deadline_unix"])
        self.assertEqual(start["inherited_host_lease"], previous)
        future = self.start_lease(host="5090")
        self.assertEqual(future["started_at_unix"], 1791794358.15949)
        self.assertEqual(future["deadline_unix"] - future["started_at_unix"], 168 * 3600)
        self.assertNotIn("inherited_host_lease", future)

    def test_expired_nonfinite_future_or_extended_inherited_lease_rejected(self):
        previous = {"started_at_unix": 1791794258.15949, "deadline_unix": 1792399058.15949}
        for bad in ({**previous, "started_at_unix": float("nan")}, {**previous, "deadline_unix": float("inf")},
            {**previous, "deadline_unix": previous["deadline_unix"] + 1},
            {"started_at_unix": 1791794359.15949, "deadline_unix": 1792399159.15949}):
            with self.assertRaises(ValueError):
                self.start_lease(previous=bad)
        with self.assertRaises(ValueError):
            self.start_lease(previous=previous, now=1792399058.15949)


if __name__ == "__main__":
    unittest.main()
