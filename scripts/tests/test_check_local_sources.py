"""Local checker tests use fabricated source and a temporary Git repository."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "local_checker", Path(__file__).resolve().parents[1] / "check_local_sources.py"
)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class LocalSourceChecks(unittest.TestCase):
    def test_compilation_never_executes_source_or_writes_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("raise RuntimeError('must not execute')\n")
            self.assertEqual(CHECKER.check_paths([source], root)["status"], "passed")
            self.assertEqual(list(root.iterdir()), [source])

    def test_invalid_return_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.py"
            source.write_text("return 1\n")
            result = CHECKER.check_paths([source], root)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["errors"][0]["path"], "bad.py")

    def test_symlink_outside_repository_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "repo"
            root.mkdir()
            outside = parent / "outside.py"
            outside.write_text("x = 1\n")
            source = root / "link.py"
            source.symlink_to(outside)
            self.assertEqual(CHECKER.check_paths([source], root)["status"], "failed")

    def test_discovery_preserves_ignored_and_historical_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(
                    ["git", *args], cwd=root, stderr=subprocess.STDOUT
                )

            git("init", "-q")
            (root / ".gitignore").write_text("ignored.py\n")
            (root / "scripts").mkdir()
            tracked = root / "scripts" / "tracked.py"
            tracked.write_text("x = 1\n")
            git("add", ".gitignore", "scripts/tracked.py")
            git(
                "-c",
                "user.name=Local Test",
                "-c",
                "user.email=local@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            )
            tracked.write_text("x = 2\n")
            (root / "scripts" / "new file.py").write_text("x = 3\n")
            (root / "scripts" / "ignored.py").write_text("invalid syntax\n")
            (root / "reports").mkdir()
            (root / "reports" / "historical.py").write_text("invalid syntax\n")
            expected = ["scripts/new file.py", "scripts/tracked.py"]
            for changed in (False, True):
                self.assertEqual(
                    [
                        str(p.relative_to(root))
                        for p in CHECKER.source_paths(root, changed=changed)
                    ],
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
