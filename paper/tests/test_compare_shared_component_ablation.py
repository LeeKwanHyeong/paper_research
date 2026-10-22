"""Pure local tests for the component-ablation comparison helper."""
import gzip
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

from paper.scripts import compare_shared_component_ablation as c


class CompareTests(unittest.TestCase):
    def test_prediction_metrics_recomputes_overall_large_and_time_nll(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.jsonl.gz"
            rows = [
                {"split": "test", "target_index": 0, "entity_id": "a", "target_seq": 1, "raw_quantity": 1., "predicted_raw_quantity": 3., "time_nll": .5},
                {"split": "test", "target_index": 1, "entity_id": "b", "target_seq": 2, "raw_quantity": 10., "predicted_raw_quantity": 7., "time_nll": 1.5},
            ]
            with gzip.open(path, "wt") as stream:
                for row in rows: stream.write(json.dumps(row) + "\n")
            value = c.prediction_metrics(path, 10., "test")
            self.assertEqual(value["overall"]["count"], 2)
            self.assertAlmostEqual(value["overall"]["qty_mae"], 2.5)
            self.assertAlmostEqual(value["overall"]["qty_rmse"], (6.5) ** .5)
            self.assertEqual(value["large"]["count"], 0)
            self.assertIsNone(value["large"]["qty_mae"])
            self.assertAlmostEqual(value["overall"]["time_nll"], 1.)

    def test_prediction_metrics_rejects_split_drift_and_empty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.jsonl.gz"
            with gzip.open(path, "wt") as stream: stream.write(json.dumps({"split": "validation", "target_index": 0, "entity_id": "a", "target_seq": 1, "raw_quantity": 1., "predicted_raw_quantity": 1., "time_nll": 1.}) + "\n")
            with self.assertRaisesRegex(ValueError, "split"):
                c.prediction_metrics(path, 1., "test")
            with gzip.open(path, "wt") as stream: pass
            with self.assertRaisesRegex(ValueError, "Empty"):
                c.prediction_metrics(path, 1., "test")

    def test_prediction_metrics_rejects_duplicate_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.jsonl.gz"; row = {"split": "test", "target_index": 0, "entity_id": "a", "target_seq": 1, "raw_quantity": 1., "predicted_raw_quantity": 1., "time_nll": 1.}
            with gzip.open(path, "wt") as stream: stream.write(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "identity/order"):
                c.prediction_metrics(path, 1., "test")

    def test_mapping_rejects_paths_outside_retrieved_host_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "run").mkdir(); (root / "run/a.json").write_text("x")
            self.assertEqual(c.mapped(root, "/remote/root", "/remote/root/run/a.json"), (root / "run/a.json").resolve())
            with self.assertRaisesRegex(ValueError, "escaped"):
                c.mapped(root, "/remote/root", "/elsewhere/a.json")

    def test_cached_report_formats_are_pure(self):
        result = {"table": [{"dataset": "d", "split": "test", "arm": "full", "overall": {"qty_rmse": {"mean": 1., "sample_sd_ddof1": .1, "n": 3}, "qty_mae": {"mean": 2., "sample_sd_ddof1": .2, "n": 3}, "time_nll": {"mean": 3., "sample_sd_ddof1": .3, "n": 3}}, "large": {"qty_rmse": {"mean": None, "sample_sd_ddof1": None, "n": 0}, "qty_mae": {"mean": None, "sample_sd_ddof1": None, "n": 0}, "time_nll": {"mean": None, "sample_sd_ddof1": None, "n": 0}}}]}
        self.assertIn("Shared15", c.render_markdown(result))
        self.assertIn("overall.qty_rmse", c.render_csv(result))
        adverse = {**result, "adverse_seed_results": [{"dataset": "d", "split": "test", "seed": 42, "comparison": "full-no_history", "scope": "overall", "metric": "qty_rmse", "left": 2.0, "right": 1.0, "absolute": 1.0, "relative": 1.0}]}
        self.assertIn("full-no_history", c.render_markdown(adverse))

    def test_collection_rejects_tar_receipt_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); contract = {"source": {"files": {"s": "x"}}, "input_files": {"i": "y"}, "legacy_files": {}}
            for host in ("5080", "5090"):
                host_dir = root / "test_cost" / host; originals = host_dir / "originals"; originals.mkdir(parents=True)
                (originals / "test_cost.complete.json").write_text("{}")
                (originals / "test_cost.supervisor_exit.json").write_text(json.dumps({"actual_os_returncode": 0}))
                files = {}
                for name in ("test_cost.complete.json", "test_cost.supervisor_exit.json"):
                    item = originals / name; files[name] = {"sha256": c.sha(item), "bytes": item.stat().st_size}
                manifest = {"files": files, "source_files": contract["source"]["files"], "input_files": contract["input_files"], "legacy_files": {}}
                manifest_path = originals / "collection_test_cost.manifest.json"; manifest_path.write_text(json.dumps(manifest))
                archive = host_dir / "collection_test_cost.tar.gz"
                with tarfile.open(archive, "w:gz") as packed:
                    packed.add(manifest_path, arcname="collection_test_cost.manifest.json")
                    for name in files: packed.add(originals / name, arcname=name)
                (host_dir / "pack.stdout").write_text("ok")
                (host_dir / "transport.json").write_text(json.dumps({"actual_os_returncode": 0, "stdout_sha256": c.sha(host_dir / "pack.stdout")}))
                (host_dir / "collection_receipt.json").write_text(json.dumps({"status": "complete", "host": host, "phase": "test_cost", "contract_sha256": c.canonical(contract), "archive_sha256": c.sha(archive), "archive_bytes": archive.stat().st_size, "manifest_sha256": c.sha(manifest_path)}))
            with patch("paper.scripts.advance_shared_component_ablation.validate_collection", side_effect=lambda _c, _host, _phase, folder, **_kwargs: Path(folder) / "originals"):
                self.assertEqual(set(c.verify_collection(root, contract)), {"5080", "5090"})
            receipt = root / "test_cost/5080/collection_receipt.json"; value = json.loads(receipt.read_text()); value["archive_sha256"] = "0" * 64; receipt.write_text(json.dumps(value))
            with patch("paper.scripts.advance_shared_component_ablation.validate_collection", side_effect=ValueError("Collection archive bytes or SHA differ")):
                with self.assertRaisesRegex(ValueError, "archive"):
                    c.verify_collection(root, contract)


if __name__ == "__main__": unittest.main()
