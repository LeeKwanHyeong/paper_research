from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, artifact_record, write_json


def main() -> None:
    data_root = ROOT / "data"
    artifacts = []
    for path in sorted(candidate for candidate in data_root.rglob("*") if candidate.is_file()):
        record = artifact_record(path)
        record["relative_path"] = str(path.relative_to(ROOT))
        artifacts.append(record)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts),
        "total_size_bytes": sum(item["size_bytes"] for item in artifacts),
        "artifacts": artifacts,
    }
    output = ROOT / "manifests" / "data_file_inventory.json"
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
