"""S15d: hash the formal prediction cache for external archive verification.

The analysis/report bundle is small and excludes the 459 MB run cache.  This
manifest lets a recipient verify a separately transferred cache archive before
rerunning s15.  It does not copy or publish the cache.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import vision_data as vd  # noqa: E402
import vision_runs as vr  # noqa: E402

ROOT = HERE / "results" / "visual_alarm_v1"
FORMAL = ROOT / "formal"


def main():
    files = []
    roles = {}
    for ds in ("ntu2012", "modelnet40"):
        base = FORMAL / ds / "batch01"
        for role in ("ref", "cal", "eval"):
            folder = base / role
            paths = sorted(folder.glob("*.npz"))
            roles[f"{ds}/{role}"] = len(paths)
            for p in paths:
                files.append({
                    "path": p.relative_to(ROOT).as_posix(),
                    "bytes": p.stat().st_size,
                    "sha256": vd.sha256_file(p),
                })
        summary = base / "reference_summary.npz"
        files.append({
            "path": summary.relative_to(ROOT).as_posix(),
            "bytes": summary.stat().st_size,
            "sha256": vd.sha256_file(summary),
        })
    manifest = {
        "protocol_sha256": vd.sha256_file(ROOT / "protocol_frozen.json"),
        "n_files": len(files),
        "total_bytes": int(sum(f["bytes"] for f in files)),
        "roles": roles,
        "files": files,
        "note": ("Prediction cache is not included in the analysis/report "
                 "package. Verify an externally transferred cache against "
                 "these hashes, then rerun s15 with the frozen protocol."),
    }
    path = ROOT / "cache_manifest.json"
    vr._write_json_atomic(path, manifest)
    print(json.dumps({"path": str(path), "n_files": len(files),
                      "total_bytes": sum(f["bytes"] for f in files),
                      "roles": roles}, indent=2))


if __name__ == "__main__":
    main()
