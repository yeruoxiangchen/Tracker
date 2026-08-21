#!/usr/bin/env python3
"""Verify migrated result trees really bind SS30K, SLat30K and ReconViaGen."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from manual_mesh_reconstruction.defaults import (
    SLAT30K_CHECKPOINT,
    SS30K_CHECKPOINT,
)
from manual_mesh_reconstruction.migrate_existing_outputs import (
    CANONICAL,
    DEFAULT_DESTINATION,
)


def main() -> None:
    root = DEFAULT_DESTINATION.resolve(strict=True)
    records = []
    for name in CANONICAL:
        directory = root / name
        json_files = sorted(directory.rglob("*.json"))
        texts = [path.read_text(encoding="utf-8", errors="strict") for path in json_files]
        ss = sum(SS30K_CHECKPOINT.sha256 in text for text in texts)
        slat = sum(SLAT30K_CHECKPOINT.sha256 in text for text in texts)
        no_vggt = sum(
            '"vggt_model_executed": false' in text.lower() for text in texts
        )
        recon = sum(
            any(
                marker in text
                for marker in (
                    '"method": "reconviagen_original"',
                    '"reconviagen_endpoint"',
                    '"reconviagen_mesh"',
                )
            )
            for text in texts
        )
        if min(ss, slat, no_vggt, recon) <= 0:
            raise RuntimeError(
                f"migrated result identity incomplete: {name} "
                f"ss={ss} slat={slat} no_vggt={no_vggt} recon={recon}"
            )
        records.append(
            {
                "name": name,
                "json_file_count": len(json_files),
                "ss30k_checkpoint_binding_file_count": ss,
                "slat30k_checkpoint_binding_file_count": slat,
                "no_vggt_execution_binding_file_count": no_vggt,
                "strict_reconviagen_result_file_count": recon,
            }
        )
    report = {
        "format": "manual_mesh_reconstruction.existing_output_model_identity.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "root": str(root),
        "ss30k_checkpoint_sha256": SS30K_CHECKPOINT.sha256,
        "slat30k_checkpoint_sha256": SLAT30K_CHECKPOINT.sha256,
        "directory_count": len(records),
        "directories": records,
    }
    destination = root / "MODEL_IDENTITY_AUDIT.json"
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": True, "report": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
