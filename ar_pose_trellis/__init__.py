from __future__ import annotations

import sys
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
if str(RECONVIAGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(RECONVIAGEN_ROOT))
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
if str(VGGT_WHEEL_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_WHEEL_ROOT))

from .condition import ARDinoRayCond
from .pipeline import TrellisARPoseTo3DPipeline

__all__ = ["ARDinoRayCond", "TrellisARPoseTo3DPipeline"]
