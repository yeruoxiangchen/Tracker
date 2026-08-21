"""Read-only static preflight for the isolated strict with-VGGT runtime."""

from __future__ import annotations

import json

from .runtime import validate_runtime_sources


def main() -> None:
    identity = validate_runtime_sources()
    print(
        json.dumps(
            {
                "passed": True,
                "runtime": identity,
                "cuda_executed": False,
                "next_gate": "8-object cache then 1/2/8-GPU CUDA smoke",
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
