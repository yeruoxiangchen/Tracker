from __future__ import annotations


def strict_decision_exit_code(passed: bool) -> int:
    """Return the process status used by automated experimental gates."""

    return 0 if bool(passed) else 2
