#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); output = Path(args.output_dir)
    summary = output / "summary.json"; config = output / "run_config.json"
    progress = output / "progress.json"
    if summary.is_file():
        print(summary.read_text(encoding="utf-8")); return
    failures = sorted(output.glob("failure_rank_*.json"))
    newest_failure = max(failures, key=lambda path: path.stat().st_mtime) if failures else None
    if newest_failure and (not progress.is_file() or
                           newest_failure.stat().st_mtime >= progress.stat().st_mtime):
        print(newest_failure.read_text(encoding="utf-8")); return
    if progress.is_file():
        print(progress.read_text(encoding="utf-8")); return
    state = {"status": "initializing_or_not_started", "output_dir": str(output.resolve())}
    if config.is_file():
        run = json.loads(config.read_text(encoding="utf-8"))
        state.update(status="running_or_interrupted", records=run.get("records"),
                     optimizer_steps_total=run.get("total_steps"))
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
