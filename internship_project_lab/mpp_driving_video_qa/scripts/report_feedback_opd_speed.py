#!/usr/bin/env python3
"""Read sampled phase timings from every rank, without importing the model stack."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--windows', type=int, default=20)
    args = parser.parse_args()
    ranks, phases = {}, defaultdict(lambda: defaultdict(list))
    for path in sorted(args.output_dir.glob('perf_rank_*.jsonl')):
        rows = []
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A concurrently appended last line may be incomplete.
        rows = rows[-args.windows:]
        if not rows:
            continue
        rank = rows[-1]['rank']
        ranks[str(rank)] = {'samples':len(rows), 'latest_step':rows[-1]['optimizer_step'],
                            'median_profiled_update_seconds':statistics.median(r['update_wall_ms'] for r in rows)/1000}
        for row in rows:
            for name, value in row['phases'].items():
                phases[name]['wall'].append(value['wall_ms'])
                if value['gpu_ms'] is not None:
                    phases[name]['gpu'].append(value['gpu_ms'])
    result = {'output_dir':str(args.output_dir.resolve()), 'rank_timings':ranks,
              'phases':{k:{'mean_wall_ms_per_update':statistics.mean(v['wall']),
                           'mean_gpu_ms_per_update':statistics.mean(v['gpu']) if v['gpu'] else None}
                        for k,v in sorted(phases.items())},
              'note':'Sampled phases can overlap host/GPU work; do not add wall and GPU times. Rank update times include waiting for other ranks.'}
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
