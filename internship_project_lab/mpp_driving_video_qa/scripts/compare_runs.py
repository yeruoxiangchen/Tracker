#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read(path):
    return {row["question_id"]: row for row in
            (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base, candidate = read(args.baseline), read(args.candidate)
    common = sorted(set(base) & set(candidate))
    rows = []
    for qid in common:
        old = base[qid]; new = candidate[qid]
        row = {"question_id": qid, "question": new["question"],
               "references": new["references"],
               "baseline_prediction": old["prediction"],
               "candidate_prediction": new["prediction"]}
        for metric in ("normalized_exact", "token_f1", "lingo_judge_correct"):
            if metric in old and metric in new:
                row[metric + "_baseline"] = old[metric]
                row[metric + "_candidate"] = new[metric]
                if isinstance(old[metric], bool):
                    row[metric + "_change"] = int(new[metric]) - int(old[metric])
                else:
                    row[metric + "_change"] = new[metric] - old[metric]
        rows.append(row)
    summary = {"records": len(rows)}
    for metric in ("normalized_exact", "token_f1", "lingo_judge_correct"):
        changes = [x[metric + "_change"] for x in rows if metric + "_change" in x]
        if changes:
            summary[metric] = {"fixed": sum(x > 0 for x in changes),
                               "regressed": sum(x < 0 for x in changes),
                               "net_change_mean": sum(changes) / len(changes)}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2,
                                 ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

