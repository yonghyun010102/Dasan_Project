#!/usr/bin/env python
"""Aggregate evaluation runs into the mean +/- std table reported in the paper."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

METRICS = [
    ("precision", "Precision", True), ("recall", "Recall", True), ("f1", "F1", True),
    ("accuracy", "Acc", True), ("balanced_accuracy", "Bal-Acc", True),
    ("exact_match", "EM", True), ("balanced_exact_match", "Bal-EM", True),
    ("normal_wer", "N-WER", False), ("error_wer", "E-WER", False),
    ("balanced_wer", "Bal-WER", False),
]


def mean_std(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_root", default="runs")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None, help="write the summary as JSON")
    args = p.parse_args()

    groups = defaultdict(dict)
    for path in sorted(Path(args.run_root).glob("eval_*/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("split") != args.split:
            continue
        key = (data.get("backbone", "-"), data.get("variant", "?"))
        groups[key][int(data.get("seed", -1))] = data

    if not groups:
        raise SystemExit(f"no metrics.json found under {args.run_root}/eval_*/ for split={args.split}")

    summary = {}
    header = f"{'backbone':<9}{'variant':<9}{'seeds':<8}" + "".join(f"{label:>18}" for _, label, _ in METRICS)
    print(header)
    print("-" * len(header))
    for (backbone, variant), per_seed in sorted(groups.items()):
        seeds = sorted(per_seed)
        row = f"{backbone:<9}{variant:<9}{','.join(map(str, seeds)):<8}"
        entry = {"seeds": seeds, "metrics": {}}
        for key, label, higher_is_better in METRICS:
            values = [per_seed[s][key] * 100 for s in seeds if key in per_seed[s]]
            mean, std = mean_std(values)
            best = (max(values) if higher_is_better else min(values)) if values else float("nan")
            best_seed = None
            if values:
                pick = best
                for s in seeds:
                    if key in per_seed[s] and abs(per_seed[s][key] * 100 - pick) < 1e-9:
                        best_seed = s
                        break
            entry["metrics"][key] = {
                "mean": mean, "std": std, "best": best, "best_seed": best_seed,
                "per_seed": {str(s): per_seed[s].get(key) for s in seeds},
            }
            row += f"{mean:>11.2f}±{std:<6.2f}"
        print(row)
        summary[f"{backbone}/{variant}"] = entry

    # which seed to release: lowest Bal-WER
    print("\nbest seed per configuration (by Bal-WER):")
    for name, entry in summary.items():
        info = entry["metrics"].get("balanced_wer", {})
        if info.get("best_seed") is not None:
            print(f"  {name:<18} seed {info['best_seed']}  Bal-WER {info['best']:.2f}")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[aggregate] summary -> {args.out}")


if __name__ == "__main__":
    main()
