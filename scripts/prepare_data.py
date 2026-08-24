#!/usr/bin/env python
"""Build the DasanCallDial splits and report the dataset statistics.

Dialogue-level group split (0.80/0.05/0.15) followed by utterance-level
deduplication inside each partition, exactly as described in Sec. 3.1.3.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcsc import config
from dcsc.data import build_splits, deduplicate, load_splits, save_splits, split_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default=config.DATASET_DIR,
                        help="released dialogue-level partition (default input)")
    parser.add_argument("--xlsx", default=None,
                        help="re-derive the partition from the source workbook instead")
    parser.add_argument("--out_dir", default=config.SPLIT_DIR)
    parser.add_argument("--split_seed", type=int, default=config.SPLIT_SEED)
    parser.add_argument("--test_size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--val_share_of_holdout", type=float, default=config.VAL_SHARE_OF_HOLDOUT)
    parser.add_argument("--no_dedup", action="store_true", help="keep duplicate utterances")
    args = parser.parse_args()

    if args.xlsx:
        dialogue_level, utterance_level = build_splits(
            args.xlsx,
            split_seed=args.split_seed,
            test_size=args.test_size,
            val_share_of_holdout=args.val_share_of_holdout,
            dedup=not args.no_dedup,
        )
        save_splits(dialogue_level, Path(args.dataset_dir))
        print(f"[prepare_data] re-derived the partition -> {args.dataset_dir}")
    else:
        dialogue_level = load_splits(args.dataset_dir)
        utterance_level = {
            name: (deduplicate(part) if not args.no_dedup else part)
            for name, part in dialogue_level.items()
        }

    out_dir = Path(args.out_dir)
    save_splits(utterance_level, out_dir)

    stats = split_statistics(dialogue_level, utterance_level)
    stats["config"] = {
        "source": str(args.xlsx) if args.xlsx else str(args.dataset_dir),
        "split_seed": args.split_seed,
        "test_size": args.test_size,
        "val_share_of_holdout": args.val_share_of_holdout,
        "dedup": not args.no_dedup,
    }
    with open(out_dir / "split_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[prepare_data] dataset (dialogue level) <- {args.dataset_dir}")
    print(f"[prepare_data] deduplicated splits     -> {out_dir}")
    print("\nDialogue level")
    print(f"  {'split':<8}{'#dialogues':>12}{'#utterances':>13}{'avg turns':>11}")
    for name in ("train", "val", "test", "total"):
        row = stats["dialogue_level"][name]
        print(f"  {name:<8}{row['n_dialogues']:>12,}{row['n_utterances']:>13,}{row['avg_turns']:>11.2f}")
    print("\nUtterance level (deduplicated)")
    print(f"  {'split':<8}{'#samples':>12}{'avg chars':>11}{'error %':>10}")
    for name in ("train", "val", "test", "total"):
        row = stats["utterance_level"][name]
        print(f"  {name:<8}{row['n_samples']:>12,}{row['avg_chars']:>11.2f}{row['error_rate']:>10.2f}")
    print("\nSpeaker breakdown (total, deduplicated)")
    for speaker, row in stats["speaker"]["total"].items():
        label = {"tx": "counselor", "rx": "customer"}.get(speaker, speaker)
        print(f"  {label:<10}{row['n_samples']:>10,}  error {row['error_rate']:>6.2f}%")


if __name__ == "__main__":
    main()
