#!/usr/bin/env python
"""Train the token-level ASR error detector (paper Sec. 4.1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcsc import config
from dcsc.data import load_splits
from dcsc.detector import train_detector


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split_dir", default=config.SPLIT_DIR)
    p.add_argument("--out_dir", default=None, help="default: runs/detector_seed<seed>")
    p.add_argument("--run_root", default="runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_model", default=config.DETECTOR_BASE)
    p.add_argument("--max_length", type=int, default=config.DETECTOR["max_length"])
    p.add_argument("--epochs", type=float, default=config.DETECTOR["epochs"])
    p.add_argument("--lr", type=float, default=config.DETECTOR["lr"])
    p.add_argument("--batch_size", type=int, default=config.DETECTOR["batch_size"])
    p.add_argument("--eval_batch_size", type=int, default=config.DETECTOR["eval_batch_size"])
    p.add_argument("--lambda_error", type=float, default=config.DETECTOR["lambda_error"])
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no_label_all_tokens", action="store_true",
                   help="supervise only the first sub-token of each word (default: all sub-tokens)")
    p.add_argument("--metric_for_best", default="f1",
                   choices=["f1", "utt_f1", "utt_recall", "utt_precision", "utt_balanced_accuracy"],
                   help="validation metric for checkpoint selection. Validation F1 is nearly flat "
                        "across epochs while precision/recall moves a lot; the detector reported in "
                        "the paper sits at a high-recall point, reproduced with utt_recall.")
    p.add_argument("--keep_intermediate", action="store_true",
                   help="keep the Trainer checkpoints dir (default: delete after saving best)")
    p.add_argument("--limit_train", type=int, default=0, help="pilot mode: cap training rows")
    p.add_argument("--limit_val", type=int, default=0, help="pilot mode: cap validation rows")
    args = p.parse_args()

    out_dir = Path(args.out_dir or f"{args.run_root}/detector_seed{args.seed}")
    splits = load_splits(args.split_dir, ["train", "val"])
    train_df, val_df = splits["train"], splits["val"]
    if args.limit_train:
        train_df = train_df.head(args.limit_train)
    if args.limit_val:
        val_df = val_df.head(args.limit_val)

    print(f"[detector] base={args.base_model} seed={args.seed} "
          f"train={len(train_df):,} val={len(val_df):,} lambda={args.lambda_error}")
    metrics = train_detector(
        train_df, val_df, out_dir,
        base_model=args.base_model, seed=args.seed, max_length=args.max_length,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, lambda_error=args.lambda_error,
        save_total_limit=args.save_total_limit, fp16=args.fp16,
        keep_intermediate=args.keep_intermediate,
        label_all_tokens=not args.no_label_all_tokens,
        metric_for_best=args.metric_for_best,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"[detector] best checkpoint -> {out_dir / 'best'}")
    print(f"[detector] validation utterance F1 = {metrics.get('eval_utt_f1', float('nan')):.4f}")


if __name__ == "__main__":
    main()
