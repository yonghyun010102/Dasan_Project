#!/usr/bin/env python
"""Train the span-level corrector for any pipeline variant (paper Sec. 4.2-4.3).

``--variant full`` reproduces DCSC: the detector is applied to the training set
first so that its false positives are folded into the corrector's training data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcsc import config
from dcsc.data import GOLD_COL, STT_COL, load_splits
from dcsc.pipeline import (
    build_corrector_frame,
    curate_detector_gated_frame,
    get_variant,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="full", choices=["u", "s", "con_s", "full", "dcsc"])
    p.add_argument("--backbone", default="pkot5", choices=["pkot5", "llama"])
    p.add_argument("--base_model", default=None, help="override the HF model id")
    p.add_argument("--split_dir", default=config.SPLIT_DIR)
    p.add_argument("--dataset_dir", default=config.DATASET_DIR)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--run_root", default="runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context_n", type=int, default=config.CONTEXT_N)
    p.add_argument("--corrector_no_dedup", action="store_true",
                   help="train the corrector on the FULL (non-deduplicated) partition. The paper's "
                        "utterance-level dataset is deduplicated (default), but repeated domain errors "
                        "are then seen only once; this flag reproduces the alternative.")
    p.add_argument("--detector_dir", default=None, help="required for --variant full")
    p.add_argument("--detector_threshold", type=float, default=config.DETECTOR["threshold"])
    p.add_argument("--curation", default="fp_aware", choices=["fp_aware", "balanced", "none"],
                   help="how the corrector's training set is built for --variant full. 'fp_aware' "
                        "keeps all true errors and fills the error-free half with the detector's "
                        "false positives (paper Sec. 4.3); 'balanced' samples that half at random; "
                        "'none' skips curation and trains on the whole partition, leaving the "
                        "detector as an inference-time gate only (no --detector_dir needed here).")
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--val_max_samples", type=int, default=None)
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--keep_intermediate", action="store_true",
                   help="keep the Trainer checkpoints dir (default: delete after saving best)")
    p.add_argument("--limit_train", type=int, default=0, help="pilot mode: cap training rows")
    p.add_argument("--limit_val", type=int, default=0, help="pilot mode: cap validation rows")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    variant = get_variant(args.variant)
    out_dir = Path(
        args.out_dir
        or f"{args.run_root}/corrector_{args.backbone}_{variant.name}_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    utt = load_splits(args.split_dir, ["train", "val"])
    dlg = load_splits(args.dataset_dir, ["train", "val"])
    if args.corrector_no_dedup:
        utt = {name: dlg[name].copy() for name in ("train", "val")}
        print("[corrector] training on the NON-deduplicated partition "
              f"(train={len(utt['train']):,}, val={len(utt['val']):,})")

    frames = {}
    for name in ("train", "val"):
        frames[name] = build_corrector_frame(
            utt[name], dlg[name], variant, context_n=args.context_n
        )

    curate = variant.use_detector and args.curation != "none"
    if variant.use_detector and not curate:
        print("[corrector] --curation none: training on the whole partition "
              f"(train={len(frames['train']):,}, val={len(frames['val']):,}); "
              "the detector gates at inference only")
    if curate:
        if not args.detector_dir:
            raise SystemExit("--variant full requires --detector_dir unless --curation none")
        from dcsc.detector import DetectorPredictor

        predictor = DetectorPredictor(
            args.detector_dir, device=args.device,
            max_length=config.DETECTOR["max_length"], threshold=args.detector_threshold,
        )
        before = {k: len(v) for k, v in frames.items()}
        for name in ("train", "val"):
            flags = predictor.predict_flags(frames[name][STT_COL].tolist())
            frames[name] = curate_detector_gated_frame(frames[name], flags, seed=args.seed,
                                                       strategy=args.curation)
        print(f"[corrector] detector-gated curation: "
              f"train {before['train']:,} -> {len(frames['train']):,}, "
              f"val {before['val']:,} -> {len(frames['val']):,}")
        del predictor

    train_df, val_df = frames["train"], frames["val"]
    if args.limit_train:
        train_df = train_df.head(args.limit_train)
    if args.limit_val:
        val_df = val_df.head(args.limit_val)

    print(f"[corrector] variant={variant.name} ({variant.description})")
    print(f"[corrector] backbone={args.backbone} seed={args.seed} "
          f"train={len(train_df):,} val={len(val_df):,}")

    if args.backbone == "pkot5":
        from dcsc.corrector_seq2seq import train_seq2seq_corrector

        cfg = dict(config.SEQ2SEQ)
        if args.epochs is not None: cfg["epochs"] = args.epochs
        if args.lr is not None: cfg["lr"] = args.lr
        if args.batch_size is not None:
            cfg["batch_size"] = args.batch_size
            cfg["eval_batch_size"] = args.batch_size
        metrics = train_seq2seq_corrector(
            train_df, val_df, out_dir,
            base_model=args.base_model or config.SEQ2SEQ_BASE,
            stt_col=STT_COL, gold_col=GOLD_COL, seed=args.seed,
            save_total_limit=args.save_total_limit,
            val_max_samples=args.val_max_samples or 0,
            keep_intermediate=args.keep_intermediate,
            **cfg,
        )
    else:
        from dcsc.corrector_llm import train_llm_corrector

        cfg = dict(config.LLM)
        if args.epochs is not None: cfg["epochs"] = args.epochs
        if args.lr is not None: cfg["lr"] = args.lr
        if args.batch_size is not None: cfg["per_device_batch_size"] = args.batch_size
        if args.val_max_samples is not None: cfg["val_max_samples"] = args.val_max_samples
        metrics = train_llm_corrector(
            train_df, val_df, out_dir,
            base_model=args.base_model or config.LLM_BASE,
            stt_col=STT_COL, gold_col=GOLD_COL, seed=args.seed,
            save_total_limit=args.save_total_limit,
            **cfg,
        )

    with open(out_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"[corrector] best checkpoint -> {out_dir / 'best'}")


if __name__ == "__main__":
    main()
