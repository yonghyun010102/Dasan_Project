#!/usr/bin/env python
"""Run inference for a pipeline variant and report the paper's metrics.

Detection is scored as an utterance-level binary decision (did the system touch
the utterance?) and correction with EM / Bal-EM / N-WER / E-WER / Bal-WER.
Metrics are computed on the deduplicated utterance-level split, while dialogue
context is reconstructed from the full dialogue partition.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcsc import config
from dcsc.data import KEY_COL, STT_COL, load_splits
from dcsc.pipeline import evaluate_pipeline, get_variant, run_pipeline

METRIC_ORDER = [
    ("precision", "Precision", 1), ("recall", "Recall", 1), ("f1", "F1", 1),
    ("accuracy", "Acc", 1), ("balanced_accuracy", "Bal-Acc", 1),
    ("exact_match", "EM", 1), ("balanced_exact_match", "Bal-EM", 1),
    ("normal_wer", "N-WER", 0), ("error_wer", "E-WER", 0),
    ("balanced_wer", "Bal-WER", 0),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="full", choices=["u", "s", "con_s", "full", "dcsc", "zero_rule"])
    p.add_argument("--backbone", default="pkot5", choices=["pkot5", "llama"])
    p.add_argument("--base_model", default=None)
    p.add_argument("--corrector_dir", default=None)
    p.add_argument("--detector_dir", default=None, help="required for --variant full")
    p.add_argument("--split_dir", default=config.SPLIT_DIR)
    p.add_argument("--dataset_dir", default=config.DATASET_DIR)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out_dir", default=None)
    p.add_argument("--run_root", default="runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context_n", type=int, default=config.CONTEXT_N)
    p.add_argument("--detector_threshold", type=float, default=config.DETECTOR["threshold"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--score_all", action="store_true",
                   help="score every utterance instead of the deduplicated subset")
    p.add_argument("--limit_dialogues", type=int, default=0, help="pilot mode: cap dialogues")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    utt = load_splits(args.split_dir, [args.split])[args.split]
    dlg = load_splits(args.dataset_dir, [args.split])[args.split]
    if args.limit_dialogues:
        keys = dlg[KEY_COL].unique()[: args.limit_dialogues]
        dlg = dlg[dlg[KEY_COL].isin(keys)].copy()
        utt = utt[utt[KEY_COL].isin(keys)].copy()

    # ------------------------------------------------ zero-rule reference
    if args.variant == "zero_rule":
        predictions = dlg.copy()
        predictions["final"] = predictions[STT_COL]
        predictions["detector_flag"] = 0
        predictions["prediction"] = ""
        out_dir = Path(args.out_dir or f"{args.run_root}/eval_zero_rule_{args.split}")
        variant_name = "zero_rule"
    else:
        variant = get_variant(args.variant)
        variant_name = variant.name
        if not args.corrector_dir:
            raise SystemExit("--corrector_dir is required")
        out_dir = Path(
            args.out_dir
            or f"{args.run_root}/eval_{args.backbone}_{variant_name}_seed{args.seed}_{args.split}"
        )

        detector_flags = None
        if variant.use_detector:
            if not args.detector_dir:
                raise SystemExit("--variant full requires --detector_dir")
            from dcsc.detector import DetectorPredictor

            predictor = DetectorPredictor(
                args.detector_dir, device=args.device,
                max_length=config.DETECTOR["max_length"], threshold=args.detector_threshold,
            )
            flags = predictor.predict_flags(dlg[STT_COL].tolist())
            detector_flags = {
                (k, int(i)): int(f)
                for k, i, f in zip(dlg[KEY_COL], dlg["utt_index"], flags)
            }
            fired = int(sum(detector_flags.values()))
            print(f"[evaluate] detector fired on {fired:,}/{len(dlg):,} utterances "
                  f"(threshold {args.detector_threshold})")
            del predictor

        if args.backbone == "pkot5":
            from dcsc.corrector_seq2seq import Seq2SeqCorrector

            corrector = Seq2SeqCorrector(
                args.corrector_dir, device=args.device,
                max_source_len=config.SEQ2SEQ["max_source_len"],
                max_new_tokens=config.SEQ2SEQ["generation_max_len"],
                num_beams=config.SEQ2SEQ["num_beams"], batch_size=args.batch_size,
            )
        else:
            from dcsc.corrector_llm import LLMCorrector

            corrector = LLMCorrector(
                args.corrector_dir, base_model=args.base_model or config.LLM_BASE,
                device=args.device, max_prompt_len=config.LLM["max_length"],
                batch_size=min(args.batch_size, config.LLM["gen_batch_size"]),
            )

        predictions = run_pipeline(
            dlg, variant, corrector, detector_flags=detector_flags,
            context_n=args.context_n, batch_size=args.batch_size,
        )

    metrics = evaluate_pipeline(predictions, score_subset=None if args.score_all else utt)
    metrics["variant"] = variant_name
    metrics["backbone"] = args.backbone if variant_name != "zero_rule" else "-"
    metrics["split"] = args.split
    metrics["seed"] = args.seed
    metrics["scored_on"] = "all" if args.score_all else "deduplicated"

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"\n=== {variant_name} / {metrics['backbone']} / {args.split} "
          f"(n={metrics['n_samples']:,}, scored on {metrics['scored_on']}) ===")
    print("  Detection : " + "  ".join(
        f"{label} {metrics[key]*100:6.2f}" for key, label, _ in METRIC_ORDER[:5]))
    print("  Correction: " + "  ".join(
        f"{label} {metrics[key]*100:6.2f}" for key, label, _ in METRIC_ORDER[5:]))
    print(f"\n[evaluate] metrics     -> {out_dir / 'metrics.json'}")
    print(f"[evaluate] predictions -> {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
