#!/usr/bin/env python
"""Publish the released checkpoints to the Hugging Face Hub.

Model weights are too large for GitHub, so the detector, the pkoT5 corrector and
the Llama LoRA adapter are distributed as Hub repositories. Requires
``huggingface_hub`` and a token with write access (``huggingface-cli login``).

Example
-------
    python scripts/upload_checkpoints.py --org my-org \
        --detector  runs/detector_seed42/best \
        --pkot5     runs/corrector_pkot5_full_seed42/best \
        --llama     runs/corrector_llama_full_seed42/best
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CARD = """---
language: ko
license: mit
tags:
  - korean
  - asr-post-editing
  - error-correction
  - dcsc
---

# {name}

{role} of **DCSC** (Detector-Gated Contextual Span Correction), the Korean ASR
post-editing framework introduced in *"Leveraging Fine-grained Error Correction
in Korean Speech Recognition for Consultation Services"* (Engineering
Applications of Artificial Intelligence).

* base model: `{base}`
* trained on: DasanCallDial (dialogue-level split 0.80/0.05/0.15, `split_seed=77`)
* training seed: `{seed}`

Usage, training code and evaluation scripts: <{repo_url}>

```bash
python scripts/evaluate.py --variant full --backbone {backbone} \\
    --detector_dir <detector snapshot> --corrector_dir <corrector snapshot>
```

{metrics_block}
"""


def _metrics_block(path: Path) -> str:
    candidates = list(path.parent.glob("../eval_*/metrics.json")) + list(
        path.parent.glob("val_metrics.json")
    )
    for candidate in candidates:
        try:
            data = json.loads(Path(candidate).read_text(encoding="utf-8"))
        except Exception:
            continue
        keys = ["f1", "balanced_accuracy", "exact_match", "balanced_exact_match",
                "normal_wer", "error_wer", "balanced_wer"]
        rows = [f"| {k} | {data[k]*100:.2f} |" for k in keys if k in data]
        if rows:
            return "## Test metrics\n\n| metric | value (%) |\n|---|---|\n" + "\n".join(rows)
    return ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--org", required=True, help="Hub user or organisation")
    p.add_argument("--detector", default=None)
    p.add_argument("--pkot5", default=None)
    p.add_argument("--llama", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo_url", default="https://github.com/<user>/dcsc")
    p.add_argument("--prefix", default="dcsc")
    p.add_argument("--private", action="store_true")
    p.add_argument("--card", default=None,
                   help="path to a model card you wrote yourself; used verbatim for every repo")
    p.add_argument("--overwrite_card", action="store_true",
                   help="regenerate README.md from the template even if one already exists")
    p.add_argument("--write_card_only", action="store_true",
                   help="write the generated card into the checkpoint dir and exit (nothing is uploaded)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    targets = []
    if args.detector:
        targets.append((args.detector, f"{args.prefix}-koelectra-detector",
                        "Token-level error detector", "monologg/koelectra-base-v3-discriminator", "pkot5"))
    if args.pkot5:
        targets.append((args.pkot5, f"{args.prefix}-pkot5-corrector",
                        "Span-level corrector", "paust/pko-t5-base", "pkot5"))
    if args.llama:
        targets.append((args.llama, f"{args.prefix}-llama31-lora-corrector",
                        "Span-level corrector (LoRA adapter)", "meta-llama/Llama-3.1-8B-Instruct", "llama"))
    if not targets:
        raise SystemExit("nothing to upload: pass at least one of --detector/--pkot5/--llama")

    if not args.dry_run:
        from huggingface_hub import HfApi

        api = HfApi()

    for local, name, role, base, backbone in targets:
        local_path = Path(local)
        if not local_path.is_dir():
            raise SystemExit(f"not a directory: {local_path}")
        repo_id = f"{args.org}/{name}"
        card_path = local_path / "README.md"
        if args.card:
            card = Path(args.card).read_text(encoding="utf-8")
            source = f"your file ({args.card})"
        elif card_path.exists() and not args.overwrite_card:
            card = card_path.read_text(encoding="utf-8")
            source = f"existing {card_path} (left untouched)"
        else:
            card = CARD.format(name=name, role=role, base=base, seed=args.seed,
                               backbone=backbone, repo_url=args.repo_url,
                               metrics_block=_metrics_block(local_path))
            source = "generated from the built-in template"
        print(f"[upload] {local_path}  ->  https://huggingface.co/{repo_id}")
        print(f"[upload] model card: {source}")
        if args.dry_run:
            print(card)
            continue
        if args.card or not card_path.exists() or args.overwrite_card:
            card_path.write_text(card, encoding="utf-8")
        if args.write_card_only:
            print(f"[upload] card written to {card_path}; edit it and re-run without --write_card_only")
            continue
        api.create_repo(repo_id, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(local_path), repo_id=repo_id)
        print(f"[upload] done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
