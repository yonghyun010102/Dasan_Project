"""Default hyper-parameters, taken from the paper (Sec. 5.1.1, App. D)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Repository root, so the defaults below work from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- data
DATASET_DIR = str(REPO_ROOT / "data" / "dasancalldial")  # released, dialogue-level, partitioned
SPLIT_DIR = str(REPO_ROOT / "data" / "splits")           # generated: deduplicated utterance level
XLSX = str(REPO_ROOT / "data" / "call_center5.xlsx")     # optional source workbook
SPLIT_SEED = 77            # fixed so every training seed sees the same partition
TEST_SIZE = 0.20           # 20 % hold-out ...
VAL_SHARE_OF_HOLDOUT = 0.25  # ... of which 25 % is validation -> 0.80/0.05/0.15
CONTEXT_N = 10             # up to 10 preceding utterances (Sec. 4.2.2)

# ---------------------------------------------------------------- detector
DETECTOR_BASE = "monologg/koelectra-base-v3-discriminator"
DETECTOR = dict(
    max_length=128,
    epochs=15,
    lr=2e-5,
    batch_size=24,
    eval_batch_size=24,
    lambda_error=8.0,     # weight on the erroneous-token class
    threshold=0.5,        # utterance is forwarded if any token fires
    label_all_tokens=True,  # every sub-token of a word carries the word label
)

# ---------------------------------------------------------------- correctors
SEQ2SEQ_BASE = "paust/pko-t5-base"
SEQ2SEQ = dict(
    epochs=15,
    lr=1e-4,
    weight_decay=1e-3,
    batch_size=24,
    eval_batch_size=24,
    max_source_len=512,
    max_target_len=128,
    generation_max_len=128,
    num_beams=1,          # greedy decoding
)

LLM_BASE = "meta-llama/Llama-3.1-8B-Instruct"
LLM = dict(
    epochs=6,
    lr=1e-4,
    per_device_batch_size=3,
    grad_accum=8,         # effective batch size 24
    max_length=768,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bf16=True,
    gradient_checkpointing=True,
    val_max_samples=300,  # generation-based validation is expensive for an 8B model
    gen_batch_size=8,
)

TRAIN_SEEDS = (42, 43, 44)


@dataclass
class RunPaths:
    """Canonical on-disk layout for one (variant, backbone, seed) run."""
    root: str = str(REPO_ROOT / "runs")
    variant: str = "full"
    backbone: str = "pkot5"
    seed: int = 42

    @property
    def detector_dir(self) -> str:
        return f"{self.root}/detector_seed{self.seed}"

    @property
    def corrector_dir(self) -> str:
        return f"{self.root}/corrector_{self.backbone}_{self.variant}_seed{self.seed}"

    @property
    def eval_dir(self) -> str:
        return f"{self.root}/eval_{self.backbone}_{self.variant}_seed{self.seed}"
