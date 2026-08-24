"""Dialogue-level context augmentation (paper Sec. 4.2.2).

The corrector never sees the whole dialogue. Instead the target utterance is
prefixed with up to ``context_n`` preceding utterances, separated by two marker
strings so the model can tell reference history from the sentence to repair::

    [Dialogue Context]
    <speaker>: <utterance>
    ...
    [Target Utterance]
    <speaker>: <utterance>

During **training** the history comes from the ground truth; during
**inference** the ground-truth history is unavailable, so it is filled
autoregressively with the model's own previously corrected utterances.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

CONTEXT_MARKER = "[대화 문맥]"       # [Dialogue Context]
TARGET_MARKER = "[수정될 문장]"      # [Target Utterance]
EMPTY_CONTEXT = "(없음)"             # (none)

SPEAKER_LABELS = {"tx": "상담원", "rx": "고객"}  # counselor / customer

__all__ = [
    "CONTEXT_MARKER", "TARGET_MARKER", "EMPTY_CONTEXT",
    "normalize_speaker", "format_line", "build_corrector_input", "ContextBuffer",
]


def normalize_speaker(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return SPEAKER_LABELS.get(str(value).strip().lower())


def format_line(text: str, speaker: Optional[str] = None) -> str:
    text = str(text).strip()
    label = normalize_speaker(speaker)
    return f"{label}: {text}" if label else text


def build_corrector_input(
    current_line: str,
    context_lines: Iterable[str] = (),
    use_context: bool = True,
) -> str:
    """Assemble the corrector input.

    With ``use_context=False`` the raw utterance is returned, which is what the
    context-free ablations (``u``, ``s``) consume.
    """
    current = str(current_line).strip()
    if not use_context:
        return current
    lines = [str(x).strip() for x in context_lines if str(x).strip()]
    context = "\n".join(lines) if lines else EMPTY_CONTEXT
    return f"{CONTEXT_MARKER}\n{context}\n{TARGET_MARKER}\n{current}"


class ContextBuffer:
    """Rolling window over the most recent ``context_n`` utterances of a dialogue."""

    def __init__(self, context_n: int = 10):
        self.context_n = int(context_n)
        self._lines: List[str] = []

    def window(self) -> List[str]:
        if self.context_n <= 0:
            return []
        return self._lines[-self.context_n :]

    def append(self, text: str, speaker: Optional[str] = None) -> None:
        self._lines.append(format_line(text, speaker))

    def reset(self) -> None:
        self._lines.clear()
