"""DCSC — Detector-Gated Contextual Span Correction for Korean ASR post-editing.

Reference implementation for the paper
*"Leveraging Fine-grained Error Correction in Korean Speech Recognition for
Consultation Services"* (Engineering Applications of Artificial Intelligence).
"""
__version__ = "1.0.0"

from .edits import (  # noqa: F401
    ARROW_TOKEN,
    NO_ERROR,
    SEP_TOKEN,
    apply_spans,
    make_span_target,
    word_error_mask,
    word_wer,
)
from .pipeline import VARIANTS, get_variant  # noqa: F401
