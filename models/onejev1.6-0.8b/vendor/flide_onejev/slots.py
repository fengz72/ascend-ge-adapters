"""Answer-slot token verification (the ``OneJev._verify_slots`` logic of
``flide_onejev/format.py``, extracted into a standalone function).

Upstream: flide 0.1.16.  The body is kept verbatim apart from taking the
tokenizer as an argument instead of ``self`` and returning the id list
explicitly; the all-or-nothing two-letter rule and its warning are unchanged.
"""
from __future__ import annotations

import logging

from .prompt import LETTERS, SLOT_LABELS

log = logging.getLogger("flide.onejev")


def verify_slot_token_ids(tokenizer) -> list[int]:
    """Slot token ids for every label the tokenizer holds as one round-trip token. A..Z are required. The
    two-letter labels are all-or-nothing: a tokenizer that splits any of them keeps 26 slots, and choices
    with more options go through the chunked path."""

    def one(label):
        enc = tokenizer.encode(label, add_special_tokens=False)
        return enc[0] if len(enc) == 1 and tokenizer.decode(enc) == label else None

    ids = [one(l) for l in LETTERS]
    if any(i is None for i in ids):
        bad = [l for l, i in zip(LETTERS, ids) if i is None]
        raise ValueError(f"answer slots {bad} are not single round-trip tokens")
    extra = [one(l) for l in SLOT_LABELS[len(LETTERS):]]
    if all(i is not None for i in extra):
        ids += extra
    else:
        log.warning("tokenizer splits some two-letter labels; %d slots, larger choices are chunked", len(ids))
    if len(set(ids)) != len(ids):
        raise ValueError("answer slot tokens collide")
    return ids
