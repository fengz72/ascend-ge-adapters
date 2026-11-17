"""Media placeholders in a System One request (prompt-rendering slice of
``flide_onejev/media.py``).

Upstream: flide 0.1.16 ``flide_onejev/media.py``.  This slice keeps the two
pieces the input generator reaches -- splitting a user turn at its
``<image:N>`` / ``<video:N>`` placeholders and chat-rendering one branch with
one vision block per item -- byte-for-byte.  Removed: reference decoding
(``open_image``), video loading, size limits and the ``processor_kwargs``
helper; the generator decodes local fixture paths itself.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

PLACEHOLDER = re.compile(r"<(image|video):(\d+)>")


class MediaError(ValueError):
    pass


def content_parts(text: str, media: Sequence[dict]) -> list[dict]:
    """Split a user-turn text at the media placeholders into chat content parts."""
    parts, pos, seen = [], 0, []
    for m in PLACEHOLDER.finditer(text):
        if m.start() > pos:
            parts.append({"type": "text", "text": text[pos:m.start()]})
        kind, n = m.group(1), int(m.group(2))
        if n < 1 or n > len(media) or media[n - 1]["type"] != kind:
            raise MediaError(f"placeholder {m.group(0)} does not match media {n} of {len(media)}")
        parts.append({"type": kind})
        seen.append(n)
        pos = m.end()
    if pos < len(text):
        parts.append({"type": "text", "text": text[pos:]})
    if seen != list(range(1, len(media) + 1)):
        raise MediaError(f"media must be referenced once each, in order, by <image:N> / <video:N>; saw {seen} for "
                         f"{len(media)} items")
    return parts


def render_prompt(processor, template_kwargs: dict, messages: list[dict], media: Sequence[dict]) -> str:
    """Chat-rendered prompt with one vision block per media item (pads are expanded later by the processor). Without
    media the text is exactly the text-only prompt."""
    user = messages[-1]["content"]
    msgs = messages[:-1] + [{"role": "user", "content": content_parts(user, media) if media else user}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **template_kwargs)


def check_items(media: Any) -> list[dict]:
    if media is None:
        return []
    if not isinstance(media, list):
        raise MediaError("media must be a list of {type: image | video, ...} items")
    out = []
    for i, m in enumerate(media):
        if not isinstance(m, dict) or m.get("type") not in ("image", "video"):
            raise MediaError(f"media[{i}] must be an object with type image or video")
        if m["type"] == "image" and not any(k in m for k in ("path", "url", "data", "image")):
            raise MediaError(f"media[{i}] (image) needs one of path, url or data")
        if m["type"] == "video" and not (isinstance(m.get("frames"), list) and m["frames"]):
            raise MediaError(f"media[{i}] (video) needs a non-empty frames list")
        out.append(m)
    return out
