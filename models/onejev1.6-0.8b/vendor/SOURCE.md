# Vendored source: flide OneJev prompt slice

Upstream: https://github.com/OmniJev/Flide
Package:  `flide==0.1.16` (the published wheel, unpacked)
License:  Apache-2.0 (see LICENSE, the wheel's `licenses/LICENSE`, unchanged)

This directory holds the minimal audit surface needed to render OneJev
`onejev-labels-v2` prompts and read the calibrated answer slots.  Everything
that carries prompt or calibration semantics is copied verbatim from the
0.1.16 wheel; only packaging glue and validation were reshaped (below).

Extracted files:

    flide/__init__.py                   packaging stub (upstream __init__ pulls the serving runtime)
    flide/api/__init__.py               packaging stub
    flide/api/schema.py                 NoulQuestion / ChoiceQuestion / ScoreQuestion, NoulCriteria, Entry, Question
    flide_onejev/__init__.py            packaging stub (upstream __init__ imports format.py)
    flide_onejev/prompt.py              verbatim: SYSTEM_PROMPT, LETTERS, SLOT_LABELS, PromptStyle, DEFAULT_STYLE,
                                        render_entry, render_state, render_question, state_block, build_messages
    flide_onejev/calibrate.py           verbatim: Calibration, softmax
    flide_onejev/media.py               content_parts, render_prompt, check_items (verbatim bodies); PLACEHOLDER,
                                        MediaError
    flide_onejev/slots.py               OneJev._verify_slots body, extracted as verify_slot_token_ids(tokenizer)

Reshaped (no behavioural change for prompt rendering):

- `flide/api/schema.py`: upstream is pydantic v2.  The vendored copy is a
  keyword-only dataclass tree with the same field names/order, defaults,
  option limits, entry union and the two `__post_init__` checks (choice has at
  least one and at most 256 options; score has at least one and at most 256
  levels).  The pydantic dependency and the unanswered SystemOneResponse
  types are not vendored.
- `flide_onejev/slots.py`: `_verify_slots` moved out of the `OneJev` class so
  the input generator can derive the A..Z ids without the format engine.
- `flide_onejev/media.py`: only the two prompt-rendering functions and
  `check_items` are kept.  Removed reference decoding (`open_image`), video
  loading, `settings.max_media_bytes` limits and `processor_kwargs`, all of
  which belong to serving / media admission rather than prompt semantics.

Not vendored (deliberately): `TokenCache`, branch planning, chunked choices,
permutation debiasing, the GPU fast path (`_DeferredResize`, `_qwen_images`,
`_pixel_scale`), `SlotHead`, format/backbone registration, HTTP server and CLI.

Extraction date: 2026-11-17.

Do not modify anything in this directory.  The supported integration points are
`flide_onejev.prompt`, `flide_onejev.calibrate`, `flide_onejev.media` and
`flide_onejev.slots`; upgrades happen by re-vendoring a new pinned wheel.
