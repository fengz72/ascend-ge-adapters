"""StartLux-Decision v1.1 0.8B Instant: CPU business-semantics bridge (P2a).

This module is the thin integration point between this repository and the vendored,
pinned official package ``startlux_decide`` (see vendor/SOURCE.md, upstream commit
a99977d9332decc9fb27a2488b4a83b533b6cd5f).  It adds no decision logic of its own:
every prompt, control, temperature, None/wide-list rule and answer formula is the
official one.  What it provides:

* ``StartluxDecision``: one Instant model folder (``decision_config.json`` +
  ``config.json`` read by the official ``load_model_folder``) plus a tokenizer plus
  an injected read-out callback, wired into the official ``Decider``.
* ``ReadoutCallbackBackend``: the official ``Backend`` protocol with ``readout``
  answered by a caller-supplied callback ``readout(prompt, letter_ids) -> logits``
  (sync or async; only differences matter).  This is the seam the future NPU
  runtime plugs into: the graph outputs the 26 letter logits for a prompt, and the
  same callback can also drive online/offline multi-round decision orchestration
  later.  Instant models never generate, so ``generate`` is refused.

Scope: text decision semantics only (choice / noul / score, none_check, allow_none,
single-option, option order, wide-list knock-out, Instant think refusal,
request/question/read-out alignment).  No model-weight inference, no export, no
HTTP service: this bridge never runs a model forward.  The pinned real materials
live under work/startlux-materials (model folder revision 6496ae64..., official
code commit a99977d...): the real tokenizer, chat template, letter token ids
(32..57) and per-type temperatures of decision_config.json are covered by the
tests in tests/test_startlux_decision.py; the independent CPU FP32 HF oracle for
model-weight goldens is reference.py next to this file (P2b).

Images (the unified AIR route): ``decide`` / ``decide_async`` accept a
keyword-only ``images`` argument forwarded verbatim to the official
Decider.decide(images=...) -- the official place_images state layout, the
official media decode and every official decision rule stay in the official
package.  The DEFAULT backend constructed by this bridge
(ReadoutCallbackBackend) is text-only and still refuses media; image requests
require an image-capable Backend passed at construction -- the production one
is unified_readout.UnifiedReadoutBackend (wired by
unified_readout.startlux_unified_decision), whose readout(prompt, letter_ids,
media) receives the official Media object and encodes it through the official
preprocessing.  Callers never need to reach a private _decider for images.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

VENDOR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR) not in sys.path:               # make the pinned official package importable
    sys.path.insert(0, str(VENDOR))

from startlux_decide.backends.base import Backend, BackendRequestError    # noqa: E402
from startlux_decide.controls import Controls, RequestError               # noqa: E402
from startlux_decide.modelcfg import ModelFolder, load_model_folder       # noqa: E402
from startlux_decide.pipeline import Decision, Decider, default_controls  # noqa: E402

Readout = Callable[[str, Sequence[int]], Any]
"""readout(prompt, letter_ids) -> one logit/log-probability per letter id, in order.
Sync or async; a value's absolute level does not matter, only differences do."""


class ReadoutCallbackBackend(Backend):
    """A Backend whose read-outs come from the injected callback (the NPU/runtime seam).

    Text only: a call with images is refused. Instant models answer in one pass, so
    there is no generation primitive.
    """

    name = "readout-callback"

    def __init__(self, tokenizer: Any, readout: Readout, context_length: int | None = None):
        self.tokenizer = tokenizer
        self.readout_fn = readout          # not self.readout: that name is the Backend method
        self.context_length = context_length

    async def readout(self, prompt: str, letter_ids: Sequence[int], media: Any = None) -> list[float]:
        if media is not None:
            raise BackendRequestError("the P2a text bridge does not read images")
        z = self.readout_fn(prompt, tuple(letter_ids))
        if inspect.isawaitable(z):
            z = await z
        z = list(z)
        if len(z) != len(letter_ids):
            raise BackendRequestError(f"the readout callback returned {len(z)} values for "
                                      f"{len(letter_ids)} letters")
        return z

    async def generate(self, prompt: str, max_tokens: int, sampling: Any, stop: Sequence[str],
                       seed: int) -> Any:
        raise NotImplementedError("Instant models answer in one read-out and never generate")


class StartluxDecision:
    """The official Decider for one Instant folder, driven by an injected read-out callback.

    ``model`` is a model folder path (read with the official ``load_model_folder``:
    temperatures, letter token ids and wide-choice settings come from its
    decision_config.json, never hard-coded) or an already-loaded ``ModelFolder``.
    ``none_check`` defaults to True and ``allow_none`` to False, the official serve
    defaults.  ``body`` of ``decide`` carries the per-request control fields of
    /v1/systemone (think_mode, gate, mix, allow_none, none_check, return_details,
    ...) parsed by the official ``parse_controls``; a request that asks an Instant
    model to think (think_mode "auto" or "always") is refused with RequestError.
    """

    def __init__(self, model: str | Path | ModelFolder, tokenizer: Any, readout: Readout | None = None,
                 *, backend: Backend | None = None, none_check: bool = True, allow_none: bool = False,
                 context_length: int | None = None):
        folder = model if isinstance(model, ModelFolder) else load_model_folder(str(model))
        if not folder.bidirectional:
            raise ValueError(f"{folder.path}: this bridge is for Instant (prompt-bidirectional) folders; "
                             f"folder kind is {folder.kind!r}")
        if backend is None:
            if readout is None:
                raise ValueError("give either a readout callback or a full backend")
            backend = ReadoutCallbackBackend(tokenizer, readout, context_length=context_length)
        self.folder = folder
        self.backend = backend
        self.defaults = default_controls(folder, none_check=none_check, allow_none=allow_none or None)
        self._decider = Decider(folder, backend, self.defaults, context_length=context_length)

    # what the folder says (for verification and orchestration; not for re-deriving answers)
    @property
    def letter_ids(self) -> tuple:
        """The 26 letter token ids, checked against the tokenizer and decision_config.json."""
        return tuple(self._decider.letter_ids)

    @property
    def temperature_by_type(self) -> dict:
        return dict(self.folder.temperature_by_type)

    def temperature(self, question_type: str) -> float:
        return self.folder.temperature(question_type)

    def controls(self, body: dict) -> Controls:
        return self._decider.controls(body)

    async def decide_async(self, state: Any, questions: Any, body: dict | None = None,
                           *, images: Any = None) -> Decision:
        """One /v1/systemone request on the event loop (the official Decider path).

        ``images`` (keyword-only, default None): the request's images -- base64
        strings or data URIs, placed into the state by the official
        images.place_images, exactly what the official Decider.decide(images=...)
        accepts.  It is forwarded verbatim; no prompt, control, temperature or
        grouping logic is reimplemented here.  The DEFAULT backend of this
        bridge is the text-only ReadoutCallbackBackend, which still REFUSES
        media; image requests need an image-capable Backend injected at
        construction (the unified AIR's UnifiedReadoutBackend in
        unified_readout.py, whose readout(prompt, letter_ids, media) receives
        the official Media object the official pipeline decoded).  With
        images=None the behavior is byte-identical to the previous text-only
        bridge.
        """
        ctl = self._decider.controls(body) if body is not None else self.defaults
        return await self._decider.decide(state, questions, ctl, images=images)

    def decide(self, state: Any, questions: Any, body: dict | None = None,
               *, images: Any = None) -> Decision:
        """One /v1/systemone request, synchronous (a fresh event loop per call; use
        decide_async from inside a running loop).  ``images``: see decide_async."""
        return asyncio.run(self.decide_async(state, questions, body=body,
                                             images=images))
