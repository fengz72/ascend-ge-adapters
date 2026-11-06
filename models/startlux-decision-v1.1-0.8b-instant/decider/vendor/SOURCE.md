# Vendored source: startlux_decide

Upstream: https://github.com/StartLuxLabs/StartLux-Decision
Commit:   a99977d9332decc9fb27a2488b4a83b533b6cd5f (pinned; every blob hash re-verified at vendor time)
License:  Apache-2.0 (see LICENSE, the upstream file, unchanged)

This directory holds unmodified, byte-identical copies of the upstream modules that
carry the text-decision business semantics used by ../decision.py:

    startlux_decide/__init__.py
    startlux_decide/prompt.py        the trained prompt format and the tokenizer check
    startlux_decide/modelcfg.py      decision_config.json / config.json reading and validation
    startlux_decide/readout.py       softmax, confidences, answers, None split, wide-list knock-out
    startlux_decide/pipeline.py      the Decider: request -> questions -> read-outs -> answers
    startlux_decide/controls.py      per-request controls (think_mode, allow_none, none_check, ...)
    startlux_decide/loops.py         thought loop detector (imported by pipeline.py)
    startlux_decide/images.py        image slots in evidence (imported by prompt.py; text-only here)
    startlux_decide/backends/__init__.py
    startlux_decide/backends/base.py          the Backend protocol (readout/generate primitives)
    startlux_decide/backends/fake.py          FakeTokenizer/FakeBackend, the official test tokenizer
    startlux_decide/backends/transformers.py  the official in-process PyTorch reference engine (P2b)
    startlux_decide/backends/bidirectional.py the official Instant prompt-bidirectional attention patch (P2b)

NOT vendored (not needed for the CPU business layer or the CPU FP32 reference; they pull in
aiohttp/vllm/mlx or belong to serving, launching, benchmarking and quantization):
server.py, serving.py, cli.py, launch.py, bench.py, evaluate.py, quantize.py,
vllm_plugin.py, backends/{http,llamacpp,mlx,vllm}.py,
and the docs/ tree of the upstream repository.

Do not modify anything in this directory: ../decision.py is the only supported
integration point, and upgrades happen by re-vendoring a new pinned commit.
