# Vendored source: startlux_decide

Upstream: https://github.com/StartLuxLabs/StartLux-Decision
Commit:   a99977d9332decc9fb27a2488b4a83b533b6cd5f (pinned; every blob hash re-verified at vendor time)
License:  Apache-2.0 (see LICENSE, the upstream file, unchanged)

This directory holds unmodified, byte-identical copies of the upstream modules that
carry the text-decision business semantics.  The whole pinned subset is kept even
though the standalone flow reaches only part of it, so the vendored set stays a
faithful slice of one upstream commit and re-vendoring a new commit stays mechanical.

Reached by the current standalone flow (`run.sh`):

    startlux_decide/__init__.py
    startlux_decide/images.py        image slots / load_media (generate_inputs.py, unified_preprocess.py)
    startlux_decide/prompt.py        the trained prompt format (generate_inputs.py; imports images.py)
    startlux_decide/backends/__init__.py
    startlux_decide/backends/base.py          the Backend protocol (imported by backends/__init__.py)
    startlux_decide/backends/bidirectional.py the official Instant prompt-bidirectional attention patch (golden.py)

Not reached by the current flow (retained as part of the pinned subset):

    startlux_decide/modelcfg.py      decision_config.json / config.json reading and validation
    startlux_decide/readout.py       softmax, confidences, answers, None split, wide-list knock-out
    startlux_decide/pipeline.py      the Decider: request -> questions -> read-outs -> answers
    startlux_decide/controls.py      per-request controls (think_mode, allow_none, none_check, ...)
    startlux_decide/loops.py         thought loop detector (imported by pipeline.py)
    startlux_decide/backends/fake.py          FakeTokenizer/FakeBackend, the official test tokenizer
    startlux_decide/backends/transformers.py  the official in-process PyTorch reference engine

NOT vendored (they pull in aiohttp/vllm/mlx or belong to serving, launching,
benchmarking and quantization):
server.py, serving.py, cli.py, launch.py, bench.py, evaluate.py, quantize.py,
vllm_plugin.py, backends/{http,llamacpp,mlx,vllm}.py,
and the docs/ tree of the upstream repository.

Do not modify anything in this directory: the supported integration points are the
imports listed above (`generate_inputs.py`, `model/unified_preprocess.py`,
`decider/golden.py`), and upgrades happen by re-vendoring a new pinned commit.
