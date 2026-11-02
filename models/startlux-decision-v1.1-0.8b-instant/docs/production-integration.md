# StartLux unified ONE-AIR production integration (P6 production)

Status: delivered and formally accepted (P7, 2026-10-31) -- the main
functionality is green.  The two items that stayed open after P7 have since
CLOSED: the B4 long [1,65,129,2048] resource corner that real-ran OOM on the
shared card was re-verified on an idle card by P8 (2026-11-01) with a fresh
GESession, 4/4 rows passed (min cosine 0.9999947830951117 / max relL2
0.0040037066209951165; the original OOM evidence is kept as history), and
performance (P5) was formally measured and accepted by the P8 single-instance
warm benchmark.  Current authoritative status and numbers:
[../validation.md](../validation.md) and [../performance.md](../performance.md).
This note documents the production thin bridge for the StartLux-Decision
v1.1 0.8B Instant unified multimodal ONE-AIR route.  The real unified-AIR GE
execution has since been formally accepted on NPU (Ascend910_9382): the six
regular CLI stages (data/baseline/prepare/build/infer/compare) on the FORMAL
models/startlux-decision-v1.1-0.8b-instant/config/model.yaml, the formal AIR
work/startlux-decision/build/model.air (1,907,534,423 bytes, sha256
c1f8816ee25e03119b2522e55adb49ebf42a88692791b179920f5f60221edba1), the
boundary matrix (18 cases / 27 rows within the gate; the B4 long OOM corner as
above) and the 13-request/17-readout business acceptance -- the business gate
is GE vs the matched-dtype official BF16 reference (33 comparisons passed,
max prob err 0.009854891487914852 / max relL2 0.009469937773648804,
work/startlux-cannbot/review-p7-business-independent.json), with GE-vs-FP32
same-prompt rows 16/16 and the wide_27 last-round cross-dtype divergence kept
disclosed (full-FP32 business is NOT claimed).  Evidence:
work/startlux-cannbot/review-p7-cli-independent.json,
review-p7-air-structure-independent.json, review-p7-boundary-independent.json,
review-p7-business-independent.json,
work/startlux-validation/p7-final/.  Performance numbers are in
[../performance.md](../performance.md) (P8, the single formal benchmark).

## What was delivered (all under models/startlux-decision-v1.1-0.8b-instant/)

| file | role |
|---|---|
| `unified_adapter.py` | the production framework adapter: `describe` (= multimodal.describe + the accepted pad-256 encode policy), `prepare_candidate` (shared `build_unified_model` + explicit `.to(source_device)` + per-name mixed-dtype/device verification), `prepare_reference` (independent official CPU FP32 oracle), `prepare_runtime` (official-preprocess encode, provenance-free decode) |
| `unified_preprocess.py` | the official CPU preprocessing seam: official tokenizer/processor composition (AutoImageProcessor(backend="pil") + AutoTokenizer + AutoVideoProcessor -- `Qwen3VLProcessor` rejects `video_processor=None`), official `encode_media_row` recipe, shared official rope provider, strict payload/bounds validation, the accepted `pad_graph_inputs` min-physical-width helper |
| `unified_readout.py` | the business read-out seam: `UnifiedGraphSpec` contract, `UnifiedReadoutSession` (one cold `ge_runtime` subprocess per read-out, raw bins kept, timeout/returncode/one-GE_RESULT enforced, honest cold-cost records), `UnifiedReadoutBackend` (official Backend, `reads_images`, Media objects, letter-column mapping), `startlux_unified_decision` public factory |
| `generate_unified_cases.py` | the unified case pool: 7 pre-encoded P2b text read-outs FIRST, then real image cases (red / blue / two-image / mixed text+image B2) on self-regenerating deterministic 256x256 fixtures; official prompt rendering; prompt+paths payloads, no pixel JSON, no goldens |
| `unified_verify.py` | the business verifier: core CPU baseline `golden.npz` + per-row `rows_meta` active-K/temperature gate (prob<=0.01, top1 when reference margin>0.02), missing references always FAIL |
| `decision.py` | (sanctioned minimal extension) `decide`/`decide_async` gained keyword-only `images=None`, forwarded verbatim to the official `Decider.decide(images=...)`; text behavior byte-identical, default callback still text-only |
| `tests/test_startlux_unified_*.py` | 5 files, 91 CPU tests in the current tree (the P6 handoff baseline ran 90; re-counted and re-run green 2026-10-31, see below) |
| `work/startlux-validation/p6-production/` | candidate YAML, fixtures, golden, logs |

## How to run (the formal canonical flow)

Environment: source the formal scoped env
`work/startlux-validation/p7-final/env.sh` (the P7 acceptance env; that file
is gitignored, so the equivalent rebuildable snippet is below -- nothing is
system-installed, and the task-local Pillow/torchvision are PREPENDED so the
CANN/A3 PYTHONPATH entries such as TBE and site-packages survive):

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source models/qwen2.5-0.5b/scripts/env_a3.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export STARTLUX_MODEL_DIR="$PWD/work/startlux-materials/model-6496ae641cc71b149ce9dad69bd97f2a44e7bf11"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-1}
# 1 = the npu-smi "info -m" Chip Logic ID (board 0 chip 1, logical device 1 --
# NOT a "board 1" card number); the selected chip is exposed to this process
# as runtime device 0.  Configurable per card availability.
export PYTHONPATH="$PWD/work/startlux-validation/p6-official/deps/pillow:$PWD/work/startlux-validation/p6-official/deps/torchvision:${PYTHONPATH:-}"
mkdir -p work/tmp && export TMPDIR="$PWD/work/tmp"
```

The formal config (the one canonical config since P7) and the correct six
stages; the canonical chunk-proto load happens exactly once, AFTER prepare:

```bash
CFG=models/startlux-decision-v1.1-0.8b-instant/config/model.yaml
python3 -m core.pipeline data     --config $CFG --target Ascend910_9382
python3 -m core.pipeline baseline --config $CFG --target Ascend910_9382
python3 -m core.pipeline prepare  --config $CFG --target Ascend910_9382
source models/startlux-decision-v1.1-0.8b-instant/scripts/env_chunk_proto.sh \
  work/startlux-decision
python3 -m core.pipeline build    --config $CFG --device 0
python3 -m core.pipeline infer    --config $CFG --device 0
python3 -m core.pipeline compare  --config $CFG --target Ascend910_9382
```

Do NOT source older task envs that already carry a different compat.so
(e.g. the p6-unified env): double REG_OP registration of two prototype builds
is a known crash; only the single canonical load above is valid.

Historical note (P6 handoff, CPU-only, kept for provenance): the handoff ran
`data` (11 cases / 12 samples) and `baseline` (11-case `golden.npz` by the
independent official FP32 oracle) on the candidate YAML
`work/startlux-validation/p6-production/model.yaml` (evidence in its
`model-work/results/`); the 7 text-case goldens were BIT-EXACT with the
accepted P2b fixture oracle (maxabs 0.0) and the image cases answered
semantically through the real official recipe (red-dominant -> option A,
blue-dominant -> option C).  The handoff draft's Ascend950PR_958b target was
an unvalidated placeholder -- no NPU command was ever run against a 950
device; the formal target is Ascend910_9382 as above.

The business read-out (official Decider on the exported AIR):

```python
from unified_readout import (UnifiedReadoutSession, UnifiedGraphSpec,
                             read_air_io, startlux_unified_decision)
io = read_air_io("runtime/build/ge_runtime", air, log)
spec = UnifiedGraphSpec.from_io(io, max_seq_len=2048)
session = UnifiedReadoutSession(device, run_dir, air_path=air, spec=spec,
                                model_dir=model_folder)   # min_width=256 default
decision, backend = startlux_unified_decision(model_folder, session)
decision.decide(state, questions, images=[data_uri])       # the public entry
```

## Verified on CPU (P6 handoff record; re-checked 2026-10-31)

- 91/91 unified tests green in the current tree (`python3 -m unittest` over
  the five `tests/test_startlux_unified_*.py` files; the P6 handoff baseline
  ran 90, the 2026-10-31 re-count is 91 -- part of the 13-module / 317-test
  independent record in `work/startlux-cannbot/review-p7-unittest-final.log`),
  including:
  - REAL official processor composition in a subprocess with the validated
    task-local Pillow 13 / torchvision 0.27 (64 image tokens per officially
    processed 256x256 image, [256,1536] pixel block per image, grid [1,16,16],
    strict cap/grid rejections, `video_processor=None` TypeError documented);
  - the public factory image path end-to-end (real tokenizer + real processor +
    real official Decider + real Media objects, marker letter-column precision
    at the official temperature, wide-list knock-out re-sending media every
    round, two-image 128-token requests, Instant generate refusal, default
    text callback refusing real media);
  - the accepted pad-256 policy: official encode at TRUE lengths, true-prefix
    bit-identity, 0/-1 pad region, true lengths/pixels untouched, floor capped
    by max_seq_len, L>=256 at true length, mixed-batch deterministic rule
    (evidence: work/startlux-cannbot/review-p4-pad256-independent.json);
  - the independent reference vs the validated base P2b handle (bit-exact text
    path on tiny models AND bit-exact on the real 0.8B baseline above);
  - the candidate placement contract: per-name mixed dtypes (vision F32 /
    trunk BF16 / A_log F32 / head26 F32), all params+buffers on the source
    device after the explicit `.to`, stray and unknown-parameter detection;
  - the business verifier on the REAL golden + REAL case pool: identical
    candidate passes, a 0.9-logit perturbation fails exactly that case.
- Real CPU pipeline stages (P6 handoff): `data` + `baseline` on the candidate YAML; since P7 all six stages run on the formal config above.

## Current limits and open points

1. Real unified-AIR GE execution: DONE in the formal P7 acceptance (six CLI
   stages, boundary matrix, 13-request/17-readout business on the formal AIR --
   see the status header).  `encode_policy.real_ge_validated_here` no longer
   hardcodes a validation status (it is null; the evidence pointers in
   `encode_policy.evidence` are authoritative).  Performance numbers: measured
   and accepted by P8 (2026-11-01 single-instance warm benchmark, see
   [../performance.md](../performance.md)); the earlier "P5 open" was stage
   history.
2. The mixed static-pixel + dynamic-text IO export through
   `core.exporters.torchair` (static pixel_values [512,1536] alongside dynamic
   B/S): DONE and formally accepted.  The formal ONE AIR embeds exactly this
   contract (single graph: input_ids / position_ids / actual_seq_lengths /
   image_feature_index dynamic, pixel_values [512,1536] f32 STATIC,
   `work/startlux-cannbot/review-p7-air-structure-independent.json`); the
   exporter's example_inputs path is CPU-gated by
   `tests/test_torchair_example_inputs.py` (12 tests, part of the 317 record)
   and the six formal CLI stages above built and ran it
   (`review-p7-cli-independent.json`).
3. Vision inv_freq / device buffers inside `build_unified_model`: DONE and
   accepted.  The non-persistent rotary inv_freq is regenerated with the
   official constructor (the core load path's final `model.to(dtype)`
   quantizes it irrecoverably, see
   `work/startlux-cannbot/review-vision-rotary-buffer-cast.json`), the
   official constant buffers are computed on a CPU float32 copy, and the
   wrapper's device-only `.to(original_source_device)` keeps every parameter
   and buffer on the source device with the mixed dtypes intact (report
   fields `rotary_inv_freq_restored` / `buffers_on_source_device`); covered by
   the unified placement tests (part of the 91/317 record) and by the P7
   formal acceptance on NPU (single-graph GE, boundary matrix, business
   read-outs).
4. The business read-out session serves one request (B1) per graph call; batch
   business read-outs stay the pipeline path (the AIR itself stays dynamic
   B1..4).
5. `bench` counts in the candidate YAML are placeholders (1 instance / 1
   iteration) -- no performance claim; the session's records report cold
   compile/load costs per read-out instead of warm QPS.
6. Wide-list media rounds: each knock-out round re-sends the full media
   (Instant folders have no prefix cache); this is the official protocol, not
   an optimization.
