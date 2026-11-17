# OneJev1.6-0.8B fixtures

Fourteen System One requests and the screenshot-style images they reference.
`model/generate_inputs.py` turns every row into one B=1 graph-input case; the
images are generated, not downloaded, and are reproducible with
`fixtures/make_images.py`.

## Row format (this adapter's simplification)

One JSON object per line:

```json
{
  "state": {"task": "Pay the open invoice from ACME", "screen": "<image:1>"},
  "media": [{"type": "image", "path": "images/invoice-512.png"}],
  "questions": {"next": {"type": "choice", "instructions": "...",
                         "criteria": {"click": "click an element", "stop": "stop"}}}
}
```

- `state` — a JSON object; it may be a plain string in the official API, but
  every fixture here is an object.  The image placeholder `<image:1>` sits in a
  state field (here `screen`), exactly where the official examples put it.
- `media` — exactly one image per row in this v1 flow.  `path` is relative to
  this directory (the directory containing `cases.jsonl`), not an absolute path
  and not a base64 `data:` URI.  Other official fields (`url`, `data`, `image`)
  are not accepted by the strict row validator.
- `questions` — exactly one question per row (v1 limit), keyed by its qid.
  - `choice`: `criteria` is an object of 2..26 options; keys are the answer
    labels, values are the option descriptions (string or null).
  - `noul`: `criteria` is optional (`true` / `false` descriptions).
  - `score`: `criteria` is a list of levels; the fixtures use 3..5.
- The prompt is rendered with the vendored official `onejev-labels-v2`
  template: `build_messages(state, render_question(qid, question).suffix)`,
  then the chat template with `add_generation_prompt=True,
  enable_thinking=False`.

The only intentional difference from an official flide request is `media`:
official requests carry base64 `data:` URIs (or URLs), these fixtures carry a
relative file path so the repository stays small and diffable.  The path is
decoded to an RGB PIL image before the official image processor sees it, so the
graph inputs are identical to what the same request with embedded data would
produce.

## Images

Deterministic, drawn with Pillow at fixed per-image seeds.  All are ≥ 65536
pixels; the official `smart_resize` factor is `patch_size × merge_size = 32`,
which yields five distinct patch grids:

| file                       | source size | grid `(t, h, w)` | patches P | merged P/4 |
| -------------------------- | ----------- | ---------------- | --------: | ---------: |
| `invoice-512.png`          | 512×512     | 1×32×32          |      1024 |        256 |
| `invoice-640x480.png`      | 640×480     | 1×30×40          |      1200 |        300 |
| `job-queue-896x512.png`    | 896×512     | 1×32×56          |      1792 |        448 |
| `customers-1024x768.png`   | 1024×768    | 1×48×64          |      3072 |        768 |
| `travel-app-1280x704.png`  | 1280×704    | 1×44×80          |      3520 |        880 |

Regenerate them (byte-for-byte identical) with:

```bash
python3 fixtures/make_images.py
```

## Coverage

| case       | question type | K / levels | image                      |
| ---------- | ------------- | ---------- | -------------------------- |
| `onejev-00` | choice        | 4          | `invoice-512.png`          |
| `onejev-01` | choice        | 3          | `invoice-512.png`          |
| `onejev-02` | score         | 4          | `invoice-512.png`          |
| `onejev-03` | noul          | 2          | `invoice-512.png`          |
| `onejev-04` | choice        | 5          | `job-queue-896x512.png`    |
| `onejev-05` | noul          | 2          | `job-queue-896x512.png`    |
| `onejev-06` | score         | 4          | `job-queue-896x512.png`    |
| `onejev-07` | choice        | 6          | `customers-1024x768.png`   |
| `onejev-08` | choice        | 3          | `customers-1024x768.png`   |
| `onejev-09` | score         | 5          | `travel-app-1280x704.png`  |
| `onejev-10` | choice        | 8          | `travel-app-1280x704.png`  |
| `onejev-11` | choice        | 2          | `invoice-640x480.png`      |
| `onejev-12` | noul          | 2          | `travel-app-1280x704.png`  |
| `onejev-13` | choice        | 4          | `travel-app-1280x704.png`  |
