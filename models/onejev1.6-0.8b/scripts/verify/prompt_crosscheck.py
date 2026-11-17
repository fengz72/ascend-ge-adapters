"""Cross-check the vendored OneJev prompt rendering against official flide 0.1.16.

Runs the same 14 fixture requests through the vendored slice and through the
official wheel installed at $FLIDE_CHECK_DIR (default /tmp/opencode/flide-check,
provisioned with `pip install --no-deps --target <dir> flide==0.1.16 pydantic`),
in separate subprocesses so the two same-named packages never collide.
Compares the rendered prompts character by character.

Usage:
    python3 scripts/verify/prompt_crosscheck.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ADAPTER = Path(__file__).resolve().parents[2]
DEPS = ADAPTER / "work" / "deps" / "py"
VENDOR = ADAPTER / "vendor"
OFFICIAL = Path(os.environ.get("FLIDE_CHECK_DIR", "/tmp/opencode/flide-check"))
if not (OFFICIAL / "flide_onejev").is_dir():
    raise SystemExit(
        f"prompt_crosscheck: no official flide slice at {OFFICIAL}; install it with "
        f"`python3 -m pip install --no-deps --target {OFFICIAL} flide==0.1.16 pydantic` "
        "or set FLIDE_CHECK_DIR to an existing install")
MODEL = Path(os.environ.get("ONEJEV_MODEL_DIR", "")) if os.environ.get("ONEJEV_MODEL_DIR") else None


def dump(impl, out):
    sys.path.insert(0, str(DEPS))
    if impl == "vendor":
        sys.path.insert(0, str(VENDOR))
    else:
        sys.path.insert(0, str(OFFICIAL))
    from transformers import AutoTokenizer
    from flide.api.schema import ChoiceQuestion, NoulCriteria, NoulQuestion, ScoreQuestion
    from flide_onejev.media import render_prompt
    from flide_onejev.prompt import build_messages, render_question

    folder = MODEL
    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    lines = [json.loads(x) for x in (ADAPTER / "fixtures" / "cases.jsonl").read_text().splitlines()
             if x.strip()]
    results = []
    for row in lines:
        qid, spec = next(iter(row["questions"].items()))
        if spec["type"] == "choice":
            q = ChoiceQuestion(type="choice", instructions=spec.get("instructions"),
                               criteria=spec["criteria"])
        elif spec["type"] == "noul":
            crit = spec.get("criteria")
            q = NoulQuestion(type="noul", instructions=spec.get("instructions"),
                             criteria=NoulCriteria(**crit) if crit else None)
        else:
            q = ScoreQuestion(type="score", instructions=spec.get("instructions"),
                              criteria=spec["criteria"])
        rendered = render_question(qid, q)
        prompt = render_prompt(tokenizer, {"enable_thinking": False},
                               build_messages(row["state"], rendered.suffix), row["media"])
        results.append({"qid": qid, "suffix": rendered.suffix,
                        "labels": list(rendered.labels), "prompt": prompt})
    Path(out).write_text(json.dumps(results, ensure_ascii=False))


def main():
    if MODEL is None:
        raise SystemExit("prompt_crosscheck: set ONEJEV_MODEL_DIR")
    if len(sys.argv) == 3 and sys.argv[1] == "--dump":
        dump(os.environ["CROSSCHECK_IMPL"], sys.argv[2])
        return
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for impl in ("vendor", "official"):
            out = Path(tmp) / f"{impl}.json"
            env = dict(os.environ, CROSSCHECK_IMPL=impl, PYTHONPATH="")
            proc = subprocess.run([sys.executable, __file__, "--dump", str(out)],
                                  env=env, capture_output=True, text=True)
            if proc.returncode:
                print(proc.stdout)
                print(proc.stderr)
                raise SystemExit(f"prompt_crosscheck: {impl} dump failed")
            paths[impl] = out
        vendored = json.loads(paths["vendor"].read_text())
        official = json.loads(paths["official"].read_text())

    bad = 0
    for i, (v, o) in enumerate(zip(vendored, official)):
        same = v == o
        print(f"prompt_crosscheck: line {i + 1:02d} qid={v['qid']!r} "
              f"{'OK' if same else 'DIFF'} ({len(v['prompt'])} chars)")
        if not same:
            bad += 1
            for field in ("suffix", "labels", "prompt"):
                if v[field] != o[field]:
                    print(f"  {field} differs")
                    if field == "prompt":
                        for j, (a, b) in enumerate(zip(v[field], o[field])):
                            if a != b:
                                print(f"    first diff at char {j}: {v[field][j:j+120]!r} "
                                      f"vs {o[field][j:j+120]!r}")
                                break
                        else:
                            print(f"    length differs: {len(v[field])} vs {len(o[field])}")
    if bad:
        raise SystemExit(f"prompt_crosscheck: {bad} of {len(vendored)} cases differ")
    print(f"prompt_crosscheck: all {len(vendored)} cases byte-identical")


if __name__ == "__main__":
    main()
