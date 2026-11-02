#!/usr/bin/env python3
"""HOST-side idle-NPU monitor for the StartLux formal AIR benchmark (P8).

Runs on host 100 (uid 1130, /usr/bin/python3 == 3.9.9, STDLIB ONLY, no numpy /
Pillow / torch, no repository imports) as the authoritative source copy in
models/startlux-decision-v1.1-0.8b-instant/scripts/; the deployed copy under
/export/home/ext.hejun23/.local/state/startlux-air-monitor/ must stay
sha256-identical to this file (deploy: docker cp, see performance.md).

Why host-side: the host user cannot read the repository or run host npu-smi
(real dcmi initialize -8005).  EVERY read-only sampling goes through
``docker exec hejun-cann910-jev npu-smi ...``; every repository check
(sha256 / report gate files) goes through ``docker exec`` as well.  The host
monitor never calls host npu-smi and never reads /proc of container processes;
process ownership comes from ``docker top hejun-cann910-jev -eo pid,ppid,args``
(host PID/PPID view; the worker argv carries the unique job id, the GE runtime
subprocess is its descendant).

Behavior (fail-closed everywhere; a parse or command error is NEVER idle):

  * every cycle: ``npu-smi info -m`` (board/chip -> logic/phy mapping; Mcu
    rows with '-' logic/phy are ignored; exactly 16 usable device rows are
    expected -- board numbering is NEVER assumed to equal the logical device
    id) + ``npu-smi info`` (per-chip AICore util, HBM used/total, and the
    per-chip device process table, which must be PRESENT and COMPLETE: every
    mapped board covered by process rows or by the exact 'No running
    processes found in NPU <id>' empty-board declaration; a truncated or
    missing process table is a parse error, never 'idle').
  * a logical device is IDLE only if ALL hold: no device process resolved to
    it, hbm_used <= 4096 MB (actual threshold; the 5% alternative would be
    3276.8 MB of 65536 MB), hbm_total - hbm_used >= 49152 MB (48 GiB), and
    AICore util <= 5%.  A card with util 0 but ~57 GB resident workers is
    NEVER idle.
  * daemon triggers only for a logical device that was idle for
    --idle-consecutive CONSECUTIVE cycles (PER-DEVICE counts: devices idle in
    alternating cycles never accumulate a streak; a busy cycle resets that
    device's count; a sampling error resets ALL counts) PLUS one extra fresh
    pre-trigger re-sample of the SAME device (race reduction; the monitor has
    no device-lock / grab capability -- documented honestly).
  * before triggering it re-validates the gates through docker exec: the
    formal AIR sha256/bytes, the ge_runtime sha256, and the P7 acceptance
    evidence: either work/startlux-cannbot/p7-final-ready.json with the REAL
    schema (startlux-p7-final-ready/1: statuses.<key>.status passed for the
    five acceptance entries + the AIR sha256/bytes binding) or BOTH
    main-independent reports review-p7-cli-independent.json and
    review-p7-business-independent.json with explicit passed flags and the
    AIR hash binding.  Missing/empty/malformed evidence never passes;
    generic top-level status flags are NOT accepted in the ready file.
  * the job dir lives under the host monitor state dir (bind-mounted at the
    same path inside the container); the monitor pre-creates it, the worker
    (container, root) writes worker-status.json 0644 there and polls the
    shared abort file; the monitor polls device processes during the run and
    writes the abort file when an EXTERNAL process (not the worker's own
    docker-top subtree) occupies the chosen card -- the worker then kills
    ONLY its own ge_runtime child (process group); the monitor never kills
    unknown or external PIDs.  Contamination is STICKY: once a racing
    occupant or attribution failure is seen, the run ends 'contaminated'
    even if the external process later leaves; with NO device process on
    the chosen card no attribution is required (worker not yet visible in
    docker top at startup / already exited is fine).  Terminal protocol:
    'bench-done' requires worker-status.json bench-done AND a zero
    docker-exec client exit code; the worker only writes bench-done when
    the bench record passed all publish-integrity gates (runtime summary,
    requests.csv accounting and the planned draw counts all consistent,
    finite metrics, wall_ms > 0, no errors/first-hits).
  * SIGTERM/SIGINT/``stop`` during an ACTIVE job write the shared abort
    marker (the worker ends its own child); the monitor waits out its grace
    window and only then kills its OWN docker-exec client process group.  A
    signal during the idle-wait phase persists state and releases the lock.
  * the launcher command is a single ``exec python3 ...`` argv line built
    with shlex (all arguments present, no multi-line continuation bug); the
    worker's stdout+stderr go to jobdir/launcher.log (never DEVNULL); a pure
    Bash guard rejects any non-canonical ChunkGatedDeltaRule LD_PRELOAD
    entry BEFORE the canonical proto env is sourced and before Python starts
    (the old + canonical double load SIGILLs before any Python preflight
    could run).
  * one successful benchmark ends the daemon (single-shot); any terminal
    failure/contamination is recorded and ends the daemon as well (no silent
    retry loops).

Commands: daemon | once | dryrun | sample | status | stop.  ``dryrun`` runs
the full decision path (gates + sampling + command construction) but NEVER
launches the worker; ``once`` takes exactly one sampling cycle; ``sample``
dumps raw npu-smi outputs read-only; ``status``/``stop`` read/request state.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

# --------------------------------------------------------------------- pins

CONTAINER = "hejun-cann910-jev"
REPO = "/export/home/ext.hejun23/workspace/ascend-ge-adapters"
STATE_DIR = "/export/home/ext.hejun23/.local/state/startlux-air-monitor"
HOST_UID_HOME = "/export/home/ext.hejun23"

AIR_REL = "work/startlux-decision/build/model.air"
AIR_BYTES = 1907534423
AIR_SHA256 = "c1f8816ee25e03119b2522e55adb49ebf42a88692791b179920f5f60221edba1"
RUNTIME_REL = "runtime/build/ge_runtime"
RUNTIME_SHA256 = "692996b5df8a7ed8fdebdc365cdaa9383b6b538868ecf3e019303930d178eecc"
WORKER_REL = ("models/startlux-decision-v1.1-0.8b-instant/scripts/"
              "run_unified_performance.py")
MODEL_FOLDER_REL = ("work/startlux-materials/"
                    "model-6496ae641cc71b149ce9dad69bd97f2a44e7bf11")
PILLOW_DEPS = "work/startlux-validation/p6-official/deps/pillow"
TORCHVISION_DEPS = "work/startlux-validation/p6-official/deps/torchvision"
RUN_ROOT_REL = "work/startlux-validation/p8-idle-performance/runs"

P7_READY = "work/startlux-cannbot/p7-final-ready.json"
P7_READY_SCHEMA = "startlux-p7-final-ready/1"
P7_READY_KEY_STATUSES = ("formal_cli_six_stages", "align_bug_fix_and_revalidate",
                         "oracle", "business_compare_failclosed",
                         "compare_negative_fixtures")
P7_CLI = "work/startlux-cannbot/review-p7-cli-independent.json"
P7_BIZ = "work/startlux-cannbot/review-p7-business-independent.json"

LOGICAL_DEVICES_EXPECTED = 16        # 8 boards x 2 chips; still PARSED, never assumed
BUS_ID_RE = re.compile(r"^[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]$")
HEALTH_ROW_RE = re.compile(r"^(\d+)\s+(\S+)$")
VERSION_ROW_RE = re.compile(r"^npu-smi\s+\S+\s+Version:\s+\S+$")
NO_RUNNING_RE = re.compile(r"^No running processes found in NPU (\d+)$")


class MonitorError(Exception):
    """Any sampling/parsing/attribution failure (fail-closed; never 'idle')."""


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def utc_ts():
    return time.time()


# ------------------------------------------------------------- docker layer

class DockerLayer:
    """All host->container read access (npu-smi, cat, sha256sum, docker top)."""

    def __init__(self, container=CONTAINER, timeout=60.0):
        self.container = container
        self.timeout = float(timeout)

    def _exec(self, argv, timeout=None):
        try:
            done = subprocess.run(["docker", "exec", self.container] + list(argv),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, timeout=timeout or self.timeout)
        except subprocess.TimeoutExpired:
            raise MonitorError("docker exec %r timed out" % (argv,))
        except OSError as error:
            raise MonitorError("docker exec %r failed: %s" % (argv, error))
        if done.returncode != 0:
            raise MonitorError("docker exec %r exit %d: %s"
                                % (argv, done.returncode, done.stderr.strip()[:500]))
        return done.stdout

    def smi(self, *args):
        return self._exec(["npu-smi"] + list(args))

    def cat(self, path):
        text = self._exec(["cat", path])
        if not text.strip():
            raise MonitorError("container file %s is empty or missing" % path)
        return text

    def sha256(self, path):
        text = self._exec(["sha256sum", path])
        parts = text.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            raise MonitorError("sha256sum %s output unparsable: %r" % (path, text[:200]))
        return parts[0]

    def size(self, path):
        text = self._exec(["stat", "-c", "%s", path])
        value = text.strip()
        if not value.isdigit():
            raise MonitorError("stat %s output unparsable: %r" % (path, text[:200]))
        return int(value)

    def top(self):
        text = subprocess.run(["docker", "top", self.container, "-eo", "pid,ppid,args"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, timeout=self.timeout)
        if text.returncode != 0:
            raise MonitorError("docker top failed (%d): %s"
                               % (text.returncode, text.stderr.strip()[:500]))
        rows = []
        lines = [line for line in text.stdout.splitlines() if line.strip()]
        if not lines or "PID" not in lines[0]:
            raise MonitorError("docker top header missing: %r" % text.stdout[:200])
        for line in lines[1:]:
            fields = line.split(None, 2)
            if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
                raise MonitorError("docker top row unparsable: %r" % line[:200])
            rows.append((int(fields[0]), int(fields[1]), fields[2]))
        return rows


# ------------------------------------------------------------ npu-smi parsing

def parse_mapping(text):
    """``npu-smi info -m`` -> list of {npu, chip, logic, phy, name}.

    Mcu rows (logic/phy '-') are ignored; the parsed logic ids are the ONLY
    accepted device numbering -- board/chip numbering is never assumed equal.
    """
    devices = []
    lines = [line.strip() for line in text.splitlines()]
    body = [line for line in lines if line and not line.startswith("NPU ID")]
    for line in body:
        fields = line.split()
        if len(fields) != 5:
            raise MonitorError("npu-smi -m: unparsable row %r" % line[:120])
        npu, chip, logic, phy, name = fields
        if logic == "-" and phy == "-" and name == "Mcu":
            continue                      # the documented Mcu rows: ignored
        try:
            npu, chip, logic, phy = int(npu), int(chip), int(logic), int(phy)
        except ValueError:
            raise MonitorError("npu-smi -m: non-integer ids in row %r" % line[:120])
        devices.append({"npu": npu, "chip": chip, "logic": logic, "phy": phy,
                         "name": name})
    if not devices:
        raise MonitorError("npu-smi -m: no usable device rows")
    if len(devices) != LOGICAL_DEVICES_EXPECTED:
        raise MonitorError("npu-smi -m: %d device rows != the expected %d; a "
                           "truncated mapping is fail-closed"
                           % (len(devices), LOGICAL_DEVICES_EXPECTED))
    for dev in devices:
        if not 0 <= dev["logic"] < LOGICAL_DEVICES_EXPECTED:
            raise MonitorError("npu-smi -m: logic id %d out of range"
                               % dev["logic"])
    logics = [dev["logic"] for dev in devices]
    if len(set(logics)) != len(logics):
        raise MonitorError("npu-smi -m: duplicate logical ids %r" % sorted(logics))
    phys = [dev["phy"] for dev in devices]
    if len(set(phys)) != len(phys):
        raise MonitorError("npu-smi -m: duplicate phy ids %r" % sorted(phys))
    if len(set((dev["npu"], dev["chip"]) for dev in devices)) != len(devices):
        raise MonitorError("npu-smi -m: duplicate (npu, chip) keys")
    return devices


def _row_cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_overview(text, mapping):
    """``npu-smi info`` -> ({(npu,phy): chip stats}, [process rows]).

    Chip rows carry the Phy-ID and a PCI bus id; the per-chip AICore util and
    the two 'used / total' pairs (memory, HBM) are parsed strictly.  Process
    rows are keyed (npu, chip) and RESOLVED to logical ids through the -m
    mapping (never assumed chip==phy==logic).

    The per-chip process table is REQUIRED and must be COMPLETE: the real
    'Process id' header must be present and EVERY board of the -m mapping must
    be covered either by real process rows or by the exact known empty-board
    declaration 'No running processes found in NPU <id>' (confirmed from this
    host's /usr/local/sbin/npu-smi binary strings).  A missing or truncated
    process table is a parse error -- with a low HBM reading it must NEVER be
    mistaken for 'no processes'.  Any row that is not one of the known shapes
    (version banner, the two chip-table headers, a per-board health row, a
    chip row, the process header, a process row, the No-running declaration)
    is rejected fail-closed instead of being generically skipped.
    """
    by_chip = {(dev["npu"], dev["chip"]): dev["logic"] for dev in mapping}
    by_phy = {dev["phy"]: dev for dev in mapping}
    board_names = {dev["name"] for dev in mapping}
    npu_ids = {dev["npu"] for dev in mapping}
    chips, processes = {}, []
    saw_process_header = False
    covered_boards = set()
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.startswith("|"):
            continue
        cells = _row_cells(line)
        norm = [" ".join(cell.split()) for cell in cells]
        if saw_process_header:
            if len(norm) == 1 and NO_RUNNING_RE.match(norm[0]):
                board = int(NO_RUNNING_RE.match(norm[0]).group(1))
                if board not in npu_ids:
                    raise MonitorError("npu-smi info: no-process declaration "
                                       "for unknown NPU %d" % board)
                covered_boards.add(board)
                continue
            if len(cells) != 4:
                raise MonitorError("npu-smi info: unparsable process row %r"
                                   % line[:160])
            head = cells[0].split()
            if len(head) != 2 or not all(part.lstrip("-").isdigit() for part in head):
                raise MonitorError("npu-smi info: bad process device key %r"
                                   % line[:160])
            if not cells[1].isdigit() or not cells[3].isdigit():
                raise MonitorError("npu-smi info: bad process pid/memory %r"
                                   % line[:160])
            npu, chip = int(head[0]), int(head[1])
            logic = by_chip.get((npu, chip))
            if logic is None:
                raise MonitorError("process row (npu %d, chip %d) has no -m "
                                   "mapping row" % (npu, chip))
            covered_boards.add(npu)
            processes.append({"npu": npu, "chip": chip, "logic": logic,
                              "pid": int(cells[1]), "name": cells[2],
                              "memory_mb": int(cells[3])})
            continue
        if "Process id" in norm and "Process name" in norm \
                and norm and norm[0] == "NPU Chip":
            saw_process_header = True
            continue
        first_tokens = norm[0].split() if norm else []
        if len(first_tokens) >= 2 and tuple(first_tokens[:2]) in (
                ("NPU", "Name"), ("Chip", "Phy-ID")):
            continue                      # the two chip-table header rows
        if len(norm) == 1 and VERSION_ROW_RE.match(norm[0]):
            continue                      # the version banner row
        health = HEALTH_ROW_RE.match(cells[0]) if cells else None
        if len(cells) == 3 and health is not None \
                and health.group(2) in board_names:
            continue                      # per-board health row (name from -m)
        if len(cells) == 3 and BUS_ID_RE.match(cells[1]):
            head = cells[0].split()
            if len(head) != 2 or not head[1].isdigit():
                raise MonitorError("npu-smi info: bad chip row key %r"
                                   % line[:160])
            pairs = re.findall(r"(\d+)\s*/\s*(\d+)", cells[2])
            head_tokens = cells[2].split()
            if not head_tokens or \
                    not head_tokens[0].lstrip("-").replace(".", "", 1).isdigit():
                raise MonitorError("npu-smi info: bad AICore util %r"
                                   % line[:160])
            if len(pairs) != 2:
                raise MonitorError("npu-smi info: expected memory+HBM pairs, "
                                   "got %r" % line[:160])
            util = float(head_tokens[0])
            # the chip row's first cell is "Chip  Phy-ID"; the globally unique
            # phy resolves the device, the chip id is cross-checked (board
            # numbering is never assumed)
            chip_id, phy = int(head[0]), int(head[1])
            dev = by_phy.get(phy)
            if dev is None:
                raise MonitorError("chip row (chip %d, phy %d) has no -m "
                                   "mapping row" % (chip_id, phy))
            if dev["chip"] != chip_id:
                raise MonitorError("chip row says chip %d phy %d, mapping says "
                                   "chip %d (npu %d)" % (chip_id, phy, dev["chip"],
                                                          dev["npu"]))
            logic = dev["logic"]
            npu = dev["npu"]
            if logic in chips:
                raise MonitorError("duplicate chip row for logic %d" % logic)
            mem_used, mem_total = int(pairs[0][0]), int(pairs[0][1])
            hbm_used, hbm_total = int(pairs[1][0]), int(pairs[1][1])
            if hbm_total <= 0:
                raise MonitorError("logic %d: HBM total %d unusable"
                                   % (logic, hbm_total))
            chips[logic] = {"npu": npu, "phy": phy, "util_pct": util,
                           "memory_used_mb": mem_used,
                           "memory_total_mb": mem_total,
                           "hbm_used_mb": hbm_used, "hbm_total_mb": hbm_total}
            continue
        raise MonitorError("npu-smi info: unrecognized row (fail-closed): %r"
                           % line[:160])
    if not chips:
        raise MonitorError("npu-smi info: no chip rows parsed")
    if set(chips) != {dev["logic"] for dev in mapping}:
        raise MonitorError("npu-smi info: chip rows %r != mapped logic devices %r"
                            % (sorted(chips), sorted(dev["logic"] for dev in mapping)))
    if not saw_process_header:
        raise MonitorError("npu-smi info: process table header missing; a "
                           "truncated output is never idle")
    if covered_boards != npu_ids:
        raise MonitorError("npu-smi info: process table covers boards %r but "
                           "the mapping declares %r; a missing/truncated "
                           "process table is never idle"
                           % (sorted(covered_boards), sorted(npu_ids)))
    return chips, processes


def take_sample(layer):
    """One full read-only sampling cycle (mapping + overview + processes)."""
    mapping = parse_mapping(layer.smi("info", "-m"))
    chips, processes = parse_overview(layer.smi("info"), mapping)
    devices = {}
    for dev in mapping:
        logic = dev["logic"]
        chip = chips[logic]
        holders = [p for p in processes if p["logic"] == logic]
        devices[logic] = {
            "npu": dev["npu"], "chip": dev["chip"], "phy": dev["phy"],
            "name": dev["name"], "util_pct": chip["util_pct"],
            "hbm_used_mb": chip["hbm_used_mb"], "hbm_total_mb": chip["hbm_total_mb"],
            "hbm_free_mb": chip["hbm_total_mb"] - chip["hbm_used_mb"],
            "processes": [{"pid": p["pid"], "memory_mb": p["memory_mb"]}
                           for p in holders],
        }
    return {"ts": now_iso(), "devices": devices,
            "logical_devices": sorted(devices)}


def device_idle(info, cfg):
    """The explicit idle rule; every criterion must hold (fail-closed)."""
    reasons = []
    if info["processes"]:
        reasons.append("device processes %r" % [p["pid"] for p in info["processes"]])
    if info["hbm_used_mb"] > cfg["hbm_used_max_mb"]:
        reasons.append("hbm_used %dMB > %dMB"
                       % (info["hbm_used_mb"], cfg["hbm_used_max_mb"]))
    if info["hbm_free_mb"] < cfg["hbm_free_min_mb"]:
        reasons.append("hbm_free %dMB < %dMB"
                       % (info["hbm_free_mb"], cfg["hbm_free_min_mb"]))
    if info["util_pct"] > cfg["util_max_pct"]:
        reasons.append("util %.1f%% > %.1f%%" % (info["util_pct"], cfg["util_max_pct"]))
    return (not reasons), reasons


def idle_devices(sample, cfg):
    out = {}
    for logic, info in sample["devices"].items():
        ok, reasons = device_idle(info, cfg)
        if ok:
            out[logic] = info
    return out


# ------------------------------------------------------------------- gates

def _load_json(layer, rel):
    """A gate JSON read; unreadable/missing/malformed -> (None, reason)."""
    try:
        return json.loads(layer.cat("%s/%s" % (REPO, rel))), None
    except MonitorError as error:
        return None, str(error)
    except ValueError:
        return None, "%s is not valid JSON (fail-closed)" % rel


def _p7_ready_check(ready):
    """The REAL p7-final-ready.json schema (no generic top-level flags).

    schema=startlux-p7-final-ready/1; the air block binds sha256+bytes to the
    pinned AIR; each of the five acceptance entries inside statuses must
    carry a 'status' that is literally 'passed' or 'passed (...)'.
    """
    if not isinstance(ready, dict):
        return False, "not a JSON object"
    if ready.get("schema") != P7_READY_SCHEMA:
        return False, ("schema %r != %r (generic top-level status/passed "
                       "flags are not accepted)"
                       % (ready.get("schema"), P7_READY_SCHEMA))
    air = ready.get("air")
    if not isinstance(air, dict) or air.get("sha256") != AIR_SHA256 \
            or air.get("bytes") != AIR_BYTES:
        return False, "air sha256/bytes binding does not match the pinned AIR"
    statuses = ready.get("statuses")
    if not isinstance(statuses, dict):
        return False, "statuses block missing"
    for key in P7_READY_KEY_STATUSES:
        entry = statuses.get(key)
        status = entry.get("status") if isinstance(entry, dict) else None
        if not isinstance(status, str) or \
                not (status == "passed" or status.startswith("passed (")):
            return False, "statuses.%s.status %r is not passed" % (key, status)
    return True, None


def check_gates(layer, cfg):
    """AIR/runtime identity + P7 acceptance evidence, all via docker exec.

    Evidence paths (either one suffices): the real-schema p7-final-ready.json,
    or the explicit fallback pair of main-independent CLI + business reports.
    """
    reasons, details = [], {}
    air_path = "%s/%s" % (REPO, AIR_REL)
    try:
        size = layer.size(air_path)
        sha = layer.sha256(air_path)
    except MonitorError as error:
        reasons.append("AIR unreadable: %s" % error)
        size = sha = None
    details["air"] = {"bytes": size, "sha256": sha}
    if size is not None and size != AIR_BYTES:
        reasons.append("AIR bytes %d != pinned %d" % (size, AIR_BYTES))
    if sha is not None and sha != AIR_SHA256:
        reasons.append("AIR sha256 %s != pinned" % sha)
    try:
        runtime_sha = layer.sha256("%s/%s" % (REPO, RUNTIME_REL))
    except MonitorError as error:
        reasons.append("ge_runtime unreadable: %s" % error)
        runtime_sha = None
    details["runtime_sha256"] = runtime_sha
    if runtime_sha is not None and runtime_sha != RUNTIME_SHA256:
        reasons.append("ge_runtime sha256 %s != pinned" % runtime_sha)

    ready, ready_error = _load_json(layer, P7_READY)
    if ready is None:
        ready_ok, ready_why = False, ready_error or "absent"
    else:
        ready_ok, ready_why = _p7_ready_check(ready)
    details["p7_final_ready"] = {"ok": ready_ok, "reason": ready_why}

    cli, cli_error = _load_json(layer, P7_CLI)
    biz, biz_error = _load_json(layer, P7_BIZ)

    cli_ok = (isinstance(cli, dict) and cli.get("all_passed") is True
              and cli.get("cases") == 11 and cli.get("rows") == 12
              and cli.get("air_sha256") == AIR_SHA256)
    details["cli_report"] = {"ok": cli_ok, "error": cli_error}
    biz_ok = (isinstance(biz, dict) and biz.get("passed") is True
              and biz.get("requests") == 13 and biz.get("native_readouts") == 17
              and biz.get("comparisons") == 33
              and isinstance(biz.get("max_prob_error"), float)
              and biz.get("max_prob_error") <= 0.01)
    details["business_report"] = {"ok": biz_ok, "error": biz_error}
    if ready_ok:
        evidence_ok, evidence_path = True, "p7-final-ready"
    elif cli_ok and biz_ok:
        evidence_ok, evidence_path = True, "report-pair-fallback"
    else:
        evidence_ok, evidence_path = False, None
        if cli_error:
            reasons.append("cli-report: %s" % cli_error)
        elif not cli_ok:
            reasons.append("cli-report: not (all_passed & 11 cases & 12 rows & "
                           "air sha pinned)")
        if biz_error:
            reasons.append("business-report: %s" % biz_error)
        elif not biz_ok:
            reasons.append("business-report: not (passed & 13 requests & 17 "
                           "readouts & 33 comparisons & max_prob_error<=0.01)")
        if ready_error:
            reasons.append("p7-final-ready: %s (fallback reports incomplete)"
                           % ready_error)
        elif not ready_ok:
            reasons.append("p7-final-ready: %s (fallback reports incomplete)"
                           % ready_why)
    details["evidence_path"] = evidence_path
    return {"ok": not reasons, "reasons": reasons, "details": details,
            "evidence_ok": evidence_ok}


# --------------------------------------------------- ownership / attribution

def own_pids(top_rows, job_id):
    """The worker's host PID + every docker-top descendant of it.

    The worker argv carries the unique ``--job-id``; the GE runtime subprocess
    is its child.  Other same-container tasks are NEVER counted as ours.  If
    the worker cannot be located the caller must fail closed.
    """
    marker = "--job-id %s" % job_id
    workers = [row for row in top_rows if marker in row[2]]
    if len(workers) != 1:
        raise MonitorError("docker top: found %d processes carrying job id %r"
                           % (len(workers), job_id))
    root = workers[0][0]
    children = {}
    for pid, ppid, _args in top_rows:
        children.setdefault(ppid, []).append(pid)
    own = set()
    stack = [root]
    while stack:
        pid = stack.pop()
        if pid in own:
            continue
        own.add(pid)
        stack.extend(children.get(pid, []))
    return own


def external_processes(processes, logic, own):
    """Device pids on the chosen card that are NOT the job's own subtree."""
    return [p for p in processes if p["logic"] == logic and p["pid"] not in own]


# ------------------------------------------------------ worker construction

def worker_argv(cfg, logic, job_id, job_dir):
    """The exact python argv the worker is exec'd with (every argument)."""
    return [cfg["python"], "%s/%s" % (cfg["repo"], WORKER_REL),
            "--phase", "gate,bench", "--device", "0",
            "--job-id", str(job_id), "--job-dir", str(job_dir),
            "--run-root", RUN_ROOT_REL,
            "--warmup", str(int(cfg["warmup"])),
            "--iterations", str(int(cfg["iterations"])),
            "--seed", str(int(cfg["seed"])),
            "--timeout", "%.0f" % float(cfg["worker_timeout"])]


def worker_command(cfg, logic, job_id, job_dir):
    """The full docker exec worker launch (printed by dryrun verbatim).

    The final exec line is ONE single line built with shlex.join so every
    worker argument really reaches the interpreter (the previous multi-line
    exec lost everything after --device).  Before the canonical proto env is
    sourced, a pure Bash guard rejects any LD_PRELOAD entry whose basename
    contains chunk_gated_delta_rule unless it IS the canonical proto library
    (the old compat lib + canonical proto double load can SIGILL inside the
    loader, before any Python preflight could run); unrelated preload entries
    are kept, and the canonical lib is loaded exactly once.
    """
    repo = cfg["repo"]
    script = "\n".join([
        "set -euo pipefail",
        "cd %s" % shlex.quote(repo),
        "source %s" % shlex.quote(cfg["ascend_env"]),
        "source %s" % shlex.quote(cfg["qwen_env"]),
        "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8",
        "export STARTLUX_MODEL_DIR=%s/%s" % (shlex.quote(repo), MODEL_FOLDER_REL),
        "export PYTHONPATH=%s:%s:\"${PYTHONPATH:-}\""
        % (shlex.quote("%s/%s" % (repo, PILLOW_DEPS)),
           shlex.quote("%s/%s" % (repo, TORCHVISION_DEPS))),
        "export TMPDIR=%s/work/tmp" % shlex.quote(repo),
        "mkdir -p \"$TMPDIR\"",
        "proto_lib_dir=%s/%s/dependencies/chunk_gated_delta_rule_proto/install/lib"
        % (shlex.quote(repo), shlex.quote(cfg["proto_work_dir"])),
        "if [ ! -d \"$proto_lib_dir\" ]; then",
        "  echo \"p8 launcher: canonical chunk proto lib dir missing: "
        "$proto_lib_dir\" >&2",
        "  exit 82",
        "fi",
        "proto_lib=\"$(cd \"$proto_lib_dir\" && pwd)/"
        "libchunk_gated_delta_rule_compat_proto.so\"",
        "bad_proto=\"\"",
        "IFS_BAK=\"$IFS\"",
        "IFS=:",
        "for part in ${LD_PRELOAD:-}; do",
        "  [ -n \"$part\" ] || continue",
        "  case \"$(basename \"$part\")\" in",
        "    *chunk_gated_delta_rule*)",
        "      [ \"$part\" = \"$proto_lib\" ] || bad_proto=\"$bad_proto $part\"",
        "      ;;",
        "  esac",
        "done",
        "IFS=\"$IFS_BAK\"",
        "if [ -n \"$bad_proto\" ]; then",
        "  echo \"p8 launcher: refusing non-canonical ChunkGatedDeltaRule "
        "preload(s):$bad_proto; canonical: $proto_lib\" >&2",
        "  exit 82",
        "fi",
        "source %s %s" % (shlex.quote(cfg["proto_env"]),
                          shlex.quote(cfg["proto_work_dir"])),
        "export ASCEND_RT_VISIBLE_DEVICES=%d" % int(logic),
        "exec %s" % shlex.join(worker_argv(cfg, logic, job_id, job_dir)),
    ])
    return ["docker", "exec", cfg["container"], "bash", "-lc", script]


# --------------------------------------------------------------- state/log

def atomic_write(path, payload):
    path = str(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_json(path, data):
    atomic_write(path, json.dumps(data, indent=1, sort_keys=True) + "\n")


def rotate(path, limit, keep):
    try:
        if os.path.getsize(path) < limit:
            return
    except OSError:
        return
    for index in range(keep - 1, 0, -1):
        if os.path.exists("%s.%d" % (path, index)):
            os.replace("%s.%d" % (path, index), "%s.%d" % (path, index + 1))
    os.replace(path, path + ".1")


class Monitor:
    """The daemon core: lock, state.json, bounded logs, job lifecycle."""

    def __init__(self, cfg, layer=None):
        self.cfg = cfg
        self.layer = layer or DockerLayer(cfg["container"])
        self.state_dir = cfg["state_dir"]
        self.lock_path = os.path.join(self.state_dir, "monitor.lock")
        self.state_path = os.path.join(self.state_dir, "state.json")
        self.log_path = os.path.join(self.state_dir, "monitor.log")
        self.history_path = os.path.join(self.state_dir, "history.jsonl")
        self.stop_path = os.path.join(self.state_dir, "stop-requested")
        self.jobs_dir = os.path.join(self.state_dir, "jobs")
        self.lock = None
        self.signal_received = None
        self.current_job = None
        self.state = {"schema": "startlux-p8-monitor-state/1",
                      "pid": os.getpid(), "started_at": now_iso(),
                      "status": "starting", "idle_counts": {},
                      "cycles": 0, "last_cycle": None, "job": None,
                      "last_error": None, "last_idle_logics": []}
        for needed in (self.state_dir, self.jobs_dir):
            os.makedirs(needed, exist_ok=True)

    # -- infrastructure ------------------------------------------------
    def acquire(self):
        import fcntl as _fcntl
        self.lock = open(self.lock_path, "a+")
        try:
            _fcntl.flock(self.lock.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except OSError:
            raise MonitorError("another monitor already holds %s" % self.lock_path)
        self.lock.seek(0)
        self.lock.truncate()
        self.lock.write("%d\n" % os.getpid())
        self.lock.flush()

    def log(self, message):
        line = "%s %s\n" % (now_iso(), message)
        with open(self.log_path, "a", encoding="utf-8") as stream:
            stream.write(line)
        rotate(self.log_path, self.cfg["log_max_bytes"], self.cfg["log_keep"])
        sys.stdout.write("[monitor] " + message + "\n")
        sys.stdout.flush()

    def save_state(self, **fields):
        self.state.update(fields)
        self.state["updated_at"] = now_iso()
        write_json(self.state_path, self.state)

    def history(self, record):
        with open(self.history_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        rotate(self.history_path, self.cfg["log_max_bytes"], self.cfg["log_keep"])

    def stop_requested(self):
        return os.path.exists(self.stop_path)

    def install_signals(self):
        def handler(signum, _frame):
            self.signal_received = signum
            job = self.current_job
            if job is not None:
                try:
                    self.write_abort(job, "monitor received signal %d" % signum)
                except OSError:
                    pass
            self.log("signal %d received%s" % (
                signum, "; abort marker written for the active job"
                if job is not None else "; stopping at the next check"))
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, handler)

    # -- sampling / decision -------------------------------------------
    def cycle(self):
        """One sampling cycle -> (sample, idle_now, error)."""
        try:
            sample = take_sample(self.layer)
        except MonitorError as error:
            self.state["cycles"] += 1
            self.state["idle_counts"] = {}
            self.save_state(status="monitoring", last_error=str(error),
                            last_cycle=now_iso(), cycles=self.state["cycles"])
            self.history({"ts": now_iso(), "cycle": self.state["cycles"],
                          "result": "error", "error": str(error)})
            self.log("sampling FAILED (fail-closed, ALL idle counts reset): "
                     "%s" % error)
            return None, [], error
        idle = idle_devices(sample, self.cfg)
        self.state["cycles"] += 1
        self.state["last_cycle"] = sample["ts"]
        self.state["last_idle_logics"] = sorted(idle)
        counts = self.state["idle_counts"]
        for logic in sample["devices"]:
            counts[logic] = (counts.get(logic, 0) + 1) if logic in idle else 0
        self.save_state(status="monitoring")
        self.history({"ts": sample["ts"], "cycle": self.state["cycles"],
                      "result": "idle" if idle else "busy",
                      "idle_logics": sorted(idle),
                      "idle_counts": {str(logic): counts[logic]
                                      for logic in sorted(idle)},
                      "max_hbm_used_mb": max(d["hbm_used_mb"]
                                             for d in sample["devices"].values()),
                      "total_device_processes": sum(len(d["processes"])
                                                    for d in sample["devices"].values())})
        self.log("cycle %d: %d/%d idle %r (per-device streaks %r)"
                 % (self.state["cycles"], len(idle), len(sample["devices"]),
                    sorted(idle), {logic: counts[logic] for logic in sorted(idle)}))
        return sample, sorted(idle), None

    def trigger_candidates(self, idle):
        """Devices idle for --idle-consecutive CONSECUTIVE cycles right now."""
        counts = self.state["idle_counts"]
        return sorted(logic for logic in idle
                      if counts.get(logic, 0) >= self.cfg["idle_consecutive"])

    def pretrigger_resample(self, logic):
        """The extra fresh sample of the SAME device right before launch.

        Returns 'ok' (still idle), 'busy' (a racing occupant appeared: that
        device's streak resets), or 'error' (fail-closed: ALL counts reset).
        """
        try:
            sample = take_sample(self.layer)
        except MonitorError as error:
            self.log("pre-trigger re-sample failed (fail-closed): %s" % error)
            return "error"
        info = sample["devices"].get(logic)
        if info is None:
            self.log("pre-trigger re-sample: logic %d vanished from the "
                     "mapping (fail-closed)" % logic)
            return "error"
        ok, reasons = device_idle(info, self.cfg)
        if not ok:
            self.log("pre-trigger re-sample: logic %d no longer idle: %s"
                     % (logic, "; ".join(reasons)))
            return "busy"
        return "ok"

    # -- job lifecycle -------------------------------------------------
    def launch_worker_process(self, command, job_dir):
        """Popen with the launcher's stdout+stderr captured in jobdir.

        The launcher log keeps CANN env / source / exec / argparse failures
        diagnosable even when the worker dies before writing its own status
        file; DEVNULL would silently lose them.
        """
        log_path = os.path.join(job_dir, "launcher.log")
        stream = open(log_path, "a", encoding="utf-8")
        try:
            process = subprocess.Popen(command, stdout=stream,
                                       stderr=subprocess.STDOUT,
                                       start_new_session=True)
        except BaseException:
            stream.close()
            raise
        stream.close()
        return process

    def run_job(self, logic):
        job_id = "p8-%s-%d" % (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
                                os.getpid())
        job_dir = os.path.join(self.jobs_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        abort_path = os.path.join(job_dir, "abort")
        status_path = os.path.join(job_dir, "worker-status.json")
        command = worker_command(self.cfg, logic, job_id, job_dir)
        job = {"job_id": job_id, "logic": logic, "job_dir": job_dir,
               "command": command, "started_at": now_iso(),
               "abort_path": abort_path, "status_path": status_path}
        self.save_state(status="running", job=job)
        write_json(os.path.join(job_dir, "job.json"), job)
        self.log("launching worker job %s on logical device %d (stdout/stderr "
                 "-> %s)" % (job_id, logic, os.path.join(job_dir, "launcher.log")))
        self.log("worker command: %s" % json.dumps(command))
        self.current_job = job
        try:
            process = self.launch_worker_process(command, job_dir)
        except OSError as error:
            self.current_job = None
            outcome = "launch-error: %s" % error
            self.finish_job(job, outcome, None)
            return outcome, None
        try:
            outcome, contamination = self.supervise(process, job)
        finally:
            self.current_job = None
        self.finish_job(job, outcome, contamination)
        return outcome, contamination

    def supervise(self, process, job):
        """Poll the worker: timeout, abort/contamination, external processes.

        Contamination is STICKY: the first racing-occupant / attribution
        failure reason is kept until the terminal state -- even if the
        external process later leaves, this run's data is never published.
        The terminal protocol: worker-status.json may claim 'bench-done'
        ONLY when the docker-exec client also exited 0; a nonzero client
        exit code overrides any stale 'bench-done' claim (bench-failed...).
        """
        started = utc_ts()
        contamination = None
        last_error = None
        while True:
            if self.signal_received is not None:
                return self.stop_active_job(
                    process, job, contamination,
                    "monitor received signal %d" % self.signal_received,
                    "aborted-signal")
            if self.stop_requested():
                return self.stop_active_job(
                    process, job, contamination, "stop requested via stop file",
                    "aborted-stop")
            code = process.poll()
            if code is not None:
                break
            elapsed = utc_ts() - started
            if elapsed >= self.cfg["worker_timeout"]:
                self.log("worker timeout after %.0fs: writing abort marker" % elapsed)
                self.write_abort(job, "monitor worker timeout")
                deadline = utc_ts() + self.cfg["abort_grace"]
                while process.poll() is None and utc_ts() < deadline:
                    time.sleep(2.0)
                if process.poll() is None:
                    self.log("worker did not exit within grace; killing ONLY the "
                             "monitor's own docker-exec client (pid %d)" % process.pid)
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except OSError as error:
                        last_error = str(error)
                    process.wait()
                return "timeout", contamination
            try:
                found = self.check_contamination(job)
                if found and contamination is None:
                    contamination = found
            except MonitorError as error:
                self.log("attribution failure during run (fail-closed): %s" % error)
                self.write_abort(job, "attribution failure: %s" % error)
                if contamination is None:
                    contamination = str(error)
            time.sleep(self.cfg["poll_interval"])
        worker_status = self.read_worker_status(job)
        claimed = worker_status.get("status") if worker_status else None
        if claimed == "bench-done" and process.returncode != 0:
            outcome = "bench-failed-client-exit-%d" % process.returncode
        elif claimed:
            outcome = claimed
        else:
            outcome = "exited-%s" % process.returncode
        if contamination:
            outcome = "contaminated"
        if last_error:
            outcome += "+kill-error"
        return outcome, contamination

    def stop_active_job(self, process, job, contamination, reason, outcome):
        """Signal/stop path: shared abort marker, then ONLY our own client."""
        self.log("stopping job: %s -> abort marker written (the worker ends "
                 "its OWN ge_runtime child)" % reason)
        self.write_abort(job, reason)
        deadline = utc_ts() + self.cfg["abort_grace"]
        while process.poll() is None and utc_ts() < deadline:
            time.sleep(1.0)
        kill_error = None
        if process.poll() is None:
            self.log("job did not exit within grace; killing ONLY the "
                     "monitor's own docker-exec client (pid %d)" % process.pid)
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except OSError as error:
                kill_error = str(error)
            process.wait()
        if kill_error:
            outcome += "+kill-error"
        return outcome, contamination

    def check_contamination(self, job):
        """External device processes on our card -> abort marker + reason.

        Strict npu-smi parsing FIRST (any parse error stays fail-closed).
        The chosen card's device processes are then filtered: with NO device
        process on it there is nothing to attribute, so the worker's presence
        in docker top is NOT required (the docker-exec client may not have
        spawned the worker yet, or it may already have exited).  The moment
        any device PID exists on the chosen card, ownership MUST resolve
        through the unique worker tree; an unknown PID is fail-closed.
        docker-top command errors are never broadly ignored: they only
        occur (and only matter) on the path where attribution is required.
        """
        mapping = parse_mapping(self.layer.smi("info", "-m"))
        _chips, processes = parse_overview(self.layer.smi("info"), mapping)
        holders = [p for p in processes if p["logic"] == job["logic"]]
        if not holders:
            return None
        top_rows = self.layer.top()
        ours = own_pids(top_rows, job["job_id"])
        external = external_processes(holders, job["logic"], ours)
        if external:
            reason = ("external device pids %r on logic %d"
                      % ([p["pid"] for p in external], job["logic"]))
            self.log("CONTAMINATION: %s -> abort marker" % reason)
            self.write_abort(job, reason)
            return reason
        return None

    def write_abort(self, job, reason):
        atomic_write(job["abort_path"],
                     json.dumps({"ts": now_iso(), "reason": reason}) + "\n")

    def read_worker_status(self, job):
        try:
            with open(job["status_path"], "r", encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, ValueError):
            return None

    def finish_job(self, job, outcome, contamination):
        record = dict(job)
        record.update({"finished_at": now_iso(), "outcome": outcome,
                       "contamination": contamination,
                       "worker_status": self.read_worker_status(job)})
        write_json(os.path.join(job["job_dir"], "job.json"), record)
        status = "done" if outcome == "bench-done" else outcome
        self.save_state(status=status, job=record)
        self.history({"ts": now_iso(), "job": job["job_id"], "outcome": outcome,
                      "contamination": contamination})
        self.log("job %s finished: outcome=%s contamination=%r"
                 % (job["job_id"], outcome, contamination))

    # -- top-level flows ----------------------------------------------
    def daemon(self):
        self.acquire()
        self.install_signals()
        self.save_state(status="monitoring")
        self.log("daemon start: cfg=%s" % json.dumps(self.cfg_public()))
        try:
            while True:
                if self.signal_received is not None:
                    self.log("signal %d during idle-wait: persisting state and "
                             "stopping (nothing launched)" % self.signal_received)
                    self.save_state(status="stopped",
                                    stopped_by_signal=self.signal_received)
                    return 130 if self.signal_received == 2 else 143
                if self.stop_requested():
                    self.log("stop requested; exiting cleanly")
                    self.save_state(status="stopped")
                    return 0
                gates = check_gates(self.layer, self.cfg)
                if not gates["ok"]:
                    self.state["idle_counts"] = {}
                    self.save_state(status="waiting-evidence",
                                    last_error="; ".join(gates["reasons"]))
                    self.log("gates not ok (fail-closed): %s"
                             % "; ".join(gates["reasons"]))
                else:
                    sample, idle, _error = self.cycle()
                    candidates = self.trigger_candidates(idle)
                    if candidates:
                        logic = candidates[0]
                        pre = self.pretrigger_resample(logic)
                        if pre == "ok":
                            outcome, contamination = self.run_job(logic)
                            if self.signal_received is not None:
                                self.save_state(
                                    stopped_by_signal=self.signal_received)
                            self.log("terminal outcome %r; daemon ends "
                                     "(single-shot)" % outcome)
                            return 0 if outcome == "bench-done" else 1
                        if pre == "error":
                            self.state["idle_counts"] = {}
                        else:
                            self.state["idle_counts"][logic] = 0
                        self.save_state(status="monitoring")
                time.sleep(self.cfg["interval"])
        finally:
            if self.lock is not None:
                try:
                    self.lock.close()
                except OSError:
                    pass
                self.lock = None
            try:
                os.unlink(self.stop_path)
            except OSError:
                pass

    def cfg_public(self):
        keys = ("interval", "idle_consecutive", "hbm_used_max_mb",
                "hbm_free_min_mb", "util_max_pct", "warmup", "iterations",
                "seed", "worker_timeout", "poll_interval", "container", "repo")
        return {key: self.cfg[key] for key in keys}

    def once(self):
        self.acquire()
        try:
            gates = check_gates(self.layer, self.cfg)
            sample, idle, error = self.cycle()
            record = {"ts": now_iso(), "gates": gates, "sample_error": error,
                      "idle_logics": idle,
                      "devices": sample["devices"] if sample else None}
            write_json(os.path.join(self.state_dir, "once-report.json"), record)
            self.log("once: gates_ok=%s idle=%r error=%r"
                     % (gates["ok"], idle, error))
            return 0 if (gates["ok"] and sample is not None) else 1
        finally:
            pass

    def dryrun(self):
        self.acquire()
        try:
            gates = check_gates(self.layer, self.cfg)
            sample, idle, error = self.cycle()
            candidates = self.trigger_candidates(idle)
            command = worker_command(self.cfg, 0, "DRYRUN-JOB-ID",
                                     os.path.join(self.jobs_dir, "dryrun"))
            record = {"ts": now_iso(), "mode": "dryrun", "gates": gates,
                      "sample_error": error, "idle_logics": idle,
                      "trigger_candidates": candidates,
                      "would_launch": bool(gates["ok"] and candidates),
                      "worker_command_preview": command,
                      "devices": sample["devices"] if sample else None}
            write_json(os.path.join(self.state_dir, "dryrun-report.json"), record)
            self.log("dryrun: gates_ok=%s idle=%r candidates=%r error=%r "
                     "(worker NEVER launched)"
                     % (gates["ok"], idle, candidates, error))
            print(json.dumps({"gates_ok": gates["ok"], "gates_reasons": gates["reasons"],
                              "idle_logics": idle, "trigger_candidates": candidates,
                              "sample_error": error,
                              "worker_command_preview": command}, indent=1))
            return 0
        finally:
            pass


# ---------------------------------------------------------------- commands

def cmd_sample(cfg):
    layer = DockerLayer(cfg["container"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = os.path.join(cfg["state_dir"], "samples", stamp)
    os.makedirs(out_dir, exist_ok=True)
    mapping_text = layer.smi("info", "-m")
    overview_text = layer.smi("info")
    with open(os.path.join(out_dir, "npu-smi-info-m.txt"), "w") as stream:
        stream.write(mapping_text)
    with open(os.path.join(out_dir, "npu-smi-info.txt"), "w") as stream:
        stream.write(overview_text)
    mapping = parse_mapping(mapping_text)
    chips, processes = parse_overview(overview_text, mapping)
    summary = {"ts": now_iso(), "sample_dir": out_dir,
               "logical_devices": [d["logic"] for d in mapping],
               "chips": {str(k): v for k, v in chips.items()},
               "processes": processes}
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print(json.dumps(summary, indent=1))
    return 0


def cmd_status(cfg):
    path = os.path.join(cfg["state_dir"], "state.json")
    if not os.path.isfile(path):
        print(json.dumps({"status": "never-run", "state_file": path}, indent=1))
        return 1
    with open(path, "r", encoding="utf-8") as stream:
        print(stream.read())
    return 0


def cmd_stop(cfg):
    path = os.path.join(cfg["state_dir"], "stop-requested")
    atomic_write(path, json.dumps({"ts": now_iso(), "requested_by_pid": os.getpid()}) + "\n")
    print("stop requested: %s (the daemon exits at its next cycle)" % path)
    return 0


DEFAULTS = {
    "state_dir": STATE_DIR,
    "container": CONTAINER,
    "repo": REPO,
    "python": "python3",
    "ascend_env": "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh",
    "qwen_env": "models/qwen2.5-0.5b/scripts/env_a3.sh",
    "proto_env": "models/startlux-decision-v1.1-0.8b-instant/scripts/"
                 "env_chunk_proto.sh",
    "proto_work_dir": "work/startlux-decision",
    "interval": 60.0,
    "idle_consecutive": 3,
    "hbm_used_max_mb": 4096,
    "hbm_free_min_mb": 49152,
    "util_max_pct": 5.0,
    "warmup": 32,
    "iterations": 2048,
    "seed": 42,
    "worker_timeout": 7200.0,
    "poll_interval": 15.0,
    "abort_grace": 120.0,
    "log_max_bytes": 1 << 20,
    "log_keep": 3,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["daemon", "once", "dryrun", "sample",
                                            "status", "stop"])
    parser.add_argument("--state-dir", default=DEFAULTS["state_dir"])
    parser.add_argument("--container", default=DEFAULTS["container"])
    parser.add_argument("--repo", default=DEFAULTS["repo"])
    parser.add_argument("--interval", type=float, default=DEFAULTS["interval"])
    parser.add_argument("--idle-consecutive", type=int,
                        default=DEFAULTS["idle_consecutive"])
    parser.add_argument("--hbm-used-max-mb", type=int,
                        default=DEFAULTS["hbm_used_max_mb"],
                        help="actual threshold: hbm_used <= this (4096 MB; the 5%% "
                             "alternative would be 3276.8 MB of 65536 MB)")
    parser.add_argument("--hbm-free-min-mb", type=int,
                        default=DEFAULTS["hbm_free_min_mb"],
                        help="hbm_total - hbm_used >= this (49152 MB = 48 GiB)")
    parser.add_argument("--util-max-pct", type=float, default=DEFAULTS["util_max_pct"])
    parser.add_argument("--warmup", type=int, default=DEFAULTS["warmup"])
    parser.add_argument("--iterations", type=int, default=DEFAULTS["iterations"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--worker-timeout", type=float,
                        default=DEFAULTS["worker_timeout"])
    parser.add_argument("--poll-interval", type=float,
                        default=DEFAULTS["poll_interval"])
    args = parser.parse_args(argv)
    cfg = dict(DEFAULTS)
    for key, value in vars(args).items():
        if key != "command" and value is not None:
            cfg[key] = value
    try:
        if args.command == "sample":
            return cmd_sample(cfg)
        if args.command == "status":
            return cmd_status(cfg)
        if args.command == "stop":
            return cmd_stop(cfg)
        monitor = Monitor(cfg)
        if args.command == "daemon":
            return monitor.daemon()
        if args.command == "once":
            return monitor.once()
        return monitor.dryrun()
    except MonitorError as error:
        sys.stderr.write("monitor: %s\n" % error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
