#!/usr/bin/env python3
"""
Restore operators lost from op_summary / msprof db of a CANN PROF directory.

Problem this fixes: on some msprof builds, CubeCore/MIX tasks (e.g. MatMulV3)
are recorded with an SQE type the profiler cannot classify, so their
task_time rows are tagged UNKNOWN(-1). As a result COMPUTE_TASK_INFO rows
for those ops get globalTaskId values absent from TASK and connectionCats
rows are never written, and the op_summary/op_statistic export drops them.

The timing is still fully present in TASK (keyed by connectionId, matching
the unnamed KERNEL_AICORE events of msprof_*.json) and TASK_PMU_INFO, so the
lost rows can be reconstructed and the db linkage repaired.

What it does on <profiling_dir> (a PROF_XXX produced after
`msprof --export=on`):
  1. Detect broken ops: COMPUTE_TASK_INFO rows whose globalTaskId is absent
     from TASK (in this failure mode they are exactly the dropped ops).
  2. Recover per-task timing by pairing unnamed json timeline events
     (name == args["Task Type"]) with launch records and TASK rows; validate
     via Task Id occurrence sequences against ge_summary.
  3. Write op_summary_*<suffix>.csv = original rows + restored rows and
     op_statistic_*<suffix>.csv (ratio re-normalized).
  4. Repair msprof_*.db in place (COMPUTE_TASK_INFO.globalTaskId,
     connectionCats) so joins/re-export see the ops again.
  5. Repair device_*/sqlite task_time/rts_task UNKNOWN rows with the sibling
     AI_CORE timings and ge_summary task_type.
  6. Write restore_report.txt with all validations.

Backups: modified dbs are copied to <name>.bak before the first modification
(never overwritten). CSVs are never modified; restored ones are new files.

Usage:
    python3 tools/restore_matmulv3.py --profiling_dir PROF_XXX
    python3 tools/restore_matmulv3.py --profiling_dir PROF_XXX --copy
    python3 tools/restore_matmulv3.py --profiling_dir PROF_XXX --suffix _fixed
"""

import argparse
import csv
import glob
import io
import json
import os
import re
import shutil
import sqlite3
import sys

TIMED_EVENT_PAT = re.compile(
    r'\{"name":"[^"]+","pid":\d+,"tid":\d+,"ts":"[0-9.]+","dur":[0-9.]+,"ph":"X",'
    r'"args":\{[^{}]*\}\}')
LAUNCH_PAT = re.compile(r'\{[^{}]*"Mode":"launch"[^{}]*\}')

REPORT_LINES = []


def P(msg):
    print(msg)
    REPORT_LINES.append(msg)


def fail(msg):
    print("[FAIL] " + msg)
    sys.exit(1)


def f3(v):
    return str(round(float(v), 3))


def fmt_ns_us3(ns):
    s = str(int(ns))
    if len(s) <= 3:
        return "0." + s.zfill(3)
    return s[:-3] + "." + s[-3:]


def discover(prof, suffix):
    msprof_dbs = sorted(glob.glob(os.path.join(prof, "msprof_*.db")))
    jsons = sorted(glob.glob(os.path.join(prof, "mindstudio_profiler_output", "msprof_*.json")))

    def latest(patt):
        files = [p for p in sorted(glob.glob(patt), key=os.path.getmtime)
                 if not p.endswith(suffix + ".csv")]
        return files[-1] if files else None

    opsum = latest(os.path.join(prof, "mindstudio_profiler_output", "op_summary_*.csv"))
    opstat = latest(os.path.join(prof, "mindstudio_profiler_output", "op_statistic_*.csv"))
    devices = sorted(glob.glob(os.path.join(prof, "device_*")))
    if len(msprof_dbs) != 1:
        fail("expected exactly one msprof_*.db in %s, found %d" % (prof, len(msprof_dbs)))
    if not jsons:
        fail("no msprof_*.json under mindstudio_profiler_output/; run `msprof --export=on` first")
    if len(devices) != 1:
        fail("expected exactly one device_* dir, found %d: %s (multi-device profiles "
             "are not supported yet)" % (len(devices), devices))
    return {
        "db": msprof_dbs[0], "json": jsons[0], "opsum": opsum, "opstat": opstat,
        "device_dir": devices[0],
    }


def backup_file(path):
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)


def checkpoint(db_path):
    con = sqlite3.connect(db_path, isolation_level=None)
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()


def scan_json(path):
    """Return (unnamed_task_events, launch_by_conn, n_timed).

    unnamed_task_events: ph:X events whose name equals args['Task Type']
    (the exporter lost the op name for these; timing is still valid).
    launch_by_conn: connection_id -> node item_id from launch records.
    """
    events, launches = [], {}
    n_timed = 0
    tail = ""
    with open(path) as f:
        while True:
            chunk = f.read(64 * 1024 * 1024)
            if not chunk:
                break
            buf = tail + chunk
            for m in TIMED_EVENT_PAT.finditer(buf):
                n_timed += 1
                try:
                    ev = json.loads(m.group(0))
                except ValueError:
                    continue
                args = ev.get("args", {})
                if ev.get("name") != args.get("Task Type"):
                    continue
                events.append({
                    "ts": float(ev["ts"]), "dur": float(ev["dur"]),
                    "model": int(args.get("Model Id", 0)),
                    "stream": int(args["Physic Stream Id"]),
                    "task": int(args["Task Id"]), "conn": int(args["connection_id"])})
            for m in LAUNCH_PAT.finditer(buf):
                try:
                    rec = json.loads(m.group(0))
                except ValueError:
                    continue
                conn, item = rec.get("connection_id"), rec.get("item_id")
                if conn is None or item is None:
                    continue
                old = launches.get(conn)
                if old is not None and old != item:
                    fail("conflicting launch records for connection_id %s: %r vs %r"
                         % (conn, old, item))
                launches[conn] = item
            tail = buf[-8192:]
    return events, launches, n_timed


def dedup_by_conn(events):
    """Keep one event per connection_id; identical duplicates are dropped."""
    first_by_conn = {}
    out = []
    for e in events:
        first = first_by_conn.get(e["conn"])
        if first is None:
            first_by_conn[e["conn"]] = e
            out.append(e)
        elif (first["ts"], first["dur"], first["task"]) != (e["ts"], e["dur"], e["task"]):
            fail("conflicting duplicate timeline events for conn %d" % e["conn"])
    return out


def restore_csvs(records, pmu_by_gid, paths, suffix):
    if not paths["opsum"]:
        P("op_summary csv missing; csv restore skipped")
        return
    P("== restoring op_summary csv")
    new_lines = []
    w = io.StringIO()
    cw = csv.writer(w, lineterminator="\n")
    for r in records:
        p = pmu_by_gid.get(r["gtid"], {})
        c, g = r["cti"], r["ge"]
        dur_us = (r["endNs"] - r["startNs"]) / 1000.0
        aic_total_us = p.get("aic_total_time", 0) / 1000.0
        cube = aic_total_us / dur_us * 100.0 if dur_us > 0 else 0.0
        row = [
            str(r["device"]), str(r["model"]), str(r["task"]), str(r["stream"]),
            r["name"], r["optype"], c["state"], g["type"] or "",
            fmt_ns_us3(r["startNs"]) + "\t",
            fmt_ns_us3(r["endNs"] - r["startNs"]),
            "0",
            str(c["block"] if c["block"] is not None else g["block"]),
            str(c["mix"] if c["mix"] is not None else g["mix"]),
            c["hf32"],
            c["inS"] or g["inS"], c["inD"] or g["inD"], c["inF"] or g["inF"],
            c["outS"] or g["outS"], c["outD"] or g["outD"], c["outF"] or g["outF"],
            str(g["ctx"]),
            f3(aic_total_us), str(int(p.get("aic_total_cycles", 0))),
            f3(p.get("aic_mac_time", 0) / 1000.0), f3(p.get("aic_mac_ratio", 0)),
            f3(p.get("aic_scalar_time", 0) / 1000.0), f3(p.get("aic_scalar_ratio", 0)),
            f3(p.get("aic_mte1_time", 0) / 1000.0), f3(p.get("aic_mte1_ratio", 0)),
            f3(p.get("aic_mte2_time", 0) / 1000.0), f3(p.get("aic_mte2_ratio", 0)),
            f3(p.get("aic_fixpipe_time", 0) / 1000.0), f3(p.get("aic_fixpipe_ratio", 0)),
            f3(p.get("aic_icache_miss_rate", 0)),
            f3(p.get("aiv_total_time", 0) / 1000.0), str(int(p.get("aiv_total_cycles", 0))),
            f3(p.get("aiv_vec_time", 0) / 1000.0), f3(p.get("aiv_vec_ratio", 0)),
            f3(p.get("aiv_scalar_time", 0) / 1000.0), f3(p.get("aiv_scalar_ratio", 0)),
            f3(p.get("aiv_mte2_time", 0) / 1000.0), f3(p.get("aiv_mte2_ratio", 0)),
            f3(p.get("aiv_mte3_time", 0) / 1000.0), f3(p.get("aiv_mte3_ratio", 0)),
            f3(p.get("aiv_icache_miss_rate", 0)),
            "%.3f" % cube]
        w.seek(0)
        w.truncate()
        cw.writerow(row)
        new_lines.append((r["startNs"], w.getvalue()))
    with open(paths["opsum"]) as f:
        orig_lines = f.read().splitlines()
    header, data = orig_lines[0], orig_lines[1:]
    starts = [float(row[8].strip()) for row in csv.reader(data)]
    merged = list(zip(starts, range(len(data)), data)) + [
        (ns / 1000.0, 10 ** 9 + i, ln.rstrip("\n")) for i, (ns, ln) in enumerate(new_lines)]
    merged.sort(key=lambda x: (x[0], x[1]))
    out_csv = paths["opsum"].replace(".csv", suffix + ".csv")
    with open(out_csv, "w", newline="") as f:
        f.write(header + "\n")
        for _, _, ln in merged:
            f.write(ln + "\n")
    P("wrote %s (%d rows = %d original + %d restored)" % (
        os.path.basename(out_csv), len(merged), len(data), len(new_lines)))

    if not paths["opstat"]:
        P("op_statistic csv missing; statistic restore skipped")
        return
    P("== restoring op_statistic csv")
    agg = {}
    for r in records:
        a = agg.setdefault(r["optype"], {"core": r["ge"]["type"] or "", "durs": []})
        a["durs"].append((r["endNs"] - r["startNs"]) / 1000.0)
    with open(paths["opstat"]) as f:
        stat_lines = f.read().splitlines()
    stat_header, stat_rows = stat_lines[0], stat_lines[1:]
    total_all = sum(sum(a["durs"]) for a in agg.values())
    for l in stat_rows:
        total_all += float(l.split(",")[4])
    out_rows = []
    for l in stat_rows:
        parts = l.split(",")
        parts[8] = f3(float(parts[4]) / total_all * 100.0)
        out_rows.append(",".join(parts))
    device = str(records[0]["device"])
    for optype, a in agg.items():
        ds = a["durs"]
        tot = sum(ds)
        out_rows.append("%s,%s,%s,%d,%s,%s,%s,%s,%s" % (
            device, optype, a["core"], len(ds), f3(tot), f3(min(ds)),
            f3(tot / len(ds)), f3(max(ds)), f3(tot / total_all * 100.0)))
    out_rows.sort(key=lambda l: -float(l.split(",")[4]))
    out_stat = paths["opstat"].replace(".csv", suffix + ".csv")
    with open(out_stat, "w", newline="") as f:
        f.write(stat_header + "\n" + "\n".join(out_rows) + "\n")
    for optype, a in sorted(agg.items(), key=lambda kv: -sum(kv[1]["durs"])):
        ds = a["durs"]
        P("%s: count=%d total=%.3fus min=%.3f avg=%.3f max=%.3f ratio=%.3f%%" % (
            optype, len(ds), sum(ds), min(ds), sum(ds) / len(ds), max(ds),
            sum(ds) / total_all * 100.0))
    P("wrote " + os.path.basename(out_stat))


def repair_msprof_db(mdb, by_name, records, optypes, db_path):
    P("== repairing msprof db")
    backup_file(db_path)
    existing_ids = set(r[0] for r in mdb.execute("SELECT globalTaskId FROM COMPUTE_TASK_INFO"))
    new_ids = set(r["gtid"] for r in records)
    clash = new_ids & existing_ids
    if clash:
        fail("target globalTaskId values already used by other CTI rows: %d" % len(clash))
    rec_by_name = {}
    for r in records:
        rec_by_name.setdefault(r["name"], []).append(r)
    updates = []
    for nm, info in by_name.items():
        ids = sorted(info["gids"])
        tasks = sorted(rec_by_name[nm], key=lambda r: (r["startNs"], r["conn"]))
        if len(ids) != len(tasks):
            fail("per-name count mismatch for %s" % nm)
        for old, rec in zip(ids, tasks):
            updates.append((rec["gtid"], old))
    upd_ids = [u[0] for u in updates]
    if len(set(upd_ids)) != len(upd_ids):
        fail("duplicate target globalTaskId")
    has_conn_cats = mdb.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='connectionCats'"
    ).fetchone()[0] > 0
    if not has_conn_cats:
        P("connectionCats table absent in this msprof db; connectionCats insert skipped")
    mdb.execute("BEGIN IMMEDIATE")
    mdb.executemany("UPDATE COMPUTE_TASK_INFO SET globalTaskId=? WHERE globalTaskId=?", updates)
    if has_conn_cats:
        mdb.executemany("INSERT INTO connectionCats VALUES (?, 'HostToDevice')",
                        [(r["conn"],) for r in records])
    mdb.commit()
    qmarks = ",".join("?" * len(optypes))
    joined = mdb.execute(
        "SELECT COUNT(*) FROM COMPUTE_TASK_INFO c JOIN TASK t ON c.globalTaskId=t.globalTaskId "
        "JOIN STRING_IDS s ON c.opType=s.id WHERE s.value IN (%s)" % qmarks,
        tuple(optypes)).fetchone()[0]
    P("CTI rows updated: %d; connectionCats +%d; TASK<->CTI join for restored ops: %d/%d" % (
        len(updates), len(records) if has_conn_cats else 0, joined, len(records)))
    if joined != len(records):
        fail("join verification failed")
    P("msprof db integrity: " + mdb.execute("PRAGMA integrity_check").fetchone()[0])


def repair_device_dbs(dev_summary_db, dev_counter_db):
    P("== repairing device dbs")
    ge_type = {}
    ddb = sqlite3.connect(dev_summary_db)
    for (task_id, stream_id, index_id, task_type) in ddb.execute(
            "SELECT task_id, stream_id, index_id, task_type FROM ge_summary"):
        ge_type[(task_id, stream_id, index_id)] = task_type
    ddb.close()
    for path, table in ((dev_summary_db, "task_time"), (dev_counter_db, "rts_task")):
        if not os.path.exists(path):
            P("skip missing " + path)
            continue
        dur_col = "duration_time" if table == "task_time" else "duration"
        d = sqlite3.connect(path)
        n_unk = d.execute("SELECT COUNT(*) FROM %s WHERE task_type='UNKNOWN'" % table).fetchone()[0]
        if n_unk == 0:
            P("%s.%s: no UNKNOWN rows" % (os.path.basename(path), table))
            d.close()
            continue
        backup_file(path)
        sib = {}
        for row in d.execute(
                "SELECT task_id, stream_id, index_id, batch_id, start_time, %s FROM %s "
                "WHERE task_type='AI_CORE'" % (dur_col, table)):
            sib[(row[0], row[1], row[2], row[3])] = (row[4], row[5])
        upd = []
        skipped = 0
        for rowid, tk, st, ix, ba in d.execute(
                "SELECT rowid, task_id, stream_id, index_id, batch_id FROM %s "
                "WHERE task_type='UNKNOWN'" % table):
            v = sib.get((tk, st, ix, ba))
            tt = ge_type.get((tk, st, ix))
            if v is None or tt is None:
                skipped += 1
                continue
            upd.append((tt, v[0], v[1], rowid))
        del sib
        d.execute("BEGIN IMMEDIATE")
        d.executemany(
            "UPDATE %s SET task_type=?, start_time=?, %s=? WHERE rowid=?" % (table, dur_col), upd)
        d.commit()
        left = d.execute("SELECT COUNT(*) FROM %s WHERE task_type='UNKNOWN'" % table).fetchone()[0]
        d.close()
        P("%s.%s: fixed %d UNKNOWN rows (skipped %d, remaining UNKNOWN=%d)" % (
            os.path.basename(path), table, len(upd), skipped, left))
        if skipped:
            P("warning: %d UNKNOWN rows had no AI_CORE sibling / ge_summary match; left as-is"
              % skipped)
    del ge_type


def main():
    ap = argparse.ArgumentParser(
        description="Restore ops lost from op_summary/msprof db (e.g. MatMulV3 with UNKNOWN task_type)")
    ap.add_argument("--profiling_dir", required=True, help="PROF_XXX directory to restore")
    ap.add_argument("--copy", action="store_true",
                    help="work on a sibling copy <dir>_restored instead of in place")
    ap.add_argument("--suffix", default="_restored",
                    help="suffix for restored CSVs (default _restored)")
    args = ap.parse_args()

    prof = os.path.abspath(args.profiling_dir)
    if not os.path.isdir(prof):
        fail("not a directory: " + prof)
    if args.copy:
        target = prof.rstrip("/") + "_restored"
        if os.path.exists(target):
            fail("copy target already exists: " + target)
        P("copying %s -> %s" % (prof, target))
        shutil.copytree(prof, target)
        prof = target

    paths = discover(prof, args.suffix)
    P("profiling dir : " + prof)
    P("msprof db     : " + os.path.basename(paths["db"]))
    P("timeline json : " + os.path.basename(paths["json"]))
    P("op_summary    : " + (os.path.basename(paths["opsum"]) if paths["opsum"]
                            else "(missing, csv restore skipped)"))
    P("device        : " + os.path.basename(paths["device_dir"]))
    dev_summary_db = os.path.join(paths["device_dir"], "sqlite", "ai_core_op_summary.db")
    dev_counter_db = os.path.join(paths["device_dir"], "sqlite", "op_counter.db")

    checkpoint(paths["db"])
    mdb = sqlite3.connect(paths["db"], isolation_level=None)

    sid = dict(mdb.execute("SELECT id, value FROM STRING_IDS"))
    by_name = {}
    name_to_optype = {}
    n_broken = 0
    for (gid, name, optype, blk, mix, hf32, opstate, inf, ind, ins, outf, outd, outs) in mdb.execute(
            "SELECT c.globalTaskId, c.name, c.opType, c.blockNum, c.mixBlockNum, c.hf32Eligible, "
            "c.opState, c.inputFormats, c.inputDataTypes, c.inputShapes, c.outputFormats, "
            "c.outputDataTypes, c.outputShapes FROM COMPUTE_TASK_INFO c "
            "WHERE c.globalTaskId NOT IN (SELECT globalTaskId FROM TASK)"):
        n_broken += 1
        nm = sid[name]
        name_to_optype[nm] = sid[optype]
        info = by_name.get(nm)
        if info is None:
            info = by_name[nm] = {"gids": [], "static": {
                "block": blk, "mix": mix,
                "state": sid.get(opstate, "dynamic") if opstate is not None else "dynamic",
                "hf32": sid.get(hf32, "") if hf32 is not None else "",
                "inF": sid.get(inf, "") if inf is not None else "",
                "inD": sid.get(ind, "") if ind is not None else "",
                "inS": sid.get(ins, "") if ins is not None else "",
                "outF": sid.get(outf, "") if outf is not None else "",
                "outD": sid.get(outd, "") if outd is not None else "",
                "outS": sid.get(outs, "") if outs is not None else ""}}
        info["gids"].append(gid)
    P("COMPUTE_TASK_INFO rows not joinable with TASK: %d" % n_broken)
    if not by_name:
        P("nothing to restore (already fixed or no loss in this profile)")
        mdb.close()
        write_report(prof)
        return
    P("broken ops: %d names (%s)" % (
        len(by_name), ", ".join(sorted(name_to_optype)[:6]) +
        (" ..." if len(name_to_optype) > 6 else "")))

    events, launches, n_timed = scan_json(paths["json"])
    P("timeline ph:X events scanned: %d; unnamed task events: %d; launch records: %d" % (
        n_timed, len(events), len(launches)))
    broken_names = set(by_name)
    events = [e for e in events if e["conn"] in launches and launches[e["conn"]] in broken_names]
    events = dedup_by_conn(events)
    P("unnamed events on broken-op launches: %d (unique conns)" % len(events))

    task_by_conn = {}
    for row in mdb.execute(
            "SELECT connectionId, startNs, endNs, globalTaskId, deviceId, streamId, taskId, "
            "modelId FROM TASK"):
        task_by_conn.setdefault(row[0], []).append(row)
    absent = [e["conn"] for e in events if e["conn"] not in task_by_conn]
    if absent:
        fail("%d recovered conns missing in TASK, e.g. %s" % (len(set(absent)), absent[:3]))
    dupconns = [e["conn"] for e in events if len(task_by_conn[e["conn"]]) != 1]
    if dupconns:
        fail("%d recovered conns have multiple TASK rows, e.g. %s"
             % (len(set(dupconns)), sorted(set(dupconns))[:3]))
    bad_ts = 0
    for e in events:
        t = task_by_conn[e["conn"]][0]
        if abs(t[1] / 1000.0 - e["ts"]) > 0.3 or t[6] != e["task"] or t[5] != e["stream"]:
            bad_ts += 1
    if bad_ts:
        fail("%d TASK rows disagree with json events" % bad_ts)
    P("TASK vs json event cross-check: OK (%d/%d)" % (len(events), len(events)))

    ev_by_name = {}
    for e in events:
        ev_by_name.setdefault(launches[e["conn"]], []).append(e)
    if set(ev_by_name) != broken_names:
        fail("launch names on recovered events != broken CTI names: %s"
             % (set(ev_by_name) ^ broken_names))

    optypes = set(name_to_optype.values())
    ddb = sqlite3.connect(dev_summary_db)
    qmarks = ",".join("?" * len(optypes))
    ge = {}
    for (task_id, stream_id, index_id, batch_id, op_name, task_type, op_state, model_id,
         block_num, mix_block_num, inF, inD, inS, outF, outD, outS, context_id) in ddb.execute(
            "SELECT task_id, stream_id, index_id, batch_id, op_name, task_type, op_state, model_id, "
            "block_num, mix_block_num, input_formats, input_data_types, input_shapes, "
            "output_formats, output_data_types, output_shapes, context_id "
            "FROM ge_summary WHERE op_type IN (%s)" % qmarks, tuple(optypes)):
        ge.setdefault(op_name, []).append({
            "task": task_id, "stream": stream_id, "index": index_id, "batch": batch_id,
            "type": task_type, "state": op_state, "model": model_id,
            "block": block_num, "mix": mix_block_num,
            "inF": inF, "inD": inD, "inS": inS, "outF": outF, "outD": outD, "outS": outS,
            "ctx": context_id})
    ddb.close()
    if set(ge) != broken_names:
        fail("ge_summary names != broken names: %s" % (set(ge) ^ broken_names))

    for nm, rows in ge.items():
        rows.sort(key=lambda r: r["index"])
        evs = sorted(ev_by_name[nm], key=lambda e: e["ts"])
        if len(rows) != len(evs) or len(rows) != len(by_name[nm]["gids"]):
            fail("count mismatch for %s: ge %d, events %d, CTI %d" % (
                nm, len(rows), len(evs), len(by_name[nm]["gids"])))
        for r, e in zip(rows, evs):
            if r["task"] != e["task"]:
                fail("occurrence pairing broke for %s: ge task %d vs event task %d"
                     % (nm, r["task"], e["task"]))
    P("occurrence pairing validated on all %d names (Task Id sequences identical)" % len(ge))

    records = []
    for nm, rows in ge.items():
        evs = sorted(ev_by_name[nm], key=lambda e: e["ts"])
        for r, e in zip(rows, evs):
            t = task_by_conn[e["conn"]][0]
            records.append({
                "conn": e["conn"], "name": nm, "optype": name_to_optype[nm],
                "task": e["task"], "stream": e["stream"], "model": r["model"],
                "device": t[4], "index": r["index"], "batch": r["batch"],
                "ge": r, "cti": by_name[nm]["static"],
                "startNs": t[1], "endNs": t[2], "gtid": t[3]})
    del task_by_conn, events, launches

    mdb.execute("CREATE TEMP TABLE recids (gid INTEGER PRIMARY KEY, conn INTEGER)")
    mdb.executemany("INSERT INTO recids VALUES (?,?)", [(r["gtid"], r["conn"]) for r in records])
    pmu_by_gid = {}
    for gid, name, value in mdb.execute(
            "SELECT p.globalTaskId, s.value, p.value FROM TASK_PMU_INFO p "
            "JOIN STRING_IDS s ON p.name=s.id JOIN recids m ON p.globalTaskId=m.gid"):
        pmu_by_gid.setdefault(gid, {})[name] = value
    n_nopmu = sum(1 for r in records if r["gtid"] not in pmu_by_gid)
    if n_nopmu:
        P("warning: %d recovered TASK rows have no TASK_PMU_INFO (metrics default to 0)" % n_nopmu)

    restore_csvs(records, pmu_by_gid, paths, args.suffix)
    del pmu_by_gid

    repair_msprof_db(mdb, by_name, records, optypes, paths["db"])
    del by_name, records
    repair_device_dbs(dev_summary_db, dev_counter_db)
    mdb.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    mdb.close()
    P("done. modified dbs have .bak backups; restored CSVs use suffix %r" % args.suffix)
    write_report(prof)


def write_report(prof):
    path = os.path.join(prof, "restore_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(REPORT_LINES) + "\n")
    print("report written to " + path)


if __name__ == "__main__":
    main()
