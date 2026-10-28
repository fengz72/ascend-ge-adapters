#!/usr/bin/env python3
"""Run one bounded POSIX job with physical-device flock locks and local evidence.

Run this on the execution host, inside the chosen container when applicable.
It coordinates cooperating jobs; it does not inspect NPU work or sandbox code.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid


POLL_SECONDS = 0.05
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def proc_stat(pid):
    """Linux state, process group, and start ticks; None on other POSIX hosts."""
    try:
        raw = Path("/proc/{}/stat".format(pid)).read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, ProcessLookupError, NotADirectoryError):
        return None
    fields = raw[raw.rfind(")") + 2:].split()
    return {"state": fields[0], "pgid": int(fields[2]), "start_ticks": fields[19]}


def process_identity(pid):
    result = {"pid": pid}
    try:
        state = proc_stat(pid)
    except PermissionError:
        state = None
    if state is not None:
        result["linux_start_ticks"] = state["start_ticks"]
        boot_id = Path("/proc/sys/kernel/random/boot_id")
        if boot_id.is_file():
            result["linux_boot_id"] = boot_id.read_text().strip()
    else:
        # Recorded evidence only. Cancellation never signals a PID read from disk.
        result["observed_at"] = utc_now()
    return result


def live_group_members(pgid):
    """Return non-zombie members without reaping the owned group leader.

    Keeping that child unreaped reserves its PID/PGID until all group signals
    have finished, avoiding a stale-PID process-group kill.
    """
    if sys.platform.startswith("linux") and Path("/proc").is_dir():
        try:
            members = []
            for entry in Path("/proc").iterdir():
                if entry.name.isdigit():
                    state = proc_stat(int(entry.name))
                    if state and state["pgid"] == pgid and state["state"] not in ("Z", "X"):
                        members.append(int(entry.name))
            return members
        except PermissionError:
            # A hidden process must not be silently counted as absent.
            pass
    # macOS and other POSIX hosts: ps contains no command arguments or environment.
    result = subprocess.run(
        ["ps", "-axo", "pid=,pgid=,stat="], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    members = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and int(fields[1]) == pgid and not fields[2].startswith("Z"):
            members.append(int(fields[0]))
    return members


def group_leader_exited(pid):
    state = proc_stat(pid)
    if state is not None:
        return state["state"] in ("Z", "X")
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("cannot inspect owned task process")
    status = result.stdout.strip()
    return not status or status.startswith("Z")


def send_group(proc, number):
    # The supervisor has not called poll/wait yet, so the child PID cannot be reused.
    try:
        os.killpg(proc.pid, number)
    except ProcessLookupError:
        pass


def cancellation_requested(run_dir, task_id):
    try:
        return read_json(run_dir / "cancel.json").get("task_id") == task_id
    except FileNotFoundError:
        return False


def stop_group(proc, grace, publish):
    """Stop only this owned process group, retaining locks until verified stopped."""
    send_group(proc, signal.SIGTERM)
    deadline = time.monotonic() + grace
    sent_kill = False
    reported_blocked = False
    while True:
        # Escalation of our still-owned PGID must not depend on an observer
        # recovering. Failed verification keeps locks held, not the task alive.
        if time.monotonic() >= deadline and not sent_kill:
            send_group(proc, signal.SIGKILL)
            sent_kill = True
        try:
            members = live_group_members(proc.pid)
            if not members:
                # Recheck after another scheduling interval; the leader remains unreaped.
                time.sleep(POLL_SECONDS)
                if not live_group_members(proc.pid):
                    return proc.wait()
        except Exception as error:
            # Losing observability must not unlock a potentially busy device.
            try:
                publish("verification_blocked", detail=str(error))
            except Exception:
                # A status-write failure must not suppress task cleanup either.
                pass
            time.sleep(POLL_SECONDS)
            continue
        if sent_kill and time.monotonic() >= deadline + 5 and not reported_blocked:
            publish("cleanup_blocked", remaining_pids=members)
            reported_blocked = True
        time.sleep(POLL_SECONDS)


def supervise(request_path):
    request = read_json(request_path)
    run_dir = request_path.parent
    signal_number = [None]

    def remember_signal(number, _frame):
        signal_number[0] = number

    for number in STOP_SIGNALS:
        signal.signal(number, remember_signal)
    state = {
        "task_id": request["task_id"], "host": socket.gethostname(),
        "supervisor": process_identity(os.getpid()),
        "cwd": request["cwd"], "argv": request["argv"],
        "devices": request["devices"], "lock_root": request["lock_root"],
        "started_at": utc_now(), "device_release_checked": False,
    }

    def publish(phase, **fields):
        state.update(fields)
        state["state"] = phase
        state["updated_at"] = utc_now()
        write_json(run_dir / "status.json", state)

    deadline = time.monotonic() + request["timeout"]
    locks = []
    proc = None
    reason = "supervisor_error"
    exit_code = 70
    task_returncode = None

    def stop_reason():
        if signal_number[0] is not None:
            return "signal_{}".format(signal_number[0]), 128 + signal_number[0]
        if cancellation_requested(run_dir, request["task_id"]):
            return "cancel_requested", 130
        if time.monotonic() >= deadline:
            return "timeout", 124
        return None

    try:
        publish("waiting_for_locks")
        lock_root = Path(request["lock_root"])
        lock_root.mkdir(parents=True, exist_ok=True)
        for device in request["devices"]:
            filename = hashlib.sha256(device.encode("utf-8")).hexdigest() + ".lock"
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(lock_root / filename, flags, 0o600)
            locks.append(fd)
            while True:
                stopped = stop_reason()
                if stopped:
                    reason, exit_code = stopped
                    return exit_code
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(POLL_SECONDS)
            # This owner hint is not a lease; never unlink or steal a lock file.
            owner = {"task_id": request["task_id"], "run_dir": str(run_dir),
                     "device": device, "supervisor": state["supervisor"]}
            os.ftruncate(fd, 0)
            os.write(fd, (json.dumps(owner, ensure_ascii=False) + "\n").encode("utf-8"))
        stopped = stop_reason()
        if stopped:
            reason, exit_code = stopped
            return exit_code
        with (run_dir / "task.log").open("ab", buffering=0) as output:
            proc = subprocess.Popen(
                request["argv"], cwd=request["cwd"], stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=tuple(locks),
            )
        publish("running", task=process_identity(proc.pid), pgid=proc.pid)
        while True:
            stopped = stop_reason()
            if stopped:
                reason, exit_code = stopped
                publish("stopping", reason=reason)
                task_returncode = stop_group(proc, request["grace"], publish)
                break
            if group_leader_exited(proc.pid):
                # A command may exit before its background children. Clean those
                # children before reaping the stable group leader or dropping locks.
                residual = live_group_members(proc.pid)
                if residual:
                    publish("stopping", reason="background_children", remaining_pids=residual)
                    task_returncode = stop_group(proc, request["grace"], publish)
                    reason, exit_code = "background_children", 125
                else:
                    task_returncode = proc.wait()
                    reason = "exited"
                    exit_code = task_returncode if task_returncode >= 0 else 128 - task_returncode
                break
            time.sleep(POLL_SECONDS)
    except Exception as error:
        traceback.print_exc()
        state["error"] = str(error)
    finally:
        if proc is not None and task_returncode is None:
            # Unexpected wrapper errors do not make a running task safe to unlock.
            while task_returncode is None:
                try:
                    task_returncode = stop_group(proc, request["grace"], publish)
                except Exception as error:
                    traceback.print_exc()
                    state["cleanup_error"] = str(error)
                    time.sleep(POLL_SECONDS)
        for fd in locks:
            os.close(fd)
        publish("finished", reason=reason, exit_code=exit_code,
                task_returncode=task_returncode, process_group_stopped=True,
                finished_at=utc_now(), locks_released_at=utc_now())
    return exit_code


def positive_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def run(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("provide command argv after --")
    cwd = Path(args.cwd).resolve(strict=True)
    if not cwd.is_dir():
        raise ValueError("cwd must be a directory")
    devices = sorted(set(device.strip() for device in args.device))
    if not all(devices):
        raise ValueError("device identity must not be empty")
    run_dir = Path(args.run_dir).absolute()
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    run_dir = run_dir.resolve(strict=True)
    request = {
        "task_id": uuid.uuid4().hex, "created_at": utc_now(),
        "cwd": str(cwd), "argv": command, "devices": devices,
        "timeout": args.timeout, "grace": args.grace,
        "lock_root": str(Path(args.lock_root).resolve()),
    }
    request_path = run_dir / "request.json"
    write_json(request_path, request)
    received = [None]

    def remember_signal(number, _frame):
        received[0] = number

    for number in STOP_SIGNALS:
        signal.signal(number, remember_signal)
    with (run_dir / "supervisor.log").open("ab", buffering=0) as output:
        supervisor = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_supervise", str(request_path)],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    forwarded = False
    while supervisor.poll() is None:
        if received[0] is not None and not forwarded:
            # The owned child is not reaped; this is not a PID read from old metadata.
            supervisor.send_signal(received[0])
            forwarded = True
        time.sleep(POLL_SECONDS)
    try:
        status = read_json(run_dir / "status.json")
    except FileNotFoundError:
        status = {}
    finished = status.get("state") == "finished"
    print(json.dumps({
        "task_id": request["task_id"], "run_dir": str(run_dir),
        "state": status.get("state", "unconfirmed"),
        "exit_code": status.get("exit_code") if finished else None,
        "supervisor_returncode": supervisor.returncode,
    }, ensure_ascii=False))
    return status["exit_code"] if finished else 70


def cancel(args):
    run_dir = Path(args.run_dir).resolve(strict=True)
    request = read_json(run_dir / "request.json")
    write_json(run_dir / "cancel.json", {"task_id": request["task_id"], "requested_at": utc_now()})
    print(json.dumps({"task_id": request["task_id"], "cancel_requested": True,
                      "run_dir": str(run_dir)}, ensure_ascii=False))
    # This acknowledges a request, not completion or device release.
    return 0


def main():
    # SIG_IGN survives exec. On Linux it auto-reaps children, destroying both
    # their real exit status and the unreaped-leader PGID ownership guarantee.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    if len(sys.argv) == 3 and sys.argv[1] == "_supervise":
        return supervise(Path(sys.argv[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="operation", required=True)
    launch = subcommands.add_parser("run", help="run on the actual execution host")
    launch.add_argument("--cwd", required=True)
    launch.add_argument("--run-dir", required=True, help="a new, private evidence directory")
    launch.add_argument("--lock-root", required=True, help="shared by all users of the same physical devices")
    launch.add_argument("--device", action="append", required=True, help="stable physical identity, repeat for multiple devices")
    launch.add_argument("--timeout", type=positive_seconds, required=True, help="lock wait plus command time, in seconds")
    launch.add_argument("--grace", type=positive_seconds, default=2.0, help="TERM to KILL grace in seconds")
    launch.add_argument("command", nargs=argparse.REMAINDER)
    cancellation = subcommands.add_parser("cancel", help="request cancellation; then inspect status.json")
    cancellation.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    try:
        if args.operation == "run":
            return run(args)
        return cancel(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
