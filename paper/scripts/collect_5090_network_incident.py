#!/usr/bin/env python3
"""Bounded, read-only network incident evidence for the dedicated 5090 host.

No repair, network reconfiguration, experiment import, or remote upload is done.
The same lock is used by user and privileged installations to prevent duplicates.
"""
from __future__ import annotations

import argparse
import datetime as dt
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import tempfile
import time
from typing import Any


MAX_BYTES = 10 * 1024 * 1024
RETAINED_FILES = 8
COMMAND_BYTES = 256 * 1024
PEER_IPS = ("100.74.94.125", "100.118.62.81")
TOKEN_RE = re.compile(r"(?i)(?:tskey|authkey|nodekey|privkey|machinekey|secret)[-_:][A-Za-z0-9_:=+/.\-]{6,}")
ASSIGN_RE = re.compile(r"(?i)\b(password|passwd|token|authorization|authkey|privatekey|secret)[\"']?\s*[=:]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)")
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def redact(value: Any) -> Any:
    """Redact common credential forms before any captured text reaches disk."""
    if isinstance(value, str):
        return BEARER_RE.sub("Bearer [REDACTED]", ASSIGN_RE.sub(lambda m: m.group(1) + "=[REDACTED]", TOKEN_RE.sub("[REDACTED]", value)))
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {key: ("[REDACTED]" if re.search(r"(?i)^(password|passwd|token|authorization|authkey|privatekey|secret|nodekey|machinekey)$", str(key)) else redact(item)) for key, item in value.items()}
    return value


class AlreadyRunning(RuntimeError):
    pass


class CollectorLock:
    def __init__(self, path: Path, owner_uid: int):
        # Linux fs.protected_regular can reject root O_CREAT on a user-owned
        # regular file in sticky /tmp. Existing files require no create flag.
        try:
            self.fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        except FileNotFoundError:
            try:
                self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                self.fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            info = os.fstat(self.fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid not in (0, owner_uid):
                raise RuntimeError("collector lock has unexpected type or owner")
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AlreadyRunning("another network collector holds the lock") from exc
            os.ftruncate(self.fd, 0)
            os.write(self.fd, (str(os.getpid()) + "\n").encode())
            os.fsync(self.fd)
        except BaseException:
            os.close(self.fd)
            raise

    def close(self) -> None:
        os.close(self.fd)


class JsonlWriter:
    def __init__(self, directory: Path, max_bytes: int = MAX_BYTES, retained: int = RETAINED_FILES):
        if max_bytes < 1024 or retained < 1:
            raise ValueError("invalid retention bounds")
        directory.mkdir(parents=True, exist_ok=True, mode=0o750)
        if directory.is_symlink():
            raise RuntimeError("output directory must not be a symlink")
        self.directory, self.max_bytes, self.retained = directory, max_bytes, retained
        self.path = directory / "network.jsonl"

    def append(self, row: dict[str, Any]) -> None:
        payload = (json.dumps(redact(row), ensure_ascii=True, separators=(",", ":")) + "\n").encode()
        if len(payload) > self.max_bytes:
            payload = (json.dumps({"kind": "record_oversize", "utc": utc_now(), "original_bytes": len(payload)}) + "\n").encode()
        # Reserve a newline so a power-loss partial last record stays isolated.
        if self.path.exists() and self.path.stat().st_size + len(payload) + 1 > self.max_bytes:
            for index in range(self.retained - 1, 0, -1):
                source = self.path if index == 1 else self.directory / ("network.jsonl." + str(index - 1))
                target = self.directory / ("network.jsonl." + str(index))
                if source.exists():
                    os.replace(source, target)
            if self.retained == 1:
                self.path.unlink()
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o640)
        try:
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                os.write(fd, b"\n")
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)


class ProbeRunner:
    def __init__(self, deadline_seconds: float = 60.0):
        self.deadline = time.monotonic() + deadline_seconds

    def run(self, argv: list[str], timeout: float = 3.0) -> dict[str, Any]:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            return {"error": "snapshot_deadline_exceeded"}
        start = time.monotonic()
        result: dict[str, Any] = {}
        # File-backed streams avoid retaining an unexpected journal burst in RAM.
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                command_deadline = start + min(timeout, remaining)
                while process.poll() is None:
                    if stdout.tell() >= COMMAND_BYTES or stderr.tell() >= COMMAND_BYTES:
                        result["error"] = "output_limit_reached"
                        process.kill()
                        break
                    if time.monotonic() >= command_deadline:
                        result["error"] = "timeout"
                        process.kill()
                        break
                    time.sleep(max(0.0, min(0.02, command_deadline - time.monotonic())))
                result["returncode"] = process.wait()
            except OSError as exc:
                result["error"] = type(exc).__name__
                result["detail"] = str(exc)
            for name, stream in (("stdout", stdout), ("stderr", stderr)):
                size = stream.tell()
                stream.seek(0)
                result[name] = stream.read(COMMAND_BYTES).decode("utf-8", errors="replace")
                if size > COMMAND_BYTES:
                    result[name + "_truncated"] = True
        result["elapsed_seconds"] = round(time.monotonic() - start, 6)
        return result


def read_local(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            return {"value": stream.read(16384).decode("utf-8", errors="replace").strip()}
    except OSError as exc:
        return {"error": type(exc).__name__, "detail": str(exc)}


def probe_tcp(address: str, port: int, deadline: float, timeout: float = 1.0) -> dict[str, Any]:
    """TCP handshake only: no authentication or application payload is sent."""
    started = time.monotonic()
    result: dict[str, Any] = {"address": address, "port": port, "path_reachable": None, "listener_open": None}
    remaining = deadline - started
    if remaining <= 0:
        return {**result, "status": "skipped", "error": "snapshot_deadline_exceeded"}
    connection = None
    try:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.settimeout(min(1.0, timeout, remaining))
        code = connection.connect_ex((address, port))
        result["errno"] = code
        if code == 0:
            result.update(status="connected", path_reachable=True, listener_open=True)
        elif code == errno.ECONNREFUSED:
            result.update(status="connection_refused", path_reachable=True, listener_open=False)
        elif code in (errno.ETIMEDOUT, errno.EAGAIN, errno.EWOULDBLOCK):
            result["status"] = "timeout"
        elif code in (errno.EPERM, errno.EACCES):
            result["status"] = "permission_denied"
        else:
            result["status"] = "connect_error"
    except (TimeoutError, socket.timeout):
        result["status"] = "timeout"
    except OSError as exc:
        result.update(status="connect_error", error=type(exc).__name__, errno=exc.errno)
    finally:
        if connection is not None:
            connection.close()
    result["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return result


def summarize_tailscale(result: dict[str, Any]) -> dict[str, Any]:
    """Never persist the full peer inventory, node keys, or user accounts."""
    summary = {key: value for key, value in result.items() if key not in ("stdout", "stderr")}
    if result.get("returncode") != 0:
        # Even CLI stderr could echo authentication material; retain its class only.
        summary["status"] = "unavailable"
        return summary
    try:
        raw = json.loads(result["stdout"])
        own = raw.get("Self") or {}
        summary.update({"backend_state": raw.get("BackendState"), "self_online": own.get("Online"), "self_ips": own.get("TailscaleIPs"), "health": redact(raw.get("Health", [])), "peers": []})
        for peer in (raw.get("Peer") or {}).values():
            selected = sorted(set(peer.get("TailscaleIPs") or []) & set(PEER_IPS))
            if selected:
                summary["peers"].append({"ips": selected, "online": peer.get("Online"), "active": peer.get("Active"), "last_seen": peer.get("LastSeen"), "relay": peer.get("Relay"), "rx_bytes": peer.get("RxBytes"), "tx_bytes": peer.get("TxBytes")})
    except (ValueError, TypeError, AttributeError) as exc:
        summary["error"] = "invalid_tailscale_status:" + type(exc).__name__
    return summary


def collect_snapshot(interface: str, gateway: str, deadline_seconds: float = 60.0, runner: ProbeRunner | None = None) -> dict[str, Any]:
    started = time.monotonic()
    runner = runner or ProbeRunner(deadline_seconds)
    row: dict[str, Any] = {"kind": "snapshot", "utc": utc_now(), "monotonic_seconds": started, "pid": os.getpid(), "uid": os.getuid(), "boot_id": read_local(Path("/proc/sys/kernel/random/boot_id")), "interface": interface, "probes": {}}
    probes = {
        "link": ["ip", "-j", "-s", "link", "show", "dev", interface],
        "address": ["ip", "-j", "address", "show", "dev", interface],
        "route": ["ip", "-j", "route", "show"],
        "gateway_neighbour": ["ip", "-j", "neigh", "show", "to", gateway],
        "ssh_listen": ["ss", "-H", "-lnt", "( sport = :22 )"],
        "service_state": ["systemctl", "show", "ssh.service", "tailscaled.service", "NetworkManager.service", "systemd-networkd.service", "-p", "Id", "-p", "ActiveState", "-p", "SubState", "-p", "Result"],
        "clock": ["timedatectl", "show", "-p", "NTPSynchronized", "-p", "TimeUSec", "-p", "Timezone"],
        "ping_gateway": ["ping", "-n", "-c", "1", "-W", "1", gateway],
        "ping_mac_tailscale": ["ping", "-n", "-c", "1", "-W", "1", PEER_IPS[0]],
        "ping_5080_tailscale": ["ping", "-n", "-c", "1", "-W", "1", PEER_IPS[1]],
    }
    for name, argv in probes.items():
        try:
            row["probes"][name] = runner.run(argv)
        except Exception as exc:
            row["probes"][name] = {"error": type(exc).__name__, "detail": str(exc)}
    try:
        row["probes"]["tailscale"] = summarize_tailscale(runner.run(["tailscale", "status", "--json"], timeout=5.0))
    except Exception as exc:
        row["probes"]["tailscale"] = {"error": type(exc).__name__}
    for name, address, port in (("tcp_gateway", gateway, 80), ("tcp_mac_tailscale", PEER_IPS[0], 22), ("tcp_5080_tailscale", PEER_IPS[1], 22), ("tcp_mac_lan", "192.168.0.45", 22)):
        try:
            row["probes"][name] = probe_tcp(address, port, getattr(runner, "deadline", started + deadline_seconds))
        except Exception as exc:
            row["probes"][name] = {"error": type(exc).__name__, "path_reachable": None}
    row["local"] = {name: read_local(Path(path)) for name, path in {"carrier": f"/sys/class/net/{interface}/carrier", "operstate": f"/sys/class/net/{interface}/operstate", "loadavg": "/proc/loadavg", "meminfo": "/proc/meminfo", "uptime": "/proc/uptime"}.items()}
    row["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return row


JOURNAL_FIELDS = ("__REALTIME_TIMESTAMP", "_BOOT_ID", "_SYSTEMD_UNIT", "SYSLOG_IDENTIFIER", "PRIORITY", "MESSAGE")


def journal_event(runner: ProbeRunner, cursor: str | None) -> tuple[dict[str, Any], str | None]:
    # --no-tail preserves the oldest outstanding entry after the cursor. The
    # runner caps bytes/time, so a burst is drained in bounded complete prefixes.
    fields = ",".join(("__CURSOR",) + JOURNAL_FIELDS)
    argv = ["journalctl", "--no-pager", "--no-tail", "-q", "-o", "json", "--output-fields=" + fields, "_TRANSPORT=kernel", "+", "_SYSTEMD_UNIT=NetworkManager.service", "+", "_SYSTEMD_UNIT=systemd-networkd.service", "+", "_SYSTEMD_UNIT=tailscaled.service", "+", "_SYSTEMD_UNIT=ssh.service", "+", "SYSLOG_IDENTIFIER=sshd"]
    argv += ["--after-cursor", cursor] if cursor else ["--since", "-2 minutes"]
    result = runner.run(argv, timeout=8.0)
    event: dict[str, Any] = {"kind": "journal", "utc": utc_now(), "uid": os.getuid(), "returncode": result.get("returncode"), "error": result.get("error"), "stderr": result.get("stderr", ""), "entries": []}
    next_cursor = cursor
    output = result.get("stdout", "")
    bounded = bool(result.get("stdout_truncated") or result.get("error") in ("output_limit_reached", "timeout"))
    lines = output.splitlines(keepends=True)
    if bounded and lines and not lines[-1].endswith(("\n", "\r")):
        lines.pop()
        event["incomplete_last_line_deferred"] = True
    for line in lines:
        try:
            entry = json.loads(line)
            if not isinstance(entry, dict) or not isinstance(entry.get("__CURSOR"), str):
                raise ValueError("journal entry must include a string cursor")
            event["entries"].append({key: entry[key] for key in JOURNAL_FIELDS if key in entry})
            next_cursor = entry["__CURSOR"]
        except (ValueError, TypeError):
            event["parse_error"] = True
            break
    event["truncated"] = bounded
    event["byte_limit"] = COMMAND_BYTES
    # A timeout/output bound can still contain a durable, complete prefix.
    if result.get("returncode") != 0 and result.get("error") not in ("output_limit_reached", "timeout"):
        next_cursor = cursor
    error_text = result.get("stderr", "").lower()
    if cursor and result.get("returncode") not in (0, None) and "cursor" in error_text and "seek" in error_text:
        event["gap"] = {"reason": "journal_cursor_unavailable", "prior_cursor": cursor, "next_regular_check_fallback_seconds": 120}
        next_cursor = None
    event["cursor_advance_after_evidence_fsync"] = next_cursor != cursor
    # An empty successful journal in user mode does not prove system-log access.
    event["system_journal_access"] = "root" if os.geteuid() == 0 else "user_scope_only_unless_groups_grant_access"
    return event, next_cursor


def save_cursor(path: Path, cursor: str | None) -> None:
    """Atomically replace the cursor only after the caller fsyncs its evidence."""
    fd, temporary_name = tempfile.mkstemp(prefix=".journal-cursor-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o640)
        with os.fdopen(fd, "w") as stream:
            json.dump({"cursor": cursor, "utc": utc_now()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interface", default="enp130s0")
    parser.add_argument("--gateway", default="192.168.0.1")
    parser.add_argument("--owner-uid", type=int, default=1001)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--deadline", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not (1 <= args.interval <= 3600 and 1 <= args.deadline <= 60) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", args.interface):
        parser.error("invalid interval, deadline, or interface")
    import ipaddress
    try:
        ipaddress.IPv4Address(args.gateway)
    except ipaddress.AddressValueError:
        parser.error("gateway must be an IPv4 address")
    os.umask(0o027)
    lock = CollectorLock(Path(f"/tmp/paper-5090-network-{args.owner_uid}.lock"), args.owner_uid)
    writer = JsonlWriter(args.output)
    stopped = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    cursor_path = args.output / "journal_cursor.json"
    cursor = None
    try:
        cursor = json.loads(cursor_path.read_text()).get("cursor")
    except (OSError, ValueError, AttributeError):
        pass
    writer.append({"kind": "collector_start", "utc": utc_now(), "pid": os.getpid(), "uid": os.getuid(), "boot_id": read_local(Path("/proc/sys/kernel/random/boot_id")), "interval_seconds": args.interval, "snapshot_deadline_seconds": args.deadline, "max_bytes_per_file": MAX_BYTES, "retained_files": RETAINED_FILES, "privileged_journal": os.geteuid() == 0, "network_repair": False})
    last_journal = -float("inf")
    try:
        while not stopped:
            start = time.monotonic()
            runner = ProbeRunner(args.deadline)
            try:
                writer.append(collect_snapshot(args.interface, args.gateway, args.deadline, runner=runner))
            except Exception as exc:
                writer.append({"kind": "snapshot_error", "utc": utc_now(), "error": type(exc).__name__, "detail": str(exc)})
            if start - last_journal >= 30:
                event, new_cursor = journal_event(runner, cursor)
                writer.append(event)
                if new_cursor != cursor:
                    cursor = new_cursor
                    save_cursor(cursor_path, cursor)
                last_journal = start
            if args.once:
                break
            while not stopped and time.monotonic() - start < args.interval:
                time.sleep(max(0.0, min(0.25, args.interval - (time.monotonic() - start))))
        writer.append({"kind": "collector_stop", "utc": utc_now(), "pid": os.getpid()})
    finally:
        lock.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AlreadyRunning as error:
        print(str(error), file=__import__("sys").stderr)
        raise SystemExit(73)
