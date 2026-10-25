"""Resumable, bounded parallel downloads from the user's authorised IDD links.

Source tokens stay in the ignored dataset folder and never enter logs. Each
completed byte range has a SHA-256 record. Existing completed ranges are reused.
No archive content is extracted and no training is started.
"""
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import msvcrt
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "dataset/idd117k_raw"
CHUNK = 16 * 1024 * 1024


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class Transfer:
    def __init__(self):
        self.stop = threading.Event()
        self.mutex = threading.RLock()
        self.network_bytes = 0
        self.started = time.monotonic()
        self.session_start_bytes = 0
        self.phase = "preparing"
        self.workers = 0
        self.errors = []
        self.cooldown_until = 0.
        self.state_path = FOLDER / "download_state.json"
        sources = json.loads((FOLDER / "download_sources.private.json").read_text())["sources"]
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"files": {}}
        self.sources = {}
        self.retry_deadline = time.time() + 24 * 3600
        for src in sources:
            part = str(src["part"])
            headers = {k.lower(): v for k, v in src["headers"].items()}
            size = int(headers["content-length"])
            name = Path(urllib.parse.urlparse(src["url"]).path).name
            if not name.startswith("IDD117K_Detection.tar.gz.part") or Path(name).name != name:
                raise ValueError("Unexpected official archive name")
            expected = {"name": name, "size": size, "etag": headers["etag"]}
            existing = self.state["files"].setdefault(part, {**expected, "chunks": {}})
            if any(existing[k] != v for k, v in expected.items()):
                raise ValueError("Archive metadata changed; preserving existing data")
            path = FOLDER / name
            if existing["chunks"] and not path.exists():
                raise ValueError("Recorded completed ranges have no local file")
            if not path.exists():
                path.touch(exist_ok=False)
            self.sources[part] = src["url"]
            expiry = urllib.parse.parse_qs(urllib.parse.urlparse(src["url"]).query).get("expires", [])
            if expiry:
                self.retry_deadline = min(self.retry_deadline, int(expiry[0]))
        self.total = sum(f["size"] for f in self.state["files"].values())
        self.session_start_bytes = self.done_bytes()
        remaining = self.total - self.session_start_bytes
        if shutil.disk_usage(FOLDER).free < remaining + 5 * 1024**3:
            raise ValueError("Insufficient space for remaining archive bytes and 5 GiB reserve")
        self.save_state()

    def done_bytes(self):
        return sum(min(CHUNK, f["size"] - int(i) * CHUNK)
                   for f in self.state["files"].values() for i in f["chunks"])

    def save_state(self):
        with self.mutex:
            atomic(self.state_path, self.state)

    def report(self):
        with self.mutex:
            now = time.monotonic()
            done = self.done_bytes()
            elapsed = now - self.started
            speed = self.network_bytes / max(elapsed, .001)
            previous = getattr(self, "last_report", (self.started, 0))
            current_speed = (self.network_bytes - previous[1]) / max(now - previous[0], .001)
            self.last_report = (now, self.network_bytes)
            result = {"status": self.phase, "pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(),
                "connections": self.workers, "completed_bytes": done, "total_bytes": self.total,
                "percent": round(100 * done / self.total, 3), "session_network_bytes": self.network_bytes,
                "average_MB_s": round(speed / 1e6, 3), "recent_MB_s": round(current_speed / 1e6, 3),
                "estimated_remaining_hours": round((self.total - done) / speed / 3600, 2) if speed > 1000 else None,
                "files": [{"name": f["name"], "expected_bytes": f["size"],
                    "completed_ranges": len(f["chunks"]), "total_ranges": math.ceil(f["size"] / CHUNK)}
                    for f in self.state["files"].values()],
                "error_count": len(self.errors), "recent_errors": self.errors[-5:]}
            result["network_backoff_remaining_seconds"] = round(max(0., self.cooldown_until - now), 1)
            atomic(FOLDER / "download_progress.json", result)
            eta = "Measuring" if result["estimated_remaining_hours"] is None else f"About {result['estimated_remaining_hours']:.1f} hours at this session's average speed"
            summary = ["# IDD-117K download status", "",
                f"State: **{self.phase.replace('_', ' ')}**", "",
                f"Saved complete ranges: **{done / 1e9:.2f} / {self.total / 1e9:.2f} GB ({result['percent']:.2f}%)**", "",
                f"Recent transfer speed: **{result['recent_MB_s']:.2f} MB/s**; connections: **{self.workers}**.", "",
                f"Estimated remaining time: {eta}.", "",
                f"Last update (UTC): {result['updated_utc']}", "",
                "Files retain their original split-archive names. Partially written ranges are not counted as saved until complete.", "",
                "Downloads resume from completed ranges. No extraction or training has started. The transfer temporarily prevents idle system sleep; the screen may turn off.", "",
                "Final verification checks every saved range and records local SHA-256 hashes. These are not provider-published file checksums.", ""]
            if self.errors:
                summary += [f"Transfer/retry events recorded: {len(self.errors)}.", ""]
            (FOLDER / "DOWNLOAD_STATUS.md").write_text("\n".join(summary), encoding="utf-8")
            print(json.dumps({k: result[k] for k in ("status", "connections", "percent", "completed_bytes", "recent_MB_s", "average_MB_s", "estimated_remaining_hours", "error_count")}), flush=True)

    def jobs(self):
        files = self.state["files"]
        return [(p, i) for i in range(max(math.ceil(f["size"] / CHUNK) for f in files.values()))
                for p, f in files.items() if i * CHUNK < f["size"] and str(i) not in f["chunks"]]

    def fetch(self, job):
        part, index = job
        file = self.state["files"][part]
        start, end = index * CHUNK, min((index + 1) * CHUNK, file["size"]) - 1
        # Temporary loss of internet must not discard a multi-hour transfer.
        # Retry with backoff until the existing authorised links expire.
        for attempt in range(1000):
            if self.stop.is_set():
                return
            with self.mutex:
                cooldown = max(0., self.cooldown_until - time.monotonic())
            if cooldown and self.stop.wait(cooldown):
                return
            try:
                request = urllib.request.Request(self.sources[part], headers={
                    "User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity",
                    "Range": f"bytes={start}-{end}", "If-Range": file["etag"]})
                with urllib.request.urlopen(request, timeout=30) as response:
                    if (response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{file['size']}"
                            or response.headers.get("ETag") != file["etag"]):
                        raise ValueError("Server range or archive identity mismatch")
                    digest = hashlib.sha256()
                    written = 0
                    with (FOLDER / file["name"]).open("r+b", buffering=0) as stream:
                        stream.seek(start)
                        while written < end - start + 1:
                            if self.stop.is_set():
                                return
                            data = response.read1(min(256 * 1024, end - start + 1 - written))
                            if not data:
                                raise ConnectionError("Range ended early")
                            stream.write(data)
                            digest.update(data)
                            written += len(data)
                            with self.mutex:
                                self.network_bytes += len(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    with self.mutex:
                        file["chunks"][str(index)] = digest.hexdigest()
                        self.save_state()
                    return
            except Exception as error:
                record = {"part": int(part), "range": index, "attempt": attempt + 1,
                          "type": type(error).__name__, "http_status": getattr(error, "code", None)}
                reason = getattr(error, "reason", error)
                record.update(reason_type=type(reason).__name__, errno=getattr(reason,"errno",None),
                              winerror=getattr(reason,"winerror",None))
                with self.mutex:
                    self.errors.append(record)
                    with (FOLDER / "download_events.jsonl").open("a", encoding="utf-8") as events:
                        events.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **record}) + "\n")
                    if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                        try:
                            retry_after = max(60., float(error.headers.get("Retry-After", 60)))
                        except (TypeError, ValueError):
                            retry_after = 60.
                        self.cooldown_until = max(self.cooldown_until, time.monotonic() + retry_after)
                        print(json.dumps({"server_rate_limit": True, "backoff_seconds": retry_after}), flush=True)
                if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403, 404):
                    self.stop.set()
                    raise RuntimeError(f"Part {part}: authorised URL returned HTTP {error.code}") from None
                transient = isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError,
                                               http.client.HTTPException))
                if isinstance(error, urllib.error.HTTPError):
                    transient = error.code in (408, 429) or 500 <= error.code <= 599
                if not transient or time.time() >= self.retry_deadline or attempt == 999:
                    self.stop.set()
                    raise RuntimeError(f"Part {part}: transfer cannot continue ({type(error).__name__}); saved ranges retained") from None
                with self.mutex:
                    self.cooldown_until = max(self.cooldown_until,
                        time.monotonic() + min(15 * 2 ** min(attempt, 5), 300))

    def execute(self, jobs, workers):
        self.workers = workers
        self.report()
        iterator = iter(jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(self.fetch, j) for j in [next(iterator, None) for _ in range(workers)] if j is not None}
            while pending:
                finished, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                if (FOLDER / "STOP_DOWNLOAD").exists():
                    self.stop.set()
                for future in finished:
                    future.result()
                    job = next(iterator, None)
                    if job is not None and not self.stop.is_set():
                        pending.add(pool.submit(self.fetch, job))
                if time.monotonic() - getattr(self, "last_report", (0, 0))[0] >= 10:
                    self.report()
        self.report()

    def verify(self):
        self.phase = "verifying_downloaded_bytes"
        self.report()
        verified = []
        for part, file in self.state["files"].items():
            path = FOLDER / file["name"]
            if path.stat().st_size != file["size"]:
                raise ValueError("Completed archive part size differs")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for index in range(math.ceil(file["size"] / CHUNK)):
                    data = stream.read(CHUNK)
                    if hashlib.sha256(data).hexdigest() != file["chunks"][str(index)]:
                        raise ValueError("A downloaded range changed on disk")
                    digest.update(data)
            verified.append({"part": int(part), "name": file["name"], "bytes": file["size"], "sha256": digest.hexdigest()})
            print(json.dumps({"part_verified": int(part), "bytes": file["size"]}), flush=True)
        atomic(FOLDER / "download_verification.json", {"passed": True, "verification": "Exact HTTP byte ranges, stable server ETags, expected sizes, local SHA256 consistency; provider file checksum unavailable", "files": verified})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connections", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-connections", default="1,4,8")
    args = parser.parse_args()
    candidates = [int(value) for value in args.benchmark_connections.split(",")]
    if any(not 1 <= value <= 8 for value in candidates):
        parser.error("Use 1 to 8 connections; the server rate-limited the 16-connection trial")
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER / ".download.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        transfer = Transfer()
        # Keep the system awake only for this transfer; do not change power plans
        # or keep the display on. Releasing the thread restores normal sleep.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            connections = args.connections
            if args.benchmark:
                benchmarks = []
                transfer.phase = "benchmarking"
                for count in candidates:
                    jobs = transfer.jobs()[:count]
                    if not jobs:
                        break
                    started, initial = time.monotonic(), transfer.network_bytes
                    transfer.execute(jobs, count)
                    rate = (transfer.network_bytes - initial) / (time.monotonic() - started) / 1e6
                    benchmarks.append({"connections": count, "MB_s": round(rate, 3)})
                    atomic(FOLDER / "speed_benchmark.json", benchmarks)
                    print(json.dumps({"benchmark": benchmarks[-1]}), flush=True)
                if benchmarks:
                    connections = max(benchmarks, key=lambda item: item["MB_s"])["connections"]
            transfer.phase = "downloading"
            transfer.execute(transfer.jobs(), connections)
            if transfer.stop.is_set():
                transfer.phase = "paused_preserved"
            else:
                transfer.verify()
                transfer.phase = "completed"
            transfer.report()
        except BaseException as error:
            transfer.stop.set()
            transfer.phase = "failed_preserved"
            transfer.errors.append({"type": type(error).__name__})
            transfer.report()
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
