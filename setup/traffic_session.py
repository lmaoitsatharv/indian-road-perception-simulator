"""Bounded native Windows MATLAB job runner for the 4 October traffic session.

Every worker starts suspended, joins a kill-on-close Windows Job Object, then
runs. Closing that object stops only its own descendants. The MATLAB desktop
is never attached. Queue entries are immutable and never rerun automatically.
"""
import argparse
import ctypes as C
from ctypes import wintypes as W
from datetime import datetime, timezone, timedelta
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SESSION = ROOT / "setup/traffic_overnight_20261004"
DEADLINE = datetime(2026, 10, 4, 4, 30, tzinfo=timezone.utc)
K = C.WinDLL("kernel32", use_last_error=True)

class STARTUPINFO(C.Structure):
    _fields_ = [("cb", W.DWORD), ("lpReserved", W.LPWSTR), ("lpDesktop", W.LPWSTR),
        ("lpTitle", W.LPWSTR), ("dwX", W.DWORD), ("dwY", W.DWORD),
        ("dwXSize", W.DWORD), ("dwYSize", W.DWORD), ("dwXCountChars", W.DWORD),
        ("dwYCountChars", W.DWORD), ("dwFillAttribute", W.DWORD),
        ("dwFlags", W.DWORD), ("wShowWindow", W.WORD), ("cbReserved2", W.WORD),
        ("lpReserved2", C.POINTER(W.BYTE)), ("hStdInput", W.HANDLE),
        ("hStdOutput", W.HANDLE), ("hStdError", W.HANDLE)]

class PROCESSINFO(C.Structure):
    _fields_ = [("hProcess", W.HANDLE), ("hThread", W.HANDLE),
                ("dwProcessId", W.DWORD), ("dwThreadId", W.DWORD)]

class BASICLIMIT(C.Structure):
    _fields_ = [("PerProcessUserTimeLimit", C.c_int64), ("PerJobUserTimeLimit", C.c_int64),
        ("LimitFlags", W.DWORD), ("MinimumWorkingSetSize", C.c_size_t),
        ("MaximumWorkingSetSize", C.c_size_t), ("ActiveProcessLimit", W.DWORD),
        ("Affinity", C.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]

class EXTENDEDLIMIT(C.Structure):
    _fields_ = [("BasicLimitInformation", BASICLIMIT), ("IoInfo", C.c_uint64 * 6),
        ("ProcessMemoryLimit", C.c_size_t), ("JobMemoryLimit", C.c_size_t),
        ("PeakProcessMemoryUsed", C.c_size_t), ("PeakJobMemoryUsed", C.c_size_t)]

K.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
K.CreateJobObjectW.restype = W.HANDLE
K.SetInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]
K.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
K.CreateProcessW.argtypes = [W.LPCWSTR, W.LPWSTR, C.c_void_p, C.c_void_p,
    W.BOOL, W.DWORD, C.c_void_p, W.LPCWSTR, C.POINTER(STARTUPINFO), C.POINTER(PROCESSINFO)]
K.ResumeThread.argtypes = [W.HANDLE]
K.ResumeThread.restype = W.DWORD
K.CloseHandle.argtypes = [W.HANDLE]
K.GetExitCodeProcess.argtypes = [W.HANDLE, C.POINTER(W.DWORD)]
K.TerminateProcess.argtypes = [W.HANDLE, W.UINT]
K.CreateMutexW.argtypes = [C.c_void_p, W.BOOL, W.LPCWSTR]
K.CreateMutexW.restype = W.HANDLE
K.SetThreadExecutionState.argtypes = [W.DWORD]
K.SetThreadExecutionState.restype = W.DWORD

def utc():
    return datetime.now(timezone.utc)

def stamp(t=None):
    return (t or utc()).isoformat().replace("+00:00", "Z")

def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temp = Path(stream.name)
            stream.write(payload)
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists(): temp.unlink()

def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default

class OwnedJob:
    def __init__(self, argv, cwd):
        self.handle = K.CreateJobObjectW(None, None)
        self.process = None
        if not self.handle:
            raise C.WinError(C.get_last_error())
        limits = EXTENDEDLIMIT()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not K.SetInformationJobObject(self.handle, 9, C.byref(limits), C.sizeof(limits)):
            self.close()
            raise C.WinError(C.get_last_error())
        si = STARTUPINFO(); si.cb = C.sizeof(si); si.dwFlags = 1; si.wShowWindow = 0
        pi = PROCESSINFO()
        command = C.create_unicode_buffer(subprocess.list2cmdline(argv))
        if not K.CreateProcessW(None, command, None, None, False, 0x08000004,
                                None, str(cwd), C.byref(si), C.byref(pi)):
            self.close()
            raise C.WinError(C.get_last_error())
        self.process, self.pid = pi.hProcess, pi.dwProcessId
        if not K.AssignProcessToJobObject(self.handle, self.process):
            error = C.get_last_error()
            K.TerminateProcess(self.process, 1); K.CloseHandle(pi.hThread); self.close()
            raise C.WinError(error)
        resumed = K.ResumeThread(pi.hThread)
        K.CloseHandle(pi.hThread)
        if resumed == 0xFFFFFFFF:
            self.close()
            raise C.WinError(C.get_last_error())

    def poll(self):
        value = W.DWORD()
        if not K.GetExitCodeProcess(self.process, C.byref(value)):
            raise C.WinError(C.get_last_error())
        return None if value.value == 259 else value.value

    def close(self):
        if self.handle:
            K.CloseHandle(self.handle); self.handle = None
        if self.process:
            K.CloseHandle(self.process); self.process = None

def permission():
    # A final stop remains final, including after the deadline. Malformed markers
    # must not accidentally become permission to launch a worker.
    if (SESSION / "STOP.json").exists():
        stop = read(SESSION / "STOP.json")
        return (stop.get('reason') or 'user_stop') if isinstance(stop, dict) else 'invalid_stop_marker', True
    now = utc()
    if now >= DEADLINE:
        return "deadline", True
    if (SESSION / "PAUSE.json").exists():
        pause = read(SESSION / "PAUSE.json")
        return (pause.get('reason') or 'usage_pause') if isinstance(pause, dict) else 'invalid_pause_marker', False
    lease = read(SESSION / "monitor_lease.json", {})
    try:
        if not isinstance(lease, dict): raise ValueError('Lease must be an object')
        used = lease['usage_percentages']
        if not isinstance(used, list) or not used or any(
                isinstance(n, bool) or not isinstance(n, (int, float)) or
                not math.isfinite(n) or not 0 <= n <= 100 for n in used):
            raise ValueError('Available usage windows are required')
        if lease.get('ordinaryUsageAllowed') is not True or max(used) >= 95:
            return 'usage_pause', False
        checked = datetime.fromisoformat(lease['checked_at_utc'].replace('Z', '+00:00'))
        expiry = datetime.fromisoformat(lease["expires_at_utc"].replace("Z", "+00:00"))
        if checked.tzinfo is None or expiry.tzinfo is None:
            raise ValueError('Lease dates require a timezone')
        if checked > now + timedelta(seconds=10) or expiry > checked + timedelta(minutes=20, seconds=10):
            raise ValueError('Lease is outside its bounded monitoring window')
        if checked <= expiry and expiry > now:
            return None, False
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
        pass
    return "monitor_lease_expired", False

def validated_job(candidate):
    """Validate job contracts before starting a suspended child process."""
    if not isinstance(candidate,dict): raise ValueError('Queue entries must be objects')
    job_id=candidate.get('id')
    if not isinstance(job_id,str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',job_id) is None:
        raise ValueError('Invalid queue job ID')
    argv=candidate.get('argv')
    if not isinstance(argv,list) or not argv or any(not isinstance(s,str) or not s or '\0' in s for s in argv):
        raise ValueError('Job argv must be a nonempty list of strings')
    timeout=candidate.get('timeout_seconds',900)
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
        raise ValueError('Job timeout must be finite and positive')
    return dict(candidate,timeout_seconds=min(float(timeout),3600))

def run_controller():
    SESSION.mkdir(parents=True, exist_ok=True)
    C.set_last_error(0)
    mutex = K.CreateMutexW(None, False, "Local\\TeslaTrafficOvernight20261004")
    if not mutex or C.get_last_error() == 183:
        if mutex: K.CloseHandle(mutex)
        raise SystemExit("Controller already exists; no duplicate started.")
    active = None; job = None; job_record = None; last_write = 0
    state = {"controller_pid": os.getpid(), "started_at_utc": stamp(),
             "deadline_utc": stamp(DEADLINE), "status": "idle", "active_job": None}
    K.SetThreadExecutionState(0x80000001)  # Temporary system-awake request, no screen request.
    try:
        while True:
            reason, final = permission()
            if reason:
                if active:
                    active.close(); active = None
                    job_record.update(status="interrupted", reason=reason, ended_at_utc=stamp())
                    atomic(SESSION / "jobs" / (job["id"] + ".json"), job_record)
                state.update(status="finished" if final else "paused", reason=reason, active_job=None)
                if final:
                    if not (SESSION / "STOP.json").exists():
                        atomic(SESSION / "STOP.json", {"reason": reason, "at_utc": stamp()})
                    break
            else:
                if active and (active.poll() is not None or time.monotonic() >= job_end):
                    code = active.poll()
                    active.close(); active = None
                    job_record.update(status=("completed" if code == 0 else "failed"),
                        exit_code=code, reason="timeout" if code is None else "process_exit", ended_at_utc=stamp())
                    atomic(SESSION / "jobs" / (job["id"] + ".json"), job_record)
                if active is None:
                    state.update(status="idle", active_job=None, reason=None)
                    queue=read(SESSION/'queue.json',[])
                    if not isinstance(queue,list): raise ValueError('Queue must be a list')
                    for candidate in queue:
                        candidate=validated_job(candidate)
                        job_id = candidate["id"]
                        destination = SESSION / "jobs" / (job_id + ".json")
                        if destination.exists(): continue
                        job = candidate
                        job_record = dict(job, status="starting", started_at_utc=stamp())
                        atomic(destination, job_record)
                        try:
                            active = OwnedJob(job["argv"], ROOT)
                            job_end = time.monotonic() + job['timeout_seconds']
                            job_record.update(status="running", pid=active.pid)
                            state.update(status="running", active_job=job_id)
                            atomic(destination, job_record)
                        except Exception as err:
                            job_record.update(status="failed", reason=str(err), ended_at_utc=stamp())
                            atomic(destination, job_record)
                        break
            if time.monotonic() - last_write >= 5:
                state["updated_at_utc"] = stamp()
                atomic(SESSION / "session_status.json", state); last_write = time.monotonic()
            time.sleep(1)
    except BaseException as error:
        if job_record is not None and active is not None:
            job_record.update(status='interrupted', reason='controller_error', ended_at_utc=stamp())
            atomic(SESSION / 'jobs' / (job['id']+'.json'), job_record)
        state.update(status='failed', reason='controller_error', error=str(error))
        raise
    finally:
        if active: active.close()
        K.SetThreadExecutionState(0x80000000)
        state.update(updated_at_utc=stamp(), keep_awake_released=True, active_job=None)
        atomic(SESSION / "session_status.json", state)
        K.CloseHandle(mutex)

def selftest():
    # Never read a stale child PID left by an earlier guard probe.
    folder = Path(tempfile.mkdtemp(prefix='guard_test_', dir=SESSION))
    child_pid_file = folder / "child_pid.txt"
    child_code = "import time; time.sleep(90)"
    own_code = ("import subprocess,sys,time,pathlib; "
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid)); time.sleep(90)")
    independent = subprocess.Popen([sys.executable, "-c", child_code], creationflags=0x08000000)
    owned = None
    try:
        owned = OwnedJob([sys.executable, "-c", own_code], ROOT)
        for _ in range(30):
            if child_pid_file.exists(): break
            time.sleep(.1)
        assert child_pid_file.exists(), "Child failed to start"
        child_pid = int(child_pid_file.read_text())
        K.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]; K.OpenProcess.restype = W.HANDLE
        K.WaitForSingleObject.argtypes = [W.HANDLE, W.DWORD]
        child_handle = K.OpenProcess(0x100000, False, child_pid)
        root_handle = K.OpenProcess(0x100000, False, owned.pid)
        assert child_handle and root_handle
        owned.close(); owned = None
        root_stopped = K.WaitForSingleObject(root_handle, 3000) == 0
        child_stopped = K.WaitForSingleObject(child_handle, 3000) == 0
        K.CloseHandle(root_handle); K.CloseHandle(child_handle)
        unaffected = independent.poll() is None
        assert root_stopped and child_stopped and unaffected
        result = dict(status="passed", owned_root_stopped=root_stopped,
            owned_child_stopped=child_stopped, unrelated_probe_untouched=unaffected, at_utc=stamp())
        atomic(SESSION / "guard_verification.json", result)
        print(json.dumps(result))
    finally:
        if owned: owned.close()
        independent.terminate(); independent.wait(timeout=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "selftest", "lease"])
    parser.add_argument("--used", nargs="*", type=float, default=[])
    args = parser.parse_args()
    if args.command == "selftest": selftest()
    elif args.command == "run": run_controller()
    else:
        if not args.used or any(not math.isfinite(n) or not 0 <= n < 95 for n in args.used):
            parser.error('Provide current usage for every available window, each below 95%')
        if utc() >= DEADLINE or (SESSION / 'STOP.json').exists():
            parser.error('A stopped or expired session cannot be renewed')
        atomic(SESSION / "monitor_lease.json", dict(ordinaryUsageAllowed=True,
            checked_at_utc=stamp(), expires_at_utc=stamp(utc()+timedelta(minutes=20)), usage_percentages=args.used))
