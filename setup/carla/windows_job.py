"""Owned Windows process tree with visible application windows and no console."""
import ctypes as C
import subprocess

from traffic_session import K, W, STARTUPINFO, PROCESSINFO, EXTENDEDLIMIT, OwnedJob


class CarlaJob(OwnedJob):
    def pids(self):
        """Read membership of this Job Object, never infer ownership by process name."""
        class ProcessIds(C.Structure):
            _fields_ = [("assigned", W.DWORD), ("count", W.DWORD), ("ids", C.c_size_t * 64)]
        query = K.QueryInformationJobObject
        query.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.POINTER(W.DWORD)]
        query.restype = W.BOOL
        buffer, returned = ProcessIds(), W.DWORD()
        if not self.handle or not query(self.handle, 3, C.byref(buffer), C.sizeof(buffer), C.byref(returned)):
            raise C.WinError(C.get_last_error())
        return list(buffer.ids[:buffer.count])

    def __init__(self, argv, cwd):
        self.handle = K.CreateJobObjectW(None, None)
        self.process = None
        if not self.handle:
            raise C.WinError(C.get_last_error())
        limits = EXTENDEDLIMIT()
        limits.BasicLimitInformation.LimitFlags = 0x2000
        if not K.SetInformationJobObject(self.handle, 9, C.byref(limits), C.sizeof(limits)):
            self.close()
            raise C.WinError(C.get_last_error())
        si = STARTUPINFO()
        si.cb = C.sizeof(si)
        si.dwFlags = 1
        si.wShowWindow = 1  # Show the actual engine/viewer window; hide only the console.
        pi = PROCESSINFO()
        command = C.create_unicode_buffer(subprocess.list2cmdline(argv))
        if not K.CreateProcessW(None, command, None, None, False, 0x08000004,
                                None, str(cwd), C.byref(si), C.byref(pi)):
            self.close()
            raise C.WinError(C.get_last_error())
        self.process, self.pid = pi.hProcess, pi.dwProcessId
        if not K.AssignProcessToJobObject(self.handle, self.process):
            error = C.get_last_error()
            K.TerminateProcess(self.process, 1)
            K.CloseHandle(pi.hThread)
            self.close()
            raise C.WinError(error)
        resumed = K.ResumeThread(pi.hThread)
        K.CloseHandle(pi.hThread)
        if resumed == 0xFFFFFFFF:
            self.close()
            raise C.WinError(C.get_last_error())
