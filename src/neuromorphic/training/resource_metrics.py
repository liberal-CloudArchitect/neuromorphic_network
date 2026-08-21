"""Cross-platform process resource measurements without optional dependencies."""

from __future__ import annotations

import ctypes
import sys


def _windows_peak_working_set_bytes() -> int:
    """Return the current process peak working set through the Windows API."""

    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    psapi = ctypes.WinDLL("psapi", use_last_error=True)  # type: ignore[attr-defined]
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
        error_code = int(ctypes.get_last_error())  # type: ignore[attr-defined]
        raise OSError(error_code, "GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


def process_peak_rss_bytes() -> tuple[int, str]:
    """Return process peak resident memory in bytes and its measurement source."""

    if sys.platform == "win32":
        return _windows_peak_working_set_bytes(), "Windows PeakWorkingSetSize"

    import resource

    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        maximum *= 1024
    return maximum, "process ru_maxrss"
