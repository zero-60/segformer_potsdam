"""Prevent classic Windows console selection from blocking training output."""
def disable_quick_edit():
    import os
    if os.name!='nt':return
    try:
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetStdHandle.argtypes=[wintypes.DWORD];kernel.GetStdHandle.restype=wintypes.HANDLE
        kernel.GetConsoleMode.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.SetConsoleMode.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        handle=kernel.GetStdHandle(wintypes.DWORD(-10));mode=wintypes.DWORD()
        if kernel.GetConsoleMode(handle,ctypes.byref(mode)):
            if kernel.SetConsoleMode(handle,(mode.value|0x0080)&~0x0040):
                print('[CONSOLE] Quick Edit disabled for this console.',flush=True)
    except (OSError,AttributeError):pass  # Redirected output or non-classic terminal.
