import subprocess
import time
from functools import partial

_ADB_SHELL = None


def _get_shell():
    global _ADB_SHELL
    if _ADB_SHELL is None:
        _ADB_SHELL = subprocess.Popen(
            ["adb", "shell"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    return _ADB_SHELL


def _close_shell():
    global _ADB_SHELL
    if _ADB_SHELL is not None:
        _ADB_SHELL.terminate()
        _ADB_SHELL = None


def _cmd(command: str):
    shell = _get_shell()
    shell.stdin.write(command + "\n")
    shell.stdin.flush()


def make_tap(x: int, y: int):
    _cmd(f"input tap {x} {y}")


def make_swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 50):
    _cmd(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")


def screencap(path: str = "frame.png"):
    subprocess.run(["adb", "exec-out", "screencap", "-p"], check=True)
