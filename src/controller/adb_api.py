import os
import re
import shutil
import subprocess
import time

_ADB_PATHS = [
    r"C:\adb\platform-tools\adb.exe",
    r"C:\adb\adb.exe",
    r"C:\Program Files\adb\adb.exe",
    r"C:\Program Files (x86)\adb\adb.exe",
]


def _find_adb() -> str | None:
    path = shutil.which("adb")
    if path:
        return path
    for p in _ADB_PATHS:
        if os.path.isfile(p):
            return p
    return None


def _list_devices(adb: str) -> list[str]:
    try:
        result = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=5)
        serials = []
        for line in result.stdout.splitlines():
            if "\tdevice" in line:
                serial = line.split("\t")[0].strip()
                if serial:
                    serials.append(serial)
        return serials
    except Exception:
        return []


def _get_device(adb: str) -> str | None:
    serials = _list_devices(adb)
    if not serials:
        return None
    for s in serials:
        if "emulator" in s:
            return s
    return serials[0]


def _shell(adb: str, serial: str, command: str) -> str | None:
    try:
        result = subprocess.run(
            [adb, "-s", serial, "shell", command],
            capture_output=True, text=True, timeout=15,
        )
        return result.stdout
    except Exception:
        return None


class ADBShell:
    def __init__(self):
        self._adb = _find_adb()
        if self._adb is None:
            print("WARNING: adb not found. Actions will be no-ops.")
            return
        self._ensure_connected()
        self._serial = _get_device(self._adb)
        if self._serial is None:
            print("WARNING: no adb device. Actions will be no-ops.")
            return
        self._event_device, self._touch_max = self._detect_touch()
        self._screen_w, self._screen_h = self._get_screen_size()
        if self._event_device is None:
            print("WARNING: touch device not found. Actions will be no-ops.")

    def _ensure_connected(self):
        serials = _list_devices(self._adb)
        if serials:
            return
        subprocess.run([self._adb, "connect", "127.0.0.1:5555"], capture_output=True, timeout=5)
        time.sleep(1)
        serials = _list_devices(self._adb)
        if not serials:
            subprocess.run([self._adb, "connect", "127.0.0.1:5555"], capture_output=True, timeout=5)
            time.sleep(1)

    def _detect_touch(self) -> tuple[str | None, int]:
        out = _shell(self._adb, self._serial, "getevent -p")
        if out is None:
            return None, 0
        dev = None
        for line in out.splitlines():
            m = re.match(r"add device \d+: (/dev/input/event\d+)", line)
            if m:
                dev = m.group(1)
            if dev and "BlueStacks Virtual Touch" in line:
                for l2 in out.splitlines():
                    m2 = re.search(r"0035.*max (\d+)", l2)
                    if m2:
                        return dev, int(m2.group(1))
                return dev, 32767
        return None, 0

    def _get_screen_size(self) -> tuple[int, int]:
        out = _shell(self._adb, self._serial, "wm size")
        if out:
            m = re.search(r"(\d+)x(\d+)", out)
            if m:
                return int(m.group(1)), int(m.group(2))
        return 1920, 1080

    def _to_ev(self, x: int, y: int) -> tuple[int, int]:
        if self._screen_w == 0 or self._screen_h == 0 or self._touch_max == 0:
            return x, y
        return (x * (self._touch_max - 1) // self._screen_w,
                y * (self._touch_max - 1) // self._screen_h)

    def _cmd(self, cmd: str):
        if self._adb is None or self._serial is None:
            return
        try:
            subprocess.run(
                f'"{self._adb}" -s {self._serial} shell "{cmd}"',
                capture_output=True, text=True, timeout=15, shell=True,
            )
        except Exception:
            pass

    def _to_chain(self, *touches: tuple[int, int]) -> list[str]:
        parts = []
        for tx, ty in touches:
            ex, ey = self._to_ev(tx, ty)
            parts.append(f"sendevent {self._event_device} 3 53 {ex}")
            parts.append(f"sendevent {self._event_device} 3 54 {ey}")
            parts.append(f"sendevent {self._event_device} 0 2 0")
        if not touches:
            parts.append(f"sendevent {self._event_device} 0 2 0")
        parts.append(f"sendevent {self._event_device} 0 0 0")
        return parts

    def tap(self, x: int, y: int):
        if self._event_device is None:
            return
        self._cmd(" && ".join([
            *self._to_chain((x, y)),
            "sleep 0.05",
            *self._to_chain(),
        ]))

    def quick_swipe(self, x1: int, y1: int, x2: int, y2: int):
        if self._event_device is None:
            return
        self._cmd(" && ".join([
            *self._to_chain((x1, y1)),
            "sleep 0.015",
            *self._to_chain((x2, y2)),
            "sleep 0.015",
            *self._to_chain(),
        ]))

    def swipes(self, *points: tuple[int, int]):
        if self._event_device is None:
            return
        frames = []
        for p in points:
            frames.extend(self._to_chain(p))
            frames.append("sleep 0.015")
        frames.extend(self._to_chain())
        self._cmd(" && ".join(frames))

    def _chain_touches(self, *touches: tuple[int, int]):
        self._cmd(" && ".join(self._to_chain(*touches)))

    @staticmethod
    def screencap(path: str = "frame.png"):
        adb = _find_adb()
        if adb is None:
            print("WARNING: adb not found. Cannot screencap.")
            return
        serial = _get_device(adb)
        if serial is None:
            print("WARNING: no adb device. Cannot screencap.")
            return
        try:
            result = subprocess.run([adb, "-s", serial, "exec-out", "screencap", "-p"], capture_output=True, check=True, timeout=10)
            if result.stdout:
                with open(path, "wb") as f:
                    f.write(result.stdout)
        except Exception:
            print("WARNING: screencap failed.")
