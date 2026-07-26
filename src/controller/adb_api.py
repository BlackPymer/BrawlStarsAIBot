import subprocess


class ADBShell:
    def __init__(self):
        self._process = subprocess.Popen(
            ["adb", "shell"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def close(self):
        if self._process is not None:
            self._process.terminate()
            self._process = None

    def _cmd(self, command: str):
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()

    def tap(self, x: int, y: int):
        self._cmd(f"input tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 50):
        self._cmd(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    @staticmethod
    def screencap(path: str = "frame.png"):
        subprocess.run(["adb", "exec-out", "screencap", "-p"], check=True)
