import platform

import cv2
import numpy as np

TEST_IMAGE = "datasets/datasetV3/images/Train/frame_000300.jpg"


class ScreenCapture:
    def __init__(self, region: list[int] | None = None):
        self.region = region
        self._system = platform.system()
        self._mss = None
        self._window_offset = (0, 0)

        if self._system == "Windows":
            self._find_bluestacks()

        try:
            import mss
            self._mss = mss.mss()
        except ImportError:
            pass

    def _find_bluestacks(self):
        try:
            import win32gui

            def callback(hwnd, handles):
                if win32gui.IsWindowVisible(hwnd) and "BlueStacks" in win32gui.GetWindowText(hwnd):
                    handles.append(hwnd)

            handles = []
            win32gui.EnumWindows(callback, handles)
            if handles:
                rect = win32gui.GetWindowRect(handles[0])
                self.region = [rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]]
                self._window_offset = (rect[0], rect[1])
        except ImportError:
            pass

    def get_frame(self) -> np.ndarray | None:
        if self._system == "Windows":
            return self._capture_screen()
        return self._capture_test()

    def _capture_screen(self) -> np.ndarray | None:
        if self._mss is None:
            return None
        if self.region:
            mon = {
                "left": self.region[0],
                "top": self.region[1],
                "width": self.region[2],
                "height": self.region[3],
            }
        else:
            mon = self._mss.monitors[1]
        sct_img = self._mss.grab(mon)
        return cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

    def _capture_test(self) -> np.ndarray | None:
        img = cv2.imread(TEST_IMAGE)
        return img
