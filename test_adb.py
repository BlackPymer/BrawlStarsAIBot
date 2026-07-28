import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import time
from controller.adb_api import ADBShell

adb = ADBShell()
if adb._adb is None:
    print("adb not found!")
    exit(1)
if adb._serial is None:
    print("no device!")
    exit(1)

print(f"Device: {adb._serial}")
print(f"Touch device: {adb._event_device}")
print(f"Screen: {adb._screen_w}x{adb._screen_h}")
print(f"Touch max: {adb._touch_max}")

# screencap before
ADBShell.screencap("before.png")
print("before.png saved")

# tap bottom-right (where battle button should be)
x = adb._screen_w - 20
y = adb._screen_h - 20
print(f"Tapping at ({x}, {y})...")
adb.tap(x, y)
time.sleep(1)

# screencap after
ADBShell.screencap("after.png")
print("after.png saved")

# swipe test
print("Testing swipe...")
adb.swipe(100, 400, 300, 400, duration_ms=300)
time.sleep(0.5)
ADBShell.screencap("swipe.png")
print("swipe.png saved")

print("Done! Check the PNG files to see if anything changed.")
