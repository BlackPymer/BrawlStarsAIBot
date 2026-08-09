import cv2
import numpy as np

from controller.config import save_config
from controller.screen_capture import ScreenCapture

WINDOW_NAME = "Brawl Stars Bot - Border Setup"


def _draw_text(img, text, pos, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["clicks"].append((x, y))


def setup_borders():
    cap = ScreenCapture()
    frame = cap.get_frame()
    if frame is None:
        print("Cannot capture screen")
        return

    display = frame.copy()
    h, w = display.shape[:2]
    scale = min(1280 / w, 720 / h, 1.0)
    if scale < 1.0:
        display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = display.shape[:2]

    param = {"clicks": []}
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, w, h)
    cv2.setMouseCallback(WINDOW_NAME, _mouse_callback, param)

    config = {}
    steps = [
        ("move_border", "Click TOP-LEFT of MOVE joystick area"),
        ("move_border", "Click BOTTOM-RIGHT of MOVE joystick area"),
        ("shoot_border", "Click TOP-LEFT of SHOOT joystick area"),
        ("shoot_border", "Click BOTTOM-RIGHT of SHOOT joystick area"),
        ("ult_center", "Click CENTER of ULT button"),
    ]

    for key, instruction in steps:
        param["clicks"] = []
        img = display.copy()
        _draw_text(img, instruction, (20, 40), (0, 255, 0))
        for i, (k, v) in enumerate(config.items()):
            text = f"{k}: {v}"
            _draw_text(img, text, (20, h - 60 + 20 * i))
        cv2.imshow(WINDOW_NAME, img)

        while len(param["clicks"]) < (1 if key == "ult_center" else 2):
            img = display.copy()
            for pt in param["clicks"]:
                cv2.circle(img, pt, 5, (0, 0, 255), -1)
            _draw_text(img, instruction, (20, 40), (0, 255, 0))
            cv2.imshow(WINDOW_NAME, img)
            cv2.waitKey(50)

        if key == "ult_center":
            x = round(param["clicks"][0][0] / scale)
            y = round(param["clicks"][0][1] / scale)
            config[key] = [x, y]
        else:
            x1 = round(param["clicks"][0][0] / scale)
            y1 = round(param["clicks"][0][1] / scale)
            x2 = round(param["clicks"][1][0] / scale)
            y2 = round(param["clicks"][1][1] / scale)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            config[key] = [x1, y1, x2, y2]

        img = display.copy()
        _draw_text(img, f"{key} set!", (20, 40), (0, 255, 0))
        cv2.imshow(WINDOW_NAME, img)
        cv2.waitKey(500)

    cv2.destroyWindow(WINDOW_NAME)

    save_config(config)
    print(f"Config saved: {config}")
    return config
