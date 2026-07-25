from controller.game_controller import GameController
from hp_recognition.hp_recogniser import HPRecogniser, HP_PADDING
from network.network_engine import NetworkEngine
from objects_detection.object_detector import ObjectDetector, CLASS_NAMES
import time as t

BULLET_RELOADING_TIME = 1.5
MAX_BULLETS = 3


class GameEngine:
    def __init__(self):
        self.is_game = False
        self.game_controller = None
        self.hp_recogniser = HPRecogniser()
        self.object_detector = ObjectDetector()
        self.bullets_number = 0
        self.last_bullet_reloaded_time = 0
        self.network = NetworkEngine()

    def start_game(self, controller: GameController):
        self.game_controller = controller
        self.is_game = True
        self.game_controller.start_game()
        self.bullets_number = MAX_BULLETS

    def stop_game(self):
        self.game_controller.stop_game()
        self.is_game = False

    def update(self):
        if self.bullets_number < MAX_BULLETS and t.time() - self.last_bullet_reloaded_time > BULLET_RELOADING_TIME:
            self.last_bullet_reloaded_time = t.time()
            self.bullets_number += 1

        frame = self.game_controller.get_frame()
        objects = self.object_detector.detect(frame)

        hp_crops = []
        for obj in objects:
            for box in obj.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if cls_id in (0, 1):
                    h, w = frame.shape[:2]
                    cx1 = max(0, x1 - HP_PADDING)
                    cy1 = max(0, y1 - HP_PADDING)
                    cx2 = min(w, x2 + HP_PADDING)
                    cy2 = min(h, y1 + 100)
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        hp_crops.append(crop)
        hp_texts = self.hp_recogniser.recognise(image=frame, crops=hp_crops) if hp_crops else None
        hp_values = [int(i) for i in hp_texts]
        self.network.make_action(objects, hp_values)

