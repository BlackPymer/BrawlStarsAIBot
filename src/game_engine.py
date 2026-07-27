from controller.game_controller import GameController
from hp_recognition.hp_recogniser import HP_PADDING
from hp_recognition.CRNN_hp_recogniser import CRNNHpRecogniser
from network.exceptions import PlayerNotFoundException
from network.network_engine import NetworkEngine
from objects_detection.object_detector import CLASS_NAMES
from objects_detection.yolo_object_detector import YoloObjectDetector
import time as t

BULLET_RELOADING_TIME = 1.5
MAX_BULLETS = 3


class GameEngine:
    def __init__(self):
        self.is_game = False
        self.game_controller = None
        self.hp_recogniser = CRNNHpRecogniser()
        self.object_detector = YoloObjectDetector()
        self.bullets_number = 0
        self.last_bullet_reloaded_time = 0
        self.network = NetworkEngine()

    def start_game(self, controller: GameController):
        self.game_controller = controller
        self.is_game = True
        self.game_controller.start_game()
        self.bullets_number = MAX_BULLETS

    def stop_game(self):
        self.game_controller.exit_game()
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
                if CLASS_NAMES[cls_id] in ("player", "enemy"):
                    h, w = frame.shape[:2]
                    cx1 = max(0, x1 - HP_PADDING)
                    cy1 = max(0, y1 - HP_PADDING)
                    cx2 = min(w, x2 + HP_PADDING)
                    cy2 = min(h, y1 + 100)
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        hp_crops.append(crop)
        hp_texts = self.hp_recogniser.recognise(image=frame, crops=hp_crops) if hp_crops else []
        hp_values = [int(i) for i in hp_texts] if hp_texts else []
        # TODO: change 1920x1080 into real resolution
        move_action = shoot_action = ult_action = None
        try:
            move_action, shoot_action, ult_action = self.network.make_action(objects, hp_values, ult=False,
                                                                             frame_width=1920, frame_height=1080)
        except PlayerNotFoundException:
            print("Player not found in current frame")
            return

        if move_action is not None and shoot_action is not None and ult_action is not None:
            self.game_controller.make_action(move_action, shoot_action,ult_action)
