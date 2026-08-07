from controller.game_controller import GameController
from hp_recognition.hp_recogniser import HP_PADDING
from hp_recognition.CRNN_hp_recogniser import CRNNHpRecogniser
from network.exceptions import PlayerNotFoundException
from network.network_engine import NetworkEngine
from objects_detection.object_detector import CLASS_NAMES
from objects_detection.yolo_object_detector import YoloObjectDetector
from ult_recognition.ult_classifier import UltClassifierRecogniser, extract_player_crop
import time as t

SHOT_COOLDOWN = 0.3


class GameEngine:
    def __init__(self):
        self.is_game = False
        self.game_controller = None
        self.hp_recogniser = CRNNHpRecogniser()
        self.object_detector = YoloObjectDetector()
        self.ult_recogniser = UltClassifierRecogniser()
        self.last_shot_time = 0
        self.network = NetworkEngine()
        self._fps_frames = 0
        self._fps_timer = t.time()
        self.fps = 0.0

    def start_game(self, controller: GameController):
        self.game_controller = controller
        self.is_game = True
        self.game_controller.start_game()
        self.last_shot_time = t.time()
        self._fps_frames = 0
        self._fps_timer = t.time()
        print("[BOT] Game started!")

    def stop_game(self):
        self.game_controller.exit_game()
        self.is_game = False

    def update(self):
        self._fps_frames += 1
        if t.time() - self._fps_timer >= 1.0:
            self.fps = self._fps_frames / (t.time() - self._fps_timer)
            self._fps_frames = 0
            self._fps_timer = t.time()

        frame = self.game_controller.get_frame()
        if frame is None:
            return
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
            player_crop = extract_player_crop(frame, objects)
            has_ult = self.ult_recogniser.recognise(player_crop)
            move_action, shoot_action, ult_action = self.network.make_action(objects, hp_values, ult=has_ult,
                                                                             frame_width=1920, frame_height=1080)
        except PlayerNotFoundException:
            print("Player not found in current frame")
            return

        if move_action is not None and shoot_action is not None and ult_action is not None:
            fired = False
            if shoot_action < 9 and t.time() - self.last_shot_time > SHOT_COOLDOWN:
                fired = True
                self.last_shot_time = t.time()
            else:
                shoot_action = 9  # cooldown: suppress the shot
            self.game_controller.make_action(move_action, shoot_action, ult_action)
            print(f"[BOT] fps={self.fps:.1f} move={move_action} shoot={shoot_action}"
                  f" ult={ult_action} objs={len(objects)} hp={hp_values} fired={fired}")
