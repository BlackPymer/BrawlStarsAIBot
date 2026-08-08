import time

from controller.keyboard_controller import KeyboardController
from hp_recognition.hp_recogniser import HP_PADDING
from hp_recognition.CRNN_hp_recogniser import CRNNHpRecogniser
from network.exceptions import PlayerNotFoundException
from network.network_engine import NetworkEngine
from objects_detection.object_detector import CLASS_NAMES
from objects_detection.yolo_object_detector import YoloObjectDetector
from reward.frame_parser import parse_frame
from reward.match_end_detector import MatchEndDetector
from reward.reward_tracker import RewardTracker, MATCH_END_GRACE_FRAMES
from ult_recognition.ult_classifier import UltClassifierRecogniser, extract_player_crop

STATE_DIM = (8, 18, 28)
FRAME_INTERVAL = 1 / 20  # ~20 fps шаг управления


class RLEnv:
    def __init__(self, controller=None):
        self.controller = controller or KeyboardController()
        self.object_detector = YoloObjectDetector()
        self.hp_recogniser = CRNNHpRecogniser()
        self.ult_recogniser = UltClassifierRecogniser()
        self.match_end = MatchEndDetector()
        self.network = NetworkEngine()
        self.tracker = RewardTracker()
        self.last_shot_time = 0
        self.shot_cooldown = 0.3
        self.miss_frames = 0
        self.frame_width = 1920
        self.frame_height = 1080
        self._started = False
        self.saw_player = False

    def _extract_hps(self, frame, objects):
        h, w = frame.shape[:2]
        hp_crops = []
        for obj in objects:
            for box in obj.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if CLASS_NAMES[cls_id] in ("player", "enemy"):
                    cx1 = max(0, x1 - HP_PADDING)
                    cy1 = max(0, y1 - HP_PADDING)
                    cx2 = min(w, x2 + HP_PADDING)
                    cy2 = min(h, y1 + 100)
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        hp_crops.append(crop)
        hp_texts = self.hp_recogniser.recognise(image=frame, crops=hp_crops) if hp_crops else []
        return [int(t) if t.isdigit() else 0 for t in hp_texts]

    def _get_frame(self):
        return self.controller.get_frame()

    def _build_state(self, objects, hps, has_ult, device="cpu"):
        return NetworkEngine.build_state(objects, hps, has_ult,
                                         self.frame_width, self.frame_height, device)

    def start_match(self):
        if not self._started:
            self.controller.setup_binds()
            self.controller.start_game()
            self._started = True
        time.sleep(2)
        self.tracker.reset()
        self.miss_frames = 0
        self.saw_player = False

    def reset(self):
        """Начинает матч и возвращает первый снимок состояния (или None)."""
        self.start_match()
        frame = self._get_frame()
        if frame is None:
            return None
        return self._observe(frame)

    def _observe(self, frame):
        objects = self.object_detector.detect(frame)
        hps = self._extract_hps(frame, objects)
        try:
            player_crop = extract_player_crop(frame, objects)
            has_ult = self.ult_recogniser.recognise(player_crop)
            map_inp, player_hp_inp, ult_inp = self._build_state(objects, hps, has_ult)
            snap = parse_frame(objects, hps)
            return {
                "map": map_inp[0],
                "hp": player_hp_inp[0],
                "ult": ult_inp[0],
                "snap": snap,
            }
        except PlayerNotFoundException:
            return None

    def step(self, move_action, shoot_action, ult_action):
        """Выполняет действие, снимает следующий кадр, обновляет трекер.

        Возвращает dict: state (тензоры), reward, done, info.
        """
        # выстрел с кулдауном
        fired = False
        now = time.time()
        if shoot_action < 9 and now - self.last_shot_time > self.shot_cooldown:
            fired = True
            self.last_shot_time = now
        else:
            shoot_action = 9

        self.controller.make_action(move_action, shoot_action, ult_action)
        if fired:
            # направленный выстрел -> регистрируем для атрибуции попадания/промаха
            shooter = self.tracker.player_center_prev
            self.tracker.register_shot(shoot_action if shoot_action <= 7 else None, shooter)

        time.sleep(FRAME_INTERVAL)

        frame = self._get_frame()
        if frame is None:
            return None, 0.0, False, {"error": "no_frame"}

        objects = self.object_detector.detect(frame)
        hps = self._extract_hps(frame, objects)

        player_in_frame = False
        snap = parse_frame(objects, hps)
        reward, events = self.tracker.update(snap, snap["has_player"])
        player_in_frame = snap["has_player"]
        if player_in_frame:
            self.saw_player = True

        # конец матча: победа = 3+ килла за матч, иначе поражение
        done = False
        result = None
        if not player_in_frame:
            self.miss_frames += 1
            if self.saw_player and self.miss_frames > MATCH_END_GRACE_FRAMES:
                win = self.tracker.kills >= 3
                result = "win" if win else "loss"
                if win:
                    reward += self.tracker.register_win()
                else:
                    reward += self.tracker.register_death()
                done = True
                # последовательность кликов после матча (win: battle x2, loss: after + battle x2)
                self.controller.restart_match(win=win)
                self.miss_frames = 0
                self.saw_player = False
        else:
            self.miss_frames = 0

        info = {
            "events": events,
            "reward": reward,
            "result": result,
            "player_in_frame": player_in_frame,
            "hp": snap["player_hp"],
            "kills": self.tracker.kills,
        }
        state = None
        if player_in_frame:
            try:
                player_crop = extract_player_crop(frame, objects)
                has_ult = self.ult_recogniser.recognise(player_crop)
                map_inp, player_hp_inp, ult_inp = self._build_state(objects, hps, has_ult)
                state = {
                    "map": map_inp[0],
                    "hp": player_hp_inp[0],
                    "ult": ult_inp[0],
                }
            except PlayerNotFoundException:
                state = None
        return state, reward, done, info

    def close(self):
        self.controller.exit_game()