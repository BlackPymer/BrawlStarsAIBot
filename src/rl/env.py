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

FRAME_INTERVAL = 1 / 20


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
        self.in_match = False
        self.match_frames = 0  # Счётчик кадров внутри текущего матча

    def _extract_hps(self, frame, objects):
        h, w = frame.shape[:2]
        hp_crops = []
        valid_indices = []

        target_entities = []
        for obj in objects:
            for box in obj.boxes:
                cls_id = int(box.cls[0])
                if CLASS_NAMES[cls_id] in ("player", "enemy"):
                    target_entities.append(box)

        for idx, box in enumerate(target_entities):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx1 = max(0, x1 - HP_PADDING)
            cy1 = max(0, y1 - HP_PADDING)
            cx2 = min(w, x2 + HP_PADDING)
            cy2 = min(h, y1 + 100)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                hp_crops.append(crop)
                valid_indices.append(idx)

        raw_hps = self.hp_recogniser.recognise(image=frame, crops=hp_crops) if hp_crops else []
        
        hps = [None] * len(target_entities)
        for crop_idx, target_idx in enumerate(valid_indices):
            if crop_idx < len(raw_hps):
                val_str = raw_hps[crop_idx]
                hps[target_idx] = int(val_str) if (val_str.isdigit() and int(val_str) > 0) else None

        return hps

    def _get_frame(self):
        return self.controller.get_frame()

    def _build_state(self, objects, hps, has_ult, device="cpu"):
        clean_hps = [hp if hp is not None else 20000 for hp in hps]
        return NetworkEngine.build_state(objects, clean_hps, has_ult,
                                         self.frame_width, self.frame_height, device)

    def start_match(self):
        if not self._started:
            self.controller.setup_binds()
            self.controller.start_game()
            self._started = True
        
        self.tracker.reset()
        self.miss_frames = 0
        self.saw_player = False
        self.in_match = True
        self.match_frames = 0

    def reset(self):
        self.start_match()
        
        # Поллим кадры, чтобы убедиться, что игрок уверенно заспавнился
        for _ in range(100):
            frame = self._get_frame()
            if frame is not None:
                state = self._observe(frame)
                if state is not None:
                    return state
            time.sleep(0.1)
            
        return None

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
        fired = False
        now = time.time()
        
        if self.in_match:
            self.match_frames += 1
            if shoot_action < 9 and now - self.last_shot_time > self.shot_cooldown:
                fired = True
                self.last_shot_time = now
            else:
                shoot_action = 9

            self.controller.make_action(move_action, shoot_action, ult_action)
            if fired:
                shooter = self.tracker.player_center_prev
                self.tracker.register_shot(shoot_action if shoot_action <= 7 else None, shooter)

        time.sleep(FRAME_INTERVAL)

        frame = self._get_frame()
        if frame is None:
            return None, 0.0, False, {"error": "no_frame"}

        objects = self.object_detector.detect(frame)
        hps = self._extract_hps(frame, objects)

        snap = parse_frame(objects, hps)
        reward, events = self.tracker.update(snap, snap["has_player"])
        player_in_frame = snap["has_player"]
        
        if player_in_frame:
            self.saw_player = True

        done = False
        result = None

        # Защита: не фиксируем смерть в первые 40 кадров (2 секунды) старта матча
        warmup_over = self.match_frames > 40

        if self.in_match:
            if not player_in_frame:
                self.miss_frames += 1
                if warmup_over and self.saw_player and self.miss_frames > MATCH_END_GRACE_FRAMES:
                    win = self.tracker.kills >= 3
                    result = "win" if win else "loss"
                    reward += self.tracker.register_win() if win else self.tracker.register_death()
                    done = True
                    self.in_match = False
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