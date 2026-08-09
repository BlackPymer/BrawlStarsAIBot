import os

import torch

from network.exceptions import PlayerNotFoundException
from objects_detection.object_detector import CLASS_NAMES
from network.player_network import PlayerNetwork, INPUT_HEIGHT, INPUT_WIDTH

MAX_HP_VALUE = 20000
PLAYER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "player_network_best.pt")


class NetworkEngine:
    def __init__(self):
        self.player = PlayerNetwork()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.player = self.player.to(self.device)
        if os.path.isfile(PLAYER_MODEL_PATH):
            self.player.load_state_dict(torch.load(PLAYER_MODEL_PATH, map_location=self.device))
            print(f"[NET] loaded weights: {PLAYER_MODEL_PATH}")

    @staticmethod
    def build_state(objects, hps, ult, frame_width, frame_height, device):
        map_inp = torch.zeros(size=(1, 8, INPUT_HEIGHT, INPUT_WIDTH), device=device)
        hp_and_player_count = 0
        player_hp_inp = 0
        ult_inp = torch.tensor([[1.0 if ult else 0.0]], device=device)
        for obj in objects:
            for box in obj.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                nx1 = round(x1 / frame_width * INPUT_WIDTH)
                nx2 = round(x2 / frame_width * INPUT_WIDTH)
                ny1 = round(y1 / frame_height * INPUT_HEIGHT)
                ny2 = round(y2 / frame_height * INPUT_HEIGHT)
                if CLASS_NAMES[cls_id] in ("player", "enemy"):
                    map_inp[0, cls_id, ny1:ny2, nx1:nx2] = hps[hp_and_player_count] / MAX_HP_VALUE
                    if CLASS_NAMES[cls_id] == "player":
                        player_hp_inp = torch.tensor([[hps[hp_and_player_count] / MAX_HP_VALUE]], device=device)
                    hp_and_player_count += 1
                else:
                    map_inp[0, cls_id, ny1:ny2, nx1:nx2] = 1
        if isinstance(player_hp_inp, int):
            raise PlayerNotFoundException
        return map_inp, player_hp_inp, ult_inp

    def make_action(self, objects: list, hps: list, ult: bool, frame_width: int, frame_height: int):
        map_inp, player_hp_inp, ult_inp = self.build_state(objects, hps, ult, frame_width, frame_height, self.device)
        move_res, shoot_res, ult_res = self.player(map_inp, player_hp_inp, ult_inp)
        move_action = torch.multinomial(move_res.exp(), num_samples=1).item()
        shoot_res = torch.multinomial(shoot_res.exp(), num_samples=1).item()
        ult_res = torch.bernoulli(ult_res).item()
        return move_action, shoot_res, bool(ult_res)
