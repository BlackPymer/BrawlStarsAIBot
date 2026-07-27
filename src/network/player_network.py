import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

INPUT_WIDTH = 28
INPUT_HEIGHT = 18


class PlayerNetwork(nn.Module):
    def __init__(self):
        super(PlayerNetwork, self).__init__()

        self.map_layer = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),  # 28x18x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x9x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 7х5х64
            nn.ReLU(),
            nn.Flatten(),  # 2240
        )
        self.fc1 = nn.Sequential(
            nn.Linear(INPUT_WIDTH * INPUT_HEIGHT * 32, 512),  # map_layer output + hp + ult possibility
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.move_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh(),
        )
        self.shoot_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.ult_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, map_inp: Tensor, hp: Tensor, has_ult: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        map_logits = self.map_layer(map_inp)
        combined = torch.cat([map_logits, hp, has_ult], 1)
        fc1_out = self.fc1(combined)
        move = self.move_head(fc1_out)
        shoot_logits = self.shoot_head(fc1_out)
        shoot = F.log_softmax(shoot_logits, dim=1)

        if has_ult.any():
            ult_logit = self.ult_head(fc1_out)
            ult_logit = ult_logit * has_ult + (1 - has_ult) * (-1e9)
            ult_prob = torch.sigmoid(ult_logit)
        else:
            ult_prob = torch.zeros(fc1_out.size(0), 1, device=fc1_out.device)

        return move, shoot, ult_prob
