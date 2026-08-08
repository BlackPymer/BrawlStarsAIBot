import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.player_network import PlayerNetwork
from rl.env import RLEnv

MODEL_PATH = BASE / "src" / "network" / "player_network_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROLLOUT = 32
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
LR = 3e-4
PPO_EPOCHS = 4
MINIBATCH = 32
MAX_ITERATIONS = 1000
LOG_EVERY = 10
EPISODE_MAX_FRAMES = 800  # предохранитель от бесконечного матча


class ValueHead(nn.Module):
    def __init__(self, in_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def to_device_tensors(state):
    return (
        state["map"].unsqueeze(0).to(DEVICE),
        state["hp"].unsqueeze(0).to(DEVICE),
        state["ult"].unsqueeze(0).to(DEVICE),
    )


def main():
    torch.manual_seed(int(time.time()))
    env = RLEnv()
    policy = PlayerNetwork().to(DEVICE)
    if os.path.isfile(MODEL_PATH):
        policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"[RL] loaded pretrained policy: {MODEL_PATH}")
    value_net = ValueHead().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()), lr=LR)

    episode_sum = 0.0
    episode_frames = 0
    episodes = 0
    episode_logs = []
    best_ep_reward = -1e9

    print(f"[RL] env starting, device={DEVICE}...")

    state = None
    done = False
    info = {}
    _t_last = time.time()
    _t_frames = 0

    def finalize_episode(ep_info, frames):
        nonlocal episodes, episode_sum, episode_frames, best_ep_reward
        episodes += 1
        episode_logs.append(episode_sum)
        print(f"[RL] episode {episodes}: reward={episode_sum:.3f} "
              f"frames={frames} kills={ep_info.get('kills', 0)} "
              f"result={ep_info.get('result', '?')} events={ep_info.get('events', [])}")
        if episode_sum > best_ep_reward:
            best_ep_reward = episode_sum
            torch.save(policy.state_dict(), str(MODEL_PATH))
            print(f"[RL] best so far, saved -> {MODEL_PATH}")
        episode_sum = 0.0
        episode_frames = 0

    try:
        for it in range(1, MAX_ITERATIONS + 1):
            if it == 1 or done:
                state = env.reset()
                done = False

            # --- ожидание игрока: поллим кадры, награды не теряем, эпизод завершаем ---
            guard = 0
            while state is None and not done:
                if guard >= 90:
                    print("[RL] no player for too long, restarting match...")
                    state = env.reset()
                    guard = 0
                    continue
                state, reward, done, info = env.step(8, 9, False)
                episode_sum += reward
                episode_frames += 1
                guard += 1
            if done:
                finalize_episode(info, episode_frames)
                continue
            if state is None:
                continue

            obs_buf = []
            act_buf = []
            rew_buf = []
            don_buf = []
            val_buf = []
            logp_buf = []

            for _ in range(ROLLOUT):
                # игрок пропал посреди роллаута — поллим, пока не вернётся
                if state is None:
                    guard = 0
                    while state is None and not done:
                        if guard >= 90:
                            break
                        state, reward, done, info = env.step(8, 9, False)
                        episode_sum += reward
                        episode_frames += 1
                        guard += 1
                    if done:
                        finalize_episode(info, episode_frames)
                        break
                    if state is None:
                        break

                obs_t = to_device_tensors(state)
                with torch.no_grad():
                    move_logp, shoot_logp, ult_prob = policy(*obs_t)
                    move_a = torch.multinomial(move_logp.exp(), 1).squeeze(1)
                    shoot_a = torch.multinomial(shoot_logp.exp(), 1).squeeze(1)
                    ult_a = (torch.bernoulli(ult_prob).squeeze(1) > 0.5).long()
                    logp = (move_logp.gather(1, move_a.unsqueeze(1))
                            + shoot_logp.gather(1, shoot_a.unsqueeze(1))).squeeze(0)
                    value = value_net(policy.encode_state(*obs_t)).item()

                move_a = int(move_a.item())
                shoot_a = int(shoot_a.item())
                ult_a = bool(ult_a.item())

                state, reward, done, info = env.step(move_a, shoot_a, ult_a)
                _t_frames += 1

                obs_buf.append(obs_t)
                act_buf.append((move_a, shoot_a, int(ult_a)))
                rew_buf.append(reward)
                don_buf.append(done)
                val_buf.append(value)
                logp_buf.append(logp)

                episode_sum += reward
                episode_frames += 1

                if done or episode_frames >= EPISODE_MAX_FRAMES:
                    finalize_episode(info, episode_frames)
                    if done:
                        break

            if not obs_buf:
                continue

            # ---- bootstrap value ----
            if state is not None:
                last_obs_t = to_device_tensors(state)
                with torch.no_grad():
                    last_value = value_net(policy.encode_state(*last_obs_t)).item() if not done else 0.0
            else:
                last_value = 0.0

            # ---- GAE ----
            advs = np.zeros(len(rew_buf), dtype=np.float32)
            rets = np.zeros(len(rew_buf), dtype=np.float32)
            gae = 0.0
            for t in reversed(range(len(rew_buf))):
                if don_buf[t]:
                    delta = rew_buf[t] - val_buf[t]
                    gae = delta
                else:
                    next_v = last_value if t == len(rew_buf) - 1 else val_buf[t + 1]
                    delta = rew_buf[t] + GAMMA * next_v - val_buf[t]
                    gae = delta + GAMMA * LAMBDA * gae
                advs[t] = gae
                rets[t] = gae + val_buf[t]
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            # ---- PPO update ----
            map_buf = torch.cat([b[0] for b in obs_buf])
            hp_buf = torch.cat([b[1] for b in obs_buf])
            ult_inp_buf = torch.cat([b[2] for b in obs_buf])
            move_a_buf = torch.tensor([a[0] for a in act_buf], device=DEVICE)
            shoot_a_buf = torch.tensor([a[1] for a in act_buf], device=DEVICE)
            old_logp_buf = torch.stack(logp_buf).detach()
            adv_buf = torch.tensor(advs, device=DEVICE)
            ret_buf = torch.tensor(rets, device=DEVICE)

            for _ in range(PPO_EPOCHS):
                perm = torch.randperm(len(obs_buf), device=DEVICE)
                for i in range(0, len(obs_buf), MINIBATCH):
                    ids = perm[i:i + MINIBATCH]
                    move_logp, shoot_logp, ult_prob = policy(
                        map_buf[ids], hp_buf[ids], ult_inp_buf[ids])
                    new_logp = (move_logp.gather(1, move_a_buf[ids].unsqueeze(1))
                                + shoot_logp.gather(1, shoot_a_buf[ids].unsqueeze(1))).squeeze(1)
                    ratio = (new_logp - old_logp_buf[ids]).exp()
                    surr1 = ratio * adv_buf[ids]
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_buf[ids]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value = value_net(policy.encode_state(
                        map_buf[ids], hp_buf[ids], ult_inp_buf[ids])).squeeze(1)
                    value_loss = F.mse_loss(value, ret_buf[ids])

                    entropy = -(move_logp.exp() * move_logp).sum(1).mean() \
                        - (shoot_logp.exp() * shoot_logp).sum(1).mean()

                    loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(policy.parameters()) + list(value_net.parameters()), 0.5)
                    optimizer.step()

            rew_sum = float(np.sum(rew_buf)) if rew_buf else 0.0
            cur_fps = _t_frames / max(time.time() - _t_last, 1e-6)
            _t_last = time.time()
            _t_frames = 0
            print(f"[RL] rollout #{it} steps={len(rew_buf)} reward={rew_sum:.3f} "
                  f"mean_reward={rew_sum / max(len(rew_buf), 1):.4f} "
                  f"episodes_done={sum(don_buf)} fps={cur_fps:.1f}")

            if it % LOG_EVERY == 0:
                recent = np.mean(episode_logs[-10:]) if episode_logs else 0.0
                print(f"[RL] it {it}/{MAX_ITERATIONS} | episodes={episodes} "
                      f"avg_reward_last10={recent:.3f} best={best_ep_reward:.3f}")

    except KeyboardInterrupt:
        print("\n[RL] interrupted")
    finally:
        env.close()
        torch.save(policy.state_dict(), str(MODEL_PATH))
        print(f"[RL] saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()