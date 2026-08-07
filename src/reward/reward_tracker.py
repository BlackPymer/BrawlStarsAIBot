REWARD_WIN = 1.0
REWARD_KILL = 0.7
REWARD_POWERCUBE = 0.5
REWARD_HIT = 0.4
REWARD_BOX_KILL = 0.3
REWARD_MISS = -0.03
REWARD_ZONE_DAMAGE = -0.2
REWARD_DEATH = -1.0

HIT_ATTRIBUTION_FRAMES = 10
MATCH_END_GRACE_FRAMES = 20

CUBE_PICKUP_DIST = 120
ENEMY_MATCH_DIST = 240
BOX_MATCH_DIST = 240

# направления стрельбы 0-7 (N по часовой), конус ~45°
SHOOT_VECTORS = {
    0: (0, -1), 1: (1, -1), 2: (1, 0), 3: (1, 1),
    4: (0, 1), 5: (-1, 1), 6: (-1, 0), 7: (-1, -1),
}


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _in_cone(shoot_class, target_center, shooter_center):
    if shoot_class not in SHOOT_VECTORS:
        return False
    vx = target_center[0] - shooter_center[0]
    vy = target_center[1] - shooter_center[1]
    if vx == 0 and vy == 0:
        return True
    length = (vx * vx + vy * vy) ** 0.5
    sx, sy = SHOOT_VECTORS[shoot_class]
    dot = (vx / length) * sx + (vy / length) * sy
    return dot > 0.7  # ~45° вокруг направления


class RewardTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player_hp_prev = None
        self.player_center_prev = None
        self.enemies = {}      # id -> dict(center, hp)
        self.cubes = {}        # id -> center
        self.boxes = {}        # id -> center
        self.shots = []        # [frames_left, shoot_class, shooter_center]
        self.hits_this_window = []  # были ли попадания у активных выстрелов
        self.miss_grace = 0
        self.last_shoot = None

    # ---------- внешние события ----------
    def update(self, snap, player_in_frame):
        """snap — результат parse_frame. Возвращает (reward, events)."""
        events = []
        reward = 0.0
        center = snap["player_center"]
        hp = snap["player_hp"]

        # --- зона: игрок внутри зоны и HP упал ---
        if player_in_frame and hp is not None and self.player_hp_prev is not None \
                and hp < self.player_hp_prev:
            in_zone = any(_dist(center, z) < 90 for z in snap["zone_polys"])
            if in_zone:
                events.append(("zone_damage", REWARD_ZONE_DAMAGE))

        # --- трекинг врагов, киллы, попадания ---
        cur_enemies = {}
        for ent in snap["entities"]:
            if ent["cls"] != "enemy":
                continue
            best_id, best_d = None, 1e9
            for eid, prev in self.enemies.items():
                d = _dist(ent["center"], prev["center"])
                if d < best_d and d < ENEMY_MATCH_DIST:
                    best_d, best_id = d, eid
            eid = best_id if best_id is not None else f"enemy_{len(cur_enemies)}_{len(self.enemies)}"
            cur_enemies[eid] = {"center": ent["center"], "hp": ent["hp"]}

        # попадания: HP врага упало -> выстрел "попадание" только если
        # стреляли в этом направлении недавно
        shot_hit = False
        for eid, cur in cur_enemies.items():
            prev = self.enemies.get(eid)
            if prev is None:
                continue
            if cur["hp"] is not None and prev["hp"] is not None and cur["hp"] < prev["hp"] - 50:
                if center is not None and any(
                        _in_cone(s[1], cur["center"], center) for s in self.shots):
                    events.append(("hit", REWARD_HIT))
                    shot_hit = True

        # киллы: враг был и исчез
        for eid, prev in self.enemies.items():
            if eid not in cur_enemies:
                events.append(("kill", REWARD_KILL))

        # --- кубы: исчез рядом с игроком -> подбор ---
        cur_cubes = {}
        for ent in snap["entities"]:
            if ent["cls"] == "powercube":
                cur_cubes[f"cube_{len(cur_cubes)}"] = ent["center"]
        for cid, cpos in self.cubes.items():
            if cid not in cur_cubes and center is not None \
                    and _dist(cpos, center) < CUBE_PICKUP_DIST:
                events.append(("powercube", REWARD_POWERCUBE))

        # --- боксы: исчез -> убит (грубо, без атрибуции) ---
        cur_boxes = {}
        for ent in snap["entities"]:
            if ent["cls"] == "powercube-box":
                cur_boxes[f"box_{len(cur_boxes)}"] = ent["center"]
        for bid, bpos in self.boxes.items():
            if bid not in cur_boxes:
                events.append(("box_kill", REWARD_BOX_KILL))

        # --- промах: выстрел в направлении, где ни у кого не убавилось HP ---
        if self.last_shoot is not None:
            shoot_class, shooter = self.last_shoot
            hit = any(_in_cone(shoot_class, e["center"], shooter) and e["hp"] is not None
                      for e in cur_enemies.values())
            # "грубо": если враг в конусе есть, но HP не падал за окно — промах
            # если врага нет вовсе в конусе — тоже промах (выстрел в пустоту)
            if not hit:
                events.append(("miss", REWARD_MISS))
            self.last_shoot = None

        # --- состояние ---
        self.enemies = cur_enemies
        self.cubes = cur_cubes
        self.boxes = cur_boxes
        if player_in_frame:
            self.player_hp_prev = hp
            self.player_center_prev = center
        else:
            self.player_hp_prev = None
            self.player_center_prev = None

        # дрейф выстрелов (счётчик кадров после выстрела)
        self.shots = [[f - 1, s, c] for f, s, c in self.shots if f > 0]
        if shot_hit:
            self.shots = []

        reward = sum(ev[1] for ev in events)
        return reward, events

    def register_shot(self, shoot_class, shooter_center):
        self.shots.append([HIT_ATTRIBUTION_FRAMES, shoot_class, shooter_center])
        self.last_shoot = (shoot_class, shooter_center)
        self.hits_this_window = []

    def register_win(self):
        return REWARD_WIN

    def register_death(self):
        return REWARD_DEATH