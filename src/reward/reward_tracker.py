import os

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
MISS_PATIENCE = 5  # Объект должен отсутствовать 5 кадров подряд для фиксации события

CUBE_PICKUP_DIST = 120.0
ENEMY_MATCH_DIST = 240.0
BOX_MATCH_DIST = 240.0

BLOCK_PX = 1920.0 / 28.0   # ~68.6 px — один блок из сетки
HIT_RANGE = 9.0 * BLOCK_PX  # 9 блоков

# Направления стрельбы 0-7 (N по часовой), конус ~45°
SHOOT_VECTORS = {
    0: (0, -1), 1: (1, -1), 2: (1, 0), 3: (1, 1),
    4: (0, 1), 5: (-1, 1), 6: (-1, 0), 7: (-1, -1),
}


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _in_cone(shoot_class, target_center, shooter_center):
    if shoot_class not in SHOOT_VECTORS or shooter_center is None:
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
        self.enemies = {}  # eid -> {"center": (x,y), "hp": int/None, "missing": int}
        self.cubes = {}    # cid -> {"center": (x,y), "missing": int}
        self.boxes = {}    # bid -> {"center": (x,y), "missing": int}
        self.shots = []    # [[frames_left, shoot_class, shooter_center], ...]
        self.last_shoot = None
        self.kills = 0

    def _was_shot_towards(self, target_center, shooter_center):
        """Проверяет, стрелял ли агент в направлении цели за последние кадры."""
        if shooter_center is None or not self.shots:
            return False
        return any(_in_cone(s[1], target_center, shooter_center) for s in self.shots)

    def update(self, snap, player_in_frame):
        """snap — результат parse_frame. Возвращает (reward, events)."""
        events = []
        center = snap["player_center"]
        hp = snap["player_hp"]

        # --- 1. ЯДОВИТЫЙ ГАЗ (ZONE) ---
        if player_in_frame and center is not None:
            in_zone = any(_dist(center, z) < 100.0 for z in snap["zone_polys"])
            if in_zone:
                events.append(("zone_damage", REWARD_ZONE_DAMAGE))

        # --- 2. ТРЕКИНГ ВРАГОВ, ПОПАДАНИЯ И КИЛЛЫ ---
        cur_enemies_detected = [e for e in snap["entities"] if e["cls"] == "enemy"]
        updated_enemies = {}

        # Обновляем / сопоставляем имеющихся врагов
        for ent in cur_enemies_detected:
            best_id, best_d = None, 1e9
            for eid, prev in self.enemies.items():
                d = _dist(ent["center"], prev["center"])
                if d < best_d and d < ENEMY_MATCH_DIST:
                    best_d, best_id = d, eid

            eid = best_id if best_id is not None else f"enemy_{len(self.enemies)}_{len(updated_enemies)}"
            prev_data = self.enemies.get(eid, {})

            cur_hp = ent["hp"]
            prev_hp = prev_data.get("hp")

            # Попадание: HP упал на >= 100, и был выстрел в направлении врага
            if cur_hp and prev_hp and 0 < cur_hp < prev_hp - 100:
                if center is not None and _dist(ent["center"], center) <= HIT_RANGE:
                    if self._was_shot_towards(ent["center"], center):
                        events.append(("hit", REWARD_HIT))

            updated_enemies[eid] = {
                "center": ent["center"],
                "hp": cur_hp if (cur_hp and cur_hp > 0) else prev_hp,
                "missing": 0
            }

        # Исчезнувшие враги (Проверка на КИЛЛ)
        for eid, prev in self.enemies.items():
            if eid not in updated_enemies:
                missing = prev.get("missing", 0) + 1
                if missing >= MISS_PATIENCE:
                    # Убийство засчитывается ТОЛЬКО если:
                    # 1) Враг пропал окончательно (>= MISS_PATIENCE кадров)
                    # 2) Находился близко (<= HIT_RANGE)
                    # 3) В его сторону БЫЛ совершён выстрел!
                    if center is not None and _dist(prev["center"], center) <= HIT_RANGE:
                        if self._was_shot_towards(prev["center"], center):
                            events.append(("kill", REWARD_KILL))
                            self.kills += 1
                else:
                    updated_enemies[eid] = {**prev, "missing": missing}

        self.enemies = updated_enemies

        # --- 3. ТРЕКИНГ ЯЩИКОВ (BOXES) ---
        cur_boxes_detected = [e for e in snap["entities"] if e["cls"] == "powercube-box"]
        updated_boxes = {}

        for ent in cur_boxes_detected:
            best_id, best_d = None, 1e9
            for bid, prev in self.boxes.items():
                d = _dist(ent["center"], prev["center"])
                if d < best_d and d < BOX_MATCH_DIST:
                    best_d, best_id = d, bid
            bid = best_id if best_id is not None else f"box_{len(self.boxes)}_{len(updated_boxes)}"
            updated_boxes[bid] = {"center": ent["center"], "missing": 0}

        # Разрушение ящика засчитывается только если в него стреляли!
        for bid, prev in self.boxes.items():
            if bid not in updated_boxes:
                missing = prev.get("missing", 0) + 1
                if missing >= MISS_PATIENCE:
                    if center is not None and _dist(prev["center"], center) <= HIT_RANGE:
                        if self._was_shot_towards(prev["center"], center):
                            events.append(("box_kill", REWARD_BOX_KILL))
                else:
                    updated_boxes[bid] = {**prev, "missing": missing}

        self.boxes = updated_boxes

        # --- 4. ТРЕКИНГ КУБОВ (POWER CUBES) ---
        cur_cubes_detected = [e for e in snap["entities"] if e["cls"] == "powercube"]
        updated_cubes = {}

        for ent in cur_cubes_detected:
            best_id, best_d = None, 1e9
            for cid, prev in self.cubes.items():
                d = _dist(ent["center"], prev["center"])
                if d < best_d and d < CUBE_PICKUP_DIST:
                    best_d, best_id = d, cid
            cid = best_id if best_id is not None else f"cube_{len(self.cubes)}_{len(updated_cubes)}"
            updated_cubes[cid] = {"center": ent["center"], "missing": 0}

        for cid, prev in self.cubes.items():
            if cid not in updated_cubes:
                missing = prev.get("missing", 0) + 1
                if missing >= MISS_PATIENCE:
                    if center is not None and _dist(prev["center"], center) < CUBE_PICKUP_DIST:
                        events.append(("powercube", REWARD_POWERCUBE))
                else:
                    updated_cubes[cid] = {**prev, "missing": missing}

        self.cubes = updated_cubes

        # --- 5. ОБРАБОТКА ПРОМАХОВ (MISS) ---
        if self.last_shoot is not None and self.last_shoot[1] is not None:
            shoot_class, shooter = self.last_shoot
            hit_enemy = any(_in_cone(shoot_class, e["center"], shooter) for e in self.enemies.values())
            hit_box = any(_in_cone(shoot_class, b["center"], shooter) for b in self.boxes.values())
            
            # Если выстрел сделан в «пустоту» без врагов/ящиков в конусе
            if not (hit_enemy or hit_box):
                events.append(("miss", REWARD_MISS))
            self.last_shoot = None

        # --- 6. ОБНОВЛЕНИЕ СОСТОЯНИЯ И ТАЙМЕРЫ ---
        if player_in_frame:
            self.player_hp_prev = hp if (hp and hp > 0) else self.player_hp_prev
            self.player_center_prev = center
        else:
            self.player_hp_prev = None
            self.player_center_prev = None

        # Уменьшаем время жизни зарегистрированных выстрелов
        self.shots = [[f - 1, s, c] for f, s, c in self.shots if f > 0]

        reward = sum(ev[1] for ev in events)
        return reward, events

    def register_shot(self, shoot_class, shooter_center):
        """Регистрирует новый выстрел для отслеживания попал/промазал."""
        if shoot_class is not None:
            self.shots.append([HIT_ATTRIBUTION_FRAMES, shoot_class, shooter_center])
            self.last_shoot = (shoot_class, shooter_center)

    def register_win(self):
        return REWARD_WIN

    def register_death(self):
        return REWARD_DEATH