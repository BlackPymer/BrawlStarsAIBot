from objects_detection.object_detector import CLASS_NAMES

DEFAULT_HP = 20000


def parse_frame(objects, hps):
    """Разбирает YOLO-objects + hp_values в структурированный снимок кадра.

    Возвращает dict:
        has_player: bool
        player_center: (x, y) или None
        player_hp: int или None
        zone_polys: список (один элемент = список центров зоны по одному боксу)
        entities: список dict {id, cls, center, box, hp}
    """
    entities = []
    player_center = None
    player_hp = None
    zone_polys = []
    has_player = False

    hp_idx = 0
    for obj in objects:
        for box in obj.boxes:
            cid = int(box.cls[0])
            cls = CLASS_NAMES[cid]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            ent = {
                "id": cid,
                "cls": cls,
                "center": center,
                "box": (x1, y1, x2, y2),
                "hp": None,
            }
            if cls in ("player", "enemy"):
                ent["hp"] = hps[hp_idx] if hp_idx < len(hps) else DEFAULT_HP
                hp_idx += 1
            if cls == "player":
                has_player = True
                player_center = center
                player_hp = ent["hp"]
            elif cls == "zone":
                zone_polys.append(center)
            entities.append(ent)

    return {
        "has_player": has_player,
        "player_center": player_center,
        "player_hp": player_hp,
        "zone_polys": zone_polys,
        "entities": entities,
    }