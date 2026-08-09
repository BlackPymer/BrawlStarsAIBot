import os

import cv2

DEFAULT_TEMPLATE = os.path.join(os.path.dirname(__file__), "win_button_template.png")
BOTTOM_RIGHT_X = 0.45  # доля ширины, с которой начинаем искать кнопку справа


class MatchEndDetector:
    """Определяет победу/поражение по кнопке в правом нижнем углу экрана.

    Логика: если игрок не найден 20+ кадров — матч окончен.
    Ищем в правой нижней части экрана кнопку (шаблон всегда одинаков).
    Нашли — победа ('win'), не нашли — поражение ('loss').

    TODO: пользователь предоставит скрин кнопки — положить в
    win_button_template.png (или указать путь в config['win_button_template']).
    Пока шаблона нет — detect() вернёт None.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        template_path = self.config.get("win_button_template", DEFAULT_TEMPLATE)
        self.template = None
        if os.path.isfile(template_path):
            self.template = cv2.imread(template_path)
        self._last_result = None

    def detect(self, frame) -> str | None:
        """Возвращает 'win', 'loss' или None (шаблон не задан / матч идёт)."""
        if frame is None or self.template is None:
            self._last_result = None
            return None

        h, w = frame.shape[:2]
        th, tw = self.template.shape[:2]
        region = frame[int(h * 0.55):h, int(w * BOTTOM_RIGHT_X):w]

        result = cv2.matchTemplate(region, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        self._last_result = "win" if max_val > 0.8 else "loss"
        return self._last_result

    def reset(self):
        self._last_result = None