class MatchEndDetector:
    """Определяет конец матча: победу/поражение по экрану конца матча.

    TODO: реализовать после получения скринов экрана победы/поражения.
    Сейчас возвращает None (матч не закончен).
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._last_result = None

    def detect(self, frame) -> str | None:
        """Возвращает 'win', 'loss' или None (не структура)."""
        return self._last_result

    def reset(self):
        self._last_result = None