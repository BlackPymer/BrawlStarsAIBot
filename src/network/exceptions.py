class PlayerNotFoundException(Exception):
    def __init__(self):
        super("player was not found in the frame", self).__init__()
