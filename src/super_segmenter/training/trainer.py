import logging


class Trainer:
    
    def __init__(
        self,
        params_name: str,
    ):
        self.log = logging.getLogger("trainer")
        