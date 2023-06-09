from enum import Enum, auto


class Models(Enum):
    unet = auto()

    def __str__(self):
        return str(self.value)


TRAIN_CE_LOSS = "train/CE_loss"

VALID_CE_LOSS = "valid/CE_loss"
VALID_IMAGES = "valid/images"
VALID_MASKS = "valid/masks"

LOGS_DIR = "segmenter_logs"
