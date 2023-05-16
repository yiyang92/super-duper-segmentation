from multiprocessing import cpu_count
from pathlib import Path

from torch import optim

from super_segmenter.utils.params import Params, Registry, params_decorator
from super_segmenter.utils.constants import Models


@params_decorator
class DataParams(Params):
    data_dir: Path = Path("/home/lyy92/data/Pascal-part")

    classes_path: Path = Path(data_dir, "classes.txt")
    train_ids_path: Path = Path(data_dir, "train_id.txt")
    val_ids_path: Path = Path(data_dir, "val_id.txt")
    images_dir_path = Path(data_dir, "JPEGImages")
    gt_masks_dir_path = Path(data_dir, "gt_masks")
    image_size: tuple = (256, 256)

    def finalize(self):
        assert self.data_dir.exists()
        assert self.classes_path.exists()
        assert self.train_ids_path.exists()
        assert self.val_ids_path.exists()
        assert self.images_dir_path.exists()
        assert self.gt_masks_dir_path.exists()


@params_decorator
class UnetModelParams(Params):
    num_classes: int = 7


@params_decorator
class TrainingParams(Params):
    batch_size: int = 8
    num_workers: int = cpu_count()
    epochs: int = 50

    optimizer = optim.Adam
    learning_rate = 0.001
    betas = (0.9, 0.999)

    # Per epochs
    validation_interval: int = 10
    checkpoint_interval: int = 10


@params_decorator
class SegmenterParams(Params):
    model_dir: Path = Path("/home/lyy92/data/models/unet")
    model: Models = Models.unet

    data_params: DataParams
    model_params: UnetModelParams
    training_params: TrainingParams

    def overwrite_default_attributes(self):
        pass

    def __post_init__(self):
        self.overwrite_default_attributes()
        self.finalize_recursive()


@Registry.register
class UNetBaseline(SegmenterParams):
    pass
