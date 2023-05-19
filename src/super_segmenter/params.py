from multiprocessing import cpu_count
from pathlib import Path

from torch import optim

from super_segmenter.utils.params import Params, Registry, params_decorator
from super_segmenter.utils.constants import Models


@params_decorator
class DataParams(Params):
    data_dir: Path

    classes_path: Path
    train_ids_path: Path
    val_ids_path: Path
    images_dir_path: Path
    gt_masks_dir_path: Path
    image_size: tuple = (256, 256)

    def finalize(self):
        assert self.data_dir.exists()
        self.classes_path = Path(self.data_dir, "classes.txt")
        self.train_ids_path = Path(self.data_dir, "train_id.txt")
        self.val_ids_path = Path(self.data_dir, "val_id.txt")
        self.images_dir_path = Path(self.data_dir, "JPEGImages")
        self.gt_masks_dir_path = Path(self.data_dir, "gt_masks")

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
    max_val_summary: int = 10  # Number of samples for tb logging

    optimizer = optim.Adam
    learning_rate = 0.001
    betas = (0.9, 0.999)

    # Per epochs
    validation_interval: int = 10
    checkpoint_interval: int = 10

    # Per iteration
    summary_interval: int = 250
    logs_dir: Path


@params_decorator
class SegmenterParams(Params):
    model_dir: Path
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
    def overwrite_default_attributes(self):
        self.model_dir = Path("/root/models-small/unet")
        self.data_params.data_dir = Path("/root/data-small/Pascal-part")
        self.training_params.logs_dir = Path("/root/tf-logs")

        self.training_params.batch_size = 32
