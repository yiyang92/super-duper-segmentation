from dataclasses import is_dataclass, dataclass
from pathlib import Path

import torch

from super_segmenter.utils.params import (
    Params,
    Registry,
    params_decorator
)

@params_decorator
class DataParams(Params):
    data_dir: Path = Path("/home/lyy92/data/Pascal-part")
    
    classes_path: Path = Path(data_dir, "classes.txt")
    train_ids_path: Path = Path(data_dir, "train_id.txt")
    val_ids_path: Path = Path(data_dir, "val_id.txt")
    images_dir_path = Path(data_dir, "JPEGImages")
    gt_masks_dir_path = Path(data_dir, "gt_masks")
    
    def finalize(self):
        assert self.data_dir.exists()
        assert self.images_dir_path.exists()
        assert self.gt_masks_dir_path.exists()


@params_decorator
class UnetModelParams(Params):
    image_size: tuple = (256, 256)
    num_classes: int = 7
    

@params_decorator
class TrainingParams(Params):
    batch_size = 8
    learning_rate = 0.001
    epochs = 30
    device = torch.device("cuda:0")


@params_decorator
class SegmenterParams(Params):
    model_dir: Path = Path("/home/lyy92/data/models/unet")
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
