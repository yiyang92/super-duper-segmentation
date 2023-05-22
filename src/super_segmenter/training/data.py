from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset


class PascalPartDataset(Dataset):
    def __init__(
        self,
        ids_path: Path,
        images_dir_path: Path,
        masks_dir_path: Path,
        img_size: tuple = (256, 256),
    ) -> None:
        """
        Args:
            ids_path: path to {train,val}.txt
            images_dir_path: path to images directory (.png files)
            masks_dir_path: path to masks directory (.npy files)
        """
        image_paths = sorted(list(images_dir_path.glob("*.jpg")))
        mask_paths = sorted(list(masks_dir_path.glob("*.npy")))

        ids_imgmask = {
            path_img.name.split(".")[0]: (path_img, path_mask)
            for path_img, path_mask in zip(image_paths, mask_paths)
        }
        ids = list(map(lambda x: x.strip(), ids_path.open("r").readlines()))

        self._data = [ids_imgmask[id_] for id_ in ids]
        self._img_size = img_size

    def _transform_image(self, image: torch.Tensor) -> torch.Tensor:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    size=self._img_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            ]
        )
        return transform(image)

    def _transform_mask(self, mask: torch.Tensor) -> torch.Tensor:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    size=self._img_size,
                    interpolation=transforms.InterpolationMode.NEAREST,
                )
            ]
        )
        return transform(mask)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self._data[index]
        image = read_image(str(img_path)).to(torch.float32) / 255.0
        image = self._transform_image(image)
        labels = torch.tensor(np.load(mask_path)).long()
        labels = labels.unsqueeze(0)
        labels = self._transform_mask(labels).squeeze()
        return image, labels
