from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, draw_segmentation_masks


class SegmenterSummaryWriter:
    def __init__(self, logdir: Path):
        self.writer = SummaryWriter(logdir)

    def write_scalars(self, message: dict, global_step: int) -> None:
        for name in message.keys():
            self.writer.add_scalar(
                tag=name, scalar_value=message[name], global_step=global_step
            )

    def write_images(self, message: dict, global_step: int) -> None:
        # E.g. message[train/images], message[train/masks]
        images: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        subset = "train"
        for key in message:
            if key.split("/")[1] == "images":
                images = message[key]
                subset = key.split("/")[0]

            if key.split("/")[1] == "masks":
                masks = message[masks]

        outs = []
        if not images and not masks:
            return

        for image, mask in zip(images, masks):
            outs.append(draw_segmentation_masks(image=image, masks=mask))

        grid = make_grid(outs)
        self.writer.add_image(
            tag=f"{subset}/images", img_tensor=grid, global_step=global_step
        )
