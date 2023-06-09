import logging
from pathlib import Path

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from super_segmenter.params import (
    Registry,
    TrainingParams,
    DataParams,
    SegmenterParams,
    UnetModelParams,
)
from super_segmenter.utils.constants import (
    Models,
    TRAIN_CE_LOSS,
    VALID_CE_LOSS,
    LOGS_DIR,
    VALID_IMAGES,
    VALID_MASKS,
)
from super_segmenter.models import UNet
from super_segmenter.training.data import PascalPartDataset
from super_segmenter.training.summary import SegmenterSummaryWriter


class Trainer:
    def __init__(
        self,
        params_name: str,
    ) -> None:
        self._log = logging.getLogger("trainer")
        params: SegmenterParams = Registry.get_params(params_name)

        self._params: TrainingParams = params.training_params
        self._data_params: DataParams = params.data_params
        self._model_params: UnetModelParams = params.model_params
        self._model_dir: Path = params.model_dir

        # TODO: init in one place for all models
        if params.model == Models.unet:
            self._model = UNet(self._model_params)
        self._device = (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._model.to(self._device)

        self._init_datasets()
        self._init_dataloaders()
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = self._init_optimizer()
        self._iteration: int = 0
        self._summary_writer = SegmenterSummaryWriter(
            Path(self._params.logs_dir, LOGS_DIR)
        )

    def _model_summary(self) -> str:
        out = []
        for idx, m in enumerate(self._model.modules()):
            out.append(f"{idx} -> {m}\n")
        num_params = sum([param.numel() for param in self._model.parameters()])
        out.append(f"number of parameters: {num_params}\n")
        return "".join(out)

    def _init_optimizer(self) -> torch.optim.Optimizer:
        optimizer = self._params.optimizer(
            params=self._model.parameters(),
            lr=self._params.learning_rate,
            betas=self._params.betas,
        )
        return optimizer

    def _init_datasets(self) -> None:
        self._train_dataset = PascalPartDataset(
            ids_path=self._data_params.train_ids_path,
            images_dir_path=self._data_params.images_dir_path,
            masks_dir_path=self._data_params.gt_masks_dir_path,
            img_size=self._model_params.image_size,
        )
        self._valid_dataset = PascalPartDataset(
            ids_path=self._data_params.val_ids_path,
            images_dir_path=self._data_params.images_dir_path,
            masks_dir_path=self._data_params.gt_masks_dir_path,
            img_size=self._model_params.image_size,
        )
        self._log.info(
            f"data was initialized\n train: "
            f"{len(self._train_dataset)} val: {len(self._valid_dataset)}"
        )

    def _init_dataloaders(self) -> None:
        self._train_loader = DataLoader(
            dataset=self._train_dataset,
            num_workers=self._params.num_workers,
            shuffle=True,
            batch_size=self._params.batch_size,
            pin_memory=True,
            drop_last=True,
        )
        self._valid_dataloader = DataLoader(
            dataset=self._valid_dataset, shuffle=False, batch_size=1
        )

    def _save_model(self) -> None:
        if not self._model_dir.exists():
            self._model_dir.mkdir()

        model_name = Path(self._model_dir, f"checkpoint-{self._iteration}")
        torch.save(self._model.state_dict(), model_name)
        self._log.info(f"checkpoint saved: {model_name}")

    def _train_step(self, X: torch.Tensor, Y: torch.Tensor) -> dict:
        summary = {TRAIN_CE_LOSS: 0.0}
        X, Y = X.to(self._device), Y.to(self._device)
        self._optimizer.zero_grad()
        Y_pred = self._model(X)
        loss = self._criterion(Y_pred, Y)
        loss.backward()
        self._optimizer.step()
        summary[TRAIN_CE_LOSS] = loss.item()
        return summary

    def validation(self) -> tuple[dict, dict]:
        self._log.info("\n################Validation################")
        self._model.eval()
        summary = {VALID_CE_LOSS: 0.0}
        images_summary = {VALID_IMAGES: [], VALID_MASKS: []}
        with torch.no_grad():
            for X, Y in tqdm(
                self._valid_dataloader,
                total=len(self._valid_dataloader),
                leave=False,
            ):
                X, Y = X.to(self._device), Y.to(self._device)
                Y_pred = self._model(X)
                loss = self._criterion(input=Y_pred, target=Y)
                summary[VALID_CE_LOSS] += loss.item()
                if (
                    len(images_summary[VALID_IMAGES])
                    < self._params.max_val_summary
                ):
                    images_summary[VALID_IMAGES].append(X.squeeze())
                    images_summary[VALID_MASKS].append(
                        torch.argmax(Y_pred, dim=1).squeeze()
                    )

        # TODO: add mIOU metrics
        summary[VALID_CE_LOSS] = summary[VALID_CE_LOSS] / len(
            self._valid_dataloader
        )
        return summary, images_summary

    def train(self) -> None:
        self._log.info(f"Model layers: \n {self._model_summary()}")
        self._log.info("\n################START TRAINING################")
        self._model.train()
        for epoch in tqdm(range(self._params.epochs)):
            self._log.info(f"\n####### TRAIN EPOCH: {epoch} ################")
            epoch_loss = 0.0
            for X, Y in tqdm(
                self._train_loader, total=len(self._train_loader), leave=False
            ):
                train_summary = self._train_step(X, Y)
                epoch_loss += train_summary[TRAIN_CE_LOSS]
                if self._iteration % self._params.summary_interval == 0:
                    self._summary_writer.write_scalars(
                        train_summary, global_step=self._iteration
                    )
                self._iteration += 1

            summary = {
                "epoch loss": epoch_loss / len(self._train_loader),
                "iteration": self._iteration,
            }
            self._log.info(f"\n\t{summary}")
            if epoch % self._params.validation_interval == 0:
                valid_summary, valid_images = self.validation()
                self._log.info(f"\nValidation summary:\n\t{valid_summary}")
                self._summary_writer.write_scalars(
                    valid_summary, global_step=self._iteration
                )
                self._summary_writer.write_images(
                    valid_images, global_step=self._iteration
                )

            if epoch % self._params.checkpoint_interval == 0:
                self._save_model()

        self._log.info("\n##### TRAINING COMPLETED #####")
        valid_summary, valid_images = self.validation()
        self._summary_writer.write_scalars(
            valid_summary, global_step=self._iteration
        )
        self._summary_writer.write_images(
            valid_images, global_step=self._iteration
        )
        self._log.info(f"\nValidation summary:\n\t{valid_summary}")
        self._save_model()
