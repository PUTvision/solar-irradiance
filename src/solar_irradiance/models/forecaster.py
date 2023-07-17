from typing import Optional

import lightning.pytorch as pl
import torch
import torchmetrics
from torch.optim import Optimizer
from torchvision.models.video import r3d_18, R3D_18_Weights


class Forecaster(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 input_channels: int,
                 loss_function: str,
                 lr: float,
                 lr_patience: int
                 ):
        super().__init__()

        self._model_name = model_name
        self._input_channels = input_channels
        self._loss_function = loss_function
        self._lr = lr
        self._lr_patience = lr_patience

        self.network = r3d_18(weights=R3D_18_Weights.DEFAULT, progress=True)
        self.network.fc = torch.nn.Identity()
        self.network_head = torch.nn.Sequential(
            torch.nn.Linear(515, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1),
        )

        if loss_function == 'MSE':
            self.loss = torch.nn.MSELoss()
        elif loss_function == 'MAE':
            self.loss = torch.nn.L1Loss(reduction='sum')
        else:
            raise NotImplementedError(f'Unsupported loss function: {loss_function}')

        metrics = torchmetrics.MetricCollection([
            torchmetrics.MeanSquaredError(),
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanAbsolutePercentageError(),
        ])

        self.train_metrics = metrics.clone('train_')
        self.val_metrics = metrics.clone('val_')
        self.test_metrics = metrics.clone('test_')

        self.save_hyperparameters()

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x: torch.Tensor, irradiance_history: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.network_head(torch.cat([x, irradiance_history], dim=1))
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[torch.Tensor]:
        source_images, source_irradiances, target_irradiances = batch
        predicted_irradiances = self.forward(source_images, source_irradiances)

        loss = self.loss(predicted_irradiances, target_irradiances)
        if torch.isinf(loss):
            return None

        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.train_metrics.update(predicted_irradiances, target_irradiances)
        self.log_dict(self.train_metrics, sync_dist=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        source_images, source_irradiances, target_irradiances = batch
        predicted_irradiances = self.forward(source_images, source_irradiances)

        loss = self.loss(predicted_irradiances, target_irradiances)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.val_metrics.update(predicted_irradiances, target_irradiances)
        self.log_dict(self.val_metrics, sync_dist=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        source_images, source_irradiances, target_irradiances = batch
        predicted_irradiances = self.forward(source_images, source_irradiances)

        loss = self.loss(predicted_irradiances, target_irradiances)

        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics.update(predicted_irradiances, target_irradiances)
        self.log_dict(self.test_metrics, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr)
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self._lr_patience,
            min_lr=1e-6,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': reduce_lr_on_plateau,
            'monitor': 'train_loss_epoch'
        }
