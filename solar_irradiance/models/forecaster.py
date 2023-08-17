from typing import Optional

import lightning.pytorch as pl
import timm
import torch
import torchmetrics
from torch.optim import Optimizer
from torchvision.models.video import R3D_18_Weights, swin3d_t, Swin3D_T_Weights, Swin3D_B_Weights, MC3_18_Weights, R2Plus1D_18_Weights
from transformers import TimesformerConfig, TimesformerModel, TimesformerForVideoClassification

from solar_irradiance.losses.mape import MAPELoss
from solar_irradiance.models.architectures.resnet import r3d_18, mc3_18, r2plus1d_18
from solar_irradiance.models.architectures.swin_transformer import swin3d_b
# from solar_irradiance.models.architectures.timesformer import Timesformer


class Forecaster(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 input_channels: int,
                 loss_function: str,
                 lr: float,
                 lr_patience: int,
                 time_window: int,
                 history_size: int
                 ):
        super().__init__()

        self._model_name = model_name
        self._input_channels = input_channels
        self._loss_function = loss_function
        self._lr = lr
        self._lr_patience = lr_patience

        if model_name == 'swin3d_b':
            self.network = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1, progress=True)
            self.num_features = self.network.num_features
            self.network.head = torch.nn.Identity()
        elif model_name == 'r3d_18':
            self.network = r3d_18(weights=R3D_18_Weights.KINETICS400_V1, progress=True, in_channels=self._input_channels)
            self.num_features = self.network.fc.in_features
            self.network.fc = torch.nn.Identity()
        elif model_name == 'mc3_18':
            self.network = mc3_18(weights=MC3_18_Weights.KINETICS400_V1, progress=True, in_channels=self._input_channels)
            self.num_features = self.network.fc.in_features
            self.network.fc = torch.nn.Identity()
        elif model_name == 'r2plus1d_18':
            self.network = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1, progress=True, in_channels=self._input_channels)
            self.num_features = self.network.fc.in_features
            self.network.fc = torch.nn.Identity()
        elif model_name.startswith('timm-'):
            self.network = timm.create_model(
                model_name.replace('timm-', ''),
                pretrained=True,
                num_classes=1,
                in_chans=self._input_channels,
            )
            self.num_features = self.network.fc.in_features
            self.network.fc = torch.nn.Identity()

        self.num_features += 4  # Add 4 historical irradiances
        self.hidden_size = 256
        self.lstm = torch.nn.LSTM(input_size=self.num_features, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.network_head = torch.nn.Sequential(
            # torch.nn.Linear(self.num_features, 256),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(256, 1),
        )

        self.hidden_cell = (torch.zeros(1, 16, self.hidden_size).to(device='cuda'), torch.zeros(1, 16, self.hidden_size).to(device='cuda'))

        if loss_function == 'MSE':
            self.loss = torch.nn.MSELoss()
        elif loss_function == 'MAE':
            self.loss = torch.nn.L1Loss()
        elif loss_function == 'SmoothL1':
            self.loss = torch.nn.SmoothL1Loss()
        elif loss_function == 'MAPE':
            self.loss = MAPELoss()
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

        # irradiance_history = irradiance_history.unsqueeze(1).expand(-1, x.shape[1], -1)
        # x = torch.cat((x.unsqueeze(2), irradiance_history), dim=2)
        x = torch.cat((x.unsqueeze(1), irradiance_history.unsqueeze(1)), dim=2)
        # x = torch.cat([x, irradiance_history], dim=0)
        x, _ = self.lstm(x)
        x = self.network_head(x.view(-1, self.hidden_size))
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
