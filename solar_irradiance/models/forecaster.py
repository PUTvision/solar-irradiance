import lightning.pytorch as pl
import timm
import torch
from torch.optim import Optimizer
import torchmetrics

from solar_irradiance.losses import MAPELoss, MeanAdaptiveBerHuLoss


class Forecaster(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        input_channels: int,
        loss_function: str,
        lr: float,
        lr_patience: int,
    ) -> None:
        super().__init__()

        self._model_name = model_name
        self._lr = lr
        self._lr_patience = lr_patience

        if "resne" in model_name:
            self.network = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=1,
                in_chans=input_channels,
            )
            self.num_features = self.network.fc.in_features
            self.network.fc = torch.nn.Identity()
        elif "mixnet" in model_name or "efficientnet" in model_name or "mobilenet" in model_name:
            self.network = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=1,
                in_chans=input_channels,
            )
            self.num_features = self.network.classifier.in_features
            self.network.classifier = torch.nn.Identity()
        elif "mambaout" in model_name or "convnext" in model_name or "convformer" in model_name or "regnet" in model_name:
            self.network = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=1,
                in_chans=input_channels,
            )
            self.num_features = self.network.head.fc.in_features
            self.network.head.fc = torch.nn.Identity()
        elif "efficientvit" in model_name:
            self.network = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=1,
                in_chans=input_channels,
            )
            self.num_features = self.network.head.classifier[-1].in_features
            self.network.head.classifier[-1] = torch.nn.Identity()
        elif "mvitv2" in model_name:
            self.network = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=1,
                in_chans=input_channels,
            )
            self.num_features = self.network.head.fc.in_features
            self.network.head = torch.nn.Identity()
        elif model_name == "venitourakis_xception":
            from solar_irradiance.models.architectures.venitourakis_xception_image_encoder import XceptionImageEncoder

            self.network = XceptionImageEncoder(in_channels=input_channels)
            self.num_features = 128
        elif model_name == "mercier_vit":
            from solar_irradiance.models.architectures.mercier_vit import MercierViT

            network_encoder = timm.create_model(
                "deit_tiny_patch16_224",
                num_classes=0,  # remove classifier nn.Linear
                pretrained=pretrained,
                in_chans=input_channels,
                img_size=128,
            )
            self.network = MercierViT(
                inmodel=network_encoder,
                number_of_linear_layers=1,
                drop_out_lin=0.1,
                intermediate_linear_layer_shape=512,
                linear_activation_func="SiLU",
                sigmoid_on=False,
                y_shape=(1,),
            )
        elif model_name == "jonathan_attention_cnn":
            from solar_irradiance.models.architectures.jonathan_attention_cnn import AttentionCNN

            self.network = AttentionCNN(
                in_channels=input_channels,
                num_classes=1,  # feature dimension before head
            )
        elif model_name == "ansong_kalisi_cnn_lstm":
            from solar_irradiance.models.architectures.ansong_kalisi_cnn_lstm import KALiSI

            image_input_dim = (input_channels, 128, 128)
            numeric_input_dim = 5
            self.network = KALiSI(image_input_dim, numeric_input_dim)
        elif model_name == "zang_model":
            from solar_irradiance.models.architectures.zang_model import ZangModel

            image_input_dim = (input_channels, 128, 128)
            numeric_input_dim = 5
            self.network = ZangModel(
                img_c=input_channels,
                img_h=128,
                img_w=128,
                seq_len=4,
                forecast_horizon=1,
                fused_channels=64,
                tcn_channels=[32, 32, 64],
                attn_dim=64,
                val_dim=32,
            )
        elif model_name == "hendrikx_lstm":
            from solar_irradiance.models.architectures.hendrikx_lstm import LSTMPredictor

            input_features = 9  # Number of historical images
            self.network = LSTMPredictor(input_features=input_features)

        if model_name not in ["zang_model", "mercier_vit", "jonathan_attention_cnn", "ansong_kalisi_cnn_lstm"]:
            self.num_features += 4  # Add 4 historical irradiances
            self.network_head = torch.nn.Sequential(
                torch.nn.Linear(self.num_features, 256),
                torch.nn.ReLU(inplace=False),
                torch.nn.Linear(256, 1),
            )

        if loss_function == "MSE":
            self.loss = torch.nn.MSELoss()
        elif loss_function == "MAE":
            self.loss = torch.nn.L1Loss()
        elif loss_function == "SmoothL1":
            self.loss = torch.nn.SmoothL1Loss()
        elif loss_function == "Huber":
            self.loss = torch.nn.HuberLoss(delta=0.1)
        elif loss_function == "MAPE":
            self.loss = MAPELoss()
        elif loss_function == "BerHu":
            self.loss = MeanAdaptiveBerHuLoss()
        else:
            raise NotImplementedError(f"Unsupported loss function: {loss_function}")

        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.MeanSquaredError(),
                torchmetrics.MeanAbsoluteError(),
                torchmetrics.MeanAbsolutePercentageError(),
            ]
        )

        self.train_metrics = metrics.clone("train_")
        self.val_metrics = metrics.clone("val_")
        self.test_metrics = metrics.clone("test_")

        self.save_hyperparameters()

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x: torch.Tensor, optical_flows, irradiance_history: torch.Tensor) -> torch.Tensor:
        if self._model_name in ["mercier_vit", "jonathan_attention_cnn", "ansong_kalisi_cnn_lstm"]:
            x = self.network(x, irradiance_history)
        elif self._model_name == "hendrikx_lstm":
            x = self.network(x)
        elif self._model_name == "zang_model":
            x = self.network(x, optical_flows, irradiance_history)
        else:
            x = self.network(x)
            x = self.network_head(torch.cat([x, irradiance_history], dim=1))
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor | None:
        if self._model_name == "zang_model":
            source_images, optical_flows, source_irradiances, target_irradiances = batch
            predicted_irradiances = self.forward(source_images, optical_flows, source_irradiances)
        else:
            source_images, source_irradiances, target_irradiances = batch
            predicted_irradiances = self.forward(source_images, source_irradiances)

        loss = self.loss(predicted_irradiances, target_irradiances)
        if torch.isinf(loss):
            return None

        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.train_metrics.update(predicted_irradiances, target_irradiances)
        self.log_dict(self.train_metrics, sync_dist=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        if self._model_name == "zang_model":
            source_images, optical_flows, source_irradiances, target_irradiances = batch
            predicted_irradiances = self.forward(source_images, optical_flows, source_irradiances)
        else:
            source_images, source_irradiances, target_irradiances = batch
            predicted_irradiances = self.forward(source_images, source_irradiances)

        loss = self.loss(predicted_irradiances, target_irradiances)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.val_metrics.update(predicted_irradiances, target_irradiances)
        self.log_dict(self.val_metrics, sync_dist=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        if self._model_name == "zang_model":
            source_images, optical_flows, source_irradiances, target_irradiances = batch
            predicted_irradiances = self.forward(source_images, optical_flows, source_irradiances)
        else:
            source_images, source_irradiances, target_irradiances = batch
            predicted_irradiances = self.forward(source_images, source_irradiances)

        loss = self.loss(predicted_irradiances, target_irradiances)

        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics.update(predicted_irradiances, target_irradiances)
        self.log_dict(self.test_metrics, sync_dist=True)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr)
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self._lr_patience, min_lr=1e-6, verbose=True
        )

        return {"optimizer": optimizer, "lr_scheduler": reduce_lr_on_plateau, "monitor": "train_loss_epoch"}
