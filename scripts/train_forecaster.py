import os
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
import dvc.api
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
import onnx
from onnxsim import simplify
import torch
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

from solar_irradiance.datamodules.forecasting import ForecastingDataModule
from solar_irradiance.models.forecaster import Forecaster
from solar_irradiance.utils import utils

log = utils.get_logger(__name__)


@click.command()
@click.option("--data-root", type=click.Path(exists=True, path_type=Path), required=True)
def train_forecaster(data_root: Path):
    load_dotenv(find_dotenv(".env"))

    data_cfg = OmegaConf.create(dvc.api.params_show()["export_periods"])
    cfg = OmegaConf.create(dvc.api.params_show()["train_forecaster"])

    pl.seed_everything(seed=cfg.seed)

    datamodule = ForecastingDataModule(
        root_data_path=data_root,
        periods_path=data_root / "periods.pickle",
        augment=cfg.datamodule.augment,
        train_val_set_size=cfg.datamodule.train_val_set_size,
        image_size=cfg.datamodule.image_size,
        image_mean=cfg.datamodule.image_mean,
        image_std=cfg.datamodule.image_std,
        batch_size=cfg.datamodule.batch_size,
        workers=cfg.datamodule.workers,
        add_sun_mask=cfg.datamodule.add_sun_mask,
        add_irradiance_channel=cfg.datamodule.add_irradiance_channel,
        optical_flow=cfg.datamodule.optical_flow,
        cloud_mask_method=cfg.datamodule.cloud_mask_method,
        model_2d=cfg.model.model_name.startswith("timm-"),
        seed=cfg.seed,
    )

    # 3 from RGB channels
    # additional channel with sun position mask or irradiance value or their combination
    # 2 channels from opttical flow (X, Y directions)
    optical_flow_channels = 0 if cfg.datamodule.optical_flow is None else 2
    input_channels = (
        3
        + int(cfg.datamodule.add_sun_mask or cfg.datamodule.add_irradiance_channel)
        + int(bool(cfg.datamodule.cloud_mask_method))
        + optical_flow_channels
    )

    model = Forecaster(
        model_name=cfg.model.model_name,
        pretrained=cfg.model.pretrained,
        input_channels=input_channels,
        loss_function=cfg.model.loss_function,
        lr=cfg.model.lr,
        lr_patience=cfg.model.lr_patience,
        history_size=data_cfg.history_size,
        image_size=cfg.datamodule.image_size,
    )

    checkpoint_callback = ModelCheckpoint(**cfg.callbacks.model_checkpoint)
    model_summary_callback = ModelSummary(max_depth=1)
    early_stopping_callback = EarlyStopping(**cfg.callbacks.early_stopping)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [
        checkpoint_callback,
        model_summary_callback,
        early_stopping_callback,
    ]

    if not cfg.debug:
        logger = NeptuneLogger(
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            project="Vision/IrradianceRegression",
            log_model_checkpoints=True,
        )
        callbacks.append(lr_monitor)

        logger.log_hyperparams(cfg.datamodule)
    else:
        logger = None

    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        devices=cfg.trainer.devices if not None else -1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=DDPStrategy(
            ddp_comm_wrapper=default.fp16_compress_wrapper,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
            static_graph=True,
        ),
        precision=cfg.trainer.precision,
        min_epochs=cfg.trainer.min_epochs,
        max_epochs=cfg.trainer.max_epochs,
        benchmark=True,
        sync_batchnorm=cfg.trainer.devices > 0,
        check_val_every_n_epoch=1,
    )

    if not cfg.test_only:
        if cfg.model.compile:
            log.info("Compiling model")
            model = torch.compile(model)

        log.info("Starting training process")
        trainer.fit(model, datamodule)

        log.info("Starting testing process for the best checkpoint")
        trainer.test(model, datamodule, ckpt_path="best")
        log.info(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        assert cfg.restore_from_ckpt is not None
        log.info(f"Starting testing process for {cfg.restore_from_ckpt} checkpoint")
        trainer.test(model, datamodule, ckpt_path=cfg.restore_from_ckpt)

    if cfg.export.export_to_onnx:
        opset = cfg.export.opset
        use_simplifier = cfg.export.use_simplifier
        log.info(f"Exporting model to onnx with parameters: opset={opset}, use_simplifier={use_simplifier}")

        model.eval()
        input_data = next(iter(datamodule.test_dataloader()))
        image_input = input_data[0][:1]
        irradiance_history = input_data[1][:1]

        torch.onnx.export(
            model,
            (
                image_input,
                irradiance_history,
            ),  # model input (or a tuple for multiple inputs)
            f"{cfg.model.model_name}.onnx",  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=opset,  # the ONNX version to export the model to
            input_names=["image_input", "irradiance_history"],
            output_names=["output"],
            do_constant_folding=False,
        )

        if use_simplifier:
            model = onnx.load("model.onnx")
            model_simp, check = simplify(model)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_simp, "model.onnx")


if __name__ == "__main__":
    train_forecaster()
