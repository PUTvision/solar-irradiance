import os

import hydra
import onnx
import pytorch_lightning as pl
import torch
from onnxsim import simplify
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as default,
)

from solar_irradiance.datamodules.regression_data_module import RegressionDataModule
from solar_irradiance.models.regressor import Regressor
from solar_irradiance.utils import utils

log = utils.get_logger(__name__)


@hydra.main(version_base=None, config_path='../config/', config_name='config')
def main(cfg: DictConfig) -> None:
    pl.seed_everything(seed=cfg.seed)

    datamodule = RegressionDataModule(
        root_data_path=cfg.datamodule.data_path,
        augment=cfg.datamodule.augment,
        image_size=cfg.datamodule.image_size,
        padded_image_size=cfg.datamodule.padded_image_size,
        image_mean=cfg.datamodule.image_mean,
        image_std=cfg.datamodule.image_std,
        batch_size=cfg.datamodule.batch_size,
        workers=cfg.datamodule.workers,
        number_of_splits=cfg.datamodule.number_of_splits,
        current_split=cfg.datamodule.current_split,
    )

    model = Regressor(
        model_name=cfg.model.model_name,
        input_channels=cfg.model.input_channels,
        loss_function=cfg.model.loss_function,
        lr=cfg.model.lr,
        lr_patience=cfg.model.lr_patience,
    )

    if cfg.restore_from_ckpt is not None:
        model = model.load_from_checkpoint(
            checkpoint_path=cfg.restore_from_ckpt
        )

    checkpoint_callback = ModelCheckpoint(**cfg.callbacks.model_checkpoint)
    model_summary_callback = ModelSummary(max_depth=1)
    early_stopping_callback = EarlyStopping(**cfg.callbacks.early_stopping)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if not cfg.debug:
        logger = NeptuneLogger(
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            project='Vision/IrradianceRegression',
            log_model_checkpoints=True,
        )
    else:
        logger = None

    callbacks = [
        checkpoint_callback,
        model_summary_callback,
        early_stopping_callback,
        lr_monitor,
    ]

    if cfg.trainer.devices > 0:
        ddp_strategy = DDPStrategy(
            ddp_comm_wrapper=default.fp16_compress_wrapper,
            find_unused_parameters=False,
            static_graph=True,
        )

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        devices=cfg.trainer.devices if not None else -1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        strategy=ddp_strategy if cfg.trainer.devices > 1 else None,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        benchmark=True,
        sync_batchnorm=cfg.trainer.devices > 0,
        check_val_every_n_epoch=1,
    )

    if not cfg.test_only:
        trainer.fit(model, datamodule)

        trainer.predict(model, datamodule, ckpt_path='best')
    else:
        assert cfg.restore_from_ckpt is not None
        trainer.predict(model, datamodule, return_predictions=False)

    if cfg.export.export_to_onnx:
        opset = cfg.export.opset
        use_simplifier = cfg.export.use_simplifier
        log.info(f'Exporting model to onnx! Params: opset={opset}, use_simplifier={use_simplifier}')

        model.eval()
        x = next(iter(datamodule.test_dataloader()))[0][:1]

        torch.onnx.export(
            model.network,
            x,                      # model input (or a tuple for multiple inputs)
            'model.onnx',           # where to save the model (can be a file or file-like object)
            export_params=True,     # store the trained parameter weights inside the model file
            opset_version=opset,    # the ONNX version to export the model to
            input_names=['input'],
            output_names=['output'],
            do_constant_folding=False
        )
        
        if use_simplifier:
            model = onnx.load('model.onnx')
            model_simp, check = simplify(model)
            assert check, 'Simplified ONNX model could not be validated'
            onnx.save(model_simp, 'model.onnx')


if __name__ == '__main__':
    main()