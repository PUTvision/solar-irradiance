import logging
import warnings
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    # disable python warnings
    warnings.filterwarnings("ignore")

    return logger


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: list[pl.Callback],
        logger: list[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    if trainer.logger is not None:
        # send hparams to all loggers
        trainer.logger.log_hyperparams(hparams)

        # disable logging any more hyperparameters for all loggers
        # this is just a trick to prevent trainer from logging hparams of model,
        # since we already did that above
        trainer.logger.log_hyperparams = empty
