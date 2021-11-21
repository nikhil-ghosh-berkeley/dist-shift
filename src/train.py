import hydra
import wandb
import logging
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
)
from pytorch_lightning.loggers import WandbLogger
from src.utils import log_hyperparams

log = logging.getLogger(__name__)
logging.getLogger("lightning").setLevel(logging.ERROR)


def train(config: DictConfig):
    log.info(f"Instantiating logger <{config.logger._target_}>")
    logger: WandbLogger = hydra.utils.instantiate(config.logger)#, settings=wandb.Settings(start_method='fork'))
    logger.log_hyperparams({"seed": config.seed})

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        checkpoint_callback=False,
        num_sanity_val_steps=0,
    )

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    preprocess = None
    if config.model.arch.startswith('Clip'):
        preprocess = model.model.preprocess

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, preprocess_func=preprocess)


    log.info("Logging hyperparameters!")
    log_hyperparams(config=config, trainer=trainer)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    wandb.finish()
