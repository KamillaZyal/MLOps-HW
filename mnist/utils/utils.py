from pathlib import Path

import git
import pytorch_lightning as pl
from dvc.api import DVCFileSystem
from sklearn.metrics import classification_report


def get_data_dvc(path):
    if Path(path).exists()!=True:
        fs = DVCFileSystem("./")
        if Path(path).is_dir:
            fs.get(path, path,recursive=True)
        else:
            fs.get_file(path, path)

def calculate_metrics(y_true, y_pred):
    """
    Calculates metrics
    """
    clf_report = classification_report(y_true, y_pred, zero_division=0)
    return clf_report

def init_trainer(cfg):
    repo = git.Repo(search_parent_directories=True)

    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.artifacts.experiment_name,
            tracking_uri=cfg.artifacts.mlflow.tracking_uri,
            tags={"git_commit_id": repo.head.object.hexsha},
            ),
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.artifacts.experiment_name,
            tracking_uri=cfg.artifacts.mlflow.save_path,
            tags={"git_commit_id": repo.head.object.hexsha},
            )
            ]
    callbacks = [
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=cfg.callbacks.model_summary.max_depth),
        ]
    

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        max_epochs=cfg.training.epochs,
        val_check_interval=cfg.trainer.val_check_interval,
        overfit_batches=cfg.trainer.overfit_batches,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        benchmark=cfg.trainer.benchmark,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_checkpointing=cfg.artifacts.checkpoint.use,
        logger=loggers,
        callbacks=callbacks
        )

    return trainer