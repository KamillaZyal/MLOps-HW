from sklearn.metrics import classification_report
from pathlib import Path
import pytorch_lightning as pl
import git


def calculate_metrics(y_true, y_pred):
    """
    Calculates metrics
    """
    clf_report = classification_report(y_true, y_pred, zero_division=0)
    return clf_report
def init_trainer(cfg):
    # for git commit id
    repo = git.Repo(search_parent_directories=True)

    loggers = [
        pl.loggers.CSVLogger(cfg.artifacts.csv_logger.path, name=cfg.artifacts.experiment_name),
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.artifacts.experiment_name,
            tracking_uri=cfg.artifacts.mlflow.tracking_uri,
            tags={"git_commit_id": repo.head.object.hexsha},
        ),
        pl.loggers.WandbLogger(
            project="mlops-logging-mnist",
            name=cfg.artifacts.experiment_name,
            save_dir=cfg.artifacts.wandb_logger.path,
        ),
    ]
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=cfg.callbacks.model_summary.max_depth),
    ]
    
    
    if cfg.callbacks.swa.use:
        callbacks.append(
            pl.callbacks.StochasticWeightAveraging(swa_lrs=cfg.callbacks.swa.lrs)
        )

    if cfg.artifacts.checkpoint.use:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=Path(cfg.artifacts.checkpoint.dirpath, cfg.artifacts.experiment_name),
                filename=cfg.artifacts.checkpoint.filename,
                monitor=cfg.artifacts.checkpoint.monitor,
                save_top_k=cfg.artifacts.checkpoint.save_top_k,
                every_n_train_steps=cfg.artifacts.checkpoint.every_n_train_steps,
                every_n_epochs=cfg.artifacts.checkpoint.every_n_epochs,
            )
        )

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        max_epochs=cfg.training.epochs,
        val_check_interval=cfg.trainer.val_check_interval,
        overfit_batches=cfg.trainer.overfit_batches,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        benchmark=cfg.trainer.benchmark,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        # detect_anomaly=cfg.trainer.detect_anomaly,
        enable_checkpointing=cfg.artifacts.checkpoint.use,
        logger=loggers,
        callbacks=callbacks
        )

    return trainer