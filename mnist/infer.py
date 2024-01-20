from itertools import chain
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from mnist.datasets.dataset import MNISTDataModule
from mnist.models.model import LightMNISTClassifier
from mnist.utils.utils import calculate_metrics, get_data_dvc, init_trainer


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.infer.seed)
    model = LightMNISTClassifier(cfg)
    model_path=cfg.model.model_path+".pth"
    if Path(model_path).exist()!=True:
        print('File .pth will be downloaded from the DVC')
        print('In other case: you need to run train.py and restart infer.py')
        get_data_dvc(model_path)
    model.load_state_dict(torch.load(model_path))
    trainer=init_trainer(cfg)
    data_module = MNISTDataModule(val_size=cfg.training.val_size,
                         batch_size=cfg.infer.batch_size,
                         num_workers=cfg.infer.num_workers,
                         n1=cfg.preprocessing.n1,
                         n2=cfg.preprocessing.n2,
                         path_train_val=cfg.data.train_data_path,
                         path_test=cfg.data.test_data_path)
    trainer.test(model, datamodule=data_module)
    predictions_list = trainer.predict(model, datamodule=data_module)
    predictions = list(chain(*[x.detach().cpu().numpy() for x in predictions_list]))
    df_pred=pd.DataFrame(predictions, columns=['predicted_label'])
    df_test=pd.read_csv(cfg.data.test_data_path,usecols=['label'])
    metrics=calculate_metrics(df_test['label'], df_pred['predicted_label'])
    print('Metrics report')
    print(metrics)
    df_pred.to_csv(cfg.data.pred_data_path,columns=['predicted_label'],index=False)
    print(f"Predictions saved to {cfg.data.pred_data_path}")
if __name__ == "__main__":
    infer_main()