import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from mnist.models.model import LightMNISTClassifier
from  mnist.utils.utils import init_trainer,calculate_metrics
from  mnist.datasets.dataset import MNISTDataModule
from mnist.manage_dvc import dvc_push


@hydra.main(config_path="../configs", config_name="config",version_base="1.3")
def main(cfg: DictConfig) -> None:
    # test_data=get_datasets(cfg.data.test_data_path,cfg.preprocessing.n1,cfg.preprocessing.n2)
    # testloader = torch.utils.data.DataLoader(test_data,
    #                                           batch_size=cfg.infer.batch_size,
    #                                           shuffle=cfg.infer.shuffle,
    #                                           num_workers=cfg.infer.num_workers)
    pl.seed_everything(cfg.infer.seed)
    model = LightMNISTClassifier(cfg)
    model.load_state_dict(torch.load(cfg.model.model_path+".pth"))
    # trainer = pl.Trainer()
    cfg.callbacks.swa.use = False
    cfg.artifacts.checkpoint.use = False
    trainer=init_trainer(cfg)
    data_module = MNISTDataModule(val_size=cfg.training.val_size,
                         batch_size=cfg.infer.batch_size,
                         num_workers=cfg.infer.num_workers,
                         n1=cfg.preprocessing.n1,
                         n2=cfg.preprocessing.n2,
                         path_train_val=cfg.data.train_data_path,
                         path_test=cfg.data.test_data_path)
    trainer.test(model, datamodule=data_module)
    predictions = trainer.predict(model, datamodule=data_module)
    predictions = np.concatenate(predictions, axis=1).T
    df_pred=pd.DataFrame(predictions, columns=['target_label', 'predicted_label'])
    metrics=calculate_metrics(df_pred['target_label'], df_pred['predicted_label'])
    print('Metrics report')
    print(metrics)
    df_pred.to_csv(cfg.data.pred_data_path,columns=['predicted_label'],index=False)
    print(f"Predictions saved to {cfg.data.pred_data_path}")
    dvc_push(cfg.dvc_data.list_data_path)
if __name__ == "__main__":
    main()