import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from models.model import LightMNISTClassifier
from utils.utils import get_datasets,calculate_metrics

@hydra.main(config_path="../configs", config_name="config",version_base="1.3")
def main(cfg: DictConfig) -> None:
    test_data=get_datasets(cfg.data.test_data_path,cfg.preprocessing.n1,cfg.preprocessing.n2)
    testloader = torch.utils.data.DataLoader(test_data,
                                              batch_size=cfg.infer.batch_size,
                                              shuffle=cfg.infer.shuffle,
                                              num_workers=cfg.infer.num_workers)
    model = LightMNISTClassifier(cfg)
    model.load_state_dict(torch.load(cfg.model.model_path))
    trainer = pl.Trainer()
    predictions = trainer.predict(model, dataloaders=testloader)
    predictions = np.concatenate(predictions, axis=1).T
    df_pred=pd.DataFrame(predictions, columns=['target_label', 'predicted_label'])
    metrics=calculate_metrics(df_pred['target_label'], df_pred['predicted_label'])
    print('Metrics report')
    print(metrics)
    df_pred.to_csv(cfg.data.pred_data_path,columns=['predicted_label'],index=False)
    print(f"Predictions saved to {cfg.data.pred_data_path}")
if __name__ == "__main__":
    main()