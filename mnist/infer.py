import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import classification_report
# from dvc.api import DVCFileSystem
# from torchvision.transforms import transforms
# from models.dataset import MNISTDataset
from models.model import LightMNISTClassifier
from utils.utils import get_datasets,calculate_metrics


# def get_datasets(path,n1,n2):
#     fs = DVCFileSystem("./")
#     fs.get_file(path, path)
#     transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((n1,), (n2,))])
#     return MNISTDataset(path, transform=transform)


def calculate_metrics(y_true, y_pred):
    """
    Calculates metrics
    """
    clf_report = classification_report(y_true, y_pred, zero_division=0)
    return clf_report

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