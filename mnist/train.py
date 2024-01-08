import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import pytorch_lightning as pl
# from sklearn.metrics import classification_report
# from dvc.api import DVCFileSystem
# from torchvision.transforms import transforms
# from models.dataset import MNISTDataset
from models.model import LightMNISTClassifier
from utils.utils import get_datasets


# def get_datasets(path,n1,n2):
#     fs = DVCFileSystem("./")
#     fs.get_file(path, path)
#     transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((n1,), (n2,))])
#     return MNISTDataset(path,transform=transform)
@hydra.main(config_path="../configs", config_name="config",version_base="1.3")
def main(cfg: DictConfig) -> None:
    train_data=get_datasets(cfg.data.train_data_path,cfg.preprocessing.n1,cfg.preprocessing.n2)
    print('Файлы загружены')
    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=cfg.training.batch_size,
                                              shuffle=cfg.training.shuffle,
                                              num_workers=cfg.training.num_workers)
    print('Файлы преобразованы в DataLoader')
    model = LightMNISTClassifier(cfg)
    trainer = pl.Trainer(max_epochs=cfg.training.epochs)
    print('Обучение модели')
    trainer.fit(model=model, train_dataloaders=trainloader)
    torch.save(model.state_dict(), cfg.model.model_path)
    print(f"model saved to {cfg.model.model_path}")

if __name__ == "__main__":
    main()