import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from models.model import LightMNISTClassifier
from utils.utils import get_datasets

@hydra.main(config_path="../configs", config_name="config",version_base="1.3")
def main(cfg: DictConfig) -> None:
    train_data=get_datasets(cfg.data.train_data_path,cfg.preprocessing.n1,cfg.preprocessing.n2)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=cfg.training.batch_size,
                                              shuffle=cfg.training.shuffle,
                                              num_workers=cfg.training.num_workers)
    model = LightMNISTClassifier(cfg)
    trainer = pl.Trainer(max_epochs=cfg.training.epochs)
    print('Training model...')
    trainer.fit(model=model, train_dataloaders=trainloader)
    torch.save(model.state_dict(), cfg.model.model_path)
    print(f"Model saved to {cfg.model.model_path}")

if __name__ == "__main__":
    main()