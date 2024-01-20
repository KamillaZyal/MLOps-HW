import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
import pytorch_lightning as pl
from itertools import chain
from mnist.models.model import LightMNISTClassifier
from mnist.utils.utils import get_datasets,calculate_metrics

@hydra.main(config_path="../configs", config_name="config",version_base="1.3")
def main(cfg: DictConfig) -> None:
    test_data=get_datasets(cfg.data.test_data_path,cfg.preprocessing.n1,cfg.preprocessing.n2,is_pred=True)
    testloader = torch.utils.data.DataLoader(test_data,
                                              batch_size=cfg.infer.batch_size,
                                              shuffle=cfg.infer.shuffle,
                                              num_workers=cfg.infer.num_workers)
    model = LightMNISTClassifier(cfg)
    model.load_state_dict(torch.load(cfg.model.model_path))
    trainer = pl.Trainer()
    predictions_list = trainer.predict(model, dataloaders=testloader)
    predictions = list(chain(*[x.detach().cpu().numpy() for x in predictions_list]))
    df_pred=pd.DataFrame(predictions, columns=['predicted_label'])
    df_test=pd.read_csv(cfg.data.test_data_path,usecols=['label'])
    metrics=calculate_metrics(df_test['label'], df_pred['predicted_label'])
    print('Metrics report')
    print(metrics)
    df_pred.to_csv(cfg.data.pred_data_path,columns=['predicted_label'],index=False)
if __name__ == "__main__":
    main()