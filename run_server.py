from pathlib import Path

import git
import hydra
import mlflow
import onnx
import pandas as pd
import torch
from omegaconf import DictConfig

from mnist.datasets.dataset import get_datasets
from mnist.utils.utils import calculate_metrics, get_data_dvc


@hydra.main(config_path="configs", config_name="config",version_base="1.3")
def main(cfg: DictConfig) -> None:
    model_path = cfg.model.model_path+'.onnx'
    if Path(model_path)!=True:
        print('File .onnx will be downloaded from the DVC')
        print('In other case: you need to run train.py and restart run_server.py')
        get_data_dvc(model_path)
        get_data_dvc(cfg.data.logs_path)
    onnx_model = onnx.load(model_path)
    input = torch.randn((cfg.model.in_channels,cfg.model.in_channels,28, 28))
    output = torch.randn((cfg.model.in_channels, cfg.model.out_channels))
    mlflow.set_tracking_uri(cfg.artifacts.mlflow.tracking_uri) 
    with mlflow.start_run():
        signature = mlflow.models.infer_signature(input.detach().cpu().numpy(), output.detach().cpu().numpy())
        model_info = mlflow.onnx.log_model(onnx_model, "model", signature=signature)
        model_mlflow = mlflow.pyfunc.load_model(model_info.model_uri)
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        mlflow.set_tag("commit_id", sha)
        labels,data_test=get_datasets(cfg.data.test_data_path,
                                       n1=cfg.preprocessing.n1,
                                       n2=cfg.preprocessing.n2,
                                       mnist_dataset=False,
                                       has_labels=True,
                                       show_image=False,
                                       )
        predictions_list=[]
        # inference model on test_datasets
        for inp in data_test:
            inp=torch.reshape(inp, (cfg.model.in_channels,cfg.model.in_channels,28,28))
            predictions_list.append(model_mlflow.predict(inp.detach().cpu().numpy())['predictions'].argmax())
        df_pred=pd.DataFrame(predictions_list, columns=['predicted_label'])
        metrics=calculate_metrics(labels, df_pred['predicted_label'])
        print('Metrics report')
        print(metrics)
        df_pred.to_csv(cfg.data.pred_onnx_data_path,columns=['predicted_label'],index=False)
        print(f"Predictions saved to {cfg.data.pred_onnx_data_path}")

        # new examples
        labels,data_test=get_datasets(cfg.data.exp_data_path,
                                       n1=cfg.preprocessing.n1,
                                       n2=cfg.preprocessing.n2,
                                       mnist_dataset=False,
                                       has_labels=True,
                                       show_image=True,
                                       )
        for i,inp in enumerate(data_test):
            print(f'Image №{i} opened in a new window')
            print(f'True label for image № {i}:      {labels[i]}')
            inp=torch.reshape(inp, (cfg.model.in_channels,cfg.model.in_channels,28,28))
            prediction=model_mlflow.predict(inp.detach().cpu().numpy())['predictions'].argmax()
            print(f'Predicted label for image № {i}: {prediction}')
if __name__ == '__main__':
    main()