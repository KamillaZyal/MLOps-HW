import shutil
import mlflow.pyfunc
import onnx
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config",version_base="1.3")
def build_server(cfg: DictConfig) -> None:
    # model_path = f'{os.getcwd()}\data\model.onnx'
    # directory_out = f'{os.getcwd()}\data\server\'
    model_path = cfg.model.model_path+'.onnx'
    dir_out = cfg.server.outputs_dir
    try:
        shutil.rmtree(dir_out)
    except FileNotFoundError:
        pass
    onnx_model = onnx.load(model_path)
    mlflow.onnx.save_model(onnx_model, dir_out)

if __name__ == '__main__':
    build_server()