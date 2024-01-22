from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig
from mnist.utils.utils import get_data_dvc
from mnist.models.model import LightMNISTClassifier



@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    model_path = cfg.model.model_path + ".pth"
    if not Path(model_path).exists():
        print("\nFile .pth will be downloaded from the DVC")
        print("In other case: you need to run train.py and restart run_server.py")
        get_data_dvc(model_path)

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = LightMNISTClassifier(cfg)
    model.to(device)

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))

    print("Converting model to .onnx")
    model_onnx_path=cfg.triton_server.model_onnx_path
    input = torch.randn((cfg.model.in_channels, cfg.model.in_channels, 28, 28))
    torch.onnx.export(
        model,
        input,
        model_onnx_path,
        export_params=True,
        input_names=["inputs"],
        output_names=["predictions"],
    )
    print(f"Model saved to {model_onnx_path}")

if __name__ == "__main__":
    main()