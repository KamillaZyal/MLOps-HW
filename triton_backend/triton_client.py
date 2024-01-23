from pathlib import Path
from dvc.api import DVCFileSystem
import hydra
from PIL import Image
from torchvision.transforms import transforms
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import pandas as pd
import torch
from omegaconf import DictConfig


def load_data(path):
    if not Path(path).exists():
        fs = DVCFileSystem("./")
        fs.get_file(path, path)
    data = pd.read_csv(path)
    labels = []
    images = []
    for i in range(data.shape[0]):
        labels.append(data.iloc[i, 0])
        images.append(data.iloc[i, 1:])
    return labels, images


def preprocessing_data(data_image, transform_norm_1, transform_norm_2):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((transform_norm_1,), (transform_norm_2,)),
        ]
    )
    image = Image.new("L", (28, 28))
    image.putdata(data_image)
    image.show()
    transform_image = transform(image)
    inp = torch.reshape(transform_image, (1, 1, 28, 28))
    return inp.detach().cpu().numpy()


def get_label_output(pred):
    return pred.argmax()


def client_model_call(input_data, server_url):
    triton_client = InferenceServerClient(url=server_url)
    inputs = []
    inputs.append(InferInput("inputs", input_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_data)

    outputs = InferRequestedOutput("predictions")

    results = triton_client.infer("mnist-onnx", inputs, outputs=[outputs])
    prediction = get_label_output(results.as_numpy("predictions"))
    return prediction


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    labels, data_test = load_data("../" + cfg.data.exp_data_path)
    for i, data_image in enumerate(data_test):
        print(f"Image № {i} opened in a new window")
        norm_image = preprocessing_data(
            data_image, cfg.preprocessing.n1, cfg.preprocessing.n2
        )
        print(f"True label for image № {i}:      {labels[i]}")
        prediction = client_model_call(norm_image, cfg.triton_server.server_url)
        print(f"Predicted label for image № {i}: {prediction}")
        assert (
            labels[i] == prediction
        ), f"Error: image № {i} ({labels[i]} vs {prediction})"


if __name__ == "__main__":
    main()
