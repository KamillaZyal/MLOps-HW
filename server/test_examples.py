import json
import numpy as np
import requests
import hydra
from omegaconf import DictConfig
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
import random


def request_pred(inputs):
    URL = "http://localhost:8080/invocations"
    HEADERS = {"Content-Type": "application/json"}
    data = json.dumps({"inputs": inputs})
    req = requests.Request("POST", URL, data=data, headers=HEADERS).prepare()
    res = requests.Session().send(req)
    return res
def create_random_test(shape):
    N = shape[0]
    if len(shape) == 1:
        return [0.1 for _ in range(N)]
    return [create_random_test(shape[1:]) for _ in range(N)]

@hydra.main(config_path="../configs", config_name="config",version_base="1.3")
def main(cfg: DictConfig) -> None:
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((cfg.preprocessing.n1,), (cfg.preprocessing.n2,))])
    csv_data = pd.read_csv(cfg.data.test_data_path)
    image = Image.new("L",(28,28))
    idx=random.randint(0,len(csv_data)-1)
    image.putdata(csv_data.iloc[idx, 1:])
    print('Image opened in a new window')
    image.show()
    label =csv_data.iloc[idx, 0]
    print('True label: ',label)
    image = [transform(image).tolist()]
    image_text=request_pred(image).text
    try:
        data = json.loads(image_text)
    except Exception as e:
        print(f"Exception while converting response to json:\n{str(e)}")
        print(image_text)
        return
    if "error_code" in data:
        print("Error from server:")
        print(data["message"])
        print(data["stack_trace"])
        return
    data = data["predictions"]
    for _, prediction in data.items():
        prediction = np.array(prediction)
        print(f"Predicted result: {prediction}")
        print(f"Predicted label: {np.argmax(prediction)}")
if __name__ == '__main__':
    main()