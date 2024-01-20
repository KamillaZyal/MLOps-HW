from sklearn.metrics import classification_report
from dvc.api import DVCFileSystem
from mnist.datasets.dataset import MNISTDataset
from torchvision.transforms import transforms
from pathlib import Path

def get_datasets(path,n1,n2,is_pred=False):
    if Path(path).exists()!=True:
        fs = DVCFileSystem("./")
        fs.get_file(path, path)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((n1,), (n2,))])
    return MNISTDataset(path,transform=transform,is_pred=is_pred)
def calculate_metrics(y_true, y_pred):
    """
    Calculates metrics
    """
    clf_report = classification_report(y_true, y_pred, zero_division=0)
    return clf_report