from sklearn.metrics import classification_report
from dvc.api import DVCFileSystem
from models.dataset import MNISTDataset
from torchvision.transforms import transforms

def get_datasets(path,n1,n2):
    fs = DVCFileSystem("./")
    fs.get_file(path, path)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((n1,), (n2,))])
    return MNISTDataset(path,transform=transform)
def calculate_metrics(y_true, y_pred):
    """
    Calculates metrics
    """
    clf_report = classification_report(y_true, y_pred, zero_division=0)
    return clf_report