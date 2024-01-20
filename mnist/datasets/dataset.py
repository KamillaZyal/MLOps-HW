from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms

from mnist.utils.utils import get_data_dvc


def get_datasets(path,n1,n2,is_pred=False,mnist_dataset=True,has_labels=True,show_image=False):
    get_data_dvc(path)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((n1,), (n2,))])
    if mnist_dataset:
        return MNISTDataset(path,transform=transform,is_pred=is_pred)
    else:
        return get_testdata(path, transform=transform,has_labels=has_labels,show_image=show_image)
    
def get_testdata(path, transform=None,has_labels=True,show_image=False):
    '''
    if has_labels==False: return empty list `labels`
    '''
    csv_data = pd.read_csv(path)
    labels=[]
    images=[]
    for i in range(csv_data.shape[0]):
        image = Image.new("L",(28,28))
        if has_labels:
            labels.append(csv_data.iloc[i, 0])
            image.putdata(csv_data.iloc[i, 1:])
        else:
            print(f'Image №{i} opened in a new window')
            image.putdata(csv_data.iloc[i, 0:])
        if show_image:
            image.show()
        if transform:
            images.append(transform(image))
        else:
            images.append(image)
    return labels,images
class MNISTDataset(Dataset):
    def __init__(self, csv_path, transform=None,is_pred=False):
        self.csv_data = pd.read_csv(csv_path)
        self.transform = transform
        self.is_pred=is_pred
    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        '''
        Returns a tuple of (PIL.Image, int) 
        '''
        image = Image.new("L",(28,28))
        if self.is_pred:
            if self.csv_data.shape[1]>28*28:
                image.putdata(self.csv_data.iloc[idx, 1:])
            else:
                image.putdata(self.csv_data.iloc[idx, 0:])
            if self.transform:
                image = self.transform(image)
            return image
        else:
            image.putdata(self.csv_data.iloc[idx, 1:])
            label = self.csv_data.iloc[idx, 0]
            if self.transform:
                image = self.transform(image)
        return image, label
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, val_size,path_train_val,path_test,n1,n2):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.path_train_val=path_train_val
        self.path_test=path_test
        self.n1=n1
        self.n2=n2


    def setup(self, stage: Optional[str] = None):
        self.train_val_dataset=get_datasets(self.path_train_val,self.n1,self.n2)
        self.test_dataset=get_datasets(self.path_test,self.n1,self.n2)
        self.pred_dataset=get_datasets(self.path_test,self.n1,self.n2,is_pred=True)
        train_indexes = list(range(0, int(len(self.train_val_dataset) * (1 - self.val_size))))
        val_indexes = list(range(int(len(self.train_val_dataset) * (1 - self.val_size)),len(self.train_val_dataset)))
        self.train_dataset = Subset(self.train_val_dataset, train_indexes)
        self.val_dataset = Subset(self.train_val_dataset, val_indexes)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
    def predict_dataloader(self)-> DataLoader:
        return DataLoader(self.pred_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)