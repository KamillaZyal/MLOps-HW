from PIL import Image
from torch.utils.data import Dataset,Subset,DataLoader
from torchvision.transforms import transforms
import pandas as pd
import pytorch_lightning as pl
from dvc.api import DVCFileSystem
from typing import Optional

def get_datasets(path,n1,n2):
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((n1,), (n2,))])
    return MNISTDataset(path,transform=transform)
class MNISTDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.csv_data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        '''
        Returns a tuple of (PIL.Image, int) 
        '''
        image = Image.new("L",(28,28))
        
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
        return self.test_dataloader()

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