from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
class MNISTDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.csv_data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        '''
        Returns a tuple of (PIL.Image, int) if is_test is True
        Returns a PIL.Image if is_test is set to False
        '''
        image = Image.new("L",(28,28))
        
        image.putdata(self.csv_data.iloc[idx, 1:])
        label = self.csv_data.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        return image, label