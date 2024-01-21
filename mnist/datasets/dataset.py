from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class MNISTDataset(Dataset):
    def __init__(self, csv_path, transform=None, is_pred=False):
        self.csv_data = pd.read_csv(csv_path)
        self.transform = transform
        self.is_pred = is_pred

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        """
        Returns a tuple of (PIL.Image, int)
        """
        image = Image.new("L", (28, 28))
        if self.is_pred:
            if self.csv_data.shape[1] > 28 * 28:
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
