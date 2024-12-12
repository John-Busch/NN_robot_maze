import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    #__len__() returns the length of the dataset
    def __len__(self):
        pass

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()

#randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    
    #the dataloaders will be iterated over
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
