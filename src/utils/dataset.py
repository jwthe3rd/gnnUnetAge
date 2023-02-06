import os
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import random

class data_generator:

    def __init__(self, path, seed):
        self.path = path
        self.seed = seed

    def segment_data(self):
        training_cases = os.listdir(self.path)
        train_data = []
        edge_paths = []
        feats_paths = []
        label_paths = []
        unique_cases = []
        ## Adding paths for all training data
        for data_file in training_cases:
            if data_file[2:] not in unique_cases:
                unique_cases.append(data_file[2:])

        random.seed(self.seed)
        random.shuffle(unique_cases)

        for cases in unique_cases:
            edge_paths.append(f'{self.path}e_{cases}')
            feats_paths.append(f'{self.path}f_{cases}')
            label_paths.append(f'{self.path}l_{cases}')
            

        
            # if data_file[0] == 'e':
            #     edge_paths.append(self.path + data_file)
            # elif data_file[0] == 'f':
            #     feats_paths.append(self.path + data_file)
            # elif data_file[0] == 'l':
            #     label_paths.append(self.path + data_file)
        
        return label_paths, feats_paths, edge_paths

class gnnAgeDataSet(Dataset):

    def __init__(self, edge_paths, feats_paths, label_paths, transform=None):

        self.edge_paths = edge_paths
        self.feats_paths = feats_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.feats_paths)

    def __getitem__(self,idx):

        X = torch.load(self.feats_paths[idx])
        edge_index = torch.load(self.edge_paths[idx])
        Y = torch.load(self.label_paths[idx])

        data = Data(x=X, edge_index = edge_index, y=Y, Re=12)

        return data 