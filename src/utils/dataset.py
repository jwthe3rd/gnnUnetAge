import os
from torch_geometric.loader import Dataset
from torch_geometric.data import Data
import torch

class data_generator:

    def __init__(self, path):
        self.path = path

    def segment_data(self):
        training_cases = os.listdir(self.path)
        train_data = []
        edge_paths = []
        feats_paths = []
        label_paths = []
        ## Adding paths for all training data
        for data_file in training_cases:
            if data_file[0] == 'e':
                edge_paths.append(self.path + data_file)
        for case in edge_paths:
            for feat in training_cases:
                if feat[0] == 'f' and feat[9:] == case[(len(self.path) + 6):]:
                    feats_paths.append(self.path + feat)
                elif feat[0] == 'l' and feat[7:] == case[(len(self.path) + 6):]:
                    label_paths.append(self.path + feat)
                else:
                    continue

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

        data = Data(x=X, edge_index = edge_index, y=Y)

        return data 