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

    def __init__(self, edge_paths, feats_paths, label_paths, transform=None, test=False):

        self.edge_paths = edge_paths
        self.feats_paths = feats_paths
        self.label_paths = label_paths
        self.transform = transform
        self.test=test

    def __len__(self):
        return len(self.feats_paths)

    def __getitem__(self,idx):

        X = torch.load(self.feats_paths[idx])
        edge_index = torch.load(self.edge_paths[idx])
        Y = torch.load(self.label_paths[idx])

        if self.test:
            curr_path = self.feats_paths[idx]

            if curr_path.split('_')[5][0] == 'd':
                # print(curr_path.split('_'))
                # print(curr_path.split('-'))
                double = 1
                if len(curr_path.split('_')[7]) == 2:
                    bafflesze = int(curr_path.split('_')[7])
                else:
                    bafflesze = int(curr_path.split('_')[7])*10
            else:
                double = 0
                bafflesze = int((curr_path.split('_')[1]))
                if len(curr_path.split('_')[1]) == 2:
                    bafflesze = int(curr_path.split('_')[1])
                else:
                    bafflesze = int(curr_path.split('_')[1])*10


            split_check = curr_path.split('-')

            if len(split_check) == 3:
                Re_num = (int(self.feats_paths[idx][len(self.feats_paths[idx])-4]))
            else:
                Re_num = 2

            print(f'Re is: {Re_num}')
            print(f'Baffle Size is : {bafflesze}')
            print(f'Double is : {double}')
        else:
            curr_path = self.feats_paths[idx]

            if curr_path.split('_')[1][0] == 'd':
                double = 1
                if len(curr_path.split('_')[3]) == 2:
                    bafflesze = int(curr_path.split('_')[3])
                else:
                    bafflesze = int(curr_path.split('_')[3])*10
                #bafflesze = int(curr_path.split('_')[3])
            else:
                double = 0
                if len(curr_path.split('_')[2]) == 2:
                    bafflesze = int(curr_path.split('_')[2])
                else:
                    bafflesze = int(curr_path.split('_')[2])*10
                # bafflesze = int((curr_path.split('_')[2]))


            split_check = curr_path.split('-')

            if len(split_check) == 2:
                Re_num = (int(self.feats_paths[idx][len(self.feats_paths[idx])-4]))
            else:
                Re_num = 2

        # Re_num = (int(self.feats_paths[idx][len(self.feats_paths[idx])-4]))
        # bafflesze = int(self.feats_paths[idx][])

        data = Data(x=X, edge_index = edge_index, y=Y, Re=Re_num, bafflesze=bafflesze,  dbl=double)

        return data 