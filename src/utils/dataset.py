import os
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import random
from sklearn.model_selection import train_test_split
import numpy as np

"""
    This file contains the utilities for data handling by the network

    Referenced in trainer.py and inf.py

"""

class DataGenerator:
    """ Generates a list of data for x, edges, and ys from a directory of .pt tensors and segments to training and testing sets """
    def __init__(self, path, seed):
        self.path = path
        self.seed = seed

    def segment_data(self, tuning=False):
        training_cases = os.listdir(self.path)
        """ List initialization """
        edge_train = []
        feats_train = []
        label_train = []
        edge_val = []
        feats_val = []
        label_val = []
        unique_cases = []
        Re_list = []
        """ ------------------ """
        ## Adding paths for all training data
        for data_file in training_cases:
            if data_file[0] == 'f' and data_file[2:] not in unique_cases:
                unique_cases.append(data_file[2:])

        if tuning:
            num_select = (len(unique_cases) * 2) // 10
            random.seed(68)
            unique_cases = random.sample(unique_cases, num_select)

        for case in unique_cases:
            Re_list.append(1 if len(case.split('-'))==2 else 0) # Creates a list identifying high or low Re for stratification

        train_cases, val_cases, *_ = train_test_split(unique_cases, train_size=0.8, random_state=self.seed, stratify=Re_list) # gets the split with stratification
        """ Loop to add all the cases to a list with the actual file paths """
        for case in train_cases:
            edge_train.append(f'{self.path}e_{case}')
            feats_train.append(f'{self.path}f_{case}')
            label_train.append(f'{self.path}l10_{case}')
        for case in val_cases:
            edge_val.append(f'{self.path}e_{case}')
            feats_val.append(f'{self.path}f_{case}')
            label_val.append(f'{self.path}l10_{case}')
        """ ---------------------------------------------------------------"""
        if len(edge_train) != len(feats_train) or len(edge_train) != len(label_train):
            raise ValueError('mismatch in edges, labels and feats train size')
        if len(edge_val) != len(feats_val) or len(edge_val) != len(label_val):
            raise ValueError('mismatch in edges, labels and feats val size')
        
        
        return label_train, feats_train, edge_train,label_val, feats_val, edge_val

class gnnAgeDataSet(Dataset):
    """ Dataset object with specifications for this network """
    def __init__(self, edge_paths, feats_paths, label_paths, transform=None, test=False):

        self.edge_paths = edge_paths
        self.feats_paths = feats_paths
        self.label_paths = label_paths
        self.transform = transform
        self.test=test

    def __len__(self):
        return len(self.feats_paths)

    def __getitem__(self,idx):
        """ Loads in the data """
        X = torch.load(self.feats_paths[idx])
        edge_index = torch.load(self.edge_paths[idx])
        Y = torch.load(self.label_paths[idx])

        X = X[:,[0,1,2,3,4,5,8]] # Removing the x,y coords from the feature tensor

        """ The following is some very not so elegant script to get the Re, baffle size, and number of baffles (1 or 2) from the file names"""
        if self.test:
            curr_path = self.feats_paths[idx]

            if curr_path.split('_')[5][0] == 'd':
                double = 1
                if len(curr_path.split('_')[7]) == 2:
                    bafflesze = int(curr_path.split('_')[7])/10
                else:
                    bafflesze = int(curr_path.split('_')[7])
            else:
                double = 0
                bafflesze = int((curr_path.split('_')[1]))
                if len(curr_path.split('_')[1]) == 2:
                    bafflesze = int(curr_path.split('_')[1])/10
                else:
                    bafflesze = int(curr_path.split('_')[1])

            split_check = curr_path.split('-')

            if len(split_check) == 3:
                # Re_num = (int(self.feats_paths[idx][len(self.feats_paths[idx])-4]))
                Re_num = np.log10(float(f'{split_check[1][-1]}.{split_check[2][0:len(split_check[2]) - 6]}')*10**(int(self.feats_paths[idx][len(self.feats_paths[idx])-4])))
            else:
                split = curr_path.split('_')
                Re_num = np.log10(float(f'{split[-2]}.{split[-1][0:len(split[-1]) - 4]}'))

            print(f'Re is {Re_num}')
            print(f'BaffleSize is {bafflesze}')
            print(f'Double is {double}')
        else:
            curr_path = self.feats_paths[idx]

            if curr_path.split('_')[1][0] == 'd':
                double = 1
                if len(curr_path.split('_')[3]) == 2:
                    bafflesze = int(curr_path.split('_')[3])
                else:
                    bafflesze = int(curr_path.split('_')[3])/10
            else:
                double = 0
                if len(curr_path.split('_')[2]) == 2:
                    bafflesze = int(curr_path.split('_')[2])/10
                else:
                    bafflesze = int(curr_path.split('_')[2])

            split_check = curr_path.split('-')

            if len(split_check) == 2:
                # Re_num = (int(self.feats_paths[idx][len(self.feats_paths[idx])-4]))
                Re_num = np.log10(float(f'{split_check[0][-1]}.{split_check[1][0:len(split_check[1]) - 6]}')*10**(int(self.feats_paths[idx][len(self.feats_paths[idx])-4])))
            else:
                # Re_num= 0
                split = curr_path.split('_')
                Re_num = np.log10(float(f'{split[-2]}.{split[-1][0:len(split[-1]) - 4]}'))
            # print(f'Re is: {Re_num}')
            # print(f'Baffle Size is : {bafflesze}')
            # print(f'Double is : {double}')


        Re_num = torch.tensor(np.repeat(Re_num, X.shape[0]), dtype=torch.float32).reshape(-1,1)
        bafflesze = torch.tensor(np.repeat(bafflesze, X.shape[0]), dtype=torch.float32).reshape(-1,1)
        double = torch.tensor(np.repeat(double, X.shape[0]), dtype=torch.float32).reshape(-1,1)
        """ -------------------------------------------------------------------------------------------- """
        X = torch.cat((X, Re_num, bafflesze, double), -1) # add the obtained results to the feature tensor

        data = Data(x=X, edge_index = edge_index, y=Y, name=curr_path) # create a torch_geometric Data object from the data loaded in this dataset

        return data 
