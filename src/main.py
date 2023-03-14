import numpy as np
import os
import argparse
from trainer import Trainer
from network import AgeNet
import torch
import torch.nn.functional as F

"""
This is the main python script for running a training pass of the network, many of the network hyperparameters are set via a config file under the 
repo /config/. The main way to interact with this file is to run it via the train.sh or train_and_inf.sh from the repo root directory.

command to run in root, "./train.sh {CONFIG FILE SPECIFIED HERE}"

"""

def get_args():
    """
    Grabs all hyperparameter values from the config file specified when running ./train.sh
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('-seed', type=int, default=24, help='seed')
    parser.add_argument('-num_epochs', type=int, default=100, help='num_epochs')
    parser.add_argument('-n_classes', type=int, default=100, help='n_classes')
    parser.add_argument('-early_stop', type=int, default=5, help='early_stop')
    parser.add_argument('-k_p', type=float, default=0.5, help='k_p')
    parser.add_argument('-num_features', type=int, default=100, help='num_features')
    parser.add_argument('-lat_dim', type=int, default=5, help='lat_dim')
    parser.add_argument('-Re_size', type=int, default=5, help='Re_size')
    parser.add_argument('-baffle_size', type=int, default=5, help='baffle_size')
    parser.add_argument('-dbl_size', type=int, default=5, help='dbl_size')
    parser.add_argument('-lr', type=float, default=0.01, help='lr')
    parser.add_argument('-max_v', type=float, default=1.00, help='max_v')
    parser.add_argument('-max_L', type=float, default=0.01, help='max_L')
    parser.add_argument('-data_path', type=str, default='../data/', help='data_path')
    parser.add_argument('-device', type=str, default='cpu', help='device')
    parser.add_argument('-batch_norm', type=bool, default=False, help='batch_norm')
    parser.add_argument('-down_drop', type=float, nargs='+',default=[0.7], help='down_drop')
    parser.add_argument('-up_drop', type=float, nargs='+',default=[0.7], help='up_drop')
    parser.add_argument('-up_conv_dims', type=int, nargs='+',default=[200, 100, 50, 10], help='up_conv_dims')
    parser.add_argument('-down_conv_dims', type=int, nargs='+',default=[10, 50, 100, 200], help='down_conv_dims')

    args, _ = parser.parse_known_args()

    return args



if __name__=="__main__":

    args = get_args()

    options = vars(args)

    model = AgeNet(args, conv_act=F.relu, pool_act=F.relu) # Sets the args and the activations, inits the network

    trainer = Trainer(args,model, torch.optim.Adam) # trainer class with model and optimizer passed to it.
    trainer.train() #runs training loop
    