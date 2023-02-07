import numpy as np
import os
import argparse
from trainer import Trainer
from network import AgeNet
import torch

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('-seed', type=int, default=24, help='seed')
    parser.add_argument('-num_epochs', type=int, default=100, help='num_epochs')
    parser.add_argument('-n_classes', type=int, default=100, help='n_classes')
    parser.add_argument('-k_p', type=float, default=0.5, help='k_p')
    parser.add_argument('-num_features', type=int, default=100, help='num_features')
    parser.add_argument('-lat_dim', type=int, default=5, help='lat_dim')
    parser.add_argument('-lr', type=float, default=0.01, help='lr')
    parser.add_argument('-max_v', type=float, default=1.00, help='max_v')
    parser.add_argument('-max_L', type=float, default=0.01, help='max_L')
    parser.add_argument('-data_path', type=str, default='../data/', help='data_path')
    parser.add_argument('-device', type=str, default='cpu', help='device')
    parser.add_argument('-up_conv_dims', type=int, nargs='+',default=[200, 100, 50, 10], help='up_conv_dims')
    parser.add_argument('-down_conv_dims', type=int, nargs='+',default=[10, 50, 100, 200], help='down_conv_dims')

    args, _ = parser.parse_known_args()

    return args



if __name__=="__main__":

    args = get_args()

    options = vars(args)

    print(options)

    model = AgeNet(6, args, [0.7, 0.7, 0.7], torch.tanh, torch.tanh, torch.Tensor([1,1]))

    print(args.up_conv_dims)

    trainer = Trainer(args,model, torch.optim.Adam)
    trainer.train()
    