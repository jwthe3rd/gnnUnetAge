import torch
import numpy as np
import os



def gen_edge(datafile):

    feat_tensor = torch.load(datafile).T

    x_c, y_c = feat_tensor[6], feat_tensor[7]

    x_c = x_c.repeat(x_c.shape[0], 1)
    y_c = y_c.repeat(y_c.shape[0], 1)
    x_c2 = x_c.T
    y_c2 = y_c.T
    x_diff = x_c - x_c2
    y_diff = y_c - y_c2
    x_dist = torch.mul(x_diff, x_diff)
    y_dist = torch.mul(y_diff, y_diff)
    tot_dist = x_dist + y_dist
    dist = torch.sqrt(tot_dist)

    init_mat = torch.where(dist<0.05, dist, 0.)
    new_mat = torch.where(init_mat==0, init_mat, 1.)

    adj_mat = new_mat + torch.eye(new_mat.shape[0])

    adj_mat = adj_mat.long()

    degrees = torch.sum(adj_mat, 1)
    degrees = degrees.float()
    mean = torch.mean(degrees)

    print(degrees.max())

    edges = adj_mat.nonzero().t().contiguous()

    return edges


if __name__=="__main__":
     
    data_dir = './test_extrap/tri/'

    for f in os.scandir(data_dir):

        if f.is_file() and f.name[0] == 'f':


            edges = gen_edge(datafile=f.path)

            print(f'{data_dir}e{f.name[1:]}')

            torch.save(edges, f'{data_dir}e{f.name[1:]}')
            


