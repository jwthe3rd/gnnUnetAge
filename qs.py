import torch 
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str, default=None, help='file')

    args, _ = parser.parse_known_args()

    return args

def add_dist_baffle(filename):

    feat_mat = torch.load(filename).T

    feat_mat = feat_mat[0:8]

    xc, yc = feat_mat[6], feat_mat[7]
    baffle_indcs = torch.nonzero(feat_mat[3], as_tuple=True)
    xc_baf, yc_baf = xc[baffle_indcs], yc[baffle_indcs]

    x_baf, y_baf = torch.mean(xc_baf), torch.mean(yc_baf)

    xdelta, ydelta = xc - x_baf.item(), yc - y_baf.item()
    xsq, ysq = torch.mul(xdelta,xdelta), torch.mul(ydelta,ydelta)
    sum_sq = xsq+ysq
    d_baf = torch.sqrt(sum_sq).reshape(-1,1)
    feat_mat = feat_mat.T

    feat_mat = torch.cat((feat_mat,d_baf),-1)

    torch.save(feat_mat,f'./{filename}')

    return 0



if __name__ == "__main__":
    args = get_args()
    add_dist_baffle(args.file)
