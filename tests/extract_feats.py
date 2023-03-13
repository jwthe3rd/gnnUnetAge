# from lf import load_owners, load_boundaries
import lf_func as lf
import torch
import numpy as np
import argparse

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-direct', type=str, default='./', help='direct')
    args, _ = parser.parse_known_args()

    return args



def load_boundary_features(direct, save_loc):

    poly_direct = direct + '/constant/polyMesh/'
    owners = lf.load_owners(poly_direct)
    faces = lf.load_faces(poly_direct, 18)
    points = lf.load_points(poly_direct, 'points', 18 )
    boundaries = lf.load_boundaries(poly_direct, owners)
    dists = lf.calc_dist_inlet_outlet_baffle(poly_direct, boundaries)
    centers_x, centers_y = lf.calc_cell_centers(owners, faces, points)
    feats_mat = lf.create_feature_matrix(boundaries, x_c=centers_x,y_c=centers_y,num_of_nodes=len(dists), dist_matrix=dists)

    #print(faces)
    print(feats_mat)
    feats_mat = torch.Tensor(feats_mat)
    torch.save(feats_mat, f'{save_loc}f_{direct}.pt')

    print(feats_mat)

    return -1


if __name__ == "__main__":

    args = get_args()
    direct = args.direct

    load_boundary_features(direct, save_loc = "../prepData2/")
