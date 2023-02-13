import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import argparse
import torch

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-direct', type=str, default='./', help='direct')
    args, _ = parser.parse_known_args()

    return args

args = get_args()

direct = args.direct

def find_max_iter(dir):

    numbers = []

    for d in os.scandir(dir):
        if d.is_dir():
            try:
                numbers.append(int(d.name))
            except:
                continue

    return f'{max(numbers)}/'


def load_feature_vector(dir,file):
    feature_Values = []

    with open(dir+file) as f:
        lines = f.readlines()

    feature_data_raw = lines[20:]

    num_of_cells = int(feature_data_raw[0])

    feature_data_raw = feature_data_raw[2:num_of_cells+2]

    features_strings_full = []

    for i in range(num_of_cells):
        features_strings_full.append(feature_data_raw[i])

    feature_strings_split = []

    for node in features_strings_full:
        feature_strings_split.append(node.split())
    
    for i in range(len(feature_strings_split)):

        feature_strings_split[i] = [float(feature_strings_split[i][0][1:]),float(feature_strings_split[i][1]),float(feature_strings_split[i][2][:len(feature_strings_split[i][2])-1])]

    feature_Values = feature_strings_split

    return feature_Values

def load_feature_scalar(case, dir,file):
    feature_Values = []

    with open(case+dir+file) as f:
        lines = f.readlines()

    feature_data_raw = lines[21:]

    num_of_cells = int(feature_data_raw[0])

    feature_data_raw = feature_data_raw[2:num_of_cells+2]

    features_strings_full = []

    for i in range(num_of_cells):
        features_strings_full.append(feature_data_raw[i])

    feature_strings_split = []

    for node in features_strings_full:
        feature_strings_split.append(node.split())
    
    for i in range(len(feature_strings_split)):

        feature_strings_split[i] = float(feature_strings_split[i][0])

    feature_Values = feature_strings_split

    return feature_Values

def normalize(feat_array):

    norm_array = np.empty_like(feat_array)

    maxima = 22.5

    for i in range(len(feat_array)):
        value = round(feat_array[i]/maxima,1)
        if value > 1.0 and value <= 2.0:
            norm_array[i] = 1.1
        elif value >2.0 and value <= 5.0:
            norm_array[i] = 2.1
        elif value >5.0 and value <= 10.0:
            norm_array[i] = 5.1
        elif value >10.0:
            norm_array[i] = 10.1
        elif value < 0.0:
            norm_array[i] = 0.0
        else:
            norm_array[i] = value

    return norm_array
def write_Norm_Contour(case, norm_matrix,dir,file):

    with open(case+dir+file) as f:
        lines = f.readlines()
    f.close()

    begin = lines[0:21]
    begin_data = lines[21:23]
    num_of_cells = int(begin_data[0])

    end = lines[(23+num_of_cells):]
    
    full_file = []
    for val in begin:
        full_file.append(val)
    for val in begin_data:
        full_file.append(val)
    for i in norm_matrix:
        full_file.append(str(i) + '\n')
    for val in end:
        try:
            if val.split()[1] == 'nonuniform':
                print(1)
                num_nodes = int(val.split()[3].split('(')[0])
                starting = f'{num_nodes}(0.0 '
                for i in range(num_nodes-2):
                    starting += '1.0 '
                starting += f'0.0);'
                new_val = f'{val.split()[0]}    {val.split()[1]} {val.split()[2]} {starting}'
                full_file.append(new_val)
            elif val.split()[1] == 'uniform':
                print(2)
                full_file.append(f'{val.split()[0]}    uniform 0;')
            else:
                full_file.append(val)
        except:
            full_file.append(val)


    with open(case + dir + 'age_norm','w') as g:

        for i in full_file:
            g.write(i)
    g.close()


    return 0

max_iter = find_max_iter(direct)
features_list = load_feature_scalar(direct, max_iter,'age')
# features_matrix=np.array([np.array(xi) for xi in features_list])
feat_norm = normalize(features_list)
norm_matrix = np.array([np.array(xi) for xi in feat_norm])
vals = np.linspace(0, 1, 11)
for i, val in enumerate(vals):
    vals[i] = round(val, 1)

labels_mat = []
print(vals)
# vals_2 = np.delete(vals, np.where(vals == -0.0))

for k, value in enumerate(norm_matrix):
    init = len(labels_mat)
    if value > 1.0 and value <= 2.0:
        labels_mat.append(len(vals))
    elif value > 2.0 and value <=5.0:
        labels_mat.append(len(vals) + 1)
    elif value > 5.0 and value <=10:
        labels_mat.append(len(vals) + 2)
    elif value > 10:
        labels_mat.append(len(vals) + 3)
    elif value < 0.0:
        labels_mat.append(0)
    else:
        for i, j in enumerate(vals):
            if value == j:
                labels_mat.append(i)
                break

labels_mat = torch.LongTensor(labels_mat)
if labels_mat.shape[0] != norm_matrix.shape[0]:
    raise Exception("Missing nodes in label mat")
torch.save(labels_mat, f'{direct}l_{direct[:len(direct) - 1]}.pt')

# print(features_matrix)
print(norm_matrix)
print(labels_mat)
write_Norm_Contour(direct,norm_matrix,max_iter,'age')

