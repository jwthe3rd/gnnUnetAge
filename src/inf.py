from network_inf import AgeNet
import numpy as np
import os
import argparse
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from utils.dataset import gnnAgeDataSet
from torch_geometric.loader import DataLoader

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('-seed', type=int, default=24, help='seed')
    parser.add_argument('-num_epochs', type=int, default=100, help='num_epochs')
    parser.add_argument('-n_classes', type=int, default=100, help='n_classes')
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
    parser.add_argument('-test', type=str, nargs='+', default='', help='test')

    args, _ = parser.parse_known_args()

    return args


def find_max_iter(dir):

    numbers = []

    for d in os.scandir(f'tests/{dir}/'):
        if d.is_dir():
            try:
                numbers.append(int(d.name))
            except:
                continue

    return f'{max(numbers)}/'

def pred_to_contour(pred, data, max_iter):
    # vals = np.linspace(0, 1, 11)
    labels_mat = []
    # for i, val in enumerate(vals):
    #     vals[i] = round(val, 1)

    # for value in pred:
    #     if value == len(vals):
    #         labels_mat.append(1.1)
    #     elif value == len(vals) + 1:
    #         labels_mat.append(2.1)
    #     elif value == len(vals) + 2:
    #         labels_mat.append(5.1)
    #     elif value == len(vals) + 3:
    #         labels_mat.append(10.1)
    #     elif value == 0:
    #         labels_mat.append(0)
    #     else:
    #         labels_mat.append(vals[value])

    for value in pred:
        if value == 0:
            labels_mat.append(0)
        elif value == 1:
            labels_mat.append(22.5)
        elif value == 2:
            labels_mat.append(45)
        else:
            labels_mat.append(90)



    with open(f'tests/{data}/{max_iter}age') as f:
        lines = f.readlines()
    f.close()

    begin = lines[0:21]
    begin_data = lines[21:23]
    num_of_cells = int(begin_data[0])
    age_data = np.asarray(lines[23:23+num_of_cells], dtype=float)

    end = lines[(23+num_of_cells):]
    
    full_file = []
    full_file_2 = []
    for val in begin:
        full_file.append(val)
        full_file_2.append(val)
    for val in begin_data:
        full_file.append(val)
        full_file_2.append(val)
    for i in labels_mat:
        full_file.append(str(i) + '\n')
    for val in age_data:
        norm = val / 22.5
        if norm > 1 and norm <= 2.0:
            full_file_2.append('22.5\n')
        elif norm >2 and norm <= 5.0:
            full_file_2.append('45\n')
        elif norm > 5.0:
            full_file_2.append('90\n')
        else:
            full_file_2.append('0\n')
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
                full_file_2.append(new_val)
            elif val.split()[1] == 'uniform':
                print(2)
                full_file.append(f'{val.split()[0]}    uniform 0;')
                full_file_2.append(f'{val.split()[0]}    uniform 0;')
            else:
                full_file.append(val)
                full_file_2.append(val)
        except:
            full_file.append(val)
            full_file_2.append(val)


    with open(f'tests/{data}/{max_iter}age_norm_pred','w') as g:

        for i in full_file:
            g.write(i)
    g.close()

    with open(f'tests/{data}/{max_iter}age_norm_gt','w') as g:

        for i in full_file_2:
            g.write(i)
    g.close()

    return 0


@torch.no_grad()
def run_test(model, data):

    e = torch.load(f'tests/{data}/e_{data}.pt')
    x = torch.load(f'tests/{data}/f_{data}.pt')
    y = torch.load(f'tests/{data}/l_{data}.pt')
    print(data)
    print(data[2])

    datapt = gnnAgeDataSet(feats_paths=[f'tests/{data}/f_{data}.pt'], edge_paths=[f'tests/{data}/e_{data}.pt'], label_paths=[f'tests/{data}/l_{data}.pt'], test=True)
    loader = DataLoader(datapt, batch_size=1)
    # datapt.to("cuda")
    for data_test in loader:
        data_test.to("cuda")

        out = model(data_test)

        loss = F.nll_loss(out, data_test.y)
        _, preds = torch.max(out, 1)
        acc = torch.mean((preds == data_test.y).float())
        #acc += acc.item()
        #loss = loss.item()
    #print ("\033[A                             \033[A")
    return acc.item(), loss.item(), preds

if __name__ == "__main__":
    args = get_args()
    model = AgeNet(args, torch.tanh, torch.tanh)
    model.load_state_dict(torch.load('models/model4'))
    model.to("cuda")
    model.eval()

    for test in args.test:
        test_acc, test_loss, test_preds = run_test(model, data=test)
        max_iter = find_max_iter(test)
        #pred_to_contour(data=test, pred=test_preds, max_iter=max_iter)
        print(f'Loss is {test_loss}')
        print(f'Acc is {test_acc}')
