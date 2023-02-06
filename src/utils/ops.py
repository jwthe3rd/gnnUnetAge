import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, BatchNorm, GCNConv
from torch_geometric.utils import to_dense_adj
import numpy as np



class bigConv(nn.Module):

    def __init__(self, in_dim, out_dim, act, p, batchNorm):

        super(bigConv, self).__init__()
        self.conv = SAGEConv(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.batchNorm = BatchNorm(in_dim) if batchNorm else nn.Identity()

    def forward(self, x, edge_index):

        l1 = self.batchNorm(x)
        l1 = self.conv(x, edge_index)
        l1 = self.drop(l1)
        l1 = self.act(l1)

        return l1


class graphConvPool(nn.Module):

    def __init__(self, k, in_dim, act):
        super(graphConvPool, self).__init__()
        self.k = k
        self.scoregen = GCNConv(in_dim, 1)
        self.act = act

    def top_k_pool(self, g, e1):
        num_nodes = g.shape[0]
        print(num_nodes)
        print(g.shape)
        print(max(2, int(self.k*num_nodes)))
        for i in range(1000000000):
            continue
        pooled, indices = torch.topk(g, max(2, int(self.k*num_nodes)))
        g = self.act(g)
        adj_mat = to_dense_adj(e1)
        adj_mat_new = adj_mat[indices, :]
        e2 = adj_mat_new.nonzero().t().continguous()

        return pooled, e2, indices
        
    def forward(self, x, edge_index):
        p1 = self.scoregen(x, edge_index).squeeze()

        return self.top_k_pool(p1, edge_index)

class graphConvUnpool(nn.Module):

    def __init__(self,  act):
        super(graphConvUnpool, self).__init__()
        self.act = act

    def forward(self, x_skip, e_skip, indices, x):

        unpooled_x = torch.zeros(size=x_skip.shape)
        unpooled_x[indices] = x
        unpooled_x = GCNConv(unpooled_x, e_skip)
        return self.act(unpooled_x), e_skip


class maxDeltaAgeLoss:

    def __init__(self, args, e, pred):
        self.max_v = args.max_v
        self.max_L = args.max_L
        self.e = e
        self.pred = pred

    def delta_mat_gen(self):
        adj_mat = to_dense_adj(self.e)
        new_mat = adj_mat*self.pred[None, :]
        subbed = torch.sub(new_mat, self.pred)
        delta_mat = subbed*adj_mat
        return delta_mat
    
    def loss(self):
        delta_mat = self.delta_mat_gen()
        error_mat = delta_mat - (self.max_L / self.max_v)
        error_mat = torch.nn.functional.relu(error_mat, inplace=True)
        return torch.sum(error_mat) / len(error_mat[0]) 


