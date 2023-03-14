import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, BatchNorm, GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse, add_self_loops, remove_isolated_nodes
import numpy as np
import math

"""
    This file contains all the non pytorch default modules to be used in the network

    Referenced in network.py 

"""

class bigConv(nn.Module):
    """ Convolution module for the network """

    def __init__(self, in_dim, out_dim, act, p, batchNorm):

        super(bigConv, self).__init__()
        self.conv1 = SAGEConv(in_dim, out_dim, project=True, aggr='max') # SAGEConv from "Inductive Representation Learning on Large Graphs" paper
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.batchNorm = BatchNorm(out_dim) if batchNorm else nn.Identity()

    def forward(self, x, edge_index):
        edge_index = add_self_loops(edge_index)[0]

        l1 = self.conv1(x, edge_index)
        l1 = self.act(l1)
        l1 = self.batchNorm(l1)
        l1 = self.drop(l1)

        return l1
    
class feedFWD(nn.Module):
    """ Basic feed forward linear layer with batch norm, dropout, and activation"""

    def __init__(self, in_dim, out_dim, act, p, batchNorm=False):
        super(feedFWD, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.batchNorm = BatchNorm(out_dim) if batchNorm else nn.Identity()
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.act = act

    def forward(self, x):
        x = self.lin(x)
        x = self.batchNorm(x)
        x = self.drop(x)
        if self.act is not None:
            return self.act(x)


class graphConvUnpool(nn.Module):
    """ Similar unpool to that in Graph U-nets but with an added convolution to the unpooling """
    def __init__(self,  act, dim, device):
        super(graphConvUnpool, self).__init__()
        self.act = act
        self.unpoolconv = GCNConv(dim, dim)
        self.device = device

    def forward(self, x_skip, e_skip, indices, x):

        unpooled_x = torch.zeros(size=x_skip.shape).to(self.device) # generate all zeros tensor of shape up layer
        unpooled_x[indices] = x # add the values from the previous layer to the upsampled all zeros
        unpooled_x = self.unpoolconv(unpooled_x, e_skip) # do a convolution over the values to pass messages
        return self.act(unpooled_x), e_skip


class Initializer(object):
    """ This is the exact same init as in Graph U-nets with a possible bias=False switch"""
    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                m.bias.data.zero_()
                cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)