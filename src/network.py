import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ops import bigConv, graphConvPool, graphConvUnpool


class AgeNet(nn.Module):

    def __init__(self, in_dims, n_classes, args,ks, conv_act, pool_act, Re_mat):
        super(AgeNet, self).__init__()
        self.ks = ks
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.conv_act = conv_act
        self.pool_act = pool_act
        #self.up_conv_dims = [int(j) for j in args.up_conv_dims.split()]
        #self.down_conv_dims = [int(j) for j in args.down_conv_dims.split()]
        self.up_conv_dims = args.up_conv_dims
        self.down_conv_dims = args.down_conv_dims
        self.depth = len(self.up_conv_dims)
        self.in_dims = in_dims
        self.bottom_conv = bigConv(args.lat_dim+1, args.lat_dim, self.conv_act, 0, False)
        self.smooth_conv = bigConv(n_classes, n_classes, self.conv_act, 0, False)
        self.Re_mat = Re_mat
        self.num_features = args.num_features
        self.lat_dim = args.lat_dim



        for i, dim in enumerate(self.down_conv_dims):
            if i == 0:
                self.down_convs.append(bigConv(self.num_features, dim, self.conv_act, 0.2, False ))
                self.pools.append(graphConvPool(0.5, dim, self.pool_act))
            else:
                self.down_convs.append(bigConv(self.down_conv_dims[i-1], dim, self.conv_act, 0.2, False ))
                self.pools.append(graphConvPool(0.5, dim, self.pool_act))

        for i, dim in enumerate(self.up_conv_dims):
            if i == 0:
                self.up_convs.append(bigConv(2*self.lat_dim, self.up_conv_dims[i+1], self.conv_act, 0.2, False ))
                self.unpools.append(graphConvUnpool(self.pool_act))
            elif i == self.depth-1:
                self.up_convs.append(bigConv(dim*2, 102, self.conv_act, 0.0, False)) 
                self.unpools.append(graphConvUnpool(self.pool_act)) 
            else:
                self.up_convs.append(bigConv(self.up_conv_dims[i]*2, self.up_conv_dims[i+1], self.conv_act, 0.2, False ))
                self.unpools.append(graphConvUnpool(self.pool_act)) 

    def forward(self, input):

        x, edge_index = input.x, input.edge_index

        x_skips = []
        edge_skips = []
        indcs = []

        for i in range(self.depth):

            x = self.down_convs[i](x, edge_index)
            x_skips.append(x)
            edge_skips.append(edge_index)
            x, edge_index, indc = self.pools[i](x, edge_index)
            indcs.append(indc)
        Re_mat = np.repeat(input.Re[0].item(), x.shape[0])
        Re_mat = torch.reshape(torch.Tensor(Re_mat), (x.shape[0], 1))
        x = torch.cat((x, Re_mat), 1)
        x = self.bottom_conv(x, edge_index)

        for i in range(self.depth):

            up_idx = self.depth - i - 1
            skip, edge, indc = x_skips[up_idx], edge_skips[up_idx], indcs[up_idx]
            x, edge_index = self.unpools[i](skip, edge, indc, x)
            x = torch.cat((x, skip), -1)
            x = self.up_convs[i](x, edge_index)

        return F.log_softmax(x, dim=1)

        
        













