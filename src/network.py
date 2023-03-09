import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ops import bigConv, feedFWD, graphConvUnpool, Initializer
from torch_geometric.nn.pool import TopKPooling


class AgeNet(nn.Module):

    def __init__(self, args,conv_act, pool_act):
        super(AgeNet, self).__init__()
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.conv_act = conv_act
        self.pool_act = pool_act
        self.batch_norm = args.batch_norm
        self.down_drop = args.down_drop
        self.up_drop = args.up_drop
        self.up_conv_dims = args.up_conv_dims
        self.down_conv_dims = args.down_conv_dims
        self.depth = len(self.up_conv_dims)
        self.Re_size = args.Re_size
        self.baffle_size = args.baffle_size
        self.dbl_size = args.dbl_size
        self.bottom_conv_indim = args.lat_dim+self.Re_size+self.baffle_size+self.dbl_size
        self.bottom_lin = feedFWD(self.bottom_conv_indim, self.bottom_conv_indim // 2, self.conv_act, self.batch_norm)#bigConv(self.bottom_conv_indim, self.bottom_conv_indim // 2, self.conv_act, 0.2, self.batch_norm)
        self.bottom_lin2 = feedFWD(self.bottom_conv_indim // 2, args.lat_dim, self.conv_act, 0.2, self.batch_norm)#bigConv(self.bottom_conv_indim // 2, args.lat_dim, self.conv_act, 0.2, self.batch_norm)
        self.smooth_conv = feedFWD(args.n_classes, args.n_classes, self.conv_act, 0, self.batch_norm)
        self.num_features = args.num_features
        self.lat_dim = args.lat_dim
        self.n_classes = args.n_classes
        self.k_p = args.k_p
        self.device = args.device


        for i, dim in enumerate(self.down_conv_dims):
            if i == 0:
                self.down_convs.append(bigConv(self.num_features, dim, self.conv_act, self.down_drop[0], self.batch_norm))
                self.pools.append(TopKPooling(dim, self.k_p))
            else:
                self.down_convs.append(bigConv(self.down_conv_dims[i-1], dim, self.conv_act, self.down_drop[i-1], self.batch_norm ))
                self.pools.append(TopKPooling(dim, self.k_p))

        for i, dim in enumerate(self.up_conv_dims):
            if i == 0:
                self.up_convs.append(bigConv(self.lat_dim*2, self.up_conv_dims[i+1], self.conv_act, self.up_drop[i], self.batch_norm ))
                self.unpools.append(graphConvUnpool(self.pool_act, self.up_conv_dims[i], self.device))
            elif i == self.depth-1:
                self.up_convs.append(bigConv(dim*2, self.n_classes, self.conv_act, 0.0, False)) 
                self.unpools.append(graphConvUnpool(self.pool_act, self.up_conv_dims[i], self.device)) 
            else:
                self.up_convs.append(bigConv(self.up_conv_dims[i]*2, self.up_conv_dims[i+1], self.conv_act, self.up_drop[i], self.batch_norm ))
                self.unpools.append(graphConvUnpool(self.pool_act, self.up_conv_dims[i], self.device))

        Initializer.weights_init(self) 

    def forward(self, input):

        x, edge_index, batch = input.x, input.edge_index, input.batch

        x_skips = []
        edge_skips = []
        indcs = []

        for i in range(self.depth):

            x = self.down_convs[i](x, edge_index)
            x_skips.append(x)
            edge_skips.append(edge_index)
            x, edge_index,_,batch,indc,_ = self.pools[i](x, edge_index, batch=batch)
            indcs.append(indc)
        Re_mat = np.repeat(input.Re[0].item(), x.shape[0])
        Re_mat = np.repeat(Re_mat, self.Re_size)
        Re_mat = torch.reshape(torch.Tensor(Re_mat), (x.shape[0], self.Re_size))
        baffle_mat = np.repeat(input.bafflesze[0].item(), x.shape[0])
        baffle_mat = np.repeat(baffle_mat, self.baffle_size)
        baffle_mat = torch.reshape(torch.Tensor(baffle_mat), (x.shape[0], self.baffle_size))
        dbl_mat = np.repeat(input.dbl[0].item(), x.shape[0])
        dbl_mat = np.repeat(dbl_mat, self.dbl_size)
        dbl_mat = torch.reshape(torch.Tensor(dbl_mat), (x.shape[0], self.dbl_size))
        Re_mat = Re_mat.to(self.device)
        baffle_mat = baffle_mat.to(self.device)
        dbl_mat = dbl_mat.to(self.device)
        x = torch.cat((x, Re_mat, baffle_mat, dbl_mat), 1)
        x = self.bottom_conv(x, edge_index)
        x = self.bottom_conv2(x, edge_index)

        for i in range(self.depth):

            up_idx = self.depth - i - 1
            skip, edge, indc = x_skips[up_idx], edge_skips[up_idx], indcs[up_idx]
            x, edge_index = self.unpools[i](skip, edge, indc, x)
            x = torch.cat((x, skip), -1)
            x = self.up_convs[i](x, edge_index)

        return F.log_softmax(x, dim=1)

        
        













