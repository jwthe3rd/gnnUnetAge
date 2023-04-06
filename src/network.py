import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ops import bigConv, feedFWD, graphConvUnpool, Initializer
from torch_geometric.nn.pool import TopKPooling

"""
This file contains the network architecture for the GNN U-Net Age Classifier.

Referenced in trainer.py, main.py, and inf.py

"""

class AgeNet(nn.Module):

    def __init__(self, args,conv_act=F.relu, pool_act=F.relu, device='cpu'):
        """ Initializes all of the layers in the network"""
        super(AgeNet, self).__init__()
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.conv_act = conv_act
        self.pool_act = pool_act
        self.batch_norm = args.batch_norm
        self.drop = args.drop
        self.down_conv_dims = args.down_conv_dims
        self.up_conv_dims = self.down_conv_dims[::-1]#args.up_conv_dims
        self.depth = len(self.up_conv_dims)
        print(self.up_conv_dims)
        self.lat_dim = self.down_conv_dims[-1]
        self.bottom_lin = feedFWD(self.lat_dim, self.lat_dim, self.conv_act, self.batch_norm)
        self.classify = nn.Linear(args.n_classes, args.n_classes)
        self.num_features = args.num_features
        self.n_classes = args.n_classes
        self.k_p = args.k_p
        self.device = device

        """ --- Loops for generating all of the up and down convolutions --- """
        for i, dim in enumerate(self.down_conv_dims):
            if i == 0:
                self.down_convs.append(bigConv(self.num_features, dim, self.conv_act, self.drop[0], self.batch_norm))
                self.pools.append(TopKPooling(dim, self.k_p))
            else:
                self.down_convs.append(bigConv(self.down_conv_dims[i-1], dim, self.conv_act, self.drop[i-1], self.batch_norm ))
                self.pools.append(TopKPooling(dim, self.k_p))

        for i, dim in enumerate(self.up_conv_dims):
            if i == 0:
                self.up_convs.append(bigConv(self.lat_dim*2, self.up_conv_dims[i+1], self.conv_act, self.drop[i], self.batch_norm ))
                self.unpools.append(graphConvUnpool(self.pool_act, self.up_conv_dims[i], self.device))
            elif i == self.depth-1:
                self.up_convs.append(bigConv(dim*2, self.n_classes, self.conv_act, 0.0, False)) 
                self.unpools.append(graphConvUnpool(self.pool_act, self.up_conv_dims[i], self.device)) 
            else:
                self.up_convs.append(bigConv(self.up_conv_dims[i]*2, self.up_conv_dims[i+1], self.conv_act, self.drop[i], self.batch_norm ))
                self.unpools.append(graphConvUnpool(self.pool_act, self.up_conv_dims[i], self.device))
        """ ------------------- -------------------------------------"""

        Initializer.weights_init(self)# Same initialization as in the Graph U-nets Paper 

    def forward(self, input, test=False):
        """Defining the forward pass of the network"""
        x, edge_index, batch = input.x, input.edge_index, input.batch

        x_skips = []        ### Initializing
        edge_skips = []     ### Lists for
        indcs = []          ### Pooling and Unpooling

        for i in range(self.depth):
            """ Convolutions and pooling """
            x = self.down_convs[i](x, edge_index)
            x_skips.append(x)
            edge_skips.append(edge_index)
            x, edge_index,_,batch,indc,_ = self.pools[i](x, edge_index, batch=batch)
            indcs.append(indc)
    
        x = self.bottom_lin(x) # Latent dimension feed forward

        for i in range(self.depth):
            """ Up sampling and convolution"""
            up_idx = self.depth - i - 1
            skip, edge, indc = x_skips[up_idx], edge_skips[up_idx], indcs[up_idx]
            x, edge_index = self.unpools[i](skip, edge, indc, x)
            x = torch.cat((x, skip), -1)
            x = self.up_convs[i](x, edge_index)
        
        x = self.classify(x) #Final smoothing / classification layer
        if test:
            return F.log_softmax(x, dim=1), indcs

        return F.log_softmax(x, dim=1)

        
        













