import torch
import numpy as np
import matplotlib.pyplot as plt
from network import GraphConvUnet_age
from utils.dataset import data_generator, gnnAgeDataSet
from torch_geometric.loader import DataLoader
import torch.nn.functional as F


class Trainer:

    def __init__(self, args, model, optimizer):

        self.batch_size = args.batch_size
        self.path = args.path
        self.num_epochs = args.num_epochs
        self.model = model
        self.device = args.device
        self.optimizer = optimizer
        self.lr = args.lr

    def load_data(self, path):

        y, x, e = data_generator(path=path).segment_data()
        dataset = gnnAgeDataSet(e, x, y)

        loader = DataLoader(dataset, batch_size=self.batch_size)

        return loader

    def to_device(self):
        self.model.to_device(self.device)
        return 0

    def train_step(self, batch, batch_num, loader):
        optimizer = self.optimizer(self.model.parameters(),lr=self.lr)
        optimizer.zero_grad()
        out = self.model(batch)
        loss = F.mse_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print ("\033[A                             \033[A")
        print(f"{int(100*batch_num/len(loader))} % ||| Training MSE Loss = {round(train_loss / (batch_num+1),2)}                     ")
        return 0
    
    def train(self, loader):
        self.to_device()
        for epoch in range(self.num_epochs):
            print('\n')
            train_loss = 0
            val_loss = 0
            for i,batch in enumerate(loader):
                batch = batch.to(self.device)
                train_loss = self.train_step(batch_num=i, batch=batch, loader=loader, train_loss=train_loss)



