import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import data_generator, gnnAgeDataSet
from torch_geometric.loader import DataLoader
import torch.nn.functional as F


class Trainer:

    def __init__(self, args, model, optimizer):

        self.batch_size = args.batch_size
        self.path = args.data_path
        self.num_epochs = args.num_epochs
        self.model = model
        self.device = args.device
        self.seed = args.seed
        self.optimizer = optimizer
        self.lr = args.lr

    def load_data(self):

        y, x, e = data_generator(path=self.path, seed=self.seed).segment_data()
        dataset = gnnAgeDataSet(e, x, y)

        self.loader = DataLoader(dataset, batch_size=self.batch_size)

        return 0

    def to_device(self):
        self.model.to(self.device)
        return 0

    def train_step(self, batch, batch_num, loader, train_loss, accur):
        optimizer = self.optimizer(self.model.parameters(),lr=self.lr)
        optimizer.zero_grad()
        out = self.model(batch)
        loss = F.nll_loss(out, batch.y)
        _, preds = torch.max(out, 1)
        acc = torch.mean((preds == batch.y).float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        accur += acc.item()
        print ("\033[A                             \033[A")
        print(f"{int(100*batch_num/len(loader))} % ||| Training MSE Loss = {round(train_loss / (batch_num+1),2)}    ||| Training Accuracy = {100*round(accur / (batch_num+1),2)} %                ")
        return train_loss, accur

    def train(self):
        self.load_data()
        self.to_device()
        for epoch in range(self.num_epochs):
            print('\n')
            train_loss = 0
            val_loss = 0
            accur = 0
            for i,batch in enumerate(self.loader):
                batch = batch.to(self.device)
                train_loss, accur = self.train_step(batch_num=i, batch=batch, loader=self.loader, train_loss=train_loss, accur=accur)




