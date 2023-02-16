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
        training_dataset = gnnAgeDataSet(e[0:int(len(e)*0.8)], x[0:int(len(x)*0.8)], y[0:int(len(y)*0.8)])
        validation_dataset = gnnAgeDataSet(e[int(len(e)*0.8):], x[int(len(x)*0.8):], y[int(len(y)*0.8):])

        print(e[int(len(e)*0.8):])

        self.train_loader = DataLoader(training_dataset, batch_size=self.batch_size)
        self.val_loader = DataLoader(validation_dataset, batch_size=self.batch_size)

        return 0

    def to_device(self):
        self.model.to(self.device)
        return 0

    def train_step(self, batch, batch_num, loader, train_loss, accur, epoch, best_loss, best_acc):
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
        if best_acc < 100*round(accur / (batch_num+1),2):
            best_acc = 100*round(accur / (batch_num+1),2)
        if best_loss > round(train_loss / (batch_num+1),2):
            best_loss = round(train_loss / (batch_num+1),2)
        print ("\033[A                             \033[A")
        print(f"Epoch:{epoch} | {int(100*batch_num/len(loader))} % | T NLL Loss = {round(train_loss / (batch_num+1),2)}  Best Loss: {best_loss}| T Acc = {100*round(accur / (batch_num+1),2)} %  Best Acc: {best_acc} %                           ")
        return train_loss, accur, best_loss, best_acc
    
    @torch.no_grad()
    def val_step(self, batch, batch_num, loader, val_loss, accur, epoch):
        out = self.model(batch)
        loss = F.nll_loss(out, batch.y)
        _, preds = torch.max(out, 1)
        acc = torch.mean((preds == batch.y).float())
        val_loss += loss.item()
        accur += acc.item()
        #print ("\033[A                             \033[A")
        print(f"Epoch: {epoch}   {int(100*batch_num/len(loader))} % ||| Validation Negative Log-Likelihood Loss = {round(val_loss / (batch_num+1),2)}    ||| Validation Accuracy = {100*round(accur / (batch_num+1),2)} %                ")
        return val_loss, accur

    def train(self):
        self.load_data()
        self.to_device()
        self.training_loss = []
        self.validation_loss =[]
        self.training_acc = []
        self.validation_acc = []
        epoch_plot = np.linspace(0,self.num_epochs,self.num_epochs)
        for epoch in range(self.num_epochs):
            print('\n')
            train_loss = 0
            val_loss = 0
            train_accur = 0
            val_accur = 0
            best_loss = float('inf')
            best_acc = float('-inf')
            for i,batch in enumerate(self.train_loader):
                batch = batch.to(self.device)
                train_loss, train_accur, best_loss, best_acc = self.train_step(batch_num=i, batch=batch, loader=self.train_loader, train_loss=train_loss, accur=train_accur, epoch=epoch, best_loss=best_loss, best_acc=best_acc)
            for i,batch in enumerate(self.val_loader):
                batch = batch.to(self.device)
                val_loss, val_accur = self.val_step(batch_num=i, batch=batch, loader=self.val_loader, val_loss=val_loss, accur=val_accur, epoch=epoch)

            self.training_loss.append(train_loss/len(self.train_loader))
            self.validation_loss.append(val_loss/len(self.val_loader))
            self.training_acc.append(train_accur/len(self.train_loader))
            self.validation_acc.append(val_accur/len(self.val_loader))

            torch.save(self.model.state_dict(), 'models/model2')

        fig = plt.figure()
        plt.plot(epoch_plot,self.training_loss, label="training nll loss")
        plt.plot(epoch_plot, self.validation_loss, label="validation nll loss")
        plt.legend()
        plt.savefig('./figs/loss_plot.png')

        fig = plt.figure()

        plt.plot(epoch_plot,self.training_acc, label="training accuracy")
        plt.plot(epoch_plot, self.validation_acc, label="validation accuracy")
        plt.legend()
        plt.savefig('./figs/accuracy_plot.png')



