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
        self.early_stop = args.early_stop

    def load_data(self):

        y_train, x_train, e_train, y_val, x_val, e_val = data_generator(path=self.path, seed=self.seed).segment_data()
        training_dataset = gnnAgeDataSet(e_train, x_train, y_train)
        validation_dataset = gnnAgeDataSet(e_val, x_val, y_val)

        self.train_loader = DataLoader(training_dataset, batch_size=self.batch_size)
        self.val_loader = DataLoader(validation_dataset, batch_size=self.batch_size)

        return 0

    def to_device(self):
        self.model.to(self.device)
        return 0

    def train_step(self, batch, batch_num, loader, train_loss, train_loss_tot, accur, epoch, best_loss, best_acc, worst_acc):
        optimizer = self.optimizer(self.model.parameters(),lr=self.lr)
        optimizer.zero_grad()
        out = self.model(batch)
        loss = F.nll_loss(out, batch.y)
        train_loss_tot+=loss.item()
        _, preds = torch.max(out, 1)
        acc = torch.mean((preds == batch.y).float())
        # loss += torch.mean(((preds - batch.y)/1.01)**2)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        accur = acc.item()
        if best_acc < 100*accur:
            best_acc = 100*accur
        if worst_acc > 100*accur:
            worst_acc = 100*accur
        if best_loss > train_loss:
            best_loss = train_loss
        if batch_num % 10 == 0:
            print ("\033[A                             \033[A")
            print(f"Epoch:{epoch} | {int(100*batch_num/len(loader))} % | T NLL Loss = {train_loss:.2f}  Best Loss: {best_loss:.2f}| T Acc = {100*accur:.2f} %  Best Acc: {best_acc:.2f} %  Worst Acc: {worst_acc:.2f}    ")
        return  train_loss_tot / (batch_num + 1), accur, best_loss, best_acc, worst_acc, train_loss_tot
    
    @torch.no_grad()
    def val_step(self, batch, batch_num, loader, val_loss, val_loss_tot, accur, epoch, best_val_acc, worst_val_acc):
        out = self.model(batch)
        loss = F.nll_loss(out, batch.y)
        _, preds = torch.max(out, 1)
        acc = torch.mean((preds == batch.y).float())
        val_loss_tot += loss.item()
        val_loss = loss.item()
        accur = acc.item()
        if best_val_acc < 100*accur:
            best_val_acc = 100*accur
        if worst_val_acc > 100*accur:
            worst_val_acc = 100*accur
        # if best_loss > val_loss:
        #     best_loss = val_loss
        if batch_num % 10 == 0:
            print ("\033[A                             \033[A")
            print(f"Epoch: {epoch} | {int(100*batch_num/len(loader))} % ||| Validation Negative Log-Likelihood Loss = {val_loss:.2f}    ||| Validation Accuracy = {100*accur:.2f} %  | Best Acc: {best_val_acc:.2f} Worst Acc: {worst_val_acc:.2f}")
        return val_loss_tot, val_loss_tot / (batch_num + 1), accur, best_val_acc, worst_val_acc

    @torch.no_grad()
    def val_test(self, batch, batch_num, loader, val_loss, val_loss_tot, accur, epoch, best_val_acc, worst_val_acc):
        out = self.model(batch)
        loss = F.nll_loss(out, batch.y)
        _, preds = torch.max(out, 1)
        acc = torch.mean((preds == batch.y).float())
        val_loss_tot += loss.item()
        val_loss = loss.item()
        accur = acc.item()
        if best_val_acc < 100*accur:
            best_val_acc = 100*accur
        if worst_val_acc > 100*accur:
            worst_val_acc = 100*accur
        # if best_loss > val_loss:
        #     best_loss = val_loss
        print(f'{batch.name} : {100*accur:.2f}')
        # if batch_num % 10 == 0:
        #     print ("\033[A                             \033[A")
        #     print(f"Epoch: {epoch} | {int(100*batch_num/len(loader))} % ||| Validation Negative Log-Likelihood Loss = {val_loss:.2f}    ||| Validation Accuracy = {100*accur:.2f} %  | Best Acc: {best_val_acc:.2f} Worst Acc: {worst_val_acc:.2f}")
        return val_loss_tot, val_loss_tot / (batch_num + 1), accur, best_val_acc, worst_val_acc

    def train(self):
        self.load_data()
        self.to_device()
        self.training_loss = []
        self.validation_loss =[]
        self.training_acc = []
        self.validation_acc = []
        epoch_plot = [] #np.linspace(0,self.num_epochs,self.num_epochs)
        count = 0
        # self.model.load_state_dict(torch.load('models/model5_ckpt'))
        print('start')
        for epoch in range(self.num_epochs):
            epoch_plot.append(epoch)
            print('\n')
            train_loss = 0
            val_loss = 0
            train_accur = 0
            val_accur = 0
            prev_val_loss = 0
            val_loss_tot = 0
            train_loss_tot = 0
            best_loss = float('inf')
            best_acc = float('-inf')
            worst_acc = float('inf')
            worst_val_acc = float('inf')
            best_val_acc = float('-inf')
            for i,batch in enumerate(self.train_loader):
                batch = batch.to(self.device)
                train_loss, train_accur, best_loss, best_acc, worst_acc, train_loss_tot = self.train_step(batch_num=i, batch=batch, loader=self.train_loader, train_loss=train_loss, accur=train_accur, epoch=epoch, best_loss=best_loss, best_acc=best_acc, worst_acc=worst_acc, train_loss_tot=train_loss_tot)
            print('\n')
            self.model.eval()
            for i,batch in enumerate(self.val_loader):
                batch = batch.to(self.device)
                val_loss_tot, val_loss, val_accur, best_val_acc, worst_val_acc = self.val_step(batch_num=i, batch=batch, loader=self.val_loader, val_loss=val_loss, val_loss_tot=val_loss_tot, accur=val_accur, epoch=epoch, best_val_acc=best_val_acc, worst_val_acc=worst_val_acc)
            self.model.train()
            if val_loss >= prev_val_loss:
                count+=1
                prev_val_loss = val_loss
            else:
                count = 0
                prev_val_loss = val_loss

            self.training_loss.append(train_loss)
            self.validation_loss.append(val_loss)
            self.training_acc.append(train_accur)
            self.validation_acc.append(val_accur)

            if count >= self.early_stop:
                break
            torch.save(self.model.state_dict(), 'models/model6_ckpt')

        torch.save(self.model.state_dict(), 'models/model6')

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



