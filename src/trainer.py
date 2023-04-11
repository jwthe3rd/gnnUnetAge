import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import DataGenerator, gnnAgeDataSet
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

"""
This is the file that holds the Trainer class, which contains and handles most of the boiler plate code for running, observing, tracking, and plotting values
from the training run.

Referenced in the main.py file
"""


class Trainer:

    def __init__(self, args, model, optimizer, device):

        self.batch_size = args.batch_size
        self.path = args.data_path
        self.num_epochs = args.num_epochs
        self.model = model
        self.device = device
        self.seed = args.seed
        self.optimizer = optimizer
        self.lr = args.lr
        self.early_stop = args.early_stop
        self.optim = self.optimizer(self.model.parameters(), lr=self.lr) # Sets the optimizer
        self.model.to(self.device) # Sends the model to the specified device from the config file
        self.TRAIN_RUN = f'model_lr_{self.lr}_depth_{model.depth}_k_{model.k_p}'

    def load_data(self):
        """
        Loads in training data from the DataGenerator object in the utils/dataset.py file. Then this data is setup as a gnnAgeDataSet, also from the dataset.py file.

        Finally the dataset is input as a DataLoader, to allow for on the fly loading of data. 
        """
        y_train, x_train, e_train, y_val, x_val, e_val = DataGenerator(path=self.path, seed=self.seed).segment_data()
        training_dataset = gnnAgeDataSet(e_train, x_train, y_train)
        validation_dataset = gnnAgeDataSet(e_val, x_val, y_val)

        self.train_loader = DataLoader(training_dataset, batch_size=self.batch_size)
        self.val_loader = DataLoader(validation_dataset, batch_size=self.batch_size)

        return 0

    def train_step(self, batch, batch_num, loader, train_loss, train_loss_tot, accur, epoch, best_loss, best_acc, worst_acc):
        """
        Runs the training step, called for every batch in each epoch.
        """
        self.optim.zero_grad() # Zeroes out the gradients from the previous batch
        out = self.model(batch) # generates hypothesis
        loss = F.nll_loss(out, batch.y) # calculates loss based on negative log likelihood l(theta) = -sum_{i=1}^{n}(y_i\log(\hat{y_{theta,i}}) + (1-y_i)\log(1-\hat{y_{theta,i}}))
        train_loss_tot+=loss.item()
        _, preds = torch.max(out, 1) # gets the prediction
        acc = torch.mean((preds == batch.y).float()) # computes the accuracy
        loss.backward() # send back the gradient to compute local gradients
        self.optim.step() # alter the weights and biases according to the gradients
        train_loss = loss.item() #for
        accur = acc.item()          #printing
        if best_acc < 100*accur:
            best_acc = 100*accur
        if worst_acc > 100*accur:
            worst_acc = 100*accur
        if best_loss > train_loss:
            best_loss = train_loss
        if batch_num % 10 == 0:
            print ("\033[A                             \033[A")
            print(f"Epoch:{epoch} T | {int(100*batch_num/len(loader))} % | L = {train_loss:.2f}  Best L: {best_loss:.2f}| A = {100*accur:.2f} % B A: {best_acc:.2f} % W A: {worst_acc:.2f}")
        return  train_loss_tot / (batch_num + 1), accur, best_loss, best_acc, worst_acc, train_loss_tot

    @torch.no_grad()
    def val_step(self, batch, batch_num, loader, val_loss, val_loss_tot, accur, epoch, best_val_acc, worst_val_acc):
        """
        All the same ideas as in the train step, only not tracking gradients or calling loss.backward(), optim.step()
        """
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
        if batch_num % 10 == 0:
            print ("\033[A                                               \033[A")
            print(f"Epoch: V {epoch} | {int(100*batch_num/len(loader))} % | L = {val_loss:.2f}  | A = {100*accur:.2f} % | B A: {best_val_acc:.2f} % W A: {worst_val_acc:.2f}%")
        return val_loss_tot, val_loss_tot / (batch_num + 1), accur, best_val_acc, worst_val_acc

    @torch.no_grad()
    def val_test(self, batch, batch_num, loader, val_loss, val_loss_tot, accur, epoch, best_val_acc, worst_val_acc):
        """
        This function is specifically for running with a batch size of 1 to see the case and its prediction accuracy
        """
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
        print(f'{batch.name} : {100*accur:.2f}')
        return val_loss_tot, val_loss_tot / (batch_num + 1), accur, best_val_acc, worst_val_acc

    def train(self):
        """
        All the boiler plate code for performing the training run
        """
        self.load_data()
        self.training_loss = []
        self.validation_loss =[]
        self.training_acc = []
        self.validation_acc = []
        epoch_plot = [] 
        count = 0
        prev_val_loss = 0
        #self.model.load_state_dict(torch.load('models/model6'))
        #print(self.TRAIN_RUN)
        #print('start')
        for epoch in range(self.num_epochs):
            epoch_plot.append(epoch)
            print('\n')
            """ ---- All of this is for tracking loss/accuracy for plotting and early stopping -------"""
            train_loss = 0
            val_loss = 0
            train_accur = 0
            val_accur = 0
            val_loss_tot = 0
            train_loss_tot = 0
            best_loss = float('inf')
            best_acc = float('-inf')
            worst_acc = float('inf')
            worst_val_acc = float('inf')
            best_val_acc = float('-inf')
            """---------------------------------------------------------"""

            """ --- Train Step --- """
            for i,batch in enumerate(self.train_loader):
                batch = batch.to(self.device)
                train_loss, train_accur, best_loss, best_acc, worst_acc, train_loss_tot = self.train_step(batch_num=i, batch=batch, loader=self.train_loader, train_loss=train_loss, accur=train_accur, epoch=epoch, best_loss=best_loss, best_acc=best_acc, worst_acc=worst_acc, train_loss_tot=train_loss_tot)
            print('\n')
            """ ----------------------"""
            #self.model.eval()
            """ ---- Validation Step -----"""
            for i,batch in enumerate(self.val_loader):
                batch = batch.to(self.device)
                val_loss_tot, val_loss, val_accur, best_val_acc, worst_val_acc = self.val_step(batch_num=i, batch=batch, loader=self.val_loader, val_loss=val_loss, val_loss_tot=val_loss_tot, accur=val_accur, epoch=epoch, best_val_acc=best_val_acc, worst_val_acc=worst_val_acc)
            # self.model.train()
            """ -------------------------------"""
            if val_loss >= prev_val_loss:           ### Early
                count+=1                            ### Stopping
                prev_val_loss = val_loss            ### Stuff
            else:
                count = 0
                prev_val_loss = val_loss

            self.training_loss.append(train_loss)
            self.validation_loss.append(val_loss)
            self.training_acc.append(train_accur)
            self.validation_acc.append(val_accur)

            if count >= self.early_stop:
                break
            torch.save(self.model.state_dict(), f'models/final_{self.TRAIN_RUN}_ckpt')

        torch.save(self.model.state_dict(), f'models/final_{self.TRAIN_RUN}')

        """ -- Plotting training results ----"""

        fig = plt.figure()
        plt.plot(epoch_plot,self.training_loss, label="training nll loss")
        plt.plot(epoch_plot, self.validation_loss, label="validation nll loss")
        #plt.plot([0], [0], label=f"Loss: {self.validation_loss[-1]:.2f} + N: {len(epoch_plot)}")
        plt.legend()
        plt.savefig(f'./figs/final_{self.TRAIN_RUN}_loss.png')

        fig = plt.figure()

        plt.plot(epoch_plot,self.training_acc, label="training accuracy")
        plt.plot(epoch_plot, self.validation_acc, label="validation accuracy")
        #plt.plot([0], [0], label=f"Acc: {self.validation_acc[-1]:.2f}")
        plt.legend()
        plt.savefig(f'./figs/final_{self.TRAIN_RUN}_accuracy.png')
        """---------------------------------------"""



