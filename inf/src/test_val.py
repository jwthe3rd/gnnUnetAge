import torch
import numpy as np

a = torch.Tensor([[1,2],[3,4]])

Re = 100

b = np.repeat(Re, a.shape[1])
b = torch.reshape(torch.Tensor(b), (1,a.shape[1])).T

print(torch.cat((a,b),1))