
import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint  #this imports the n_ode magic


#os.chdir(r"C:\Users\Micha\torchdiffeq\examples\")

#os.chdir(r"C:\Users\Micha\torchdiffeq\examples\")

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
data1 = np.loadtxt("mol_dynam_data.dat")  #This is the ground truth data 
data = torch.from_numpy(data1) #to convert to a tensor in torch

N=4096        #Total number of molecules 2^12?

T=1001      #total number of time steps


#reshape the data into a 3d tensor 
mol_data=data.view(1001,4096,5) #this forms a 2d array for each time instance
#print(mol_data)
true_y0 = mol_data[0,0:4095,0:3]  #the initial condition
t = torch.linspace(0, T, T+1)


#another thing worth trying is to implement SInDYs on the data set

# =============================================================================
# class Lambda(nn.Module):
# 
#     def forward(self, t, y):
#         return torch.mm(y**3, true_A)
# 
# 
# with torch.no_grad():
#     true_y = odeint(Lambda(), true_y0, t, method='dopri5')
#     
# def get_batch():
#     s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
#     batch_y0 = true_y[s]  # (M, D)
#     batch_t = t[:args.batch_time]  # (T)
#     batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
#     return batch_y0, batch_t, batch_y
# 
# =============================================================================





         
         
print("success")