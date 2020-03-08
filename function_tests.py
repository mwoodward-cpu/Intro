# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:19:47 2020

@author: Micha
"""


import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint  #this imports the n_ode magic


#os.chdir(r"C:\Users\Micha\torchdiffeq\examples\")

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
data1 = np.loadtxt("mol_dynam_data.dat")  #This is the ground truth data 
data = torch.from_numpy(data1) #to convert to a tensor in torch

N=4096        #Total number of molecules 2^12?

T=1001      #total number of time steps


#reshape the data into a 3d tensor where
mol_data=data.view(1001,4096,5) #this forms a 2d array for each time instance
print(mol_data)
true_y0 = data[0,0:4095,1:4]  #the initial condition
t = torch.linspace(0, T, T+1)
print(true_y0)

print("success")


