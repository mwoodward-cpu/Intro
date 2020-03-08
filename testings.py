
import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from torchdiffeq import odeint
# =============================================================================
# T=1001
# 
# t = torch.linspace(0, T, T+1)
# 
# def log_softmax(x):
#     return x - x.exp().sum(-1).log().unsqueeze(-1)
# 
# y=log_softmax(t)
# 
# plt.plot(t,y)
# plt.ylabel('log_soft')
# plt.show()
# 
# 
# 
# print(t)
# print(y)
# =============================================================================

# =============================================================================
# k=1.1
# def F(t, y):
#     return k*y   #simple exponential solution y=e^kt
# 
# t = torch.linspace(0,5,100)   #discretized time domain
# y0 =  torch.tensor(1.0)  # the initial condition
# ysol = odeint(F, y0, t)
# ysol = np.array(ysol).flatten()
# =============================================================================

x=torch.arange(32)
z=x.view(16,2)

W=z.view(2,4,4)
print(z)
print(W)