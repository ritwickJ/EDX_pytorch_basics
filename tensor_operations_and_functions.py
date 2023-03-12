# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 18:28:07 2023

@author: ritwick
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# vector addition
a = torch.tensor([1,0])
b = torch.tensor([0.5,1.2])
z = a+b
print(a,b,z)

# vector multiplication by a scalar
x = torch.tensor([1,2])
z = 42*x
print(x,z)

# vector element vise multiplication
# also called hadamard product
u = torch.tensor([1,2])
v = torch.tensor([1.1, 1.1])
z = u*v
print(z)

# dot product
u = torch.tensor([1,2,3,4,5])
v = torch.tensor([1,1,1,1,1])
z = torch.dot(u,v)
print(z.item())

# using function on tesnors
t = torch.tensor([0,np.pi/2,np.pi])
ft = torch.sin(t)
print(ft)
#print(torch.sin(t), t.sin())
#print(t.max(), torch.max(t))

# linspace
t = torch.linspace(0,10,11)
print(t)

# plotting
x = torch.linspace(0,2*np.pi,100)
y = torch.sin(x)
plt.plot(x.numpy(),y.numpy())

