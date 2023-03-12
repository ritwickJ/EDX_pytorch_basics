# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 17:47:08 2023

@author: ritwick
"""

import torch 
import numpy as np

# create a tensor
a = torch.tensor([1,2,5,4,6,7])
# get the data type and type of tensor
print(a.dtype, a.type())
print(a)
print('1-------------------------\n')

# create a tensor
a = torch.tensor([1.0,2.1,5.7,4.0,6.0,7.0])
# get the data type and type of tensor
print(a.dtype, a.type())
print(a)
print('2-------------------------\n')

# create a tensor
a = torch.tensor([1,2,3,4], dtype=torch.float32)
# get the data type and type of tensor
print(a.dtype, a.type())
print(a)
print('3-------------------------\n')


l = [1,2,3,4]
# torch.Tensor() is a class and creates a tensor with the default data type, 
# as defined by torch.get_default_dtype().
a = torch.Tensor(l)
# torch.tensor() will infer data type from the data.
b = torch.tensor(l)
print(a.dtype, a.type())
print(a)
print(b.dtype, b.type())
print(b)
print('4-------------------------\n')


# tensor attributes
a = torch.tensor([[1,2,3],[4,5,6]])
print(a.dtype, a.type(), a.size(), a.ndimension())
print('5-------------------------\n')


# reshaping a tensor
a = torch.tensor([1,2,3,4,5,6])
a1 = a.view(6,1)
print(a.ndimension(), a1.ndimension())
a1 = a.view(-1,2)
print(a.ndimension(), a1.ndimension())
print(a1)
print('6-------------------------\n')

# working with tensors
# converting tensors
# numpy and pytorch
np_arr = np.array([7.0, 1.2, 1.1, 1.7, 2.9])
print(type(np_arr), np_arr)
torch_tensor = torch.from_numpy(np_arr)
print(type(torch_tensor), torch_tensor)
np_arr2 = torch_tensor.numpy()
print(type(np_arr2), np_arr2)
print('-------------------------')

# this is done by pointers
# torch_tensor points to np_arr and np_arr2 points to torch_tensor
np_arr[0] = 0.0
torch_tensor[1] = 0.1
np_arr2[2] = 0.2 
print(np_arr)
print(torch_tensor)
print(np_arr2)
print('-------------------------')

# convert tensor to python list
a = torch.tensor([1,2,3,4])
l = a.tolist()
print(type(a), type(l))
l[0] = 42
print(a, l)
print('-------------------------')

# individual elements of a 1d tensor are also tensors
# to get the python number use .item()
a = torch.tensor([1,2,3,4])
print(type(a[0]), type(a[0].item()))
print(a[0], a[0].item())
print('7-------------------------\n')



a = torch.tensor([1,6,5,4,2,7])
a[0] = 42
b = a[1:3]
print(a, b)

a[3:5] = torch.tensor([0,0])
print(a)
print('8-------------------------\n')



