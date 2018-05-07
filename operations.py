# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:06:42 2018

@author: Hugot
"""

from __future__ import print_function
import torch

# addition
x = torch.rand(5, 3, dtype=torch.float)
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

# provide an ouput tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# add x to y in-place
# Any operation that mutates a tensor in-place is post-fixed with an "_"
y.add_(x)
print(y)

# print the second column of x
print(x[:, 1])

# reshape or resize tensors
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# get the value of one-element tensor as a Python number
x = torch.randn(1)
print(x)
print(x.item()) 

