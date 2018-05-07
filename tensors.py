# -*- coding: utf-8 -*-
"""
Created on Mon May  7 11:43:53 2018

@author: Hugot
"""

from __future__ import print_function
import torch

# not initialized matrix
x = torch.empty(5, 3)
print(x)

# a randomly intialized matrix
x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# construct a tensor directly from data
x = torch.tensor([5.5, 3])
print(x)

# creating a tensor using existing tensor
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

# override dtype
x = torch.rand_like(x, dtype=torch.float)
print(x)
print(x.size())


