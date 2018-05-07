# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:27:27 2018

@author: Hugot
"""

from __future__ import print_function
import torch

# converting a torch tensor to a numpy array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# The Torch Tensor and NumPy array will share their underlying memory 
# locations, and changing one will change the other
a.add_(1)
print(a)
print(b)

# converting numpy array to torch tensor 
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)

# changing the np array changed the Torch Tensor automatically
np.add(a, 1, out=a)
print(a)
print(b)