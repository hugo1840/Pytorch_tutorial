# -*- coding: utf-8 -*-
"""
Neural Networks
===============

Neural networks can be constructed using the ``torch.nn`` package.

Now that you had a glimpse of ``autograd``, ``nn`` depends on
``autograd`` to define models and differentiate them.
An ``nn.Module`` contains layers, and a method ``forward(input)``\ that
returns the ``output``.

For example, look at this network that classifies digit images:

.. figure:: /_static/img/mnist.png
   :alt: convnet

   convnet

It is a simple feed-forward network. It takes the input, feeds it
through several layers one after the other, and then finally gives the
output.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or
  weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule:
  ``weight = weight - learning_rate * gradient``

Define the network 
------------------
# https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d

Let’s define this network: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # conv1: in_channels = 1, out_channels = 6, kernel_size = 5
        # if kernel_size is a single int, height = width = kernel_size
        # so here conv1 got a 5x5 square convolution kernel
        # By default: stride = 1, padding = 0, bias = True
        # shape of weight = out_channels * in_channels * height * width
        # shape of bias = out_channels
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        # int_features = size of input samples = 16*5*5
        # out_feature = size of output features = 120
        # shape of weight: out_features * int_features
        # bias = True
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # fc3: out_features = 10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # functional.relu: element-wise
        # Max pooling over a (2, 2) window
        # max_pool2d(input, kernel_size, stride=NOne, padding=0)
        # kernel_size = the size of the window to take a max over
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net() 
print(net)

########################################################################
# You just have to define the ``forward`` function, and the ``backward``
# function (where gradients are computed) is automatically defined for you
# using ``autograd``.
# You can use any of the Tensor operations in the ``forward`` function.
#
# The learnable parameters of a model are returned by ``net.parameters()``

params = list(net.parameters())
print(len(params))
print("params-0: ", params[0].size())  # conv1's .weight: 6*1*5*5
print("params-1: ", params[1].size())  # conv1's .bias: 6
print("params-4: ", params[4].size())  # fc1's weight: 120*400
print("params-5: ", params[5].size())  # fc1's bias: 120


########################################################################
# Let try a random 32x32 input
# Note: Expected input size to this net(LeNet) is 32x32. To use this net on
# MNIST dataset, please resize the images from the dataset to 32x32.

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

########################################################################
# Zero the gradient buffers of all parameters and backprops with random
# gradients:
net.zero_grad()  # set gradients of all model parameters to zero 
out.backward(torch.randn(1, 10))

########################################################################
# .. note::
#
#     ``torch.nn`` only supports mini-batches. The entire ``torch.nn``
#     package only supports inputs that are a mini-batch of samples, and not
#     a single sample.
#
#     For example, ``nn.Conv2d`` will take in a 4D Tensor of
#     ``nSamples x nChannels x Height x Width``.
#
#     If you have a single sample, just use ``input.unsqueeze(0)`` to add
#     a fake batch dimension.
#
# Before proceeding further, let's recap all the classes you’ve seen so far.
#
# **Recap:**
#   -  ``torch.Tensor`` - A *multi-dimensional array* with support for autograd
#      operations like ``backward()``. Also *holds the gradient* w.r.t. the
#      tensor.
#   -  ``nn.Module`` - Neural network module. *Convenient way of
#      encapsulating parameters*, with helpers for moving them to GPU,
#      exporting, loading, etc.
#   -  ``nn.Parameter`` - A kind of Tensor, that is *automatically
#      registered as a parameter when assigned as an attribute to a*
#      ``Module``.
#   -  ``autograd.Function`` - Implements *forward and backward definitions
#      of an autograd operation*. Every ``Tensor`` operation, creates at
#      least a single ``Function`` node, that connects to functions that
#      created a ``Tensor`` and *encodes its history*.
#
# **At this point, we covered:**
#   -  Defining a neural network
#   -  Processing inputs and calling backward
#
# **Still Left:**
#   -  Computing the loss
#   -  Updating the weights of the network
#
# Loss Function
# -------------
# A loss function takes the (output, target) pair of inputs, and computes a
# value that estimates how far away the output is from the target.
#
# There are several different
# `loss functions <http://pytorch.org/docs/nn.html#loss-functions>`_ under the
# nn package .
# A simple loss is: ``nn.MSELoss`` which computes the mean-squared error
# between the input and the target.
#
# For example:

output = net(input)
target = torch.arange(1, 11)  # a dummy target: 1,2,3,...,9,10
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

########################################################################
# Now, if you follow ``loss`` in the backward direction, using its
# ``.grad_fn`` attribute, you will see a graph of computations that looks
# like this:
#
# ::
#
#     input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#           -> view -> linear -> relu -> linear -> relu -> linear
#           -> MSELoss
#           -> loss
#
# So, when we call ``loss.backward()``, the whole graph is differentiated
# w.r.t. the loss, and all Tensors in the graph that has ``requres_grad=True``
# will have their ``.grad`` Tensor accumulated with the gradient.
#
# For illustration, let us follow a few steps backward:

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

########################################################################
# Backprop
# --------
# To backpropagate the error all we have to do is to ``loss.backward()``.
# You need to clear the existing gradients though, else gradients will be
# accumulated to existing gradients.
#
#
# Now we shall call ``loss.backward()``, and have a look at conv1's bias
# gradients before and after the backward.


net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

########################################################################
# Now, we have seen how to use loss functions.
#
# **Read Later:**
#
#   The neural network package contains various modules and loss functions
#   that form the building blocks of deep neural networks. A full list with
#   documentation is `here <http://pytorch.org/docs/nn>`_.
#
# **The only thing left to learn is:**
#
#   - Updating the weights of the network
#
# Update the weights
# ------------------
# The simplest update rule used in practice is the Stochastic Gradient
# Descent (SGD):
#
#      ``weight = weight - learning_rate * gradient``
#
# We can implement this using simple python code:
#
# .. code:: python
#
#     learning_rate = 0.01
#     for f in net.parameters():
#         f.data.sub_(f.grad.data * learning_rate)
#
# However, as you use neural networks, you want to use various different
# update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc.
# To enable this, we built a small package: ``torch.optim`` that
# implements all these methods. Using it is very simple:

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update


###############################################################
# .. Note::
#
#       Observe how gradient buffers had to be manually set to zero using
#       ``optimizer.zero_grad()``. This is because gradients are accumulated
#       as explained in `Backprop`_ section.
