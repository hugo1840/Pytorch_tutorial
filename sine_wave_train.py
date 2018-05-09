from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib
matplotlib.use('Agg')  # must be before importing pyplot or pylab
import matplotlib.pyplot as plt

# https://pytorch.org/docs/stable/nn.html?highlight=lstmcell#torch.nn.LSTMCell

class Sequence(nn.Module):
    def __init__(self):
        
        super(Sequence, self).__init__()
        
        # nn.LSTMCell(input_size, hidden_size, bias=True)
        # input_size: number of expected features in the input
        # hidden_size: number of features in the hidden state
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        
        # input.size() = (97, 999) at training 
        # input.size(0) = batch = 97, hidden_size = 51
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        
        # lstm1 = nn.LSTMCell(...)
        # h_1, c_1 = lstm1(input, (h_0, c_0))
        # input of shape (batch, input_size): tensor containing input features.
        
        # h_0 of shape (batch, hidden_size): tensor containing the initial 
        # hidden state for each element in the batch.
        # c_0 of shape (batch, hidden_size): tensor containing the initial cell 
        # state for each element in the batch.
        # h_1 of shape (batch, hidden_size): tensor containing the next hidden
        # state for each element in the batch.
        # c_1 of shape (batch, hidden_size): tensor containing the next cell 
        # state for each element in the batch.
        
        # .chunk(chunks, dim=0) splits a tensor into specified number of
        # chunks along the given dimension
        # input.size(1) = length of time series(= 999 at training)
        # here input_t.size() = (batch, input_size) = (97, 1)
        # h_t.size() = (97, 51), c_t.size() = (97, 51)
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            # update of hidden state & cell state of each lstmCell
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]  # outputs is a list of tensors
            
        # this loop is neglected if future == 0
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        
        # .stack(seq, dim) concatenates sequence of tensors along a new
        # dimension. All tensors must be of the same size.
        # .chunk() vs .stack()
        # .squeeze(dim) returns a tensor with the given dimensions
        # of input of size 1 removed.
        # here after concatenation outputs.size() = (97, 999)
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    
    # input = data[3 to the last, the first to the second last]
    # target = data[3 to the last, the second to the last]
    # one-timestep prediction
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    # build the model
    seq = Sequence()
    # print network layout
    print(seq)
    # nn.Nodule.double() casts all floating point parameters and buffers
    # to double datatype.
    seq.double() 
    # 
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    #begin to train
    for i in range(15):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)            
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        # .step() performs a single optimization step
        # for LBFGS algorithm you have to pass in a closure which
        # should clear the gradients, compute the loss, and return it.
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            # we predicted 1999 values here, the first 999 w.r.t test_target
            # and the rest 1000 are generated future values
            future = 1000
            pred = seq(test_input, future=future)
            # test_loss is calculated with only the first 999 predicted values
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            # .detach() returns a new Tensor, detached from the current graph.
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()
    
    # print names of parameters    
    params = seq.state_dict()
    for k,v in params.items():
        print(k)
    # print size or value of a given parameter    
    print(params['lstm1.weight_ih'].size())
    print(params['lstm1.weight_ih'])

