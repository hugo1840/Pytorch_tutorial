import numpy as np
import torch

np.random.seed(2)

T = 20
L = 1000
N = 100

# np.empty() returns a new array of given shape and type, without
# initializing entries.
x = np.empty((N, L), 'int64')

# np.array(range(10)) = [0, 1, 2, ..., 8, 9]
# randint(low, high, size): Return random integers from low (inclusive)
# to high (exclusive) with output_shape = size.

# addition of 2 arrays of different dimensions in numpy
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)

# ndarray.astype(): Copy of the array, cast to a specified type.
data = np.sin(x / 1.0 / T).astype('float64')

# torch.save(data, file)
# python open(name, mode)
# wb = 以二进制格式打开一个文件只用于写入
# 如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
torch.save(data, open('traindata.pt', 'wb'))
