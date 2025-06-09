# coding:utf-8

import numpy as np

arr = np.array([i for i in range(8)])
arr = arr.reshape((1, 8))
print(arr)
arr = arr.reshape((1, 4, 2))
print(arr[:, 0, :])
print(arr[:, 1, :])
print(arr[:, 2, :])
print(arr[:, 3, :])