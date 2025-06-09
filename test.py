# coding:utf-8

import numpy as np

def strides_f(ori: tuple):
    return tuple([i // 4 for i in ori])


def print_st(ori):
    print(strides_f(ori))


arr = np.array([1 + i for i in range(12)])
arr = arr.reshape((3, 1, 4))

print(arr.shape)
print_st(arr.strides)

arr = np.broadcast_to(arr, (3, 2, 4))

print()
print(arr.shape)
print_st(arr.strides)

arr = (3, 1, 4)

cnt = 11
target = (2, 0, 3)

arr = (3, 2, 4)

cnt = 11
target = (1, 0, 3)


for i in range(3-1, -1, -1):
    k = cnt % arr[i]
    cnt = cnt // arr[i]
    print(k)

