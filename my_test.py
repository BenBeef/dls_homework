# coding:utf-8
import numpy as np

# 定义一个二维数组 A
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 定义一个一维数组 b
b = np.array([0, 2, 1])

# 使用 b 的元素作为索引获取 A 的元素
result = A[np.arange(len(b)), b]

# 打印获取的结果
print(result)
