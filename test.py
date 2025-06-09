# coding:utf-8
# reads off the underlying data array in order (i.e., offset 0, offset 1, ..., offset n)
# i.e., ignoring strides
#
# import needle as ndl
# import numpy as np
# from needle import nn
# from matplotlib import pyplot as plt
import numpy as np

a = np.arange(16).reshape(4, 4)
b = np.arange(4, 0, -1).reshape(4)
c = a + b
print(a)
print(b)
print(c)