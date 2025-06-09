# coding:utf-8
import numpy as np
import gzip


def gz_open(_f, is_x):
    with gzip.open(_f, 'rb') as f:
        data = f.read()
        _ = int.from_bytes(data[0:4], 'big')
        num = int.from_bytes(data[4:8], 'big')
        rows, cols = -1, -1
        if is_x:
            rows = int.from_bytes(data[8:12], 'big')
            cols = int.from_bytes(data[12:16], 'big')
            data = data[16:]
        else:
            data = data[8:]
        data = np.frombuffer(data, dtype=np.uint8)
        return data, num, rows, cols


gz_file_1 = 'data/train-images-idx3-ubyte.gz'
gz_file_2 = 'data/train-labels-idx1-ubyte.gz'

X, nx, num_r, num_c = gz_open(gz_file_1, True)
Y, ny, _, _ = gz_open(gz_file_1, True)
X = X.reshape(nx, num_r, num_c)
X = X.astype(np.float32)
X = X / 255
print(X.shape)
print(X[1, 0:10, 0:28])