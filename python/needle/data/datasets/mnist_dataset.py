from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
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

        X, nx, num_r, num_c = gz_open(image_filename, True)
        y, _, _, _ = gz_open(label_filename, False)
        X = X.reshape(nx, num_r * num_c)
        X = X.astype(np.float32)
        X = X / 255

        self.X = X
        self.y = y
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        x = self.X[index]
        if self.transforms:
            x = np.reshape(x, (28, 28, 1))
        x = self.apply_transforms(x)
        y = self.y[index]
        return x, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION