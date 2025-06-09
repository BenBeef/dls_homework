import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(
            self,
            base_folder: str,
            train: bool,
            p: Optional[int] = 0.5,
            transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """

        ### BEGIN YOUR SOLUTION
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
            return data

        def read_cifar_10(dataset):
            data = dataset[b"data"]
            data = data.reshape(-1, 3, 32, 32)
            labels = dataset[b"labels"]
            return data, labels

        import os
        if train:
            data_1, labels_1 = read_cifar_10(unpickle(os.path.join(base_folder, "data_batch_1")))
            data_2, labels_2 = read_cifar_10(unpickle(os.path.join(base_folder, "data_batch_2")))
            data_3, labels_3 = read_cifar_10(unpickle(os.path.join(base_folder, "data_batch_3")))
            data_4, labels_4 = read_cifar_10(unpickle(os.path.join(base_folder, "data_batch_4")))
            data_5, labels_5 = read_cifar_10(unpickle(os.path.join(base_folder, "data_batch_5")))
            X = np.stack([data_1, data_2, data_3, data_4, data_5], axis=0)
            X = X.reshape((-1, 3, 32, 32))
            Y = np.concatenate([labels_1, labels_2, labels_3, labels_4, labels_5], axis=0)
        else:
            X, Y = read_cifar_10(unpickle(os.path.join(base_folder, "test_batch")))
        self.X = X
        self.y = Y
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        x = self.X[index]
        x = self.apply_transforms(x)
        y = self.y[index]
        return x, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
