import numpy as np


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.flip(img, axis=1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding + 1, size=2)
        ### BEGIN YOUR SOLUTION
        x_len, y_len = img.shape[0], img.shape[1]
        img_z = np.zeros(img.shape, dtype=img.dtype)
        for i in range(x_len):
            if i + shift_x < 0 or i + shift_x >= x_len:
                continue
            for j in range(y_len):
                if j + shift_y < 0 or j + shift_y >= y_len:
                    continue
                img_z[i, j, :] = img[i + shift_x, j + shift_y, :]
        return img_z
        ### END YOUR SOLUTION
