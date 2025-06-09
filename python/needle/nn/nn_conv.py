"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        fan_in = in_channels * out_channels
        weight = init.kaiming_uniform(fan_in, fan_in, shape=(kernel_size, kernel_size, in_channels, out_channels),
                                      device=device, dtype=dtype)
        self.weight = Parameter(weight)

        self.bias = None
        if bias:
            sqrt_k = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
            self.bias = Parameter(init.rand(out_channels, low=-sqrt_k, high=sqrt_k, device=device, dtype=dtype,
                                            requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x = (N, C, H, W)
        x = ops.transpose(x, (0, 2, 3, 1))
        pad = self.kernel_size // 2
        x = ops.conv(x, self.weight, stride=self.stride, padding=pad)  # (H, H_, W_, C_out)
        if self.bias:
            bias = ops.reshape(self.bias, (1, 1, 1, self.out_channels))
            bias = ops.broadcast_to(bias, x.shape)
            x += bias
        x = ops.transpose(x, (0, 3, 1, 2))  # -> (N, C, H, W)
        return x
        ### END YOUR SOLUTION
