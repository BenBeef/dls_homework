from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, self.axes)
        max_z_r = max_z
        axes = [] if self.axes is None else self.axes
        axes = (axes, ) if type(axes) not in (tuple, list) else axes
        if len(axes) != len(Z.shape):
            # shape=(1, 2, 3, 4), axes=(0, 2) --> new_shape = (1, 2, 1, 4) 填充sum后丢掉的字段为1,其余留下
            shape = [1 if i in axes else v for i, v in enumerate(Z.shape)]
            max_z_r = max_z.reshape(tuple(shape))
        if Z.shape != max_z_r.shape:
            max_z_r = array_api.broadcast_to(max_z_r, Z.shape)
        Z = Z - max_z_r
        Z = array_api.exp(Z)
        Z = array_api.sum(Z, self.axes)
        Z = array_api.log(Z)
        Z = Z + max_z
        return Z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, lhs_ori = node.inputs[0], node.inputs[0] # lhs.shape=(1, 2, 3, 4), axes=(0, 2)  --> out_grad.shape=(2, 4)
        new_shape = None
        if self.axes is not None and len(self.axes) != len(lhs.shape):
            new_shape = [1 if i in self.axes else v for i, v in enumerate(lhs.shape)]
            out_grad = reshape(out_grad, new_shape)
        out_grad = broadcast_to(out_grad, lhs.shape)
        lhs_max = Tensor(array_api.max(lhs.realize_cached_data(), self.axes), device=out_grad.device, dtype=out_grad.dtype)
        if new_shape is not None:
            lhs_max = reshape(lhs_max, new_shape)
        lhs_max = broadcast_to(lhs_max, lhs.shape)
        lhs = exp(lhs + (-1) * lhs_max)
        lhs_sum = summation(lhs, self.axes)
        if new_shape is not None:
            lhs_sum = reshape(lhs_sum, new_shape)
        lhs_sum = broadcast_to(lhs_sum, lhs.shape)
        lhs = divide(lhs, lhs_sum)
        return out_grad * lhs
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

