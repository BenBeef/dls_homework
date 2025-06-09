"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import needle.init as init
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar, dtype=array_api.float32)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
                node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a ** b) * array_api.log(a.data)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b, dtype=array_api.float32)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad * power_scalar(rhs, -1), -1 * out_grad * lhs * power_scalar(rhs, -2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar, dtype=array_api.float32)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        self.axes_format = None

    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        self.axes = [a.ndim - 2, a.ndim - 1] if self.axes is None else self.axes
        if len(self.axes) == 2:
            axes = [i for i in range(a.ndim)]
            axes[self.axes[0]], axes[self.axes[1]] = self.axes[1], self.axes[0]
            self.axes_format = axes
        else:
            assert a.ndim == len(self.axes), "full ndim"
            self.axes_format = self.axes
        return array_api.transpose(a, self.axes_format)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        axes = [(ax, i) for i, ax in enumerate(self.axes_format)]
        axes.sort(key=lambda x: x[0])
        axes = [i for _, i in axes]
        return array_api.transpose(out_grad, axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if tuple(a.shape) == tuple(self.shape):
            return a
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, shape = node.inputs[0], node.inputs[0].shape
        if tuple(shape) == tuple(self.shape):
            return out_grad
        shape = [1] * (len(self.shape) - len(shape)) + list(shape)
        axes = tuple(i for i, v in enumerate(shape) if v != self.shape[i])
        g = summation(out_grad, axes=axes)
        return reshape(g, lhs.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        if self.axes is None:
            k = array_api.prod(lhs.shape)
            new_shape = (k,)
            out_grad = broadcast_to(out_grad, new_shape)
            out_grad = reshape(out_grad, lhs.shape)
            return out_grad
        if self.axes == ():
            return out_grad
        self.axes = tuple(self.axes) if isinstance(self.axes, (list, tuple)) else (self.axes,)
        shape = [1 if i in self.axes else v for i, v in enumerate(lhs.shape)]
        out_grad = reshape(out_grad, shape)
        return broadcast_to(out_grad, lhs.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        g1, g2 = matmul(out_grad, transpose(rhs)), matmul(transpose(lhs), out_grad)
        if len(g1.shape) > len(lhs.shape) > 0:  # 左侧维度 < 右侧维度 --> 右侧做batch
            g1 = summation(g1, axes=tuple(i for i in range(len(g1.shape) - len(lhs.shape))))
        if len(g2.shape) - len(rhs.shape) > 0:  # 左侧维度 > 右侧维度 --> 左侧做batch
            g2 = summation(g2, axes=tuple(i for i in range(len(g2.shape) - len(rhs.shape))))
        return g1, g2
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -1 * out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return divide(out_grad, lhs)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad, node)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):

    def rule(self, a):
        zeros = a * 0
        x = array_api.maximum(a, zeros)
        return x

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        x = self.rule(a)
        return x
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        data = node.inputs[0].realize_cached_data()
        pos = data > 0
        tens = Tensor(pos, dtype="float32", device=data.device)
        return multiply(out_grad, tens)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        _2x = x * 2
        e_2x = exp(_2x)
        e_2x_1 = e_2x + 1
        e_2x_sq = e_2x_1 * e_2x_1
        e_2x_d = e_2x / e_2x_sq
        res = e_2x_d * 4
        return multiply(out_grad, res)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis
        self.l = None

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        def sl_n():
            return slice(None, None, None)

        self.l, shape, arg_0 = len(args), list(args[0].shape), args[0]
        shape.insert(self.axis, 1)
        arr = array_api.reshape(arg_0, shape)
        shape[self.axis] = self.l
        arr = array_api.broadcast_to(arr, shape)
        arr = array_api.compact(arr)
        for i in range(self.l):
            arg = args[i]
            idx = tuple([sl_n() if s != self.axis else slice(i, i + 1, None) for s in range(len(shape))])
            arg = array_api.compact(arg)
            arr[idx] = arg
        return arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        def sl_n():
            return slice(None, None, None)

        # shape, data = out_grad.shape, out_grad.realize_cached_data()
        shape, data = out_grad.shape, out_grad.realize_cached_data()
        ori_shape = node.inputs[0][0].shape
        result = []
        for i in range(self.l):
            idx = tuple([sl_n() if s != self.axis else slice(i, i + 1, None) for s in range(len(shape))])
            d = data[idx]
            shape_copy = list(ori_shape).copy()
            shape_copy.insert(self.axis, 1)
            d = array_api.broadcast_to(d, shape_copy)
            d = array_api.reshape(d, ori_shape)
            result.append(Tensor(d, device=node.device))
        return make_tuple(*result)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis
        self.len = 1

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        shape = A.shape
        self.len = shape[self.axis]
        new_shape = tuple([s for i, s in enumerate(shape) if i != self.axis])
        arr = []
        sls = [slice(None, None, None) for _ in range(len(shape))]
        for i in range(self.len):
            sls[self.axis] = slice(i, i+1, None)
            item = A[tuple(sls)]
            item = array_api.compact(item)
            item = array_api.reshape(item, new_shape)
            item = array_api.compact(item)
            arr.append(item)
        return tuple(arr)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        Z = array_api.full(lhs.shape, 0, device=lhs.device)
        sls = [slice(None, None, None) for _ in range(len(lhs.shape))]
        data = out_grad.realize_cached_data()
        for i in range(self.len):
            sls[self.axis] = slice(i, i+1, None)
            Z[tuple(sls)] = data[i]
        return Tensor(Z, dtype=Z.dtype, device=Z.device)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


def pos_2_slices(pos, dim, n):
    res = [slice(None, None, None) if i != dim else slice(pos, pos + 1, None) for i in range(n)]
    return tuple(res)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0 or len(self.axes) == 0:
            return a
        if self.dilation == 0 or len(self.axes) == 0:
            return a
        new_arr, axes, dilation = a, self.axes, self.dilation
        for axis in axes:
            shape = [s * (1 + dilation if axis == i else 1) for i, s in enumerate(new_arr.shape)]
            zeros = array_api.full(shape, 0, dtype=a.dtype, device=a.device)
            for i in range(0, shape[axis], dilation + 1):
                sls_0, sls_1 = pos_2_slices(i, axis, len(shape)), pos_2_slices(i // (dilation + 1), axis, len(shape))
                zeros[sls_0] = new_arr[sls_1]
            new_arr = zeros
        return new_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_arr, axes, dilation = a, self.axes, self.dilation
        for axis in axes:
            shape = [s // (1 + dilation if axis == i else 1) for i, s in enumerate(new_arr.shape)]
            zeros = array_api.full(shape, 0, dtype=a.dtype, device=a.device)
            for i in range(0, shape[axis]):
                sls_0, sls_1 = pos_2_slices(i, axis, len(shape)), pos_2_slices(i * (dilation + 1), axis, len(shape))
                zeros[sls_0] = new_arr[sls_1]
            new_arr = zeros
        return new_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        '''
            N,H,W,C_in = Z.shape
            K,_,_,C_out = weight.shape
            Ns, Hs, Ws, Cs = Z.strides

            inner_dim = K * K * C_in
            A = np.lib.stride_tricks.as_strided(Z, shape = (N, H-K+1, W-K+1, K, K, C_in),
                                                strides = (Ns, Hs, Ws, Hs, Ws, Cs)).reshape(-1,inner_dim)
            out = A @ weight.reshape(-1, C_out)
            return out.reshape(N,H-K+1,W-K+1,C_out)
        '''
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides

        stride = self.stride
        inner_dim = K * K * C_in
        h_len = (H - K) // self.stride + 1
        w_len = (W - K) // self.stride + 1
        Z = array_api.as_strided(A, shape=(N, h_len, w_len, K, K, C_in),
                                 strides=(Ns, stride * Hs, stride * Ws, Hs, Ws, Cs))
        Z = array_api.compact(Z)
        Z = Z.reshape((-1, inner_dim))
        Z = array_api.compact(Z)
        B = array_api.compact(B.reshape((-1, C_out)))
        out = Z @ B
        out = out.reshape((N, h_len, w_len, C_out))
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # out_grad = N, H-k+1, W-k+1, C_out
        # X.grad = X = N, H, W, C_in
        # W.grad = W = K, K, C_in, C_out
        X, W = node.inputs
        out_grad_d = dilate(out_grad, axes=(1, 2), dilation=2)  # TODO dilation=2 不正确
        W_T = transpose(W, (0, 1, 3, 2))
        X_grad = conv(out_grad_d, W_T, self.stride, self.padding)

        # TODO
        return X_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
