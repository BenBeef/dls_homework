"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import operator
from functools import reduce


def prod(x):
    return reduce(operator.mul, x, 1)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-1 * x)) ** -1
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.activator = ops.relu if nonlinearity == 'relu' else ops.tanh
        self.bias = bias

        ### BEGIN YOUR SOLUTION
        sqrt_k = (1 / hidden_size) ** 0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-sqrt_k, high=sqrt_k, device=device, dtype=dtype,
                                        requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-sqrt_k, high=sqrt_k, device=device, dtype=dtype,
                                        requires_grad=True))
        self.bias_shape = (1, hidden_size)
        self.hidden_size = hidden_size
        self.device = device
        if bias:
            bias_ih = init.rand(hidden_size, low=-sqrt_k, high=sqrt_k, device=device, dtype=dtype, requires_grad=True)
            self.bias_ih = Parameter(ops.reshape(bias_ih, (1, hidden_size)))
            bias_hh = init.rand(hidden_size, low=-sqrt_k, high=sqrt_k, device=device, dtype=dtype, requires_grad=True)
            self.bias_hh = Parameter(ops.reshape(bias_hh, (1, hidden_size)))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # h′=tanh(xWih+bih+hWhh+bhh) . If nonlinearity is 'relu', then ReLU is used in place of tanh.
        Z = X @ self.W_ih  # m x input_size, input_size x hidden_size -> m x hidden_size
        if not h:
            h = init.zeros(*(X.shape[0], self.hidden_size), requires_grad=True, device=self.device)
        Z += h @ self.W_hh
        if self.bias:
            self.bias_ih = ops.reshape(self.bias_ih, self.bias_shape)
            bias_hi = ops.broadcast_to(self.bias_ih, Z.shape)
            self.bias_hh = ops.reshape(self.bias_hh, self.bias_shape)
            bias_hh = ops.broadcast_to(self.bias_hh, Z.shape)
            Z += bias_hi + bias_hh
        Z = self.activator(Z)
        return Z
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None,
                 dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = [RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias, nonlinearity, device, dtype)
                          for i in range(num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h0:
            h_s = ops.split(h0, 0)
            h_s = [item for item in h_s]
        else:
            h_s = [None for _ in range(len(self.rnn_cells))]
        x_arr = ops.split(X, 0)
        output = []
        for x in x_arr:
            for i, layer in enumerate(self.rnn_cells):
                _h = h_s[i] if h_s[i] else None
                h_s[i] = layer.forward(x if i == 0 else h_s[i - 1], _h)
            output.append(h_s[-1])
        output = ops.stack(output, 0)
        h_n = ops.stack(h_s, 0)
        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.bias = bias

        ### BEGIN YOUR SOLUTION
        s_k = (1 / hidden_size) ** 0.5
        _hid_4 = hidden_size * 4
        self.hidden_size = hidden_size
        self.device = device

        def build(*shape):
            dt = init.rand(*shape, low=-s_k, high=s_k, device=device, dtype=dtype, requires_grad=True)
            return Parameter(dt)

        self.W_ih = build(input_size, _hid_4)
        self.W_hh = build(hidden_size, _hid_4)
        if self.bias:
            self.bias_ih = build(_hid_4)  # 1 x (4*h)
            self.bias_hh = build(_hid_4)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """

        ### BEGIN YOUR SOLUTION

        if not h:
            H = init.zeros(*(X.shape[0], self.hidden_size), requires_grad=True, device=self.device)
            C = init.zeros(*(X.shape[0], self.hidden_size), requires_grad=True, device=self.device)
        else:
            H, C = h[0], h[1]

        def func(activate, w_ih, w_hh, b_ih, b_hh):
            Z = X @ w_ih
            Z += H @ w_hh
            if self.bias:
                b_ih = ops.reshape(b_ih, (1, b_ih.shape[0]))
                b_hh = ops.reshape(b_hh, (1, b_hh.shape[0]))
                Z += ops.broadcast_to(b_ih, Z.shape) + ops.broadcast_to(b_hh, Z.shape)
            return activate(Z)

        new_shape0 = (self.W_ih.shape[0], 4, self.W_ih.shape[1] // 4)
        new_shape0_1 = (self.W_hh.shape[0], 4, self.W_hh.shape[1] // 4)

        w_ihs = ops.split(ops.reshape(self.W_ih, new_shape0), axis=1)
        w_hhs = ops.split(ops.reshape(self.W_hh, new_shape0_1), axis=1)
        w_ih_i, w_ih_f, w_ih_g, w_ih_o = w_ihs[0], w_ihs[1], w_ihs[2], w_ihs[3]
        w_hh_i, w_hh_f, w_hh_g, w_hh_o = w_hhs[0], w_hhs[1], w_hhs[2], w_hhs[3]

        b_ih_i, b_ih_f, b_ih_g, b_ih_o = None, None, None, None
        b_hh_i, b_hh_f, b_hh_g, b_hh_o = None, None, None, None
        if self.bias:
            new_shape1 = (4, self.bias_ih.shape[0] // 4)
            b_ihs = ops.split(ops.reshape(self.bias_ih, new_shape1), axis=0)
            b_hhs = ops.split(ops.reshape(self.bias_hh, new_shape1), axis=0)

            b_ih_i, b_ih_f, b_ih_g, b_ih_o = b_ihs[0], b_ihs[1], b_ihs[2], b_ihs[3]
            b_hh_i, b_hh_f, b_hh_g, b_hh_o = b_hhs[0], b_hhs[1], b_hhs[2], b_hhs[3]

        Z_i = func(Sigmoid(), w_ih_i, w_hh_i, b_ih_i, b_hh_i)
        Z_f = func(Sigmoid(), w_ih_f, w_hh_f, b_ih_f, b_hh_f)
        Z_g = func(ops.tanh, w_ih_g, w_hh_g, b_ih_g, b_hh_g)
        Z_o = func(Sigmoid(), w_ih_o, w_hh_o, b_ih_o, b_hh_o)

        C = Z_f * C + Z_i * Z_g
        H = Z_o * ops.tanh(C)
        return H, C
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.lstm_cells = [LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias, device, dtype)
                           for i in range(num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h:
            h_0, c_0 = h
            h_s = ops.split(h_0, 0)
            h_s = [item for item in h_s]
            c_s = ops.split(c_0, 0)
            c_s = [item for item in c_s]
        else:
            h_s = [None for _ in range(self.num_layers)]
            c_s = [None for _ in range(self.num_layers)]
        x_arr = ops.split(X, 0)
        output = []
        for x in x_arr:
            for i, layer in enumerate(self.lstm_cells):
                _h = (h_s[i], c_s[i]) if h_s[i] and c_s[i] else None
                h_s[i], c_s[i] = layer.forward(x if i == 0 else h_s[i - 1], _h)
            output.append(h_s[-1])
        output = ops.stack(output, 0)
        h_n = ops.stack(h_s, 0)
        c_n = ops.stack(c_s, 0)
        return output, (h_n, c_n)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.embedding_dim = embedding_dim
        self.weight = init.randn(*(num_embeddings, embedding_dim), device=device, dtype=dtype, requires_grad=True)
        self.weight = Parameter(self.weight)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        data = x.data
        seq_len, bs = data.shape
        data = ops.reshape(data, (seq_len * bs, ))
        data = data.numpy()
        result = []
        weights = ops.split(self.weight, 0)
        for i in range(seq_len * bs):
            _id = int(data[i])
            emb = weights[_id]
            result.append(emb)
        result = ops.stack(result, 0)
        result = ops.reshape(result, (seq_len, bs, self.embedding_dim))
        return result
        ### END YOUR SOLUTION
