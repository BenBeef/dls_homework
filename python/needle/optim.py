"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.u = self.u if len(self.u) > 0 or self.momentum == 0 else {w: ndl.zeros(*w.shape) for w in self.params}
        for w in self.params:
            if self.momentum > 0:
                self.u[w].data = self.momentum * self.u[w].data + (1 - self.momentum) * (w.grad + self.weight_decay * w.data)
                w.data = w.data + (-self.lr) * self.u[w].data
            else:
                w.data = (1 - self.lr * self.weight_decay) * w.data + (-self.lr) * w.grad
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.m = self.m if self.m else {w: ndl.zeros(*w.shape) for w in self.params}
        self.v = self.v if self.v else {w: ndl.zeros(*w.shape) for w in self.params}
        self.t += 1
        for w in self.params:
            grad_decay = w.grad + self.weight_decay * w.data
            self.m[w].data = self.beta1 * self.m[w].data + (1 - self.beta1) * grad_decay                  # ↓
            self.v[w].data = self.beta2 * self.v[w].data + (1 - self.beta2) * (grad_decay * grad_decay)   # ↑
            u_t_1 = self.m[w].data / (1 - self.beta1 ** self.t)
            v_t_1 = self.v[w].data / (1 - self.beta2 ** self.t)
            v_t_1_sqrt = v_t_1.data ** 0.5
            w.data = w.data + (-self.lr) * u_t_1.data / (v_t_1_sqrt.data + self.eps)
        ### END YOUR SOLUTION
