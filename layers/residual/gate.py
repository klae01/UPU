__all__ = ["Gate"]

from typing import Union

import torch
import torch.nn as nn
from torch import Tensor


class Gate_AG(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, x0: Tensor, residual: Tensor, rezero: Tensor):
        assert max(len(x0.shape), len(residual.shape)) > len(rezero.shape)
        ar = rezero.tanh()
        a0 = rezero.cosh().reciprocal()  # = (1 - ar**2)**0.5 = sech(x)
        ctx.save_for_backward(x0, residual, a0, ar)
        return x0 * a0 + residual * ar

    @staticmethod
    @torch.no_grad()
    def backward(ctx, x: Tensor):
        def grad_over_batch(x, grad):
            # x.shape = [B, *]
            # grad.shape = rezero.shape
            broad_casts = len(x.shape) - len(grad.shape)
            remove_dim = tuple(range(1, broad_casts))
            reduce_dim = tuple(
                broad_casts + i for i, I in enumerate(grad.shape) if I == 1
            )
            dim = remove_dim + reduce_dim
            g = x.sum(dim) if dim else x
            if grad.shape:
                g = g.view(x.size(0), *grad.shape)
            return g * grad

        x0, residual, a0, ar = ctx.saved_tensors

        dar = a0.square()  # d/dx(tanh(x)) = sech^2(x)
        da0 = -ar * a0  # d/dx((1 - tanh^2(x))^0.5) = -tanh(x) sech(x)

        rezero_grad = grad_over_batch(x * x0, da0) + grad_over_batch(x * residual, dar)
        return a0 * x, ar * x, rezero_grad


class Gate(nn.Module):
    def __init__(
        self,
        init: Union[float, Tensor] = 0.0,
    ):
        super().__init__()
        if isinstance(init, Tensor):
            self.rezero = nn.Parameter(init)
        else:
            self.rezero = nn.Parameter(torch.tensor(init))

    def forward(self, x0, residual):
        return Gate_AG.apply(x0, residual, self.rezero)
