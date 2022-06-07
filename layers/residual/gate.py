__all__ = ["Gate"]

import torch
import torch.nn as nn


class Gate_AG(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, x0, residual, rezero):
        ar = rezero.tanh()
        a0 = rezero.cosh().reciprocal()  # = (1 - ar**2)**0.5
        ctx.save_for_backward(x0, residual, rezero)
        return x0.mul(a0).add_(residual, alpha=ar)

    @staticmethod
    @torch.no_grad()
    def backward(ctx, x):
        x0, residual, r = ctx.saved_tensors

        ar = r.tanh()
        a0 = r.cosh().reciprocal()  # = (1 - ar**2)**0.5

        dar = a0.square()  # d/dx(tanh(x)) = sech^2(x)
        da0 = -r.tanh() * a0  # d/dx((1 - tanh^2(x))^0.5) = sinh(x) (-sech^2(x))

        v_shape = [x.size(0), -1]
        rezero_grad = (
            torch.einsum("Bf,Bf->B", x.view(*v_shape), x0.view(*v_shape)) * da0
            + torch.einsum("Bf,Bf->B", x.view(*v_shape), residual.view(*v_shape)) * dar
        )
        return a0 * x, ar * x, rezero_grad


class Gate(nn.Module):
    def __init__(self, init: float = 0.0):
        super().__init__()
        self.rezero = nn.Parameter(torch.tensor(init))

    def forward(self, x0, residual):
        return Gate_AG.apply(x0, residual, self.rezero)
