import itertools
import numpy as np
import torch


def generall_fully_denoise(x):
    r_data = x.copy()
    cases = {}
    for I in itertools.product([True], *([[True, False]] * len(r_data.shape[1:]))):
        if not all(I):
            X = cases[I] = np.median(
                r_data,
                axis=tuple(i for i, dep in enumerate(I[1:], 1) if not dep),
                keepdims=True,
            )
            r_data -= X

    for _ in range(5):
        denom = 1 / np.linalg.norm(r_data, ord=2, axis=0, keepdims=True)
        num = r_data * denom

        ops = {}
        for K in cases.keys():
            axis = tuple(i for i, dep in enumerate(K) if not dep)
            ops[K] = num.mean(axis=axis, keepdims=True) / denom.mean(
                axis=axis, keepdims=True
            )

        for K in cases.keys():
            target_sgn = sum(K) % 2
            delta = sum(
                ops[I] if sum(I) % 2 == target_sgn else -ops[I]
                for I in itertools.product(
                    *[[[False], [True, False]][dep] for dep in K]
                )
                if I in ops
            )
            cases[K] += delta
            r_data -= delta

    return x - sum(cases.values())


@torch.no_grad()
def denoise_2d(x):
    if type(x) == torch.Tensor:
        lib = torch
        axis_name = "dim"
        r_data = x - x.median(dim=-1, keepdims=True)[0]
        r_data -= r_data.median(dim=-2, keepdims=True)[0]
    else:
        lib = np
        axis_name = "axis"
        r_data = x.copy()
        r_data -= lib.median(r_data, axis=-1, keepdims=True)
        r_data -= lib.median(r_data, axis=-2, keepdims=True)

    for _ in range(5):
        denom = 1 / (1e-9 + lib.linalg.norm(r_data, ord=2, **{axis_name: 0}))
        num = r_data * denom
        for i in [-1, -2]:
            A = num.mean(**{axis_name: i}, keepdims=True)
            A /= denom.mean(**{axis_name: i}, keepdims=True)
            r_data -= A
    return r_data
