__all__ = ["Block"]

from typing import Union

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

from .. import Gate
from . import Computing_Conv2d


class GN(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        return nn.functional.group_norm(x, self.groups)


class Block(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        stride: int = 1,
        resample: nn.Module = None,
        radix: int = 2,
        cardinality: int = 2,
        dilation: int = 1,
        avd: bool = True,
        avd_first: bool = False,
        is_first: bool = False,
        avg_down: bool = True,
        local_window_size: Union[str, _size_2_t] = None,
        normalize_group_size: int = 32,
    ):
        """_summary_

        Args:
            in_features (int): _description_
            out_features (int): _description_
            kernel_size (int, optional): _description_. Defaults to 3.
            stride (int, optional): _description_. Defaults to 1.
            resample (nn.Module, optional): define downsampling module. Defaults to None.
            radix (int, optional): _description_. Defaults to 2.
            cardinality (int, optional): _description_. Defaults to 2.
            dilation (int, optional): _description_. Defaults to 1.
            avd (bool, optional): if avd and (stride > 1 or is_first), 3x3 avg pooling after conv2. it can replace conv2d stride. Defaults to True.
            avd_first (bool, optional): if True, avd perform before conv2. if False, svd perform after conv2. Defaults to False.
            is_first (bool, optional): if the block is first of residual blocks of each resolution (right after downsampling). Defaults to False.
            avg_down (bool, optional): if True, downsampling with avg pooling, if False, downsampling without avg pooling. Defaults to True.
            local_window_size (Union[str, _size_2_t], optional): _description_. Defaults to None.
            normalize_group_size (int, optional): _description_. Defaults to 32.
        """
        super().__init__()
        self.in_features = in_features
        self.channels = channels = out_features
        self.resample = resample
        self.radix = r = radix
        self.cardinality = k = cardinality
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first
        self.local_window_size = (
            _pair(local_window_size) if local_window_size is not None else None
        )
        inter_channels = max(channels // 8, 32)

        if in_features != out_features or stride != 1 and self.resample is None:
            if avg_down:
                self.resample = nn.Sequential(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False,
                    ),
                    GN(normalize_group_size),
                    nn.Conv2d(
                        in_features, out_features, kernel_size=1, stride=1, bias=False
                    ),
                    GN(normalize_group_size),
                )
            else:
                self.resample = nn.Sequential(
                    GN(normalize_group_size),
                    nn.Conv2d(
                        in_features,
                        out_features,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    GN(normalize_group_size),
                )
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.conv = nn.ModuleList(
            [
                nn.Conv2d(in_features, channels, 1, 1, groups=1),
                Computing_Conv2d(
                    channels,
                    channels * r,
                    kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) * dilation // 2,
                    dilation=dilation,
                    groups=r * k,
                ),
                nn.Conv2d(channels, inter_channels * k, 1, 1, groups=k),
                nn.Conv2d(inter_channels * k, channels * r, 1, 1, groups=k),
            ]
        )
        self.weight = nn.Parameter(torch.randn((channels, channels)) / channels**0.5)
        self.conv[-1].weight.data /= (channels / k / r) ** 0.5
        self.conv[-1].bias.data *= 0
        self.norm = [
            {"num_groups": normalize_group_size},
            {"num_groups": r * k * 4},
            {"num_groups": k},
            {"num_groups": k * 4},
            {"num_groups": normalize_group_size},
        ]
        self.gate = Gate()

    def forward(self, hidden_state):
        CONV = iter(self.conv)
        NORM = iter(self.norm)
        B, C, H, W = hidden_state.size()
        K = self.cardinality
        R = self.radix

        x = hidden_state
        x = nn.functional.group_norm(x, **next(NORM))  # [B, C, H, W]
        x = next(CONV)(x)  # [B, C/k/r, H, W] * r * k
        x = nn.functional.silu(x)

        if self.avd and self.avd_first:
            x = self.avd_layer(x)

        x = nn.functional.group_norm(x, **next(NORM))
        x = next(CONV)(x)  # [B, C/k, H, W] * r * k
        x = nn.functional.silu(x)

        if self.avd and not self.avd_first:
            x = self.avd_layer(x)

        x = nn.functional.group_norm(x, **next(NORM))
        H, W = x.shape[-2:]
        WS = self.local_window_size or [H, W]
        y = x

        x = x.view(
            B, -1, R, H // WS[0], WS[0], W // WS[1], WS[1]
        )  # [B, C/k, R, H//WS, WS, W//WS, WS] * k
        x = x.mean(dim=(-1, -3))  # [B, C/k, R, H//WS, W//WS] * k
        x = x.amax(dim=2)  # [B, C/k, H//WS, W//WS] * k
        x = next(CONV)(x)  # [B, I, H//WS, W//WS] * k
        x = nn.functional.silu(x)
        x = nn.functional.group_norm(x, **next(NORM))
        x = next(CONV)(x)  # [B, C/k, H//WS, W//WS] * r * k
        x = x.view(B, -1, R, H // WS[0], W // WS[1])  # [B, C/k, R, H//WS, W//WS] * k
        x = x.sigmoid() if R == 1 else x.softmax(dim=2)
        x = torch.einsum(
            "BCRHhWw,BCRHW,Cc->BcHhWw",
            y.view(B, -1, R, H // WS[0], WS[0], W // WS[1], WS[1]),
            x,
            self.weight,
        )
        x = nn.functional.group_norm(x, **next(NORM))

        identity = self.resample(hidden_state) if self.resample else hidden_state
        return self.gate(identity, x.view(B, -1, H, W))
