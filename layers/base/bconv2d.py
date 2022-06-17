import torch
import torch.nn as nn


class BConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        with torch.no_grad():
            expected_power = nn.functional.conv2d(
                torch.ones_like(x),
                self.weight.square(),
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            ).sqrt()
        x = self._conv_forward(x, self.weight, None) / (expected_power + 1e-6)
        if self.bias is not None:
            x = x.add(self.bias.view(-1, 1, 1))
        return x
