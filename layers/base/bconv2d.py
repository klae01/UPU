import torch
import torch.nn as nn


class BConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.padding_mode != "zeros":
            return super().forward(x)
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


class BConv2d_fast(BConv2d):
    @staticmethod
    def reweight_mapping(x, weight, kernel_size, stride, padding, dilation, groups):
        B, C, H, W = x.shape

        WS = weight.square()  # [O, I / G, Kh, Kw]
        WS = WS.sum(dim=1)  # [O, Kh, Kw]

        # Height processing
        SQ = WS.sum(dim=2).unsqueeze(1)  # [O, 1, Kh]
        S, P, D = stride[0], padding[0], dilation[0]
        H_cutoffs = torch.conv1d(
            torch.ones((1, 1, H + P * 2)), weight=SQ, stride=S, dilation=D
        ) - torch.conv1d(
            torch.ones((1, 1, H)), weight=SQ, stride=S, padding=P, dilation=D
        )

        SQ = WS.sum(dim=1).unsqueeze(1)  # [O, 1, Kw]
        S, P, D = stride[1], padding[1], dilation[1]
        W_cutoffs = torch.conv1d(
            torch.ones((1, 1, W + P * 2)), weight=SQ, stride=S, dilation=D
        ) - torch.conv1d(
            torch.ones((1, 1, W)), weight=SQ, stride=S, padding=P, dilation=D
        )

        # output = torch.empty((WS.size(0), H_cutoffs.size(2), W_cutoffs.size(2)))
        print(WS.shape, H_cutoffs.shape, W_cutoffs.shape)
        output = WS.sum((1, 2), True) - H_cutoffs.unsqueeze(3) - W_cutoffs.unsqueeze(2)

        valid_ws = [
            ((s + P * 2) % (S * D)) + K * S * D
            for s, K, S, P, D in zip([H, W], kernel_size, stride, padding, dilation)
        ]
        print(valid_ws)
        print(padding)
        edges = torch.conv2d(
            torch.ones((1, 1, *valid_ws)),
            weight=WS.unsqueeze(1),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        print(edges.shape)
        print(output.shape)

        Eh, Ew = [P for P, K in zip(padding, kernel_size)]
        output[..., :Eh, :Ew] = edges[..., :Eh, :Ew]
        output[..., :Eh, -Ew:] = edges[..., :Eh, -Ew:]
        output[..., -Eh:, :Ew] = edges[..., -Eh:, :Ew]
        output[..., -Eh:, -Ew:] = edges[..., -Eh:, -Ew:]

        return torch.nn.functional.relu(output, inplace=True).sqrt()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.padding_mode != "zeros":
            return super().forward(x)
        with torch.no_grad():
            expected_power = BConv2d_fast.reweight_mapping(
                x,
                self.weight,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        x = self._conv_forward(x, self.weight, None) / (expected_power + 1e-6)
        if self.bias is not None:
            x = x.add(self.bias.view(-1, 1, 1))
        return x
