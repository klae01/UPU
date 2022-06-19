import torch
import torch.nn as nn


class BConv2d_base(nn.Conv2d):
    @classmethod
    def reweight_mapping(
        cls, x, weight, kernel_size, stride, padding, dilation, groups
    ):
        B, C, H, W = x.shape
        return torch.conv2d(
            torch.ones((1, C, H, W), device=x.device, dtype=x.dtype),
            weight.square(),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        ).sqrt()

    def forward(self, x):
        if self.padding_mode != "zeros":
            return super().forward(x)
        expected_power = type(self).reweight_mapping(
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


class BConv2d_v1(BConv2d_base):
    @classmethod
    def reweight_mapping(
        cls, x, weight, kernel_size, stride, padding, dilation, groups
    ):
        B, C, H, W = x.shape

        WS = weight.square()  # [O, I / G, Kh, Kw]
        WS = WS.sum(dim=1)  # [O, Kh, Kw]

        # Height processing
        SQ = WS.sum(dim=2).unsqueeze(1)  # [O, 1, Kh]
        S, P, D = stride[0], padding[0], dilation[0]
        op_kwargs = {
            "device": SQ.device,
            "dtype": SQ.dtype,
        }
        H_cutoffs = torch.conv1d(
            torch.ones((1, 1, H + P * 2), **op_kwargs), SQ, stride=S, dilation=D
        ) - torch.conv1d(
            torch.ones((1, 1, H), **op_kwargs), SQ, stride=S, padding=P, dilation=D
        )

        # Width processing
        SQ = WS.sum(dim=1).unsqueeze(1)  # [O, 1, Kw]
        S, P, D = stride[1], padding[1], dilation[1]
        W_cutoffs = torch.conv1d(
            torch.ones((1, 1, W + P * 2), **op_kwargs), SQ, stride=S, dilation=D
        ) - torch.conv1d(
            torch.ones((1, 1, W), **op_kwargs), SQ, stride=S, padding=P, dilation=D
        )

        output = WS.sum((1, 2), True) - H_cutoffs.unsqueeze(3) - W_cutoffs.unsqueeze(2)

        valid_ws = []
        for s, K, S, P, D in zip([H, W], kernel_size, stride, padding, dilation):
            eff = 1 + D * (K - 1)
            field = s + 2 * P
            min_req_field = (field - eff) // S - (s - eff) // S + S + eff
            margin = (S + s % S - min_req_field % S) % S
            req_field = min_req_field + margin
            valid_ws.append(req_field)

        ws_paste = [s > vs for s, vs in zip([H, W], valid_ws)]
        rws = [min(s, vs) for s, vs in zip([H, W], valid_ws)]
        edges = torch.conv2d(
            torch.ones((1, 1, *rws), **op_kwargs),
            weight=WS.unsqueeze(1),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        Eh, Ew = [
            ((s + 2 * P - D * (K - 1) - 1) // S - (s - D * (K - 1) - 1) // S) // 2 + 1
            for s, K, S, P, D in zip([H, W], kernel_size, stride, padding, dilation)
        ]
        if all(ws_paste):
            output[..., :Eh, :Ew] = edges[..., :Eh, :Ew]
            output[..., :Eh, -Ew:] = edges[..., :Eh, -Ew:]
            output[..., -Eh:, :Ew] = edges[..., -Eh:, :Ew]
            output[..., -Eh:, -Ew:] = edges[..., -Eh:, -Ew:]
        elif not ws_paste[0]:
            output[..., :, -Ew:] = edges[..., :, -Ew:]
            output[..., :, :Ew] = edges[..., :, :Ew]
        elif not ws_paste[1]:
            output[..., :Eh, :] = edges[..., :Eh, :]
            output[..., -Eh:, :] = edges[..., -Eh:, :]
        else:
            output[...] = edges[...]

        return torch.relu(output).sqrt()


BConv2d = BConv2d_v1
