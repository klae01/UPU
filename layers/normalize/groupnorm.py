import torch


class GroupNorm1D(torch.nn.Module):
    __constants__ = [
        "num_groups",
        "window_size",
        "num_heights",
        "num_channels",
        "eps",
        "affine",
    ]

    def __init__(
        self,
        num_groups: int,
        window_size: int,
        num_heights: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(GroupNorm1D, self).__init__()
        self.num_groups = num_groups
        self.window_size = window_size
        self.num_heights = num_heights
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(
                torch.empty([num_channels, num_heights], **factory_kwargs)
            )
            self.bias = torch.nn.Parameter(
                torch.empty([num_channels, num_heights], **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # reduction width
        var, mean = torch.var_mean(input, dim=-1, unbiased=False)

        # compute average by window size
        if self.window_size > 1:
            var = torch.nn.functional.avg_pool1d(
                var,
                self.window_size,
                stride=1,
                padding=self.window_size // 2,
                count_include_pad=False,
            )
            mean = torch.nn.functional.avg_pool1d(
                mean,
                self.window_size,
                stride=1,
                padding=self.window_size // 2,
                count_include_pad=False,
            )
            if self.window_size % 2 == 0:
                var = var[:, :, :-1]
                mean = mean[:, :, :-1]

        # reduction channel by group size
        origin_shape = input.shape
        convert_shape = (
            input.shape[0],
            self.num_groups,
            input.shape[1] // self.num_groups,
            input.shape[2],
            -1,
        )

        result = (
            input.view(convert_shape)
            - mean.view(convert_shape).mean(dim=2, keepdim=True)
        ) / torch.sqrt(var.view(convert_shape).mean(dim=2, keepdim=True) + self.eps)
        result = result.view(origin_shape)
        if self.affine:
            result.mul_(self.weight[None, :, :, None]).add_(
                self.bias[None, :, :, None]
            )
        return result

    def extra_repr(self) -> str:
        return (
            "{num_groups}, {window_size}, {num_heights}, {num_channels}, eps={eps}, "
            "affine={affine}".format(**self.__dict__)
        )
