from functools import partial
from typing import Optional, Union

import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor, nn
from torch.nn.common_types import _size_1_t


def print_t(name: str, tensor: Tensor, precision: int = 10) -> None:
    values = [
        f"{value:.{precision}f}" for value in tensor.detach().reshape(-1).tolist()
    ]
    print(f"{name}: [{', '.join(values)}]")


class IntegerQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, width: int, frac_width: int, is_signed: bool = True):
        return _integer_quantize(
            x, width=width, frac_width=frac_width, is_signed=is_signed
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def integer_quantizer(
    x: Tensor | ndarray, width: int, frac_width: int, is_signed: bool = True
):
    """
    - Do linear quantization to input according to a scale and number of bits
    - Note that `bias` can be negative or larger than `bits`

    ---
    - forward: convert IEEE FP32/64 to fixed-point
    - backward: STE

    ---
    width: the bit width of the fixed-point number
    frac_width: the number of fractional bits. Note that `bias` can be negative or larger than `bits`

    ---
    For example: 0b101 . 00111, bits = 8, bias = 5

    """
    return IntegerQuantize.apply(x, width, frac_width, is_signed)


def _integer_quantize(
    x: Tensor | ndarray, width: int, frac_width: int = None, is_signed: bool = True
):
    """
    - Do linear quantization to input according to a scale and number of bits
    - Note that `bias` can be negative or larger than `bits`

    ---
    - forward: convert IEEE FP32/64 to fixed-point
    - backward: STE

    ---
    width: the bit width of the fixed-point number
    frac_width: the number of fractional bits. Note that `bias` can be negative or larger than `bits`

    ---
    For example: 0b101 . 00111, bits = 8, bias = 5

    """
    if frac_width is None:
        frac_width = width // 2

    if is_signed:
        int_min = -(2 ** (width - 1))
        int_max = 2 ** (width - 1) - 1
    else:
        int_min = 0
        int_max = 2**width - 1
    # thresh = 2 ** (width - 1)
    scale = 2**frac_width

    if isinstance(x, (Tensor, ndarray)):
        return x.mul(scale).round().clamp(int_min, int_max).div(scale)
    elif isinstance(x, int):
        return x
    else:
        return (x * scale).round().clamp(int_min, int_max) / (scale)


class _Conv1dBase(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t | str = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.bypass = False
        self.w_quantizer = None
        self.x_quantizer = None
        self.b_quantizer = None
        self.pruning_masks = None

    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return self._conv_forward(x, self.weight, self.bias)
        x = self.x_quantizer(x)
        w = self.w_quantizer(self.weight)
        bias = self.b_quantizer(self.bias) if self.bias is not None else None
        # WARNING: this may have been simplified, we are assuming here the accumulation is lossless!
        # The addition size is in_channels * K * K
        return self._conv_forward(x, w, bias)


class Conv1dInteger(_Conv1dBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        # establish quantizers
        w_width, w_frac_width = config["weight_width"], config["weight_frac_width"]
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        b_width, b_frac_width = config["bias_width"], config["bias_frac_width"]
        self.w_quantizer = partial(
            _integer_quantize, width=w_width, frac_width=w_frac_width
        )
        self.x_quantizer = partial(
            _integer_quantize, width=x_width, frac_width=x_frac_width
        )
        self.b_quantizer = partial(
            _integer_quantize, width=b_width, frac_width=b_frac_width
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class Conv1q(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class LinearInteger(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        w_width, w_frac_width = config["weight_width"], config["weight_frac_width"]
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        b_width, b_frac_width = config["bias_width"], config["bias_frac_width"]
        self.w_quantizer = partial(
            _integer_quantize, width=w_width, frac_width=w_frac_width
        )
        self.x_quantizer = partial(
            _integer_quantize, width=x_width, frac_width=x_frac_width
        )
        self.b_quantizer = partial(
            _integer_quantize, width=b_width, frac_width=b_frac_width
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.linear(
                x,
                self.weight.to(x.dtype),
                None if self.bias is None else self.bias.to(x.dtype),
            )
        x = self.x_quantizer(x)
        w = self.w_quantizer(self.weight).to(x.dtype)
        bias = (
            self.b_quantizer(self.bias).to(x.dtype) if self.bias is not None else None
        )
        return F.linear(x, w, bias)


def simple_quantise_test():
    torch.set_printoptions(precision=10, sci_mode=False)

    t = torch.tensor(
        [0.10, 0.35, 0.60, 0.90, 1.10, 1.40, 1.90, 2.20, 2.60, 2.90]
    ).float()
    print_t("input", t)
    width = 12
    frac_width = 4
    config = {
        "name": "integer",
        "data_in_width": width,
        "data_in_frac_width": frac_width,
        "weight_width": width,
        "weight_frac_width": frac_width,
        "bias_width": width,
        "bias_frac_width": frac_width,
    }
    t = t.unsqueeze(0).unsqueeze(0)

    conv1 = Conv1d(1, 1, kernel_size=3, padding=1)
    convq = Conv1dInteger(1, 1, kernel_size=3, padding=1, config=config)
    conv1.weight.data = torch.tensor([[[0.30, 0.05, -0.70]]])
    conv1.bias.data = torch.tensor([0.0])
    convq.weight.data = torch.tensor([[[0.30, 0.05, -0.70]]])
    convq.bias.data = torch.tensor([0.0])
    print_t("kernel", conv1.weight)
    print_t("bias", conv1.bias)
    print_t("input q", convq.x_quantizer(t))
    print_t("kernel q", convq.w_quantizer(convq.weight))

    # t = conv1(t)
    print_t("output", conv1(t))
    print_t("output q", convq(t))


if __name__ == "__main__":
    simple_quantise_test()
